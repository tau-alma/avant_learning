import pygame
import torch
import numpy as np
from abc import ABC, abstractmethod
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict

class GoalEnv(ABC):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError
    

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        normalized_image: bool = True,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                raise ValueError("Trying to extract image")
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = 7

    def forward(self, observations: TensorDict) -> torch.Tensor:
        achieved = self.extractors["achieved_goal"](observations["achieved_goal"])
        desired = self.extractors["desired_goal"](observations["desired_goal"])
        obs = self.extractors["observation"](observations["observation"])

        pos_residual = desired[:, :2] - achieved[:, :2]
        achieved_hdg_data = achieved[:, 2:4]
        desired_hdg_data = desired[:, 2:4]
        sin_achieved, cos_achieved = achieved_hdg_data[:, 0], achieved_hdg_data[:, 1]
        sin_desired, cos_desired = desired_hdg_data[:, 0], desired_hdg_data[:, 1]

        # Compute the position error in the "local" / wheel loader longitudinal and lateral axis
        rotation_matrix = torch.stack([
            cos_achieved, sin_achieved, 
            -sin_achieved, cos_achieved
        ], dim=1).reshape(-1, 2, 2)
        local_pos_residual = torch.bmm(rotation_matrix, pos_residual.unsqueeze(-1)).squeeze(-1)
        # Recover the heading error from sin/cos:
        hdg_error = torch.atan2(
            sin_desired * cos_achieved - cos_desired * sin_achieved,
            cos_desired * cos_achieved + sin_desired * sin_achieved
        )

        # longitudinal error
        # lateral error
        # sin(heading error)
        # cos(heading error)
        # obs
        encoded_tensor_list = [
            local_pos_residual, 
            torch.sin(hdg_error.unsqueeze(1)),
            torch.cos(hdg_error.unsqueeze(1)),
            obs
        ]
        encoded_tensor_list = torch.cat(encoded_tensor_list, dim=1)

        goal_delta = desired - achieved

        return encoded_tensor_list, goal_delta


def compute_gradient_penalty(model, obs_dict, action, lambda_gp=1e-4):
    with torch.no_grad():
        features, delta_goal = model.extract_features(obs_dict, model.features_extractor)

    noise_scale_feature = features.mean(axis=0).unsqueeze(0)
    scale_action = action.mean(axis=0).unsqueeze(0)
    # Perturb the current state and actions slightly to ensure high gradients are penalized in nearby states as well
    features_uniform = features + 0.05 * noise_scale_feature * (torch.rand_like(features) * 2 - 1)
    actions_uniform = action + 0.05 * scale_action * (torch.rand_like(action) * 2 - 1)
    features_uniform.requires_grad_(True)
    actions_uniform.requires_grad_(True)

    qvalue_input = torch.cat([features_uniform, actions_uniform], dim=1)
    model_outputs = [q_net(qvalue_input)**2 for q_net in model.q_networks]

    gradient_penalty = 0
    for model_output in model_outputs:
        gradients = torch.autograd.grad(
            outputs=model_output,
            inputs=[features_uniform, actions_uniform],
            grad_outputs=torch.ones_like(model_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )

        gradients = torch.cat([gradients[0].view(gradients[0].size(0), -1),
                               gradients[1].view(gradients[1].size(0), -1)], dim=1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # gradient_penalty += lambda_gp * ((gradient_norm - 1) ** 2).mean()
        gradient_penalty += lambda_gp * (torch.relu(gradient_norm - 1)**2).mean()

    return gradient_penalty / len(model_outputs)