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
        self._features_dim = 9

    def forward(self, observations: TensorDict) -> torch.Tensor:
        achieved = self.extractors["achieved_goal"](observations["achieved_goal"])
        desired = self.extractors["desired_goal"](observations["desired_goal"])
        obs = self.extractors["observation"](observations["observation"])

        pos_residual = desired[:, :2] - achieved[:, :2]
        achieved_hdg_data = achieved[:, 2:4]
        desired_hdg_data = desired[:, 2:4]

        encoded_tensor_list = [pos_residual, achieved_hdg_data, desired_hdg_data, obs]
        
        try:
            retval = torch.cat(encoded_tensor_list, dim=1)
        except:
            for t in encoded_tensor_list:
                print(t.shape)
            raise RuntimeError("noped out")
        
        return retval


def rotate_image(image, angle, p_rot):
   # Create a larger surface to accommodate the rotation
    w, h = image.get_size()
    large_surface = pygame.Surface((2*w, 2*h), pygame.SRCALPHA)
    
    # Calculate the position to blit the image on the large surface
    blit_pos = (large_surface.get_width() // 2 - p_rot[0], large_surface.get_height() // 2 - p_rot[1])
    
    # Blit the original image to the larger surface
    large_surface.blit(image, blit_pos)
    
    # Rotate the large surface
    rotated_large_surface = pygame.transform.rotate(large_surface, -angle)
    
    # The new position for p_place is relative to the center of the larger surface
    # after rotation, which is unchanged and equivalent to p_rot
    center_of_rotated = rotated_large_surface.get_rect().center
    
    return rotated_large_surface, center_of_rotated