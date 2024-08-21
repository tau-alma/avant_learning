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

        encoded_tensor_list = [pos_residual, achieved_hdg_data, desired_hdg_data, achieved[:, 4:7]] #, obs]
        encoded_tensor_list = torch.cat(encoded_tensor_list, dim=1)

        goal_delta = desired - achieved

        return encoded_tensor_list, goal_delta


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


class AvantRenderer:
    RENDER_RESOLUTION = 1024

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BROWN = (150, 75, 0)

    HALF_MACHINE_WIDTH = 0.6
    HALF_MACHINE_LENGTH = 1.3
    MACHINE_RADIUS = 0.8

    def __init__(self, position_bound):
        self.position_bound = position_bound
        pos_to_pixel_scaler = self.RENDER_RESOLUTION / (4*position_bound)

        pygame.init()
        self.screen = pygame.Surface((self.RENDER_RESOLUTION, self.RENDER_RESOLUTION))
        # Drawing avant:
        avant_image_pixel_scaler = np.mean([452 / 1.29, 428 / 1.29])
        avant_scale_factor = pos_to_pixel_scaler / avant_image_pixel_scaler
        self.front_center_offset = np.array([215, 430]) * avant_scale_factor
        self.rear_center_offset = np.array([226, 0]) * avant_scale_factor
        front_image = pygame.image.load('sparse_to_dense_reward/front.png')
        rear_image = pygame.image.load('sparse_to_dense_reward/rear.png')
        self.front_image = pygame.transform.scale(front_image, (avant_scale_factor*front_image.get_width(), avant_scale_factor*front_image.get_height()))
        self.rear_image = pygame.transform.scale(rear_image, (avant_scale_factor*rear_image.get_width(), avant_scale_factor*rear_image.get_height()))
        front_gray_image = pygame.image.load('sparse_to_dense_reward/front_gray.png')
        rear_gray_image = pygame.image.load('sparse_to_dense_reward/rear_gray.png')
        self.front_gray_image = pygame.transform.scale(front_gray_image, (avant_scale_factor*front_gray_image.get_width(), avant_scale_factor*front_gray_image.get_height()))
        self.rear_gray_image = pygame.transform.scale(rear_gray_image, (avant_scale_factor*rear_gray_image.get_width(), avant_scale_factor*rear_gray_image.get_height()))


    def render(self, state: np.ndarray, goal: np.ndarray, horizon: np.ndarray = np.empty([]), obstacles: np.ndarray = np.empty([])):
        x_f = state[0]
        y_f = state[1]
        theta_f = state[2]
        beta = state[3]
        x_goal = goal[0]
        y_goal = goal[1]
        theta_goal = goal[2]  

        center = self.RENDER_RESOLUTION // 2
        pos_to_pixel_scaler = self.RENDER_RESOLUTION / (4*self.position_bound)

        surf = pygame.Surface((self.RENDER_RESOLUTION, self.RENDER_RESOLUTION))
        surf.fill(self.WHITE)

        alpha_surf = pygame.Surface((self.RENDER_RESOLUTION, self.RENDER_RESOLUTION))
        alpha_surf.set_alpha(72)
        alpha_surf.fill(self.WHITE)

        # Visualizing avant front collision bound:
        pygame.draw.circle(alpha_surf, self.RED, 
                           center=(center + pos_to_pixel_scaler*(x_f), center + pos_to_pixel_scaler*(y_f)),
                           radius=pos_to_pixel_scaler*self.MACHINE_RADIUS)
        
        # Visualizing avant rear collision bound:
        x_off = pos_to_pixel_scaler*(
            x_f 
            - np.cos(theta_f) * np.cos(beta/4)*self.HALF_MACHINE_LENGTH/2 
            - np.sin(theta_f) * np.sin(beta/4)*self.HALF_MACHINE_LENGTH/2 
            - np.cos(-theta_f - beta) * self.HALF_MACHINE_LENGTH/2 
        )
        y_off = pos_to_pixel_scaler*(
            y_f 
            - np.sin(theta_f) * np.cos(beta/4)*self.HALF_MACHINE_LENGTH/2 
            + np.cos(theta_f) * np.sin(beta/4)*self.HALF_MACHINE_LENGTH/2 
            + np.sin(-theta_f - beta) * self.HALF_MACHINE_LENGTH/2 
        )
        pygame.draw.circle(alpha_surf, self.RED, 
                           center=(center + x_off, center + y_off),
                           radius=pos_to_pixel_scaler*self.MACHINE_RADIUS)

        # Black magic to shift the avant frame images correctly given the kinematics:
        x_off = pos_to_pixel_scaler*(
            x_goal - np.cos(theta_goal) * self.HALF_MACHINE_LENGTH/2
        )
        y_off = pos_to_pixel_scaler*(
            y_goal - np.sin(theta_goal) * self.HALF_MACHINE_LENGTH/2 
        )
        rotated_image, final_position = rotate_image(self.rear_gray_image, np.rad2deg(theta_goal + np.pi/2), self.rear_center_offset.tolist())
        surf.blit(rotated_image, 
                       (center + x_off - final_position[0], 
                        center + y_off - final_position[1]))
        rotated_image, final_position = rotate_image(self.front_gray_image, np.rad2deg(theta_goal + np.pi/2), self.front_center_offset.tolist())
        surf.blit(rotated_image, 
                       (center + x_off - final_position[0], 
                        center + y_off - final_position[1]))
        pygame.draw.circle(surf, self.RED, 
                           center=(center + pos_to_pixel_scaler*(x_goal), center + pos_to_pixel_scaler*(y_goal)),
                           radius=5)

        # During evaluation, draw the obstacles, if provided:
        if len(obstacles):
            for j in range(len(obstacles)):
                x_o, y_o, r_o = obstacles[j]
                pygame.draw.circle(draw_surf, self.BLACK,
                                center=(center + pos_to_pixel_scaler*(x_o), center + pos_to_pixel_scaler*(y_o)),
                                radius=pos_to_pixel_scaler*r_o)

        # During evaluation, draw the prediction horizon, if provided:
        if len(horizon):
            data = np.r_[np.c_[x_f, y_f, theta_f, beta],
                         horizon]
        else:
            data = np.c_[x_f, y_f, theta_f, beta]
        for j in range(len(data)):
            x_f_val, y_f_val, theta_f_val, beta_val = data[j]
            if j == 0:
                draw_surf = surf
            else:
                draw_surf = alpha_surf
            # Black magic to shift the image correctly given the kinematics:
            x_off = pos_to_pixel_scaler*(
                x_f_val - np.cos(theta_f_val) * np.cos(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
                - np.sin(theta_f_val) * np.sin(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
            )
            y_off = pos_to_pixel_scaler*(
                y_f_val - np.sin(theta_f_val) * np.cos(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
                + np.cos(theta_f_val) * np.sin(beta_val/4)*self.HALF_MACHINE_LENGTH/2 
            )
            rotated_image, final_position = rotate_image(self.rear_image, np.rad2deg(theta_f_val + beta_val + np.pi/2), self.rear_center_offset.tolist())
            draw_surf.blit(rotated_image, 
                        (center + x_off - final_position[0], 
                            center + y_off - final_position[1]))
            rotated_image, final_position = rotate_image(self.front_image, np.rad2deg(theta_f_val + np.pi/2), self.front_center_offset.tolist())
            draw_surf.blit(rotated_image, 
                        (center + x_off - final_position[0], 
                            center + y_off - final_position[1]))
            pygame.draw.circle(draw_surf, self.RED,
                               center=(center + pos_to_pixel_scaler*(x_f_val), center + pos_to_pixel_scaler*(y_f_val)),
                               radius=5)
        
        self.screen.blits([
            (surf, (0, 0)),
            (alpha_surf, (0, 0))
        ])
        buffer = pygame.transform.flip(self.screen, True, False)
        buffer = pygame.surfarray.array3d(buffer)
        return buffer # SHOULD BE: (512, 1024, 3), by concatting buffer and bw frame converted to color bw