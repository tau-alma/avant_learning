import pygame
import torch
import numpy as np
from pygame.math import Vector2
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from typing import Dict


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
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = 9 # total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        achieved = self.extractors["achieved_goal"](observations["achieved_goal"])
        desired = self.extractors["desired_goal"](observations["desired_goal"])
        obs = self.extractors["observation"](observations["observation"])

        pos_residual = desired[:, :2] - achieved[:, :2]
        achieved_hdg_data = achieved[:, 2:4]
        desired_hdg_data = desired[:, 2:4]

        encoded_tensor_list = [pos_residual, achieved_hdg_data, desired_hdg_data, obs]

        return torch.cat(encoded_tensor_list, dim=1)
    


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

def create_occupancy_grids(obstacles, cell_size, axis_limit):
    device = obstacles.device
    
    # Calculate grid size based on the axis limits and cell size
    grid_size = int((2 * axis_limit) / cell_size)

    # The input is expected to be in the range [-axis_limit, axis_limit]
    # We transform it to [0, 1] range for computation
    normalized_obstacles = (obstacles[:, :, :2] + axis_limit) / (2 * axis_limit)
    normalized_radii = obstacles[:, :, 2] / (2 * axis_limit)

    N, K, _ = obstacles.shape
    occupancy_grids = torch.zeros((N, grid_size, grid_size), dtype=torch.bool, device=device)

    # Generate coordinates for the centers of each grid cell in the normalized range
    coords = torch.linspace(0, 1, grid_size, device=device)
    x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')

    # Compute the occupancy grids
    for i in range(N):
        for j in range(K):
            # Obstacle center and radius in normalized coordinates
            x, y = normalized_obstacles[i, j]
            r = normalized_radii[i, j]
            # Compute squared distance from each grid cell center to the obstacle center
            distance_squared = (x_grid - x)**2 + (y_grid - y)**2

            # Update the occupancy grid: mark as occupied if within the normalized radius
            occupancy_grids[i] |= (distance_squared <= r**2)

    return occupancy_grids.to('cpu').numpy().astype(int)