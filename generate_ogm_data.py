import numpy as np
import torch
import uuid
import os
from PIL import Image
from argparse import ArgumentParser
from sparse_to_dense_reward.avant_goal_env import AvantGoalEnv


def save_bw_image(array, file_path):
    # Ensure the array contains integers in the range [0, 255]
    assert array.dtype == np.uint8, "Array should be of type np.uint8"
    assert (array >= 0).all() and (array <= 255).all(), "Array values should be in the range [0, 255]"

    # Create a Pillow Image and save it
    img = Image.fromarray(array, mode='L')  # 'L' mode is for (8-bit pixels, black and white)
    img.save(file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", default="/home/aleksi/ogm_images")
    args = parser.parse_args()

    # Initialize environment and actor
    env = AvantGoalEnv(num_envs=100, dt=0.1, time_limit_s=30, device='cpu', num_obstacles=5)

    for i in range(1000):
        if i % 50 == 0:
            print(f"iteration {i}")

        obs = env.reset()
        
        occupancy_grid = obs["occupancy_grid"]
        occupancy_grids_normalized = (occupancy_grid * 255).astype(np.uint8)
        print(occupancy_grids_normalized.shape)

        for j in range(len(occupancy_grids_normalized)):
            save_bw_image(occupancy_grids_normalized[j], os.path.join(args.dataset_dir, str(uuid.uuid4()) + ".png"))