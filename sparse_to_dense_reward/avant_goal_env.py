import time
import torch
import gymnasium
import numpy as np
import pygame
import numpy as np
from multiprocessing import Pool
from abc import ABC, abstractmethod
from typing import List, Tuple
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn
from sparse_to_dense_reward.avant_dynamics import AvantDynamics
from sparse_to_dense_reward.utils import rotate_image, create_occupancy_grids, GoalEnv, precompute_lidar_direction_vectors, simulate_lidar
from rtree import index

POSITION_BOUND = 5
OBSTACLE_MIN_RADIUS = 0.5
OBSTACLE_MAX_RADIUS = 1.5

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BROWN = (150, 75, 0)

PALLET_OFFSET = 1.0
HALF_PALLET_WIDTH = 0.4
PALLET_LENGTH = 1.2
PALET_RADIUS = PALLET_LENGTH/2 + 0.1

HALF_MACHINE_WIDTH = 0.6
HALF_MACHINE_LENGTH = 1.3
MACHINE_RADIUS = 0.8


class AvantGoalEnv(VecEnv, GoalEnv):
    RENDER_RESOLUTION = 1280

    def __init__(self, num_envs: int, dt: float, time_limit_s: float, device: str, num_obstacles: int = 0, encoder: torch.nn.Module = None, eval=False):
        self.num_envs = num_envs
        self.time_limit_s = time_limit_s
        self.device = device
        self.num_obstacles = num_obstacles
        self.encoder = encoder
        self.eval = eval
        self.dynamics = AvantDynamics(dt=dt, device=device, eval=eval)

        n_actions = len(self.dynamics.control_scalers.cpu().numpy())
        self.single_action_space = spaces.Box(
            low=-np.ones(n_actions),
            high=np.ones(n_actions), 
            dtype=np.float32)
        l_achieved, l_desired = self._compute_goals(self.dynamics.lbx.unsqueeze(0))
        u_achieved, u_desired = self._compute_goals(self.dynamics.ubx.unsqueeze(0))

        tmp_o = torch.ones([1, num_obstacles, 3], dtype=torch.float32).to(device)
        self.single_observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    low=self._observe(self.dynamics.lbx.unsqueeze(0), -2*np.inf * tmp_o).flatten(),
                    high=self._observe(self.dynamics.ubx.unsqueeze(0), 2*np.inf * tmp_o).flatten(),
                    dtype=np.float32
                ),
                achieved_goal=spaces.Box(
                    low=l_achieved.flatten(),
                    high=u_achieved.flatten(),
                    dtype=np.float32
                ),
                desired_goal=spaces.Box(
                    low=l_desired.flatten(),
                    high=u_desired.flatten(),
                    dtype=np.float32
                ),
                occupancy_grid = spaces.Box(
                    low=-np.inf * np.ones(64),
                    high=np.inf * np.ones(64),
                    dtype=np.float32
                )
            )
        )
        
        self.states = torch.empty([num_envs, len(self.dynamics.lbx)]).to(device)
        self.num_steps = torch.zeros(num_envs).to(device)
        # unused for now:
        self.reward_target = -1e-3

        # Define initial pose and goal pose sampling distribution:
        lb_initial = torch.tensor([-POSITION_BOUND, -POSITION_BOUND, 0]*2, dtype=torch.float32).to(device)
        ub_initial = torch.tensor([POSITION_BOUND, POSITION_BOUND, 2*torch.pi]*2, dtype=torch.float32).to(device)
        self.initial_pose_distribution = torch.distributions.uniform.Uniform(lb_initial, ub_initial)

        # Define circular obstacle sampling distribution:
        self.obstacles = torch.empty([num_envs, num_obstacles, 3]).to(device)
        lb_o = torch.tensor([-POSITION_BOUND, -POSITION_BOUND, OBSTACLE_MIN_RADIUS]*num_obstacles, dtype=torch.float32).to(device)
        ub_o = torch.tensor([POSITION_BOUND, POSITION_BOUND, OBSTACLE_MAX_RADIUS]*num_obstacles, dtype=torch.float32).to(device)
        self.obstacle_position_distribution = torch.distributions.uniform.Uniform(lb_o, ub_o)

        # For rendering:
        self.render_mode = "rgb_array"
        pygame.init()
        self.screen = pygame.Surface((self.RENDER_RESOLUTION, self.RENDER_RESOLUTION))
        pos_to_pixel_scaler = self.RENDER_RESOLUTION / (4*POSITION_BOUND)
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
        # Drawing pallet
        pallet_image_pixel_scaler = np.mean([1280 / 1.2, 860 / 0.8])
        pallet_scale_factor = pos_to_pixel_scaler / pallet_image_pixel_scaler
        self.pallet_center_offset = np.array([430, 650]) * pallet_scale_factor
        pallet_image = pygame.image.load('sparse_to_dense_reward/pallet.png')
        self.pallet_image = pygame.transform.scale(pallet_image, (pallet_scale_factor*pallet_image.get_width(), pallet_scale_factor*pallet_image.get_height()))

        # For 2D lidar "simulation" using R-trees:
        self.r_tree_indices = [None for _ in range(num_envs)]
        self.lidar_angles, self.lidar_direction_vectors = precompute_lidar_direction_vectors(num_rays=180, max_distance=5)
        self.lidar_points = torch.empty([num_envs, len(self.lidar_angles), 2]).to(device)

        # For "fake" vectorization (done within the env already)
        self.returns = None

        super(AvantGoalEnv, self).__init__(num_envs=num_envs, observation_space=self.single_observation_space, action_space=self.single_action_space)

    def _observe(self, states: torch.Tensor, obstacles: torch.Tensor) -> np.ndarray:
        betas = states[:, self.dynamics.beta_idx, None].cpu().numpy()
        dot_betas = states[:, self.dynamics.dot_beta_idx, None].cpu().numpy()
        velocities = states[:, self.dynamics.v_f_idx, None].cpu().numpy()

        if self.num_obstacles == 0:
            return np.c_[betas, dot_betas, velocities]
        else:
            x_f = states[:, self.dynamics.x_f_idx]
            y_f = states[:, self.dynamics.y_f_idx]
            x_o = obstacles[:, :, 0]
            y_o = obstacles[:, :, 1]
            r_o = obstacles[:, :, 2]
            d_x = x_o - x_f[:, None]
            d_y = y_o - y_f[:, None]
            d_x_border = d_x - torch.sign(d_x) * r_o
            d_y_border = d_y - torch.sign(d_y) * r_o

            return np.c_[betas, dot_betas, velocities, d_x_border.cpu().numpy(), d_y_border.cpu().numpy()]
    
    def _compute_goals(self, states: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        x_f = states[:, self.dynamics.x_f_idx].cpu().numpy()
        y_f = states[:, self.dynamics.y_f_idx].cpu().numpy()
        theta_f = states[:, self.dynamics.theta_f_idx]
        s_theta_f, c_theta_f = torch.sin(theta_f).cpu().numpy(), torch.cos(theta_f).cpu().numpy()

        x_goal = states[:, self.dynamics.x_goal_idx].cpu().numpy()
        y_goal = states[:, self.dynamics.y_goal_idx].cpu().numpy()
        theta_goal = states[:, self.dynamics.theta_goal_idx]
        s_theta_goal, c_theta_goal = torch.sin(theta_goal).cpu().numpy(), torch.cos(theta_goal).cpu().numpy()

        beta = states[:, self.dynamics.beta_idx].cpu().numpy()
        dot_beta = states[:, self.dynamics.dot_beta_idx].cpu().numpy()
        v_f = states[:, self.dynamics.v_f_idx].cpu().numpy()

        return (
            np.c_[x_f, y_f, s_theta_f, c_theta_f, beta, v_f, dot_beta], 
            np.c_[x_goal, y_goal, s_theta_goal, c_theta_goal, np.zeros_like(beta), np.zeros_like(v_f), np.zeros_like(dot_beta)]
        )


    def _construct_observation(self, states: torch.Tensor, obstacles: torch.Tensor, tmp: bool=False):
        achieved_goals, desired_goals = self._compute_goals(states)
        obs = {
            "achieved_goal": achieved_goals,
            "desired_goal": desired_goals,
            "observation": self._observe(states, self.obstacles)
        }
        if not tmp:
            obs["occupancy_grid"] = np.zeros((self.num_envs, 64), dtype=np.float32) # create_occupancy_grids(self.obstacles, cell_size=0.078, axis_limit=2*POSITION_BOUND, encoder=self.encoder)
        return obs

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: List[dict]) -> float:
        x = achieved_goal[:, 0]
        y = achieved_goal[:, 1]
        sin_c_a, cos_c_a = achieved_goal[:, 2], achieved_goal[:, 3]
        cx = desired_goal[:, 0]
        cy = desired_goal[:, 1]
        sin_c_d, cos_c_d = desired_goal[:, 2], desired_goal[:, 3]

        beta = achieved_goal[:, 4]
        desired_beta = desired_goal[:, 4]

        # TODO: append a pallet obstacle and handle this with the other obstacles:
        # cx = cx + cos_c_d*(PALLET_OFFSET + PALET_RADIUS)
        # cy = cy + sin_c_d*(PALLET_OFFSET + PALET_RADIUS)
        # dist = np.sqrt((cx - x)**2 + (cy - y)**2)
        # pallet_penalty = np.where(dist < PALET_RADIUS, 0*np.ones_like(dist), np.zeros_like(dist))

        achieved_hdg = np.arctan2(sin_c_a, cos_c_a)
        desired_hdg = np.arctan2(sin_c_d, cos_c_d)
        hdg_error_deg = 180/np.pi * np.arctan2(np.sin(achieved_hdg - desired_hdg), np.cos(achieved_hdg - desired_hdg))
        beta_error_deg = 180/np.pi * np.arctan2(np.sin(beta - desired_beta), np.cos(beta - desired_beta))
        reward = np.where(
            (np.sqrt(np.sum((achieved_goal[:, :2] - desired_goal[:, :2])**2, axis=1)) < 0.1) &
            (np.abs(hdg_error_deg) < 5) &
            (np.abs(beta_error_deg) < 5)
            , torch.zeros(len(achieved_goal))
            , -torch.ones(len(achieved_goal))
        )

        return reward
    
    def _compute_penalties(self):
        obstacle_penalty = 0
        if self.num_obstacles > 0:
            x_f = self.states[:, self.dynamics.x_f_idx]
            y_f = self.states[:, self.dynamics.y_f_idx]
            theta_f = self.states[:, self.dynamics.theta_f_idx]
            beta = self.states[:, self.dynamics.beta_idx]

            x_o = self.obstacles[:, :, 0]
            y_o = self.obstacles[:, :, 1]
            r_o = self.obstacles[:, :, 2]

            # Front link collision check:
            dist_f = torch.sqrt((x_f[:, None] - x_o)**2 + (y_f[:, None] - y_o)**2)
            obstacle_penalty_f = torch.where(torch.any(dist_f < r_o + MACHINE_RADIUS, axis=1), -100 * torch.ones(x_f.shape[0]).to(self.device), torch.zeros(x_f.shape[0]).to(self.device))
            # Rear link collision check: (TODO: should probably be computed exactly like the visual bounding circle)
            x_r = x_f - HALF_MACHINE_LENGTH/2 * torch.cos(theta_f) - HALF_MACHINE_LENGTH/2 * torch.cos(theta_f + beta) # (cos_beta * cos_c_a - sin_beta * sin_c_a)  # cos(a+b) = cos a * cos b - sin a * sin b
            y_r = y_f - HALF_MACHINE_LENGTH/2 * torch.sin(theta_f) - HALF_MACHINE_LENGTH/2 * torch.sin(theta_f + beta) # (sin_beta * cos_c_a + cos_beta * sin_c_a)  # sin(a+b) = sin a * cos b - cos a * sin b
            dist_r = torch.sqrt((x_r[:, None] - x_o)**2 + (y_r[:, None] - y_o)**2)
            obstacle_penalty_r = torch.where(torch.any(dist_r < r_o + MACHINE_RADIUS, axis=1), -100 * torch.ones(x_f.shape[0]).to(self.device), torch.zeros(x_f.shape[0]).to(self.device))
            obstacle_penalty = obstacle_penalty_f + obstacle_penalty_r

        return obstacle_penalty.cpu().numpy()
    
    def _simulate_lidar(self):
        origins = self.states[:, :2].cpu().numpy()
        results = []
        for env_i in range(self.num_envs):
            result = simulate_lidar(origins[env_i], self.r_tree_indices[env_i], self.lidar_angles, self.lidar_direction_vectors) 
            results.append(result)
        results = np.asarray(results)

        lidar_points = torch.as_tensor(results, dtype=torch.float32).to(self.states.device)
        self.lidar_points = lidar_points

    def step(self, actions):
        assert actions.dtype==np.float32
        actions = torch.from_numpy(actions).to(self.device)
        self.states = self.dynamics.discrete_dynamics_fun(self.states, actions)
        self.num_steps += 1

        info = [{} for i in range(self.num_envs)]
        tmp_obs = self._construct_observation(self.states, self.obstacles)
        reward = self.compute_reward(tmp_obs["achieved_goal"], tmp_obs["desired_goal"], info)

        if self.num_obstacles > 0:
            reward += self._compute_penalties()

        terminated = torch.from_numpy((reward > self.reward_target) | (reward <= -100)).to(self.states.device)
        truncated = (self.num_steps > self.time_limit_s / self.dynamics.dt)
        done = terminated | truncated

        # Collect terminal observation for done envs:
        done_indices = torch.argwhere(done).flatten()
        if len(done_indices) > 0:
            trunc_not_term = (truncated & ~terminated).cpu().numpy()
            for i, done_idx in enumerate(done_indices):
                terminal_obs_dict = {}
                for key, value in tmp_obs.items():
                    terminal_obs_dict[key] = value[done_idx]
                info[done_idx]["terminal_observation"] = terminal_obs_dict
                info[done_idx]["TimeLimit.truncated"] = trunc_not_term[i]
            # Reset done envs:
            self._internal_reset(done_indices)

        # self._simulate_lidar()
        obs = self._construct_observation(self.states, self.obstacles)

        return obs, reward, done.cpu().numpy(), info
    
    def _setup_lidar_r_trees(self, indices: torch.Tensor):
        obstacles = self.obstacles.cpu().numpy()
        for env_i in indices:
            r_idx = index.Index()
            for i, (x, y, r) in enumerate(obstacles[env_i]):
                # R-tree needs bounding boxes, which for circles are given by (x-r, y-r, x+r, y+r)
                r_idx.insert(i, (x-r, y-r, x+r, y+r), obj=(x, y, r))
            self.r_tree_indices[env_i] = r_idx

    def _internal_reset(self, indices: torch.Tensor):
        N_SAMPLES = self.num_envs*10000
        indices_copy = indices.clone()

        # Ensure we have valid initial pose, goal pose and obstacle location(s), if not, resample:
        envs_left = len(indices)
        while True:        
            pose_samples = self.initial_pose_distribution.sample((N_SAMPLES,))

            # Initial pose should be far enough away from goal pose:
            i_g_dist = torch.norm(pose_samples[:, :2] - pose_samples[:, 3:5], dim=1)
        
            if self.num_obstacles > 0:
                obstacles = self.obstacle_position_distribution.sample((N_SAMPLES,)).view(N_SAMPLES, self.num_obstacles, 3)
                # Initial pose should be far enough away from any obstacle:
                machine_center_points = pose_samples[:, :2].clone()
                machine_center_points[:, 0] -= torch.cos(pose_samples[:, 2]) * HALF_MACHINE_LENGTH
                machine_center_points[:, 1] -= torch.sin(pose_samples[:, 2]) * HALF_MACHINE_LENGTH
                i_f_o_diff = obstacles[:, :, :2] - machine_center_points.unsqueeze(1).expand(-1, self.num_obstacles, -1)
                i_f_o_dist = torch.norm(i_f_o_diff, dim=2) - (obstacles[:, :, 2] + HALF_MACHINE_LENGTH)
                min_i_f_o_dist = torch.amin(i_f_o_dist, dim=[1])

                # Goal pose should be far enough away from any obstacle:
                goal_center_points = pose_samples[:, 3:5]
                g_o_diff = obstacles[:, :, :2] - goal_center_points.unsqueeze(1).expand(-1, self.num_obstacles, -1)
                g_o_dist = torch.norm(g_o_diff, dim=2) - (obstacles[:, :, 2] + 2*HALF_MACHINE_LENGTH)
                min_g_o_dist = torch.amin(g_o_dist, dim=[1])

                # The distance between each obstacle should be big enough to drive through:
                a_expanded_row = obstacles[:, :, :2].unsqueeze(2).expand(N_SAMPLES, self.num_obstacles, self.num_obstacles, 2)
                a_expanded_col = obstacles[:, :, :2].unsqueeze(1).expand(N_SAMPLES, self.num_obstacles, self.num_obstacles, 2)
                diff = a_expanded_row - a_expanded_col
                radii_expanded_row = obstacles[:, :, 2].unsqueeze(2).expand(N_SAMPLES, self.num_obstacles, self.num_obstacles)
                radii_expanded_col = obstacles[:, :, 2].unsqueeze(1).expand(N_SAMPLES, self.num_obstacles, self.num_obstacles)
                edge_to_edge_distances = torch.norm(diff, p=2, dim=3) - (radii_expanded_row + radii_expanded_col)
                # Set diagonal elements to infinity to ignore self-distance
                torch.diagonal(edge_to_edge_distances, dim1=-2, dim2=-1).fill_(float('inf'))
                min_edge_to_edge_distance = torch.amin(edge_to_edge_distances, dim=[1, 2])

                # Check which samples satisfy all conditions:
                valid_indices = torch.argwhere(
                    (i_g_dist > 7.5) &
                    (min_i_f_o_dist > 2.5*HALF_MACHINE_WIDTH) &
                    (min_g_o_dist > 2.5*HALF_MACHINE_WIDTH) & 
                    (min_edge_to_edge_distance > 3*HALF_MACHINE_WIDTH)
                ).flatten()
            else:
                # Check which samples satisfy all conditions:
                valid_indices = torch.argwhere(
                    (i_g_dist > 2.5)
                ).flatten()

            # Apply all the valid env samples:
            n_valid = min(envs_left, len(valid_indices))
            if n_valid == 0:
                continue
            updated_env_indices = indices[:n_valid]
            np_valid_indices = valid_indices.cpu().numpy()[:n_valid]
            self.states[updated_env_indices, :3] = pose_samples[np_valid_indices, :3]
            self.states[updated_env_indices, 3:6] = torch.zeros((n_valid, 3)).to(self.states.device)
            self.states[updated_env_indices, 6:9] = pose_samples[np_valid_indices, 3:6]
            self.num_steps[updated_env_indices] = 0

            if self.num_obstacles > 0:
                self.obstacles[updated_env_indices] = obstacles[np_valid_indices]

            if len(valid_indices) >= envs_left:
                break

            # Resample the remaining envs:
            envs_left -= n_valid
            indices = indices[n_valid:]

        self._setup_lidar_r_trees(indices_copy)
        
    def reset(self, initial_pose: List[np.ndarray]=None, goal_pose: List[np.ndarray]=None):
        if initial_pose is not None and goal_pose is not None:
            if len(initial_pose) == self.num_envs and len(goal_pose) == self.num_envs:
                for i in range(self.num_envs):
                    assert initial_pose[i].dtype == np.float32 and goal_pose[i].dtype == np.float32
                    self.states[i] = torch.zeros(len(self.dynamics.lbx)).to(self.states.device)
                    self.states[i, :3] = torch.from_numpy(initial_pose[i]).to(self.states.device)
                    self.states[i, 6:9] = torch.from_numpy(goal_pose[i]).to(self.states.device)
                    self.num_steps[i] = 0
            else:
                raise ValueError("Need initial pose and goal pose for all environments")
        else:
            self._internal_reset(torch.arange(self.num_envs).to(self.states.device))

        return self._construct_observation(self.states, self.obstacles)

    def render(self, indices: List[int] = [0, 1], mode='rgb_array', horizon: np.ndarray = np.array([])):
        if len(horizon):
            assert len(horizon.shape) == 2 and horizon.shape[1] == 4, "Horizon should be (N, 4) array"

        x_f = self.states[indices, self.dynamics.x_f_idx].cpu().numpy()
        y_f = self.states[indices, self.dynamics.y_f_idx].cpu().numpy()
        theta_f = self.states[indices, self.dynamics.theta_f_idx].cpu().numpy()
        betas = self.states[indices, self.dynamics.beta_idx].cpu().numpy()
        x_goal = self.states[indices, self.dynamics.x_goal_idx].cpu().numpy()
        y_goal = self.states[indices, self.dynamics.y_goal_idx].cpu().numpy()
        theta_goal = self.states[indices, self.dynamics.theta_goal_idx].cpu().numpy()

        lidar_points = self.lidar_points[indices].cpu().numpy()

        center = self.RENDER_RESOLUTION // 2
        pos_to_pixel_scaler = self.RENDER_RESOLUTION / (4*POSITION_BOUND)
        
        frames = np.empty([len(indices), self.RENDER_RESOLUTION, self.RENDER_RESOLUTION, 3])
        for i in range(len(indices)):
            surf = pygame.Surface((self.RENDER_RESOLUTION, self.RENDER_RESOLUTION))
            surf.fill(WHITE)

            alpha_surf = pygame.Surface((self.RENDER_RESOLUTION, self.RENDER_RESOLUTION))
            alpha_surf.set_alpha(72)
            alpha_surf.fill(WHITE)

            # Draw obstacles
            x_obs = self.obstacles[indices, :, 0].cpu().numpy()
            y_obs = self.obstacles[indices, :, 1].cpu().numpy()
            r_obs = self.obstacles[indices, :, 2].cpu().numpy()
            for j in range(self.num_obstacles):
                pygame.draw.circle(surf, BLACK, 
                    center=(center + pos_to_pixel_scaler*x_obs[i, j], center + pos_to_pixel_scaler*y_obs[i, j]),
                    radius=pos_to_pixel_scaler*r_obs[i, j])

            # Visualizing avant front collision bound:
            pygame.draw.circle(alpha_surf, RED, 
                               center=(center + pos_to_pixel_scaler*(x_f[i]), center + pos_to_pixel_scaler*(y_f[i])),
                               radius=pos_to_pixel_scaler*MACHINE_RADIUS)
            
            # Visualizing avant rear collision bound:
            x_off = pos_to_pixel_scaler*(
                x_f[i] 
                - np.cos(theta_f[i]) * np.cos(betas[i]/4)*HALF_MACHINE_LENGTH/2 
                - np.sin(theta_f[i]) * np.sin(betas[i]/4)*HALF_MACHINE_LENGTH/2 
                - np.cos(-theta_f[i] - betas[i]) * HALF_MACHINE_LENGTH/2 
            )
            y_off = pos_to_pixel_scaler*(
                y_f[i] 
                - np.sin(theta_f[i]) * np.cos(betas[i]/4)*HALF_MACHINE_LENGTH/2 
                + np.cos(theta_f[i]) * np.sin(betas[i]/4)*HALF_MACHINE_LENGTH/2 
                + np.sin(-theta_f[i] - betas[i]) * HALF_MACHINE_LENGTH/2 
            )
            pygame.draw.circle(alpha_surf, RED, 
                               center=(center + x_off, center + y_off),
                               radius=pos_to_pixel_scaler*MACHINE_RADIUS)
            
            # Black magic to shift the pallet image correctly given the goal to pallet offset:
            x_off = pos_to_pixel_scaler*(
                x_goal[i] + np.cos(theta_goal[i]) * (PALLET_OFFSET + PALLET_LENGTH/2)
            )
            y_off = pos_to_pixel_scaler*(
                y_goal[i] + np.sin(theta_goal[i]) * (PALLET_OFFSET + PALLET_LENGTH/2)
            )
            pygame.draw.circle(alpha_surf, RED, 
                               center=(center + x_off, center + y_off),
                               radius=pos_to_pixel_scaler*PALET_RADIUS)
            rotated_image, final_position = rotate_image(self.pallet_image, np.rad2deg(theta_goal[i] + np.pi/2), self.pallet_center_offset.tolist())
            surf.blit(rotated_image, 
                           (center + x_off - final_position[0], 
                            center + y_off - final_position[1]))

            # Black magic to shift the avant frame images correctly given the kinematics:
            x_off = pos_to_pixel_scaler*(
                x_goal[i] - np.cos(theta_goal[i]) * HALF_MACHINE_LENGTH/2
            )
            y_off = pos_to_pixel_scaler*(
                y_goal[i] - np.sin(theta_goal[i]) * HALF_MACHINE_LENGTH/2 
            )
            rotated_image, final_position = rotate_image(self.rear_gray_image, np.rad2deg(theta_goal[i] + np.pi/2), self.rear_center_offset.tolist())
            surf.blit(rotated_image, 
                           (center + x_off - final_position[0], 
                            center + y_off - final_position[1]))
            rotated_image, final_position = rotate_image(self.front_gray_image, np.rad2deg(theta_goal[i] + np.pi/2), self.front_center_offset.tolist())
            surf.blit(rotated_image, 
                           (center + x_off - final_position[0], 
                            center + y_off - final_position[1]))
            pygame.draw.circle(surf, RED, 
                               center=(center + pos_to_pixel_scaler*(x_goal[i]), center + pos_to_pixel_scaler*(y_goal[i])),
                               radius=5)
            
            # Draw the lidar returns:
            for x, y in lidar_points[i]:
                pygame.draw.circle(surf, RED, 
                                center=(center + pos_to_pixel_scaler*x, center + pos_to_pixel_scaler*y),
                                radius=3)

            # During evaluation, draw the prediction horizon, if provided:
            if len(horizon):
                data = np.r_[np.c_[x_f[i], y_f[i], theta_f[i], betas[i]],
                             horizon]
            else:
                data = np.c_[x_f[i], y_f[i], theta_f[i], betas[i]]
            for j in range(len(data)):
                x_f_val, y_f_val, theta_f_val, beta_val = data[j]
                if j == 0:
                    draw_surf = surf
                else:
                    draw_surf = alpha_surf
                # Black magic to shift the image correctly given the kinematics:
                x_off = pos_to_pixel_scaler*(
                    x_f_val - np.cos(theta_f_val) * np.cos(beta_val/4)*HALF_MACHINE_LENGTH/2 
                    - np.sin(theta_f_val) * np.sin(beta_val/4)*HALF_MACHINE_LENGTH/2 
                )
                y_off = pos_to_pixel_scaler*(
                    y_f_val - np.sin(theta_f_val) * np.cos(beta_val/4)*HALF_MACHINE_LENGTH/2 
                    + np.cos(theta_f_val) * np.sin(beta_val/4)*HALF_MACHINE_LENGTH/2 
                )
                rotated_image, final_position = rotate_image(self.rear_image, np.rad2deg(theta_f_val + beta_val + np.pi/2), self.rear_center_offset.tolist())
                draw_surf.blit(rotated_image, 
                            (center + x_off - final_position[0], 
                                center + y_off - final_position[1]))
                rotated_image, final_position = rotate_image(self.front_image, np.rad2deg(theta_f_val + np.pi/2), self.front_center_offset.tolist())
                draw_surf.blit(rotated_image, 
                            (center + x_off - final_position[0], 
                                center + y_off - final_position[1]))
                pygame.draw.circle(draw_surf, RED,
                                   center=(center + pos_to_pixel_scaler*(x_f_val), center + pos_to_pixel_scaler*(y_f_val)),
                                   radius=5)
            
            self.screen.blits([
                (surf, (0, 0)),
                (alpha_surf, (0, 0))
            ])
            buffer = pygame.transform.flip(self.screen, True, False)
            buffer = pygame.surfarray.array3d(buffer)
            frames[i] = buffer
        return np.concatenate(frames, axis=1)
    
    def close(self) -> None:
        pass    
    
    def env_is_wrapped(self, wrapper_class: gymnasium.Wrapper, indices=None) -> List[bool]:
        if indices is None:
            return [False for _ in range(self.num_envs)]
        else:
            return [False for _ in indices]
    
    def step_async(self, actions: np.ndarray) -> None:
        self.returns = self.step(actions)

    def step_wait(self) -> VecEnvStepReturn:
        return self.returns
    
    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[torch.Any]:
        if indices is None:
            return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in range(self.num_envs)]
        else:
            return [getattr(self, method_name)(*method_args, **method_kwargs) for _ in indices]
    
    def get_attr(self, attr_name: str, indices=None) -> List[torch.Any]:
        if indices is None:
            return [getattr(self, attr_name) for _ in range(self.num_envs)]
        else:
            return [getattr(self, attr_name) for _ in indices]
    
    def set_attr(self, attr_name: str, value: torch.Any, indices=None) -> None:
        setattr(self, attr_name, value)