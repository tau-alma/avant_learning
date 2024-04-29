import torch
import gymnasium
import numpy as np
import pygame
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn
from sparse_to_dense_reward.avant_dynamics import AvantDynamics
from sparse_to_dense_reward.utils import rotate_image

POSITION_BOUND = 5
OBSTACLE_MIN_RADIUS = 1
OBSTACLE_MAX_RADIUS = 2

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


class AvantGoalEnv(VecEnv, GoalEnv):
    RENDER_RESOLUTION = 1280

    def __init__(self, num_envs: int, dt: float, time_limit_s: float, device: str, num_obstacles: int = 0):
        self.num_envs = num_envs
        self.time_limit_s = time_limit_s
        self.device = device
        self.dynamics = AvantDynamics(dt=dt, device=device)
        
        n_actions = len(self.dynamics.control_scalers.cpu().numpy())
        self.single_action_space = spaces.Box(
            low=-np.ones(n_actions),
            high=np.ones(n_actions), 
            dtype=np.float32)
        l_achieved, l_desired = self._compute_goals(self.dynamics.lbx.unsqueeze(0))
        u_achieved, u_desired = self._compute_goals(self.dynamics.ubx.unsqueeze(0))
        self.single_observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    low=self._observe(self.dynamics.lbx.unsqueeze(0)).flatten(),
                    high=self._observe(self.dynamics.ubx.unsqueeze(0)).flatten(),
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
                )
            )
        )
        
        self.states = torch.empty([num_envs, len(self.dynamics.lbx)]).to(device)
        self.num_steps = torch.zeros(num_envs).to(device)
        self.reward_weights = np.array([1, 1, 2, 2])
        self.reward_target = -5e-1

        # Define initial pose and goal pose sampling distribution:
        lb_initial = torch.tensor([-POSITION_BOUND, -POSITION_BOUND, 0]*2, dtype=torch.float32).to(device)
        ub_initial = torch.tensor([POSITION_BOUND, POSITION_BOUND, 2*torch.pi]*2, dtype=torch.float32).to(device)
        self.initial_pose_distribution = torch.distributions.uniform.Uniform(lb_initial, ub_initial)

        # Define circular obstacle sampling distribution:
        self.num_obstacles = num_obstacles
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

        # For "fake" vectorization (done within the env already)
        self.returns = None

        super(AvantGoalEnv, self).__init__(num_envs=num_envs, observation_space=self.single_observation_space, action_space=self.single_action_space)

    def _observe(self, states: torch.Tensor) -> np.ndarray:
        betas = states[:, self.dynamics.beta_idx].cpu().numpy()
        dot_betas = states[:, self.dynamics.dot_beta_idx].cpu().numpy()
        velocities = states[:, self.dynamics.v_f_idx].cpu().numpy()
        return np.vstack([betas, dot_betas, velocities]).T
    
    def _compute_goals(self, states: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        x_f = states[:, self.dynamics.x_f_idx].cpu().numpy()
        y_f = states[:, self.dynamics.y_f_idx].cpu().numpy()
        theta_f = states[:, self.dynamics.theta_f_idx]
        x_goal = states[:, self.dynamics.x_goal_idx].cpu().numpy()
        y_goal = states[:, self.dynamics.y_goal_idx].cpu().numpy()
        theta_goal = states[:, self.dynamics.theta_goal_idx]

        s_theta_f, c_theta_f = torch.sin(theta_f).cpu().numpy(), torch.cos(theta_f).cpu().numpy()
        s_theta_goal, c_theta_goal = torch.sin(theta_goal).cpu().numpy(), torch.cos(theta_goal).cpu().numpy()

        return np.vstack([x_f, y_f, s_theta_f, c_theta_f]).T, np.vstack([x_goal, y_goal, s_theta_goal, c_theta_goal]).T

    def _construct_observation(self, states: torch.Tensor):
        achieved_goals, desired_goals = self._compute_goals(states)
        obs = {
            "achieved_goal": achieved_goals,
            "desired_goal": desired_goals,
            "observation": self._observe(states),
        }
        return obs

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: List[dict], p=0.5) -> float:
        x = achieved_goal[:, 0]
        y = achieved_goal[:, 1]
        cx = desired_goal[:, 0]
        cy = desired_goal[:, 1]
        sin_c, cos_c = desired_goal[:, 2], desired_goal[:, 3]

        cx = cx + cos_c*(PALLET_OFFSET + PALET_RADIUS)
        cy = cy + sin_c*(PALLET_OFFSET + PALET_RADIUS)
        dist = np.sqrt((cx - x)**2 + (cy - y)**2)
        penalty = np.where(dist < PALET_RADIUS, 100*np.ones_like(dist), np.zeros_like(dist))

        reward = -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                self.reward_weights,
            ),
            p,
        )

        return reward - penalty

    def step(self, actions):
        assert actions.dtype==np.float32
        actions = torch.from_numpy(actions).to(self.device)
        self.states = self.dynamics.discrete_dynamics_fun(self.states, actions)
        self.num_steps += 1

        info = [{} for i in range(self.num_envs)]
        tmp_obs = self._construct_observation(self.states)
        reward = self.compute_reward(tmp_obs["achieved_goal"], tmp_obs["desired_goal"], info)

        terminated = torch.from_numpy(reward > self.reward_target).to(self.states.device)
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

        obs = self._construct_observation(self.states)

        return obs, reward, done.cpu().numpy(), info

    def _internal_reset(self, indices: torch.Tensor):
        samples = self.initial_pose_distribution.sample((len(indices),))
        initial_poses = samples[:, :3]
        goal_poses = samples[:, 3:6]

        while True:
            # Ensure we have a valid goal location, if not, resample:
            i_g_dist = torch.norm(goal_poses[:, :2] - initial_poses[:, :2], dim=1)
            invalid_indices = torch.argwhere(i_g_dist < 3)
            if len(invalid_indices) == 0:
                break
            samples[invalid_indices] = self.initial_pose_distribution.sample((len(invalid_indices),))
        
        self.states[indices, :3] = samples[:, :3]
        self.states[indices, 3:6] = torch.zeros((len(indices), 3)).to(self.states.device)
        self.states[indices, 6:9] = samples[:, 3:6]
        self.num_steps[indices] = 0

        obstacles = self.obstacle_position_distribution.sample((len(indices),)).view(len(indices), self.num_obstacles, 3)
        while True:
            # Ensure we have valid obstacle location(s), if not, resample:
            i_f_o_diff = obstacles[:, :, :2] - initial_poses[:, :2].unsqueeze(1).expand(-1, self.num_obstacles, -1)
            i_f_o_dist = torch.norm(i_f_o_diff, dim=2) - (obstacles[:, :, 2] + MACHINE_RADIUS)
            min_i_f_o_dist = torch.amin(i_f_o_dist, dim=[1])

            g_o_diff = obstacles[:, :, :2] - goal_poses[:, :2].unsqueeze(1).expand(-1, self.num_obstacles, -1)
            g_o_dist = torch.norm(g_o_diff, dim=2) - (obstacles[:, :, 2] + PALET_RADIUS)
            min_g_o_dist = torch.amin(g_o_dist, dim=[1])

            # Expand obstacle coordinates for pairwise broadcasting
            a_expanded_row = obstacles[:, :, :2].unsqueeze(2).expand(len(indices), self.num_obstacles, self.num_obstacles, 2)
            a_expanded_col = obstacles[:, :, :2].unsqueeze(1).expand(len(indices), self.num_obstacles, self.num_obstacles, 2)
            # Compute the difference between every pair of points
            diff = a_expanded_row - a_expanded_col
            # Calculate Euclidean distances (center to center)
            pairwise_distances = torch.norm(diff, p=2, dim=3)
            # Expand radii for pairwise operations
            radii_expanded_row = obstacles[:, :, 2].unsqueeze(2).expand(len(indices), self.num_obstacles, self.num_obstacles)
            radii_expanded_col = obstacles[:, :, 2].unsqueeze(1).expand(len(indices), self.num_obstacles, self.num_obstacles)
            # Calculate sum of radii for each pair
            sum_radii = radii_expanded_row + radii_expanded_col
            # Compute edge-to-edge distances by subtracting radii from center-to-center distances
            edge_to_edge_distances = pairwise_distances - sum_radii
            # Set diagonal elements to infinity to ignore self-distance
            torch.diagonal(edge_to_edge_distances, dim1=-2, dim2=-1).fill_(float('inf'))
            # Find the minimum edge-to-edge distance for each environment across all obstacle pairs
            min_edge_to_edge_distance = torch.amin(edge_to_edge_distances, dim=[1, 2])

            invalid_indices = torch.argwhere(
                (min_i_f_o_dist < 3*HALF_MACHINE_WIDTH) |
                (min_g_o_dist < 3*HALF_MACHINE_WIDTH) | 
                (min_edge_to_edge_distance < 3*HALF_MACHINE_WIDTH)
            ).flatten()
            if len(invalid_indices) == 0:
                break

            resamples = self.obstacle_position_distribution.sample((len(invalid_indices),))
            obstacles[invalid_indices] = resamples.view(len(invalid_indices), self.num_obstacles, 3)

        self.obstacles = obstacles

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

        return self._construct_observation(self.states)

    def render(self, indices: List[int] = [0], mode='rgb_array', horizon: np.ndarray = np.array([])):
        if len(horizon):
            assert len(horizon.shape) == 2 and horizon.shape[1] == 4, "Horizon should be (N, 4) array"

        x_f = self.states[indices, self.dynamics.x_f_idx].cpu().numpy()
        y_f = self.states[indices, self.dynamics.y_f_idx].cpu().numpy()
        theta_f = self.states[indices, self.dynamics.theta_f_idx].cpu().numpy()
        betas = self.states[indices, self.dynamics.beta_idx].cpu().numpy()
        x_goal = self.states[indices, self.dynamics.x_goal_idx].cpu().numpy()
        y_goal = self.states[indices, self.dynamics.y_goal_idx].cpu().numpy()
        theta_goal = self.states[indices, self.dynamics.theta_goal_idx].cpu().numpy()

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

            pygame.draw.circle(alpha_surf, RED, 
                               center=(center + pos_to_pixel_scaler*(x_f[i]), center + pos_to_pixel_scaler*(y_f[i])),
                               radius=pos_to_pixel_scaler*MACHINE_RADIUS)
            
            # Visualizing avant collision bound:
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
            
            # Black magic to shift the image correctly given the goal to pallet offset:
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

            # Black magic to shift the image correctly given the kinematics:
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
            buffer = pygame.surfarray.array3d(self.screen)
            buffer = np.transpose(buffer, (1, 0, 2))
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