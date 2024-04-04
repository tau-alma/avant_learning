import torch
import gymnasium
import numpy as np
import config
import pygame
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn
from dynamics import AvantDynamics

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BROWN = (150, 75, 0)

PALLET_OFFSET = 0.5
HALF_PALLET_WIDTH = 0.4
PALLET_LENGTH = 1.2
PALET_RADIUS = PALLET_LENGTH

HALF_MACHINE_WIDTH = 0.6
HALF_MACHINE_LENGTH = 1.3
JOINT_SIZE = 0.25

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

    def __init__(self, num_envs: int, time_limit_s: float, device: str):
        self.num_envs = num_envs
        self.time_limit_s = time_limit_s
        self.device = getattr(config, 'device', device) 
        self.dynamics = AvantDynamics(dt=0.2, device=device)
        
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
        self.reward_weights = np.array([1, 1, 1, 1])
        self.reward_target = -5e-1
        # For rendering:
        self.render_mode = "rgb_array"
        pygame.init()
        self.screen = pygame.Surface((self.RENDER_RESOLUTION, self.RENDER_RESOLUTION))

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
            "observation": self._observe(states),
            "achieved_goal": achieved_goals,
            "desired_goal": desired_goals,
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
        self.states[indices] = self.dynamics.generate_initial_state(len(indices))
        self.num_steps[indices] = 0

    def reset(self):
        self._internal_reset(torch.arange(self.num_envs).to(self.states.device))
        return self._construct_observation(self.states)

    def render(self, indices: List[int] = [0], mode='rgb_array'):
        x_f = self.states[indices, self.dynamics.x_f_idx].cpu().numpy()
        y_f = self.states[indices, self.dynamics.y_f_idx].cpu().numpy()
        theta_f = self.states[indices, self.dynamics.theta_f_idx].cpu().numpy()
        betas = self.states[indices, self.dynamics.beta_idx].cpu().numpy()
        x_goal = self.states[indices, self.dynamics.x_goal_idx].cpu().numpy()
        y_goal = self.states[indices, self.dynamics.y_goal_idx].cpu().numpy()
        theta_goal = self.states[indices, self.dynamics.theta_goal_idx].cpu().numpy()

        center = self.RENDER_RESOLUTION // 2
        pos_to_pixel_scaler = self.RENDER_RESOLUTION / (4*self.dynamics.position_bound)
        
        frames = np.empty([len(indices), self.RENDER_RESOLUTION, self.RENDER_RESOLUTION, 3])
        for i in range(len(indices)):
            surf = pygame.Surface((self.RENDER_RESOLUTION, self.RENDER_RESOLUTION))
            surf.fill(WHITE)

            # Draw pallet
            pygame.draw.polygon(surf, BROWN, (
                (center + pos_to_pixel_scaler*(x_goal[i] + np.cos(theta_goal[i])*PALLET_OFFSET - np.cos(theta_goal[i] - np.pi / 2)*HALF_PALLET_WIDTH), 
                center + pos_to_pixel_scaler*(y_goal[i] + np.sin(theta_goal[i])*PALLET_OFFSET - np.sin(theta_goal[i] - np.pi / 2)*HALF_PALLET_WIDTH)),
                (center + pos_to_pixel_scaler*(x_goal[i] + np.cos(theta_goal[i])*(PALLET_OFFSET + PALLET_LENGTH) - np.cos(theta_goal[i] - np.pi / 2)*HALF_PALLET_WIDTH), 
                center + pos_to_pixel_scaler*(y_goal[i] + np.sin(theta_goal[i])*(PALLET_OFFSET + PALLET_LENGTH) - np.sin(theta_goal[i] - np.pi / 2)*HALF_PALLET_WIDTH)),
                (center + pos_to_pixel_scaler*(x_goal[i] + np.cos(theta_goal[i])*(PALLET_OFFSET + PALLET_LENGTH) + np.cos(theta_goal[i] - np.pi / 2)*HALF_PALLET_WIDTH), 
                center + pos_to_pixel_scaler*(y_goal[i] + np.sin(theta_goal[i])*(PALLET_OFFSET + PALLET_LENGTH) + np.sin(theta_goal[i] - np.pi / 2)*HALF_PALLET_WIDTH)),
                (center + pos_to_pixel_scaler*(x_goal[i] + np.cos(theta_goal[i])*PALLET_OFFSET + np.cos(theta_goal[i] - np.pi / 2)*HALF_PALLET_WIDTH), 
                center + pos_to_pixel_scaler*(y_goal[i] + np.sin(theta_goal[i])*PALLET_OFFSET + np.sin(theta_goal[i] - np.pi / 2)*HALF_PALLET_WIDTH)),
            ))
            # Draw goal
            pygame.draw.polygon(surf, RED, (
                (center + pos_to_pixel_scaler*x_goal[i], 
                center + pos_to_pixel_scaler*y_goal[i]),
                (center + pos_to_pixel_scaler*(x_goal[i] - np.cos(theta_goal[i])*HALF_MACHINE_LENGTH - np.cos(theta_goal[i] + np.pi / 2)*HALF_MACHINE_WIDTH), 
                center + pos_to_pixel_scaler*(y_goal[i] - np.sin(theta_goal[i])*HALF_MACHINE_LENGTH - np.sin(theta_goal[i] + np.pi / 2)*HALF_MACHINE_WIDTH)),
                (center + pos_to_pixel_scaler*(x_goal[i] - np.cos(theta_goal[i])*HALF_MACHINE_LENGTH - np.cos(theta_goal[i] - np.pi / 2)*HALF_MACHINE_WIDTH), 
                center + pos_to_pixel_scaler*(y_goal[i] - np.sin(theta_goal[i])*HALF_MACHINE_LENGTH - np.sin(theta_goal[i] - np.pi / 2)*HALF_MACHINE_WIDTH)),
            ))
            # Draw rear link
            pygame.draw.line(
                surf, GREEN, 
                (center + pos_to_pixel_scaler*(x_f[i] - np.cos(theta_f[i]) * (JOINT_SIZE + HALF_MACHINE_LENGTH)), 
                 center + pos_to_pixel_scaler*(y_f[i] - np.sin(theta_f[i]) * (JOINT_SIZE + HALF_MACHINE_LENGTH))),
                (center + pos_to_pixel_scaler*(x_f[i] - np.cos(theta_f[i] + betas[i]) * (JOINT_SIZE + 2*HALF_MACHINE_LENGTH)), 
                 center + pos_to_pixel_scaler*(y_f[i] - np.sin(theta_f[i] + betas[i]) * (JOINT_SIZE + 2*HALF_MACHINE_LENGTH))),
                 width=int(pos_to_pixel_scaler * HALF_MACHINE_WIDTH)
            )
            # Draw center link
            pygame.draw.circle(
                surf, BLACK, 
                (center + pos_to_pixel_scaler*(x_f[i] - np.cos(theta_f[i])*(JOINT_SIZE + HALF_MACHINE_LENGTH)), 
                 center + pos_to_pixel_scaler*(y_f[i] - np.sin(theta_f[i])*(JOINT_SIZE + HALF_MACHINE_LENGTH))),
                int(pos_to_pixel_scaler*JOINT_SIZE)
            )
            # Draw front link
            pygame.draw.polygon(surf, GREEN, (
                (center + pos_to_pixel_scaler*x_f[i], 
                 center + pos_to_pixel_scaler*y_f[i]),
                (center + pos_to_pixel_scaler*(x_f[i] - np.cos(theta_f[i])*HALF_MACHINE_LENGTH + np.cos(theta_f[i] + np.pi / 2)*HALF_MACHINE_WIDTH), 
                 center + pos_to_pixel_scaler*(y_f[i] - np.sin(theta_f[i])*HALF_MACHINE_LENGTH + np.sin(theta_f[i] + np.pi / 2)*HALF_MACHINE_WIDTH)),
                (center + pos_to_pixel_scaler*(x_f[i] - np.cos(theta_f[i])*HALF_MACHINE_LENGTH - np.cos(theta_f[i] + np.pi / 2)*HALF_MACHINE_WIDTH), 
                 center + pos_to_pixel_scaler*(y_f[i] - np.sin(theta_f[i])*HALF_MACHINE_LENGTH - np.sin(theta_f[i] + np.pi / 2)*HALF_MACHINE_WIDTH))
            ))
            self.screen.blits([
                (surf, (0, 0)),
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