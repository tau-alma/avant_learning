import torch
import gymnasium
import numpy as np
import pygame
import time
import numpy as np
from typing import List, Tuple
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn
from sparse_to_dense_reward.avant_dynamics import AvantDynamics
from sparse_to_dense_reward.utils import GoalEnv, AvantRenderer

POSITION_BOUND = 5

class AvantGoalEnv(VecEnv, GoalEnv):
    def __init__(self, num_envs: int, dt: float, time_limit_s: float, device: str, eval=False):
        self.num_envs = num_envs
        self.time_limit_s = time_limit_s
        self.device = device
        self.eval = eval
        self.dynamics = AvantDynamics(dt=dt, device=device, eval=eval)

        n_actions = len(self.dynamics.control_scalers.cpu().numpy())
        self.single_action_space = spaces.Box(
            low=-np.ones(n_actions),
            high=np.ones(n_actions), 
            dtype=np.float32)
        
        l_achieved, l_desired = self._compute_goals(torch.zeros([1, 3]).to(device), self.dynamics.lbx.unsqueeze(0))
        u_achieved, u_desired = self._compute_goals(torch.zeros([1, 3]).to(device), self.dynamics.ubx.unsqueeze(0))
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
        
        self.goals = torch.empty([num_envs, 3]).to(device)
        self.states = torch.empty([num_envs, len(self.dynamics.lbx)]).to(device)
        self.num_steps = torch.zeros(num_envs).to(device)

        # Define initial pose and goal pose sampling distribution:
        lb_initial = torch.tensor([-POSITION_BOUND, -POSITION_BOUND, 0]*2, dtype=torch.float32).to(device)
        ub_initial = torch.tensor([POSITION_BOUND, POSITION_BOUND, 2*torch.pi]*2, dtype=torch.float32).to(device)
        self.initial_pose_distribution = torch.distributions.uniform.Uniform(lb_initial, ub_initial)

        # For rendering:
        self.render_mode = "rgb_array"
        self.renderer = AvantRenderer(POSITION_BOUND)
        # For "fake" vectorization (done within the env already)
        self.returns = None

        super(AvantGoalEnv, self).__init__(num_envs=num_envs, observation_space=self.single_observation_space, action_space=self.single_action_space)

    def _observe(self, states: torch.Tensor) -> np.ndarray:
        betas = states[:, self.dynamics.beta_idx, None].cpu().numpy()
        dot_betas = states[:, self.dynamics.dot_beta_idx, None].cpu().numpy()
        velocities = states[:, self.dynamics.v_f_idx, None].cpu().numpy()
        return np.c_[betas, dot_betas, velocities]
    
    def _compute_goals(self, goals: torch.Tensor, states: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        x_f = states[:, self.dynamics.x_f_idx].cpu().numpy()
        y_f = states[:, self.dynamics.y_f_idx].cpu().numpy()
        theta_f = states[:, self.dynamics.theta_f_idx]
        s_theta_f, c_theta_f = torch.sin(theta_f).cpu().numpy(), torch.cos(theta_f).cpu().numpy()

        x_goal = goals[:, 0].cpu().numpy()
        y_goal = goals[:, 1].cpu().numpy()
        theta_goal = goals[:, 2]
        s_theta_goal, c_theta_goal = torch.sin(theta_goal).cpu().numpy(), torch.cos(theta_goal).cpu().numpy()

        beta = states[:, self.dynamics.beta_idx].cpu().numpy()
        dot_beta = states[:, self.dynamics.dot_beta_idx].cpu().numpy()
        v_f = states[:, self.dynamics.v_f_idx].cpu().numpy()

        return (
            np.c_[x_f, y_f, s_theta_f, c_theta_f, beta, dot_beta, v_f], 
            np.c_[x_goal, y_goal, s_theta_goal, c_theta_goal, np.zeros_like(beta), np.zeros_like(dot_beta), np.zeros_like(v_f)]
        )

    def _construct_observation(self):
        achieved_goals, desired_goals = self._compute_goals(self.goals, self.states)
        obs = {
            "achieved_goal": achieved_goals,
            "desired_goal": desired_goals,
            "observation": self._observe(self.states)
        }
        return obs
    
    def _check_done_condition(self, achieved_goal: np.ndarray, desired_goal: np.ndarray):
        sin_c_a, cos_c_a = achieved_goal[:, 2], achieved_goal[:, 3]
        sin_c_d, cos_c_d = desired_goal[:, 2], desired_goal[:, 3]

        beta = achieved_goal[:, 4]
        dot_beta = achieved_goal[:, 5]
        v_f = achieved_goal[:, 6]
        achieved_hdg = np.arctan2(sin_c_a, cos_c_a)
        desired_hdg = np.arctan2(sin_c_d, cos_c_d)
    
        pos_error = np.sqrt(np.sum((achieved_goal[:, :2] - desired_goal[:, :2])**2, axis=1))
        hdg_error_deg = 180/np.pi * np.arctan2(np.sin(achieved_hdg - desired_hdg), np.cos(achieved_hdg - desired_hdg))
        beta_error_deg = 180/np.pi * beta
        dot_beta_error_deg = 180/np.pi * dot_beta
        velocity_error = v_f

        return (
            (pos_error < 0.1) &
            (np.abs(hdg_error_deg) < 2.5) &
            (np.abs(beta_error_deg) < 5) & 
            (np.abs(dot_beta_error_deg) < 5) &
            (np.abs(velocity_error) < 0.1) 
        )

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: List[dict]) -> float:
        reward = np.where(
            self._check_done_condition(achieved_goal, desired_goal)
            , np.zeros(len(achieved_goal))
            , -np.ones(len(achieved_goal))
        )

        return reward
    
    def step(self, actions):
        assert actions.dtype==np.float32
        actions = torch.from_numpy(actions).to(self.device)
        self.states = self.dynamics.discrete_dynamics_fun(self.states, actions)
        self.num_steps += 1

        info = [{} for i in range(self.num_envs)]
        tmp_obs = self._construct_observation()
        reward = self.compute_reward(tmp_obs["achieved_goal"], tmp_obs["desired_goal"], info)
        
        # Policy seems to learn an action sequence of [max_a, min_a, max_a, ...] to maintain a constant velocity, this should help encourage 0 acceleration instead:
        accel_penalty = 1e-2 * torch.sum(actions**2, axis=1).cpu().numpy()
        reward -= accel_penalty
        # We want to minimize janky movements, where the policy tries to zero out the beta angle only when standing still at the goal position
        standstill_turn_penalty = ((1 - torch.tanh(self.states[:, self.dynamics.v_f_idx]**2 / 0.05)) * 1e1*self.states[:, self.dynamics.dot_beta_idx]**2).cpu().numpy()
        reward -= standstill_turn_penalty

        terminated = torch.from_numpy(self._check_done_condition(tmp_obs["achieved_goal"], tmp_obs["desired_goal"])).to(self.num_steps.device)
        truncated = (self.num_steps > self.time_limit_s / self.dynamics.dt)
        done = terminated | truncated

        # Collect terminal observation for done envs:
        done_indices = torch.argwhere(done).flatten()
        if len(done_indices) > 0:
            trunc_not_term = (truncated & ~terminated).cpu().numpy()
            term_not_trunc = (terminated & ~truncated).cpu().numpy()
            for i, done_idx in enumerate(done_indices):
                terminal_obs_dict = {}
                for key, value in tmp_obs.items():
                    terminal_obs_dict[key] = value[done_idx]
                info[done_idx]["terminal_observation"] = terminal_obs_dict
                info[done_idx]["TimeLimit.truncated"] = trunc_not_term[i]
                info[done_idx]["is_success"] = term_not_trunc[i]
            # Reset done envs:
            self._internal_reset(done_indices)

        obs = self._construct_observation()

        return obs, reward, done.cpu().numpy(), info

    def _internal_reset(self, indices: torch.Tensor):
        N_SAMPLES = self.num_envs*10000

        # Ensure we have valid initial pose and goal pose locations, if not, resample:
        envs_left = len(indices)
        while True:        
            pose_samples = self.initial_pose_distribution.sample((N_SAMPLES,))
        
            # Initial pose should be far enough away from goal pose:
            i_g_dist = torch.norm(pose_samples[:, :2] - pose_samples[:, 3:5], dim=1)
            valid_indices = torch.argwhere(
                (i_g_dist > 0.5)
            ).flatten()

            # Apply all the valid env samples:
            n_valid = min(envs_left, len(valid_indices))
            if n_valid == 0:
                continue
            updated_env_indices = indices[:n_valid]
            np_valid_indices = valid_indices.cpu().numpy()[:n_valid]
            self.goals[updated_env_indices] = pose_samples[np_valid_indices, 3:6]
            self.states[updated_env_indices, :3] = pose_samples[np_valid_indices, :3]
            self.states[updated_env_indices, 3:6] = torch.zeros((n_valid, 3)).to(self.states.device)
            self.num_steps[updated_env_indices] = 0

            if len(valid_indices) >= envs_left:
                break

            # Resample the remaining envs:
            envs_left -= n_valid
            indices = indices[n_valid:]
        
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

        return self._construct_observation()

    def render(self, indices: List[int] = [0, 1], mode='rgb_array', horizon: np.ndarray = np.array([]), obstacles: np.ndarray = np.array([])):
        if len(horizon):
            assert len(horizon.shape) == 2 and horizon.shape[1] == 4, "Horizon should be (N, 4) array"
        if len(obstacles):
            assert len(horizon.shape) == 2 and horizon.shape[1] == 4, "Obstacles should be (N, 4) array"
        
        frames = []
        for i in range(len(indices)):
            frames.append(self.renderer.render(
                self.states[i, :4].cpu().numpy(),
                self.goals[i, :3].cpu().numpy(),
                horizon, obstacles
            ))
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