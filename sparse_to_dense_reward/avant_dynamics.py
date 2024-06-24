import torch
import numpy as np
import sparse_to_dense_reward.config as config
from avant_modeling.gp import GPModel

class AvantDynamics:
    # State indices:
    x_f_idx = 0
    y_f_idx = 1
    theta_f_idx = 2
    beta_idx = 3
    dot_beta_idx = 4
    v_f_idx = 5
    x_goal_idx = 6
    y_goal_idx = 7
    theta_goal_idx = 8

    # Control indices for RL:
    u_dot_beta_idx = 0
    u_v_f_idx = 1

    # Control indices for MPC:
    u_steer_idx = 0
    u_throttle_idx = 1

    def __init__(self, dt: float, device: str, eval: bool):
        self.dt = dt
        self.device = device
        self.eval = eval

        # Define control normalization constants:
        if not eval:
            self.control_scalers = torch.tensor([config.avant_max_dot_beta, config.avant_max_v], dtype=torch.float32).to(device)
        else:
            self.control_scalers = torch.ones(2, dtype=torch.float32).to(device)

        # Define state bounds:
        self.lbx = torch.tensor([
            -torch.inf, -torch.inf, -torch.inf, 
            -config.avant_max_beta, -config.avant_max_dot_beta, 
            config.avant_min_v, -torch.inf, -torch.inf, 0
        ], dtype=torch.float32).to(device)
        self.ubx = torch.tensor([
            torch.inf, torch.inf, torch.inf, 
            config.avant_max_beta, config.avant_max_dot_beta,
            config.avant_max_v, torch.inf, torch.inf, 2*torch.pi
        ], dtype=torch.float32).to(device)

        if eval:
            self.gp_dict = {}
            for name in ["omega_f", "v_f", "dot_beta"]:
                data_x = torch.load(f"sparse_to_dense_reward/{name}/{name}_gp_inputs.pth").to(device)
                data_y = torch.load(f"sparse_to_dense_reward/{name}/{name}_gp_targets.pth").to(device)
                gp = GPModel(data_x, data_y, train_epochs=0).to(device)
                gp.load_state_dict(torch.load(f"sparse_to_dense_reward/{name}/{name}_gp_model.pth"))
                self.gp_dict[name] = gp

    def _rl_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        u_values = self.control_scalers * u_values

        desired_beta_accel = (u_values[:, self.u_dot_beta_idx]) / self.dt
        desired_linear_accel = (u_values[:, self.u_v_f_idx]) / self.dt

        limited_beta_accel = torch.max(
            torch.min(
                desired_beta_accel, 
                config.avant_max_dot_dot_beta * torch.ones_like(desired_beta_accel).to(u_values.device)
            ), 
            -config.avant_max_dot_dot_beta * torch.ones_like(desired_beta_accel).to(u_values.device)
        )
        limited_linear_accel = torch.max(
            torch.min(
                desired_linear_accel, 
                config.avant_max_a * torch.ones_like(desired_linear_accel).to(u_values.device)
            ), 
            -config.avant_max_a * torch.ones_like(desired_linear_accel).to(u_values.device)
        )

        zero_beta_accel = limited_beta_accel.clone()
        zero_beta_accel[:] = 0
        limited_beta_accel = torch.where(
            (torch.abs(x_values[:, self.v_f_idx]) < 0.1) & (torch.sign(limited_beta_accel) == torch.sign(x_values[:, self.dot_beta_idx])),
            zero_beta_accel, 
            limited_beta_accel
        )

        # To avoid accumulating beta even at the limits, we scale the dot_beta accordingly:
        distance_to_limit = config.avant_max_beta - torch.abs(x_values[:, self.beta_idx])
        scaling_factor = torch.min(distance_to_limit / (torch.abs(x_values[:, self.dot_beta_idx]) * self.dt + 1e-5), torch.tensor(1.0).to(x_values.device))

        omega_f = -(
            (config.avant_lr * scaling_factor * x_values[:, self.dot_beta_idx] + x_values[:, self.v_f_idx] * torch.sin(x_values[:, self.beta_idx])) 
            / (config.avant_lf * torch.cos(x_values[:, self.beta_idx]) + config.avant_lr)
        )
        dot_state = torch.vstack([
            x_values[:, self.v_f_idx] * torch.cos(x_values[:, self.theta_f_idx]),
            x_values[:, self.v_f_idx] * torch.sin(x_values[:, self.theta_f_idx]),
            omega_f,
            x_values[:, self.dot_beta_idx],
            limited_beta_accel,
            limited_linear_accel,
            # TODO: move these away from here:
            torch.zeros(u_values.shape[0]).to(u_values.device),  # constant goal x
            torch.zeros(u_values.shape[0]).to(u_values.device),  # constant goal y
            torch.zeros(u_values.shape[0]).to(u_values.device)   # constant goal theta
        ]).T
        
        next_state = x_values + self.dt * dot_state
        clamped_next_state = torch.max(torch.min(next_state, self.ubx), self.lbx)

        zero_dot_beta = clamped_next_state.clone()
        zero_dot_beta[:, self.dot_beta_idx] = 0

        selector = ((clamped_next_state[:, self.beta_idx] == -config.avant_max_beta) & (clamped_next_state[:, self.dot_beta_idx] < 0)).unsqueeze(1).expand(-1, 9)
        clamped_next_state = torch.where(
            selector,
            zero_dot_beta,
            clamped_next_state
        )

        selector = ((clamped_next_state[:, self.beta_idx] == config.avant_max_beta) & (clamped_next_state[:, self.dot_beta_idx] > 0)).unsqueeze(1).expand(-1, 9)
        clamped_next_state = torch.where(
            selector,
            zero_dot_beta,
            clamped_next_state
        )

        return clamped_next_state
    
    def _mpc_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        u_values = self.control_scalers * u_values

        # To avoid accumulating beta even at the limits, we scale the dot_beta accordingly:
        distance_to_limit = config.avant_max_beta - torch.abs(x_values[:, self.beta_idx])
        scaling_factor = torch.min(distance_to_limit / (torch.abs(x_values[:, self.dot_beta_idx]) * self.dt + 1e-5), torch.tensor(1.0).to(x_values.device))

        # Nominal and GP for omega_f
        nominal_omega_f = -(
            (config.avant_lr * scaling_factor * x_values[:, self.dot_beta_idx] + x_values[:, self.v_f_idx] * torch.sin(x_values[:, self.beta_idx])) 
            / (config.avant_lf * torch.cos(x_values[:, self.beta_idx]) + config.avant_lr)
        )
        gp_omega_f_inputs = torch.vstack([
            x_values[:, self.beta_idx], x_values[:, self.dot_beta_idx], x_values[:, self.v_f_idx], u_values[:, self.u_steer_idx]
        ]).T
        gp_omega_f = self.gp_dict["omega_f"](gp_omega_f_inputs).mean
        omega_f = nominal_omega_f + gp_omega_f

        # Nominal and GP for v_f
        nominal_v_f = 3 * u_values[:, self.u_throttle_idx]
        gp_v_f_inputs = torch.vstack([
            x_values[:, self.beta_idx], u_values[:, self.u_throttle_idx]
        ]).T
        gp_v_f = self.gp_dict["v_f"](gp_v_f_inputs).mean
        v_f = nominal_v_f + gp_v_f

        # Nominal and GP for dot_dot_beta        
        a = 0.127                  # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
        b = 0.495                  # AFS parameter, check the paper page(1) Figure 1: AFS mechanism
        eps0 = 1.4049900478554351  # the angle from of the hydraulic sylinder check the paper page(1) Figure (1) 
        eps = eps0 - x_values[:, self.beta_idx]
        k = 10 * a * b * np.sin(eps) / np.sqrt(a**2 + b**2 - 2*a*b*np.cos(eps))
        nominal_dot_beta = u_values[:, self.u_steer_idx] / k
        gp_dot_beta_inputs = torch.vstack([
            x_values[:, self.beta_idx], x_values[:, self.v_f_idx], u_values[:, self.u_steer_idx]
        ]).T
        gp_dot_beta = self.gp_dict["dot_beta"](gp_dot_beta_inputs).mean
        dot_beta = nominal_dot_beta + gp_dot_beta

        dot_state = torch.vstack([
            v_f * torch.cos(x_values[:, self.theta_f_idx]),
            v_f * torch.sin(x_values[:, self.theta_f_idx]),
            omega_f,
            x_values[:, self.dot_beta_idx],
            (dot_beta - x_values[:, self.dot_beta_idx]) / self.dt,  # results in dot_beta after euler integration by dt
            (v_f - x_values[:, self.v_f_idx]) / self.dt,            # results in v_f after euler integration by dt
            # TODO: move these away from here:
            torch.zeros(u_values.shape[0]).to(u_values.device),  # constant goal x
            torch.zeros(u_values.shape[0]).to(u_values.device),  # constant goal y
            torch.zeros(u_values.shape[0]).to(u_values.device)   # constant goal theta
        ]).T
        next_state = x_values + self.dt * dot_state
        clamped_next_state = torch.max(torch.min(next_state, self.ubx), self.lbx)

        zero_dot_beta = clamped_next_state.clone()
        zero_dot_beta[:, self.dot_beta_idx] = 0

        clamped_next_state = torch.where(
            (clamped_next_state[:, self.beta_idx] == -config.avant_max_beta) & (clamped_next_state[:, self.dot_beta_idx] < 0),
            zero_dot_beta,
            clamped_next_state
        )

        clamped_next_state = torch.where(
            (clamped_next_state[:, self.beta_idx] == config.avant_max_beta) & (clamped_next_state[:, self.dot_beta_idx] > 0),
            zero_dot_beta,
            clamped_next_state
        )

        return clamped_next_state 

    def discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        if not self.eval:
            return self._rl_dynamics_fun(x_values, u_values)
        else:
            with torch.no_grad():
                return self._mpc_dynamics_fun(x_values, u_values)