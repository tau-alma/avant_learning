import torch
import config

class KinematicLoader:
    # State indices:
    x_f_idx = 0
    y_f_idx = 1
    theta_f_idx = 2
    beta_idx = 3
    dot_beta_idx = 4
    v_f_idx = 5

    # Control indices for RL:
    u_dot_dot_beta_idx = 0
    u_a_f_idx = 1

    def __init__(self, dt: float, device: str):
        self.dt = dt
        self.device = device

        # Define control normalization constants:
        self.control_scalers = torch.tensor([config.loader_max_dot_dot_beta, config.loader_max_a], dtype=torch.float32).to(device)

        # Define state bounds:
        self.lbx = torch.tensor([
            -torch.inf, -torch.inf, -torch.inf, 
            -config.loader_max_beta, -config.loader_max_dot_beta, 
            config.loader_min_v
        ], dtype=torch.float32).to(device)
        self.ubx = torch.tensor([
            torch.inf, torch.inf, torch.inf, 
            config.loader_max_beta, config.loader_max_dot_beta,
            config.loader_max_v
        ], dtype=torch.float32).to(device)


    # Simple kinematic model with direct control of accelerations, used to train RL policy
    def _rl_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        def continuous_dynamics(x_values: torch.Tensor, u_values: torch.Tensor):
            u_values = self.control_scalers * u_values

            desired_beta_accel = u_values[:, self.u_dot_dot_beta_idx]
            desired_linear_accel = u_values[:, self.u_a_f_idx]

            # Limit the resulting accelerations
            limited_beta_accel = torch.max(
                torch.min(
                    desired_beta_accel, 
                    config.loader_max_dot_dot_beta * torch.ones_like(desired_beta_accel).to(u_values.device)
                ), 
                -config.loader_max_dot_dot_beta * torch.ones_like(desired_beta_accel).to(u_values.device)
            )
            limited_linear_accel = torch.max(
                torch.min(
                    desired_linear_accel, 
                    config.loader_max_a * torch.ones_like(desired_linear_accel).to(u_values.device)
                ), 
                -config.loader_max_a * torch.ones_like(desired_linear_accel).to(u_values.device)
            )

            omega_f = -(
                (config.loader_lr * x_values[:, self.dot_beta_idx] + x_values[:, self.v_f_idx] * torch.sin(x_values[:, self.beta_idx])) 
                / (config.loader_lf * torch.cos(x_values[:, self.beta_idx]) + config.loader_lr)
            )
            dot_state = torch.vstack([
                x_values[:, self.v_f_idx] * torch.cos(x_values[:, self.theta_f_idx]),
                x_values[:, self.v_f_idx] * torch.sin(x_values[:, self.theta_f_idx]),
                omega_f,
                x_values[:, self.dot_beta_idx],
                limited_beta_accel,
                limited_linear_accel
            ]).T
            return dot_state
        
        k1 = continuous_dynamics(x_values, u_values)
        k2 = continuous_dynamics(x_values + self.dt / 2 * k1, u_values)
        k3 = continuous_dynamics(x_values + self.dt / 2 * k2, u_values)
        k4 = continuous_dynamics(x_values + self.dt * k3, u_values)
        state_delta = self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        next_state = x_values + state_delta
        clamped_next_state = torch.max(torch.min(next_state, self.ubx), self.lbx)

        # Zero out any center link velocity (going towards the closest joint limit direction) at joint limit
        zero_dot_beta = clamped_next_state.clone()
        zero_dot_beta[:, self.dot_beta_idx] = 0
        selector = ((clamped_next_state[:, self.beta_idx] == -config.loader_max_beta) & (clamped_next_state[:, self.dot_beta_idx] < 0)).unsqueeze(1).expand(-1, 6)
        clamped_next_state = torch.where(
            selector,
            zero_dot_beta,
            clamped_next_state
        )
        selector = ((clamped_next_state[:, self.beta_idx] == config.loader_max_beta) & (clamped_next_state[:, self.dot_beta_idx] > 0)).unsqueeze(1).expand(-1, 6)
        clamped_next_state = torch.where(
            selector,
            zero_dot_beta,
            clamped_next_state
        )

        return clamped_next_state

    def discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        return self._rl_dynamics_fun(x_values, u_values)