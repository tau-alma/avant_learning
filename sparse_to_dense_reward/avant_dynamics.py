import torch
import sparse_to_dense_reward.config as config


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

    # Control indices:
    dot_dot_beta_idx = 0
    a_f_idx = 1

    def __init__(self, dt: float, device: str):
        self.dt = dt
        # Define control normalization constants:
        self.control_scalers = torch.tensor([config.avant_max_dot_dot_beta, config.avant_max_a], dtype=torch.float32).to(device)
        
        # Define state bounds:
        self.lbx = torch.tensor([-torch.inf, -torch.inf, -torch.inf, -config.avant_max_beta, -config.avant_max_dot_beta, config.avant_min_v, -torch.inf, -torch.inf, 0], dtype=torch.float32).to(device)
        self.ubx = torch.tensor([torch.inf, torch.inf, torch.inf, config.avant_max_beta, config.avant_max_dot_beta, config.avant_max_v, torch.inf, torch.inf, 2*torch.pi], dtype=torch.float32).to(device)

    def discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        u_values = self.control_scalers * u_values
        alpha = torch.pi + x_values[:, self.beta_idx]
        omega_f = x_values[:, self.v_f_idx] * config.avant_lf / torch.tan(alpha/2)

        # Calculate the distance to the nearest limit for beta.
        distance_to_limit = config.avant_max_beta - torch.abs(x_values[:, self.beta_idx])
        # Scale factor to adjust dot_beta, with a cap to avoid division by zero issues.
        scaling_factor = torch.min(distance_to_limit / (torch.abs(x_values[:, self.dot_beta_idx]) * self.dt + 1e-5), torch.tensor(1.0).to(x_values.device))
        # Adjust dot_beta by the scaling factor and apply the 1/2 factor.
        dot_beta_omega_f = 0.5 * x_values[:, self.dot_beta_idx] * scaling_factor
        omega_f -= dot_beta_omega_f

        # Lateral movement of front tires in response to changing center link angle
        dot_beta_lateral_front_movement = torch.sin(x_values[:, self.dot_beta_idx]/4 * scaling_factor) * config.avant_lf

        dot_state = torch.vstack([
            x_values[:, self.v_f_idx] * torch.cos(x_values[:, self.theta_f_idx]) - torch.cos(x_values[:, self.theta_f_idx]) * dot_beta_lateral_front_movement,
            x_values[:, self.v_f_idx] * torch.sin(x_values[:, self.theta_f_idx]) - torch.sin(x_values[:, self.theta_f_idx]) * dot_beta_lateral_front_movement,
            omega_f,
            x_values[:, self.dot_beta_idx],
            u_values[:, self.dot_dot_beta_idx],
            u_values[:, self.a_f_idx],
            torch.zeros(u_values.shape[0]).to(u_values.device),  # constant goal x
            torch.zeros(u_values.shape[0]).to(u_values.device),  # constant goal y
            torch.zeros(u_values.shape[0]).to(u_values.device)   # constant goal theta
        ]).T
        next_state = x_values + self.dt * dot_state
        clamped_next_state = torch.max(torch.min(next_state, self.ubx), self.lbx)
        return clamped_next_state
