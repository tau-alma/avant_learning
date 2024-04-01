import torch
import config as config


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

    position_bound = 20

    def __init__(self, dt: float, device: str):
        self.dt = dt
        # Define control normalization constants:
        self.control_scalers = torch.tensor([config.avant_max_dot_dot_beta, config.avant_max_a], dtype=torch.float32).to(device)
        # Define state bounds:
        self.lbx = torch.tensor([-torch.inf, -torch.inf, -torch.inf, -config.avant_max_beta, -config.avant_max_dot_beta, config.avant_min_v, -torch.inf, -torch.inf, 0], dtype=torch.float32).to(device)
        self.ubx = torch.tensor([torch.inf, torch.inf, torch.inf, config.avant_max_beta, config.avant_max_dot_beta, config.avant_max_v, torch.inf, torch.inf, 2*torch.pi], dtype=torch.float32).to(device)
        
        # Define initial state sampling distribution:
        lbx_initial = torch.tensor([-self.position_bound, -self.position_bound, 0, -1e-5, -1e-5, -1e-5], dtype=torch.float32).to(device)
        ubx_initial = torch.tensor([self.position_bound, self.position_bound, 2*torch.pi, 1e-5, 1e-5, 1e-5], dtype=torch.float32).to(device)
        self.state_sampler = torch.distributions.uniform.Uniform(lbx_initial, ubx_initial)
        # Define goal offset sampling distribution:
        lbg = torch.tensor([5, 0.0], dtype=torch.float32).to(device)
        ubg = torch.tensor([15.0, 2*torch.pi], dtype=torch.float32).to(device)
        self.goal_offset_sampler = torch.distributions.uniform.Uniform(lbg, ubg)

    def discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        u_values = self.control_scalers * u_values
        alpha = torch.pi + x_values[:, self.beta_idx]
        omega_f = x_values[:, self.v_f_idx] * config.avant_lf / torch.tan(alpha/2)
        dot_state = torch.vstack([
            x_values[:, self.v_f_idx] * torch.cos(x_values[:, self.theta_f_idx]),
            x_values[:, self.v_f_idx] * torch.sin(x_values[:, self.theta_f_idx]),
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
    
    def generate_initial_state(self, N: int) -> torch.Tensor:
        x = self.state_sampler.sample((N,))
        goal_offset = self.goal_offset_sampler.sample((N,))
        goal_dist = goal_offset[:, 0]
        goal_angle = goal_offset[:, 1]

        goal = torch.empty((N, 3)).to(x.device)
        goal[:, 0] = x[:, AvantDynamics.x_f_idx] + goal_dist * torch.cos(goal_angle)
        goal[:, 1] = x[:, AvantDynamics.y_f_idx] + goal_dist * torch.sin(goal_angle)
        goal[:, 2] = x[:, AvantDynamics.theta_f_idx] + goal_angle
    
        return torch.hstack([x, goal])