import torch
import avant_preferences.config as config
from preference_learning.dynamics import Dynamics


class AvantDynamics(Dynamics):
    # State indices:
    x_f_idx = 0
    y_f_idx = 1
    theta_f_idx = 2
    beta_idx = 3
    dot_beta_idx = 4
    v_f_idx = 5

    # Control indices:
    dot_dot_beta_idx = 0
    a_f_idx = 1

    def __init__(self, dt: float, device: str):
        # Define control bounds:
        lbu = torch.tensor([-config.avant_max_dot_dot_beta, -config.avant_max_a]).to(device)
        ubu = torch.tensor([config.avant_max_dot_dot_beta, config.avant_max_a]).to(device)
        # Define state bounds:
        lbx = torch.tensor([-torch.inf, -torch.inf, -torch.inf, -config.avant_max_beta, -config.avant_max_dot_beta, config.avant_min_v]).to(device)
        ubx = torch.tensor([torch.inf, torch.inf, torch.inf, config.avant_max_beta, config.avant_max_dot_beta, config.avant_max_v]).to(device)
        super().__init__(dt, lbu=lbu, ubu=ubu, lbx=lbx, ubx=ubx)
        
        # Define initial state sampling distribution:
        lbx_initial = torch.tensor([-20, -20, 0, -1e-5, -1e-5, -1e-5]).to(device)
        ubx_initial = torch.tensor([20, 20, 2*torch.pi, 1e-5, 1e-5, 1e-5]).to(device)
        self.sampler = torch.distributions.uniform.Uniform(lbx_initial, ubx_initial)

    def _discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor, dt: float) -> torch.Tensor:
        alpha = torch.pi + x_values[:, self.beta_idx]
        omega_f = x_values[:, self.v_f_idx] * config.avant_lf / torch.tan(alpha/2)
        dot_state = torch.vstack([
            x_values[:, self.v_f_idx] * torch.cos(x_values[:, self.theta_f_idx]),
            x_values[:, self.v_f_idx] * torch.sin(x_values[:, self.theta_f_idx]),
            omega_f,
            x_values[:, self.dot_beta_idx],
            u_values[:, self.dot_dot_beta_idx],
            u_values[:, self.a_f_idx]
        ]).T
        return x_values + dt * dot_state

    def generate_initial_state(self) -> torch.Tensor:
        x = self.sampler.sample()
        return x