import torch
import config


class AvantDynamics:
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
        self.dt = dt
        # Define control bounds:
        self.lbu = torch.tensor([-config.avant_max_dot_dot_beta, -config.avant_max_a]).to(device)
        self.ubu = torch.tensor([config.avant_max_dot_dot_beta, config.avant_max_a]).to(device)
        # Define state bounds:
        self.lbx = torch.tensor([-torch.inf, -torch.inf, -torch.inf, -config.avant_max_beta, -config.avant_max_dot_beta, config.avant_min_v]).to(device)
        self.ubx = torch.tensor([torch.inf, torch.inf, torch.inf, config.avant_max_beta, config.avant_max_dot_beta, config.avant_max_v]).to(device)
        
        # Define initial state sampling distribution:
        lbx_initial = torch.tensor([-20, -20, 0, -1e-5, -1e-5, -1e-5]).to(device)
        ubx_initial = torch.tensor([20, 20, 2*torch.pi, 1e-5, 1e-5, 1e-5]).to(device)
        self.sampler = torch.distributions.uniform.Uniform(lbx_initial, ubx_initial)

    def propagate(self, x0: torch.Tensor, u_values: torch.Tensor):
        x_values = torch.empty((u_values.shape[0], u_values.shape[1]+1, len(x0))).to(x0.device)
        x_values[:, 0, :] = x0
        x = x0.tile(u_values.shape[0], 1)
        for n in range(u_values.shape[1]):
            x = self._discrete_dynamics_fun(x, u_values[:, n, :])
            x_values[:, n+1, :] = x
        return x_values

    def _discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
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
        return x_values + self.dt * dot_state