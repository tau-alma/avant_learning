import torch
import config
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict


class Dynamics(ABC):
    def __init__(self, dt, lbu, ubu):
        self.dt = dt
        self.lbu = lbu
        self.ubu = ubu

    def propagate(self, x0: torch.Tensor, u_values: torch.Tensor):
        x_values = torch.empty((u_values.shape[0], u_values.shape[1], len(x0))).to(x0.device)
        x = x0.tile(u_values.shape[0], 1)
        for n in range(u_values.shape[1]):
            x = self._discrete_dynamics_fun(x, u_values[:, n, :], self.dt)
            x_values[:, n, :] = x
        return x_values

    @abstractmethod
    def _discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor, dt: float) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_constraint_violation(x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def generate_initial_state(self) -> torch.Tensor:
        pass


class AvantDynamics(Dynamics):
    x_f_idx = 0
    y_f_idx = 1
    theta_f_idx = 2
    beta_idx = 3
    dot_beta_idx = 4
    v_f_idx = 5

    dot_dot_beta_idx = 0
    a_f_idx = 1

    def __init__(self, dt, device="cuda:0"):
        lbu = torch.tensor([-config.avant_max_dot_dot_beta, -config.avant_max_a]).to(device)
        ubu = torch.tensor([config.avant_max_dot_dot_beta, config.avant_max_a]).to(device)
        super().__init__(dt, lbu, ubu)
        self.device = device
        self.lbx = torch.tensor([-20, -20, 0, -config.avant_max_beta, -config.avant_max_dot_beta, config.avant_min_v]).to(device)
        self.ubx = torch.tensor([20, 20, 2*np.pi, config.avant_max_beta, config.avant_max_dot_beta, config.avant_max_v]).to(device)
        self.sampler = torch.distributions.uniform.Uniform(self.lbx, self.ubx)

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

    def compute_constraint_violation(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        # Compute absolute state violations
        state_lower_violations = torch.clamp(self.lbx - x_values, min=0).sum(dim=[1, 2])
        state_upper_violations = torch.clamp(x_values - self.ubx, min=0).sum(dim=[1, 2])
        
        # Compute absolute control violations
        control_lower_violations = torch.clamp(self.lbu - u_values, min=0).sum(dim=[1, 2])
        control_upper_violations = torch.clamp(u_values - self.ubu, min=0).sum(dim=[1, 2])

        # Aggregate violations
        total_violations = state_lower_violations + state_upper_violations + control_lower_violations + control_upper_violations

        return total_violations

    def generate_initial_state(self) -> torch.Tensor:
        x = self.sampler.sample()
        # Adjust the initial beta based on dot beta (higher dot beta -> lower beta, to avoid inevitable constraint violations):
        x[self.beta_idx] -= (config.avant_max_beta - x[self.beta_idx]) / (x[self.dot_beta_idx] / config.avant_max_dot_dot_beta)
        # x[3:] = 0
        return x
    

# Wrapper that parallelizes the optimization of two trajectories:
class DualAvantDynamics(Dynamics):
    def __init__(self, dt, device="cuda:0"):
        self.avant_dynamics = AvantDynamics(dt, device)
        lbu = self.avant_dynamics.lbu.tile(2)
        ubu = self.avant_dynamics.ubu.tile(2)
        super().__init__(dt, lbu, ubu)

    def _discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor, dt: float) -> torch.Tensor:
        # X = M x 2*n_states
        # U = M x 2*n_controls
        M = x_values.shape[0]
        n_states = len(self.avant_dynamics.lbx)
        n_controls = len(self.avant_dynamics.lbu)
        stacked_x_values = torch.vstack([x_values[:, :n_states], x_values[:, n_states:]])
        stacked_u_values = torch.vstack([u_values[:, :n_controls], u_values[:, n_controls:]])

        stacked_next_states = self.avant_dynamics._discrete_dynamics_fun(stacked_x_values, stacked_u_values, dt)

        unstacked_next_states = torch.hstack([stacked_next_states[:M, :], stacked_next_states[M:, :]])
        return unstacked_next_states

    def compute_constraint_violation(self, x_values: torch.Tensor, u_values: torch.Tensor) -> torch.Tensor:
        # X = M x N x 2*n_states
        # U = M x N x 2*n_controls
        M = x_values.shape[0]
        n_states = len(self.avant_dynamics.lbx)
        n_controls = len(self.avant_dynamics.lbu)
        stacked_x_values = torch.vstack([x_values[:, :, :n_states], x_values[:, :, n_states:]])
        stacked_u_values = torch.vstack([u_values[:, :, :n_controls], u_values[:, :, n_controls:]])
        
        stacked_total_violations = self.avant_dynamics.compute_constraint_violation(stacked_x_values, stacked_u_values)

        unstacked_total_violations = torch.vstack([stacked_total_violations[:M], stacked_total_violations[M:]])
        return unstacked_total_violations.sum(dim=0)

    
    def generate_initial_state(self) -> torch.Tensor:
        return self.avant_dynamics.generate_initial_state().tile(2)
    


if __name__ == "__main__":
    a = torch.arange(20).reshape(5, 4)
    print(a)
    b = torch.vstack([a[:, :2], a[:, 2:]])
    print(b)
    c = torch.hstack([b[:5, :], b[5:, :]])
    print(c)