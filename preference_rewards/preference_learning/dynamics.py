import torch
from abc import ABC, abstractmethod


class Dynamics(ABC):
    def __init__(self, dt: float, lbu: torch.Tensor, ubu: torch.Tensor, lbx=None, ubx=None):
        self.dt = dt
        self.lbu = lbu
        self.ubu = ubu
        self.lbx = lbx
        self.ubx = ubx

    def propagate(self, x0: torch.Tensor, u_values: torch.Tensor):
        x_values = torch.empty((u_values.shape[0], u_values.shape[1], len(x0))).to(x0.device)
        x = x0.tile(u_values.shape[0], 1)
        for n in range(u_values.shape[1]):
            x = self._discrete_dynamics_fun(x, u_values[:, n, :], self.dt)
            x_values[:, n, :] = x
        return x_values
    
    def compute_constraint_violation(self, u_values: torch.Tensor) -> torch.Tensor:
        # Compute absolute control violations
        control_lower_violations = torch.clamp(self.lbu - u_values, min=0).sum(dim=[1, 2])
        control_upper_violations = torch.clamp(u_values - self.ubu, min=0).sum(dim=[1, 2])

        # Aggregate violations
        total_violations = control_lower_violations + control_upper_violations

        return total_violations
    
    @abstractmethod
    def _discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor, dt: float) -> torch.Tensor:
        pass
    
    @abstractmethod
    def generate_initial_state(self) -> torch.Tensor:
        pass


# Wrapper that parallelizes the optimization of two trajectories:
class DualDynamics(Dynamics):
    def __init__(self, dynamics: Dynamics):
        self.dynamics = dynamics
        lbu = self.dynamics.lbu.tile(2)
        ubu = self.dynamics.ubu.tile(2)
        super().__init__(self.dynamics.dt, lbu, ubu)

    def _discrete_dynamics_fun(self, x_values: torch.Tensor, u_values: torch.Tensor, dt: float) -> torch.Tensor:
        # X = M x 2*n_states
        # U = M x 2*n_controls
        M = x_values.shape[0]
        n_states = len(self.dynamics.lbx)
        n_controls = len(self.dynamics.lbu)
        stacked_x_values = torch.vstack([x_values[:, :n_states], x_values[:, n_states:]])
        stacked_u_values = torch.vstack([u_values[:, :n_controls], u_values[:, n_controls:]])

        stacked_next_states = self.dynamics._discrete_dynamics_fun(stacked_x_values, stacked_u_values, dt)
        # Clamp states to lie between constraints:
        stacked_next_states = torch.max(torch.min(stacked_next_states, self.dynamics.ubx), self.dynamics.lbx)
    
        unstacked_next_states = torch.hstack([stacked_next_states[:M, :], stacked_next_states[M:, :]])
        return unstacked_next_states

    def compute_constraint_violation(self, u_values: torch.Tensor) -> torch.Tensor:
        # X = M x N x 2*n_states
        # U = M x N x 2*n_controls
        M = u_values.shape[0]
        n_controls = len(self.dynamics.lbu)
        stacked_u_values = torch.vstack([u_values[:, :, :n_controls], u_values[:, :, n_controls:]])
        
        stacked_total_violations = self.dynamics.compute_constraint_violation(stacked_u_values)

        unstacked_total_violations = torch.vstack([stacked_total_violations[:M], stacked_total_violations[M:]])
        return unstacked_total_violations.sum(dim=0)
    
    def generate_initial_state(self) -> torch.Tensor:
        return self.dynamics.generate_initial_state().tile(2)