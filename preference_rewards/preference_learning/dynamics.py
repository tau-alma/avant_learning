import torch
from abc import ABC, abstractmethod


class Dynamics(ABC):
    def __init__(self, dt: float, lbu: torch.Tensor, ubu: torch.Tensor):
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