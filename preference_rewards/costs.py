import torch
import numpy as np
from abc import ABC, abstractmethod
from dynamics import AvantDynamics
from utils import information_gain


class Cost(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate_p(self, x0: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x_values: torch.Tensor, u_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    
class EmpiricalCost(Cost):
    x_goal_idx = 0
    y_goal_idx = 1
    theta_goal_idx = 2

    def __init__(self, N, device="cuda:0"):
        super().__init__()

        self.steps = torch.arange(N-1).to(device)

        self.goal_dist_weight = torch.nn.Parameter(torch.randn(1))

        self.heading_error_magnitude = torch.nn.Parameter(torch.randn(1))
        self.scaling_radius = torch.nn.Parameter(torch.randn(1))

        self.perpendicular_error_magnitude = torch.nn.Parameter(torch.randn(1))
        self.perpendicual_error_shift = torch.nn.Parameter(torch.randn(1))
        self.perpendicular_error_scaler = torch.nn.Parameter(torch.randn(1))

        self.discount_power = torch.nn.Parameter(torch.randn(1))
        self.discount_divider = torch.nn.Parameter(torch.randn(1))

    def init_test_weights(self):
        self.goal_dist_weight = torch.nn.Parameter(torch.tensor([10.]))
        self.heading_error_magnitude = torch.nn.Parameter(torch.tensor([90.]))
        self.scaling_radius = torch.nn.Parameter(torch.tensor([1.5]))
        self.perpendicular_error_magnitude = torch.nn.Parameter(torch.tensor([0.]))
        self.perpendicual_error_shift = torch.nn.Parameter(torch.tensor([0.]))
        self.perpendicular_error_scaler = torch.nn.Parameter(torch.tensor([0.]))
        self.discount_power = torch.nn.Parameter(torch.tensor([2.]))
        self.discount_scaler = torch.nn.Parameter(torch.tensor([1/10]))

    def generate_p(self, x0: torch.Tensor) -> torch.Tensor:
        p = torch.empty(3)
        dist = torch.tensor([np.random.uniform(0, 10)]).to(x0.device)
        angle = torch.tensor([np.random.uniform(0, 2*np.pi)]).to(x0.device)
        p[self.x_goal_idx] = x0[AvantDynamics.x_f_idx] + dist * torch.cos(angle)
        p[self.y_goal_idx] = x0[AvantDynamics.y_f_idx] + dist * torch.sin(angle)
        p[self.theta_goal_idx] = x0[AvantDynamics.theta_f_idx] + angle
        return p.unsqueeze(0).to(x0.device)
    
    def _stage_cost(self, x_values: torch.Tensor, u_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        x_f = x_values[:, :-1, AvantDynamics.x_f_idx]
        y_f = x_values[:, :-1, AvantDynamics.y_f_idx]
        theta_f = x_values[:, :-1, AvantDynamics.theta_f_idx]

        discounts = (1 - torch.pow(self.discount_power, -self.discount_scaler * self.steps))
        
        goal_dist = (x_f - p_values[:, self.x_goal_idx])**2 + (y_f - p_values[:, self.y_goal_idx])**2
        goal_heading = self.heading_error_magnitude * (1 - torch.cos(p_values[:, self.theta_goal_idx] - theta_f))
        radius_scaler = torch.exp(-(goal_dist/self.scaling_radius)**2)

        perpendicular_dist = ((x_f - p_values[:, self.x_goal_idx]) * torch.cos(p_values[:, self.theta_goal_idx]) 
                            + (y_f - p_values[:, self.y_goal_idx]) * torch.sin(p_values[:, self.theta_goal_idx]))
        
        e_perp = self.perpendicular_error_magnitude * (
            torch.tanh((perpendicular_dist - self.perpendicual_error_shift) / self.perpendicular_error_scaler) + 1
        )

        C = discounts * (self.goal_dist_weight * goal_dist + goal_heading * radius_scaler + e_perp)
        return C.sum(dim=1)

    def _terminal_cost(self, x_values: torch.Tensor, u_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        x_f = x_values[:, -1, AvantDynamics.x_f_idx]
        y_f = x_values[:, -1, AvantDynamics.y_f_idx]
        theta_f = x_values[:, -1, AvantDynamics.theta_f_idx]
        
        goal_dist = (x_f - p_values[:, self.x_goal_idx])**2 + (y_f - p_values[:, self.y_goal_idx])**2
        goal_heading = self.heading_error_magnitude * (1 - torch.cos(p_values[:, self.theta_goal_idx] - theta_f))
        radius_scaler = torch.exp(-(goal_dist/self.scaling_radius)**2)

        C = (self.goal_dist_weight * goal_dist + goal_heading * radius_scaler)
        return C

    def forward(self, x_values: torch.Tensor, u_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        stage_cost = self._stage_cost(x_values, u_values, p_values)
        terminal_cost = self._terminal_cost(x_values, u_values, p_values)
        return stage_cost + terminal_cost
    
    
class InfoCost(Cost):
    def __init__(self, N, device="cuda:0"):
        super().__init__()
        self.empirical_cost = EmpiricalCost(N, device)
        self.empirical_cost.init_test_weights()
        self.empirical_cost.to(device)

    def generate_p(self, x0: torch.Tensor) -> torch.Tensor:
        return self.empirical_cost.generate_p(x0)

    def forward(self, x_values: torch.Tensor, u_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        # X = M x N x 2*n_states
        # U = M x N x 2*n_controls
        M = x_values.shape[0]
        n_states = int(x_values.shape[2] / 2)
        n_controls = int(u_values.shape[2] / 2)
        stacked_x_values = torch.vstack([x_values[:, :, :n_states], x_values[:, :, n_states:]])
        stacked_u_values = torch.vstack([u_values[:, :, :n_controls], u_values[:, :, n_controls:]])

        # 2*M x N costs
        stacked_costs = self.empirical_cost(stacked_x_values, stacked_u_values, p_values).unsqueeze(1)

        traj_A_cost = stacked_costs[:M, None, :]
        traj_B_cost = stacked_costs[M:, None, :]

        info_gain = information_gain(traj_A_cost, traj_B_cost)
        
        return info_gain