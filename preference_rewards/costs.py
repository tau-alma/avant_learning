import torch
import numpy as np
from abc import ABC, abstractmethod
from dynamics import AvantDynamics
from utils import information_gain, preference_loss
from torch.utils.data import DataLoader


class Cost(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate_p(self, x0: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    @abstractmethod
    def training_loop(self, dataloader: DataLoader):
        raise NotImplementedError()
    
    
class EmpiricalCost(Cost):
    x_goal_idx = 0
    y_goal_idx = 1
    theta_goal_idx = 2

    def __init__(self, N, device="cuda:0"):
        super().__init__()
        self.steps = torch.arange(N-1).to(device)

        params = [
            'goal_dist_weight',
            'heading_error_magnitude',
            'scaling_radius',
            'perpendicular_error_magnitude',
            'perpendicular_error_shift',
            'perpendicular_error_scaler',
            'discount_power',
            'discount_scaler',
        ]
        for param_name in params:
            initialized_param = torch.nn.Parameter(torch.normal(mean=torch.tensor([0.]), std=torch.tensor([10.])).to(device))
            setattr(self, param_name, initialized_param)

    def generate_p(self, x0: torch.Tensor) -> torch.Tensor:
        p = torch.empty(3)
        dist = torch.tensor([np.random.uniform(0, 10)]).to(x0.device)
        angle = torch.tensor([np.random.uniform(0, 2*np.pi)]).to(x0.device)
        p[self.x_goal_idx] = x0[AvantDynamics.x_f_idx] + dist * torch.cos(angle)
        p[self.y_goal_idx] = x0[AvantDynamics.y_f_idx] + dist * torch.sin(angle)
        p[self.theta_goal_idx] = x0[AvantDynamics.theta_f_idx] + angle
        return p.unsqueeze(0).to(x0.device)
    
    def _stage_cost(self, x_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        x_f = x_values[:, :-1, AvantDynamics.x_f_idx]
        y_f = x_values[:, :-1, AvantDynamics.y_f_idx]
        theta_f = x_values[:, :-1, AvantDynamics.theta_f_idx]

        discount_power = (1 + 4 * torch.sigmoid(self.discount_power))
        discount_scaler = torch.sigmoid(self.discount_scaler)
        heading_error_magnitude = 90 * torch.sigmoid(self.heading_error_magnitude)
        scaling_radius = (1 + 9 * torch.sigmoid(self.scaling_radius))
        perpendicular_error_magnitude = 100 * torch.sigmoid(self.perpendicular_error_magnitude)
        perpendicular_error_shift = 50 * torch.sigmoid(self.perpendicular_error_shift)
        perpendicular_error_scaler = torch.sigmoid(self.perpendicular_error_scaler)
        goal_dist_weight = (1 + 9 * torch.sigmoid(self.goal_dist_weight))

        discounts = (1 - torch.pow(discount_power, -discount_scaler * self.steps))
        
        goal_dist = (x_f - p_values[:, self.x_goal_idx])**2 + (y_f - p_values[:, self.y_goal_idx])**2
        goal_heading = heading_error_magnitude * (1 - torch.cos(p_values[:, self.theta_goal_idx] - theta_f))
        radius_scaler = torch.exp(-(goal_dist/scaling_radius)**2)

        perpendicular_dist = ((x_f - p_values[:, self.x_goal_idx]) * torch.cos(p_values[:, self.theta_goal_idx]) 
                            + (y_f - p_values[:, self.y_goal_idx]) * torch.sin(p_values[:, self.theta_goal_idx]))
        
        e_perp = perpendicular_error_magnitude * (
            torch.tanh((perpendicular_dist - perpendicular_error_shift) * perpendicular_error_scaler) + 1
        )

        C = discounts * (goal_dist_weight * goal_dist + goal_heading * radius_scaler + e_perp)
        return C.sum(dim=1)

    def _terminal_cost(self, x_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        x_f = x_values[:, -1, AvantDynamics.x_f_idx]
        y_f = x_values[:, -1, AvantDynamics.y_f_idx]
        theta_f = x_values[:, -1, AvantDynamics.theta_f_idx]

        heading_error_magnitude = 90 * torch.sigmoid(self.heading_error_magnitude)
        scaling_radius = (1 + 9 * torch.sigmoid(self.scaling_radius))
        goal_dist_weight = (1 + 9 * torch.sigmoid(self.goal_dist_weight))
        
        goal_dist = (x_f - p_values[:, self.x_goal_idx])**2 + (y_f - p_values[:, self.y_goal_idx])**2
        goal_heading = heading_error_magnitude * (1 - torch.cos(p_values[:, self.theta_goal_idx] - theta_f))
        radius_scaler = torch.exp(-(goal_dist/scaling_radius)**2)

        C = (goal_dist_weight * goal_dist + goal_heading * radius_scaler)
        return C

    def forward(self, x_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        stage_cost = self._stage_cost(x_values, p_values)
        terminal_cost = self._terminal_cost(x_values, p_values)
        return stage_cost + terminal_cost
    
    def training_loop(self, dataloader: DataLoader):
        raise NotImplementedError("Should train using the class that holds the cost ensemble")

    
class InfoCost(Cost):
    def __init__(self, N, K, device="cuda:0"):
        super().__init__()
        self.empirical_costs = []
        for _ in range(K):
            self.empirical_costs.append(EmpiricalCost(N, device).to(device))

    def generate_p(self, x0: torch.Tensor) -> torch.Tensor:
        return self.empirical_costs[0].generate_p(x0)

    def forward(self, x_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        # X = M x N x 2*n_states
        # U = M x N x 2*n_controls
        M = x_values.shape[0]
        n_states = int(x_values.shape[2] / 2)
        stacked_x_values = torch.vstack([x_values[:, :, :n_states], x_values[:, :, n_states:]])

        stacked_costs = torch.empty([2*M, len(self.empirical_costs)]).to(x_values.device)
        #stacked_terminals = torch.empty([2*M, len(self.empirical_costs)]).to(x_values.device)
        for k, cost in enumerate(self.empirical_costs):
             stacked_cost = cost(stacked_x_values, p_values)
             stacked_costs[:, k] = stacked_cost
        #    stacked_terminals[:, k] = stacked_terminal

        traj_A_cost = stacked_costs[:M, :].T
        traj_B_cost = stacked_costs[M:, :].T
        info_gain = information_gain(traj_A_cost, traj_B_cost)

        return -1e2*info_gain #+ 1e-1*(traj_A_term + traj_B_term)
    
    def training_loop(self, dataloader: DataLoader):
        for k, empirical_cost in enumerate(self.empirical_costs):
            print(f"Training empirical cost model {k+1}/{len(self.empirical_costs)}...")
            optimizer = torch.optim.Adam(empirical_cost.parameters(), lr=1e-1)
            
            total_loss = 0
            for x_values, y_values, p_values in dataloader:
                model_device = next(empirical_cost.parameters()).device
                x_values, y_values, p_values = x_values.to(model_device), y_values.to(model_device), p_values.to(model_device)
                optimizer.zero_grad()
                
                # Forward pass for the current empirical cost model
                M = x_values.shape[0]
                n_states = int(x_values.shape[2] / 2)
                stacked_x_values = torch.vstack([x_values[:, :, :n_states], x_values[:, :, n_states:]])
                stacked_costs = empirical_cost.forward(stacked_x_values, p_values)
                traj_A_cost = stacked_costs[:M]
                traj_B_cost = stacked_costs[M:]

                loss = preference_loss(traj_A_cost, traj_B_cost, y_values)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Model {k+1}, Total Loss: {total_loss}")
