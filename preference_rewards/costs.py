import torch
import numpy as np
import os
import json
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
    def forward(self, x_values: torch.Tensor, p_values: torch.Tensor, return_terminal=False) -> torch.Tensor:
        raise NotImplementedError()
    
    @abstractmethod
    def training_loop(self, dataloader: DataLoader):
        raise NotImplementedError()
    
    @abstractmethod
    def get_parameters(self):
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

    def forward(self, x_values: torch.Tensor, p_values: torch.Tensor, return_terminal=False) -> torch.Tensor:
        stage_cost = self._stage_cost(x_values, p_values)
        terminal_cost = self._terminal_cost(x_values, p_values)
        if not return_terminal:
            return stage_cost + terminal_cost
        else:
            return stage_cost + terminal_cost, terminal_cost
    
    def training_loop(self, dataloader: DataLoader):
        raise NotImplementedError("Should train using the class that holds the cost ensemble")
    
    def get_parameters(self):
        params = {
            "discount_power": 1 + 4 * torch.sigmoid(self.discount_power),
            "discount_scaler": torch.sigmoid(self.discount_scaler),
            "heading_error_magnitude": 90 * torch.sigmoid(self.heading_error_magnitude),
            "scaling_radius": 1 + 9 * torch.sigmoid(self.scaling_radius),
            "perpendicular_error_magnitude": 100 * torch.sigmoid(self.perpendicular_error_magnitude),
            "perpendicular_error_shift": 50 * torch.sigmoid(self.perpendicular_error_shift),
            "perpendicular_error_scaler": torch.sigmoid(self.perpendicular_error_scaler),
            "goal_dist_weight": 1 + 9 * torch.sigmoid(self.goal_dist_weight),
        }
        # Convert parameters to item for serialization
        return {k: v.item() for k, v in params.items()}

    
class InfoCost(Cost):
    def __init__(self, N, K, device="cuda:0"):
        super().__init__()
        self.empirical_costs = []
        for _ in range(K):
            self.empirical_costs.append(EmpiricalCost(N, device).to(device))
        self._load_models()

    def generate_p(self, x0: torch.Tensor) -> torch.Tensor:
        return self.empirical_costs[0].generate_p(x0)

    def forward(self, x_values: torch.Tensor, p_values: torch.Tensor, return_terminal=False) -> torch.Tensor:
        # X = M x N x 2*n_states
        # U = M x N x 2*n_controls
        M = x_values.shape[0]
        n_states = int(x_values.shape[2] / 2)
        stacked_x_values = torch.vstack([x_values[:, :, :n_states], x_values[:, :, n_states:]])

        stacked_costs = torch.empty([2*M, len(self.empirical_costs)]).to(x_values.device)
        stacked_terminals = torch.empty([2*M, len(self.empirical_costs)]).to(x_values.device)
        for k, cost in enumerate(self.empirical_costs):
             stacked_cost, stacked_terminal = cost(stacked_x_values, p_values, return_terminal=True)
             stacked_costs[:, k] = stacked_cost
             stacked_terminals[:, k] = stacked_terminal

        traj_A_cost = stacked_costs[:M, :].T
        traj_B_cost = stacked_costs[M:, :].T
        info_gain = information_gain(traj_A_cost, traj_B_cost)

        traj_A_term = stacked_costs[:M, :].mean(dim=1)
        traj_B_term = stacked_costs[M:, :].mean(dim=1)

        return -1e1*info_gain + (traj_A_term + traj_B_term)**2
    
    def training_loop(self, dataloader: DataLoader):
        for k, empirical_cost in enumerate(self.empirical_costs):
            print(f"Training empirical cost model {k+1}/{len(self.empirical_costs)}...")
            optimizer = torch.optim.Adam(empirical_cost.parameters(), lr=1e-2)
            
            for epoch in range(10):
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

                print(f"Model {k+1}, epoch {epoch}, Total Loss: {total_loss}")

        self._save_models()
        self._print_means()
        self._save_means_to_json()

    def _load_models(self, directory="saved_models"):
        for k, empirical_cost in enumerate(self.empirical_costs):
            path = os.path.join(directory, f"empirical_cost_{k}.pt")
            if os.path.isfile(path):
                model_device = next(empirical_cost.parameters()).device
                empirical_cost.load_state_dict(torch.load(path))
                empirical_cost.to(model_device)
                print(f"Loaded weights for empirical_cost_{k} from {path}")
            else:
                print(f"No saved weights found for empirical_cost_{k} at {path}")

    def _save_models(self, directory="saved_models"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for k, empirical_cost in enumerate(self.empirical_costs):
            path = os.path.join(directory, f"empirical_cost_{k}.pt")
            torch.save(empirical_cost.state_dict(), path)
            print(f"Saved empirical_cost_{k} to {path}")

    def _aggregate_and_compute_means(self):
        aggregated_params = {}
        for cost_func in self.empirical_costs:
            params = cost_func.get_parameters()
            for k, v in params.items():
                if k not in aggregated_params:
                    aggregated_params[k] = []
                aggregated_params[k].append(v)
        means = {k: np.mean(v) for k, v in aggregated_params.items()}
        vars = {k: np.var(v) for k, v in aggregated_params.items()}
        return means, vars
    
    def _print_means(self):
        means, vars = self._aggregate_and_compute_means()
        print("Ensemble Mean Parameter Values:")
        for k, v in means.items():
            print(f"{k}: {v} (+/- {2*np.sqrt(vars[k])})")
    
    def _save_means_to_json(self, file_path="ensemble_mean_parameters.json"):
        means, _ = self._aggregate_and_compute_means()
        with open(file_path, 'w') as json_file:
            json.dump(means, json_file, indent=4)
        print(f"Saved ensemble mean parameter values to {file_path}")

    def get_parameters(self):
        pass
