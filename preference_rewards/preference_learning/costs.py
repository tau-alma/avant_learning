import torch
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict
from preference_learning.utils import information_gain, preference_loss


class Cost(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x_values: torch.Tensor, p_values: torch.Tensor, return_terminal=False) -> torch.Tensor:
        raise NotImplementedError()
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        raise NotImplementedError()
    
class InfoCost:
    def __init__(self, N: int, K: int, cost_module: Cost, parameters_path: str, device: str):
        super().__init__()
        self.empirical_costs = []
        for _ in range(K):
            self.empirical_costs.append(cost_module(N=N, device=device).to(device))

        self.parameters_path = parameters_path
        if not os.path.exists(self.parameters_path):
            os.makedirs(self.parameters_path, exist_ok=True)

        self._load_models()

    def get_cost(self, x_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
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
    
    def training_loop(self, dataloader: DataLoader, n_epochs=10, lr=1e-1):
        for k, empirical_cost in enumerate(self.empirical_costs):
            print(f"Training empirical cost model {k+1}/{len(self.empirical_costs)}...")
            optimizer = torch.optim.Adam(empirical_cost.parameters(), lr=lr)

            for epoch in range(n_epochs):
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

    def _load_models(self):
        for k, empirical_cost in enumerate(self.empirical_costs):
            path = os.path.join(self.parameters_path, f"empirical_cost_{k}.pt")
            if os.path.isfile(path):
                empirical_cost.load_state_dict(torch.load(path, map_location=self.device))
                empirical_cost.to(self.device)
                print(f"Loaded weights for empirical_cost_{k} from {path}")
            else:
                print(f"No saved weights found for empirical_cost_{k} at {path}")

    def _save_models(self):
        if not os.path.exists(self.parameters_path):
            os.makedirs(self.parameters_path)
        for k, empirical_cost in enumerate(self.empirical_costs):
            path = os.path.join(self.parameters_path, f"empirical_cost_{k}.pt")
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
    
    def _save_means_to_json(self):
        means, _ = self._aggregate_and_compute_means()
        file_path = os.path.join(self.parameters_path, "ensemble_mean_parameters.json")
        with open(file_path, 'w') as json_file:
            json.dump(means, json_file, indent=4)
        print(f"Saved ensemble mean parameter values to {file_path}")
