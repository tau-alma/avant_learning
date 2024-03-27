import torch
from preference_learning.costs import Cost
from avant_preferences.dynamics import AvantDynamics
from avant_preferences.problem import AvantInfoGainProblem
from typing import Dict

class AvantEmpiricalCost(Cost):
    def __init__(self, N: int, device: str):
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
        
        goal_dist = (x_f - p_values[:, AvantInfoGainProblem.x_goal_idx])**2 + (y_f - p_values[:, AvantInfoGainProblem.y_goal_idx])**2
        goal_heading = heading_error_magnitude * (1 - torch.cos(p_values[:, AvantInfoGainProblem.theta_goal_idx] - theta_f))
        radius_scaler = torch.exp(-(goal_dist/scaling_radius)**2)

        perpendicular_dist = ((x_f - p_values[:, AvantInfoGainProblem.x_goal_idx]) * torch.cos(p_values[:, AvantInfoGainProblem.theta_goal_idx]) 
                            + (y_f - p_values[:, AvantInfoGainProblem.y_goal_idx]) * torch.sin(p_values[:, AvantInfoGainProblem.theta_goal_idx]))
        
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
        
        goal_dist = (x_f - p_values[:, AvantInfoGainProblem.x_goal_idx])**2 + (y_f - p_values[:, AvantInfoGainProblem.y_goal_idx])**2
        goal_heading = heading_error_magnitude * (1 - torch.cos(p_values[:, AvantInfoGainProblem.theta_goal_idx] - theta_f))
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
    
    def get_parameters(self) -> Dict[str, float]:
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
