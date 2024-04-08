import torch


class AvantCost:
    def __init__(self) -> None:
        self.model = torch.load("critic").eval().cuda()

    def stage_cost(self, x_values: torch.Tensor, u_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        dot_dot_deta = u_values[:, 0]
        a_f = u_values[:, 1]

        C = (1e-3*dot_dot_deta)**2 + (1e-3*a_f)**2

        return C

    def terminal_cost(self, x_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        x_f = x_values[:, 0]
        y_f = x_values[:, 1]
        theta_f = x_values[:, 2]
        beta = x_values[:, 3]
        dot_beta = x_values[:, 4]
        v_f = x_values[:, 5]

        zeros = torch.zeros_like(x_f).to(x_values.device)
        
        x_goal = p_values[0].tile(x_values.shape[0])
        y_goal = p_values[1].tile(x_values.shape[0])
        theta_goal = p_values[2].tile(x_values.shape[0])

        input_tensor = torch.vstack([
            x_f, y_f, torch.sin(theta_f), torch.cos(theta_f),
            x_goal, y_goal, torch.sin(theta_goal), torch.cos(theta_goal),
            beta, dot_beta, v_f,
            zeros, zeros
        ]).T

        C = -self.model(input_tensor)

        return C
        