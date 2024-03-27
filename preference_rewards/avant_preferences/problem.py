import torch
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from preference_learning.optimization import InfoGainProblem
from preference_learning.dynamics import Dynamics
from preference_learning.costs import Cost
from preference_learning.training_data import TrajectoryDataset
from avant_preferences.dynamics import AvantDynamics


class AvantInfoGainProblem(InfoGainProblem):
    # Parameter indices:
    x_goal_idx = 0
    y_goal_idx = 1
    theta_goal_idx = 2

    def __init__(self, N: int, dynamics: Dynamics, cost_module: Cost, dataset: TrajectoryDataset, n_ensembles: int, parameters_path: str, device: str):
        super().__init__(N, dynamics, cost_module, dataset, n_ensembles, parameters_path, device)

    def visualize(self, x_values: torch.Tensor, u_values: torch.Tensor, p_values: torch.Tensor) -> torch.Tensor:
        num_paths = x_values.shape[0]  # Determine the number of paths.

        if x_values.shape[2] == 12:
            x_values = torch.vstack([x_values[:, :, :6], x_values[:, :, 6:]])
            num_paths = 2

        fig, axs = plt.subplots(1, num_paths, figsize=(10 * num_paths, 10), squeeze=False)

        # Prepare all positions including goals to calculate comprehensive limits
        all_x_positions = np.concatenate([x_values[:, :, 0].numpy().flatten(), p_values[:, 0].numpy()])
        all_y_positions = np.concatenate([x_values[:, :, 1].numpy().flatten(), p_values[:, 1].numpy()])
        x_min, x_max = all_x_positions.min(), all_x_positions.max()
        y_min, y_max = all_y_positions.min(), all_y_positions.max()

        # Add a margin for visibility
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05

        norm = plt.Normalize(vmin=x_values[:, :, 5].numpy().min(), vmax=x_values[:, :, 5].numpy().max())

        for i in range(num_paths):
            x_positions = x_values[i, :, 0].numpy()
            y_positions = x_values[i, :, 1].numpy()
            front_link_angles = x_values[i, :, 2].numpy()
            center_link_angles = x_values[i, :, 3].numpy()
            velocities = x_values[i, :, 5].numpy()

            # Calculate direction for front link
            dx_front = np.cos(front_link_angles)
            dy_front = np.sin(front_link_angles)

            # Calculate adjusted center link angles for direction lines
            adjusted_center_link_angles = front_link_angles + center_link_angles + np.pi
            dx_center = np.cos(adjusted_center_link_angles) * 0.1
            dy_center = np.sin(adjusted_center_link_angles) * 0.1

            quiver = axs[0, i].quiver(x_positions, y_positions, dx_front, dy_front, velocities, cmap='viridis', norm=norm)

            # Plotting center link direction lines
            for j in range(len(x_positions)):
                axs[0, i].plot([x_positions[j], x_positions[j] + dx_center[j]], 
                                [y_positions[j], y_positions[j] + dy_center[j]], 
                                color='gray', linestyle='-', linewidth=2)

            # Plot goal positions with direction
            axs[0, i].quiver(p_values[:, 0].numpy(), p_values[:, 1].numpy(), np.cos(p_values[:, 2].numpy()), np.sin(p_values[:, 2].numpy()), color='red', scale=50)
            axs[0, i].scatter(x_positions[0], y_positions[0], color='red', marker='o')  # Start position

            axs[0, i].set_xlim(x_min - x_margin, x_max + x_margin)
            axs[0, i].set_ylim(y_min - y_margin, y_max + y_margin)
            axs[0, i].grid()
            axs[0, i].set_title(f'Path {i + 1}')
        
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        fig.colorbar(quiver, cax=cbar_ax)

        answer_que = Queue()
        def on_key(event):
            if event.key == 'left':
                answer_que.put(torch.Tensor([1, 0]))
                plt.close(fig)
            elif event.key == 'right':
                answer_que.put(torch.Tensor([0, 1]))
                plt.close(fig)
            elif event.key == 'up':
                answer_que.put(torch.Tensor([1, 1]))
                plt.close(fig)
            elif event.key == 'down':
                answer_que.put(torch.Tensor([0, 0]))
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
        answer = answer_que.get()
        return answer
    
    def generate_p(self, x0: torch.Tensor) -> torch.Tensor:
        p = torch.empty(3)
        dist = torch.tensor([np.random.uniform(0, 10)]).to(x0.device)
        angle = torch.tensor([np.random.uniform(0, 2*np.pi)]).to(x0.device)
        p[self.x_goal_idx] = x0[AvantDynamics.x_f_idx] + dist * torch.cos(angle)
        p[self.y_goal_idx] = x0[AvantDynamics.y_f_idx] + dist * torch.sin(angle)
        p[self.theta_goal_idx] = x0[AvantDynamics.theta_f_idx] + angle
        return p.unsqueeze(0).to(x0.device)
    