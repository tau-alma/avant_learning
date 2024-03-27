import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
from matplotlib.widgets import Button
from queue import Queue


class TruncatedNormal:
    def __init__(self, lower_bound, upper_bound, mean, std):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mean = mean
        self.std = std

        # Convert actual bounds to z-scores
        self.a = (self.lower_bound - self.mean) / self.std
        self.b = (self.upper_bound - self.mean) / self.std

    def sample(self, size=1):
        return truncnorm.rvs(self.a, self.b, loc=self.mean, scale=self.std, size=size)


# TODO: is this computed correctly when shape = MxN?
def preference_probability(trajA_cost: torch.Tensor, trajB_cost: torch.Tensor) -> torch.Tensor:
    return torch.exp(-trajA_cost) / (torch.exp(-trajA_cost) + torch.exp(-trajB_cost))


def preference_loss(trajA_costs: torch.Tensor, trajB_costs: torch.Tensor, preference_labels: torch.Tensor):
    """
    Parameters:
    - trajA_costs: length M tensor, where M is the number of trajectory pairs
    - trajB_costs: length M tensor, where M is the number of trajectory pairs
    - preference_labels: Mx2 tensor, where M is the number of trajectory pairs
    """
    assert len(preference_labels.shape) == 2 and (len(trajA_costs) == len(preference_labels) and len(trajB_costs) == len(preference_labels))
    assert trajA_costs.amin().item() >= 0 and trajB_costs.amin().item() >= 0

    # Normalize so we don't end up with NAN values due to exp(-big number):
    max_A = trajA_costs.amax()
    max_B = trajB_costs.amax()
    scaler = torch.amax(torch.vstack([max_A, max_B]))
    # Avoid in-place operations
    norm_trajA_costs = trajA_costs / scaler
    norm_trajB_costs = trajB_costs / scaler

    return -(
        preference_labels[:, 0] * torch.log(preference_probability(norm_trajA_costs, norm_trajB_costs)) 
        + preference_labels[:, 1] * torch.log(preference_probability(norm_trajB_costs, norm_trajA_costs))
    )    


def information_gain(trajA_costs: torch.Tensor, trajB_costs: torch.Tensor) -> torch.Tensor:
    """
    Compute the information gain for each pair of trajectories.
    
    Parameters:
    - trajA_costs: KxM tensor, where K is the number of cost function ensembles, and M is the number of trajectory pairs
    - trajB_costs: KxM tensor, where K is the number of cost function ensembles, and M is the number of trajectory pairs
    
    Returns:
    - An M-element tensor containing the information gain for each trajectory pair.
    """

    assert trajA_costs.amin().item() >= 0 and trajB_costs.amin().item() >= 0

    # Normalize so we don't end up with NAN values due to exp(-big number):
    max_A = trajA_costs.amax()
    max_B = trajB_costs.amax()
    scaler = torch.amax(torch.vstack([max_A, max_B]))
    trajA_costs /= scaler
    trajB_costs /= scaler

    # Calculate preference probabilities for both orders
    pref_prob_a_b = preference_probability(trajA_costs, trajB_costs)
    pref_prob_b_a = preference_probability(trajB_costs, trajA_costs)

    # Sum the preference probabilities over all ensembles for each trajectory
    pref_probs_sum_a = torch.sum(pref_prob_a_b, dim=0)
    pref_probs_sum_b = torch.sum(pref_prob_b_a, dim=0)

    # Compute the log probabilities for each preference
    M = trajA_costs.size(0)
    log_prob_a_b = torch.log2(M * pref_prob_a_b / (pref_probs_sum_a + 1e-9))
    log_prob_b_a = torch.log2(M * pref_prob_b_a / (pref_probs_sum_b + 1e-9))

    # Calculate the information gain for each preference
    info_gain_a = pref_prob_a_b * log_prob_a_b
    info_gain_b = pref_prob_b_a * log_prob_b_a
    
    # Sum the information gain over all ensembles for each pair of trajectories
    info_gain = torch.sum(info_gain_a + info_gain_b, dim=0)
    
    return info_gain

def draw_2d_solution(x_values, u_values, p_values):
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