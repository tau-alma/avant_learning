import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm


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
    - trajA_costs: Nx1 tensor, where N is the number of trajectory pairs
    - trajB_costs: Nx1 tensor, where N is the number of trajectory pairs
    - preference_labels: Nx2 tensor, where N is the number of trajectory pairs
    """
    assert len(preference_labels.shape) == 2 and (len(trajA_costs) == len(preference_labels) and len(trajB_costs) == len(preference_labels))
    return -(
        preference_labels[:, 0] * preference_probability(trajA_costs, trajB_costs) 
        + preference_labels[:, 1] * preference_probability(trajB_costs, trajA_costs)
    )    


def information_gain(trajA_costs: torch.Tensor, trajB_costs: torch.Tensor) -> torch.Tensor:
    """
    Compute the information gain for each pair of trajectories.
    
    Parameters:
    - trajA_costs: MxN tensor, where M is the number of cost function ensembles, and N is the number of trajectory pairs
    - trajB_costs: MxN tensor, where M is the number of cost function ensembles, and N is the number of trajectory pairs
    
    Returns:
    - An N-element tensor containing the information gain for each trajectory pair.
    """

    print("gosts", trajA_costs, trajB_costs)

    assert trajA_costs.amin().item() >= 0 and trajB_costs.amin().item() >= 0

    max_A = trajA_costs.amax()
    max_B = trajB_costs.amax()
    scaler = torch.amax(torch.vstack([max_A, max_B]))

    trajA_costs /= scaler
    trajB_costs /= scaler

    print("gosts 2", trajA_costs, trajB_costs)

    # Calculate preference probabilities for both orders
    pref_prob_a_b = preference_probability(trajA_costs, trajB_costs)
    pref_prob_b_a = preference_probability(trajB_costs, trajA_costs)

    print(pref_prob_a_b, pref_prob_b_a)

    # Sum the preference probabilities over all ensembles for each trajectory
    pref_probs_sum_a = torch.sum(pref_prob_a_b, dim=1)
    pref_probs_sum_b = torch.sum(pref_prob_b_a, dim=1)
    
    # Compute the log probabilities for each preference
    M = trajA_costs.size(0)
    log_prob_a_b = torch.log2(M * pref_prob_a_b / (pref_probs_sum_a + 1e-9))
    log_prob_b_a = torch.log2(M * pref_prob_b_a / (pref_probs_sum_b + 1e-9))

    print(log_prob_a_b, log_prob_b_a)
    
    # Calculate the information gain for each preference
    info_gain_a = pref_prob_a_b * log_prob_a_b
    info_gain_b = pref_prob_b_a * log_prob_b_a
    
    # Sum the information gain over all ensembles for each pair of trajectories
    info_gain = torch.sum(info_gain_a + info_gain_b, dim=0)

    print(info_gain)
    
    return info_gain.squeeze(1)


def draw_2d_solution(x_values, u_values, p_values):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    # Extracting necessary data
    x_positions = x_values[0, :, 0].numpy()
    y_positions = x_values[0, :, 1].numpy()
    front_link_angles = x_values[0, :, 2].numpy()
    center_link_angles = x_values[0, :, 3].numpy()  # Center link angle in the machine frame

    velocities = x_values[0, :, 5].numpy()

    # Calculate the arrow directions based on the heading angle for the front link
    dx_front = np.cos(front_link_angles)
    dy_front = np.sin(front_link_angles)

    # Correctly adjust center link angles by adding π to the front link angles for global orientation,
    # making the rear link point directly opposite when the center link angle is zero.
    adjusted_center_link_angles = front_link_angles + center_link_angles + np.pi  # Adding π to reverse direction

    # Calculate the direction vectors for the center links, applying a scale factor for length
    dx_center = np.cos(adjusted_center_link_angles) * 0.1  # Example scale factor for visualization
    dy_center = np.sin(adjusted_center_link_angles) * 0.1

    # Velocity for color coding the quiver plot for x_values
    norm = plt.Normalize(vmin=velocities.min(), vmax=velocities.max())
    axs.quiver(x_positions, y_positions, dx_front, dy_front, velocities, cmap='viridis', norm=norm)

    # Plotting center link lines using adjusted angles
    for i in range(len(dx_center)):
        axs.plot([x_positions[i], x_positions[i] + dx_center[i]], 
                 [y_positions[i], y_positions[i] + dy_center[i]], 
                 color='gray', linestyle='-', linewidth=2)

    # Assuming p_values also has headings at index 2
    dx_p = np.cos(p_values[:, 2].numpy())
    dy_p = np.sin(p_values[:, 2].numpy())
    axs.quiver(p_values[:, 0].numpy(), p_values[:, 1].numpy(), dx_p, dy_p, color='red')
    
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axs)
    axs.scatter(x_positions[0], y_positions[0], color='red', marker='o')
    axs.grid()
    axs.autoscale_view()
    axs.set_title('AFS machine visualization')

    plt.tight_layout()
    plt.show()




import time
if __name__ == "__main__":
    t1 = time.time_ns()

    M, N = 5, 1000  # 5 cost function ensembles, 10 trajectories
    trajA_costs = torch.randn(M, N)#.cuda()
    trajB_costs = torch.randn(M, N)#.cuda()
    ig = information_gain(trajA_costs, trajB_costs)
    print(ig)

    t2 = time.time_ns()
    print(f"Took {(t2-t1)/1e6} ms")
