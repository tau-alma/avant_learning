import torch
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


def preference_probability(trajA_cost: torch.Tensor, trajB_cost: torch.Tensor) -> torch.Tensor:
    return torch.exp(-trajA_cost) / (torch.exp(-trajA_cost) + torch.exp(-trajB_cost))


def preference_loss(trajA_costs: torch.Tensor, trajB_costs: torch.Tensor, preference_labels: torch.Tensor):
    """
    Parameters:
    - trajA_costs: length M tensor, where M is the number of trajectory pairs
    - trajB_costs: length M tensor, where M is the number of trajectory pairs
    - preference_labels: Mx2 tensor, where M is the number of trajectory pairs
    """
    assert len(preference_labels.shape) in [2, 3] and (len(trajA_costs) == len(preference_labels) and len(trajB_costs) == len(preference_labels))
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
