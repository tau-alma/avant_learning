from gymnasium import spaces
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ContinuousCritic
from typing import List, Type


class SquaredContinuousCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True
    ):
        super(SquaredContinuousCritic, self).__init__(
            observation_space = observation_space,
            action_space = action_space,
            net_arch = net_arch,
            features_extractor = features_extractor,
            features_dim = features_dim,
            activation_fn = activation_fn,
            normalize_images = normalize_images,
            n_critics = n_critics,
            share_features_extractor = share_features_extractor
        )
        
    def forward(self, obs, actions):
        q_values = super(SquaredContinuousCritic, self).forward(obs, actions)
        # Square the Q-values
        squared_q_values = [q**2 for q in q_values]
        return tuple(squared_q_values)