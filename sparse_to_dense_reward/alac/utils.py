import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.preprocessing import get_action_dim 
from stable_baselines3.common.policies import BaseModel, ContinuousCritic
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
)
from typing import List, Tuple, Type


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


class LyapunovCritic(BaseModel):
    """
    Constrained critic for Lyapunov RL algorithms. Enforces L(goal) = 0, and L(x) >= 0 for all x.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param num_out: Number of output units
    :param activation_fn: Activation function
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        num_out: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        share_features_extractor: bool = True,
        normalize_images: bool = True
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.num_out = num_out
        self.share_features_extractor = share_features_extractor

        action_dim = get_action_dim(self.action_space)
        self.q_net = th.nn.Sequential( *create_mlp(features_dim + action_dim, num_out, net_arch, activation_fn) )
        self.add_module(f"Qf", self.q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features, goal_delta = self.extract_features(obs, self.features_extractor)
        
        qvalue_input = th.cat([features, actions], dim=1)
        qvalue_ouput = self.q_net(qvalue_input)

        scaler = 1 / (th.sum(goal_delta, dim=1) + 1e-5)
        scaler = scaler.view(-1, 1, 1)
        F = qvalue_ouput.unsqueeze(1).repeat(1, goal_delta.shape[1], 1)
        GQ = th.bmm(goal_delta.unsqueeze(1), F)
        GQ *= scaler
        L_value = GQ @ GQ.transpose(1, 2)
        return L_value.view(-1, 1)