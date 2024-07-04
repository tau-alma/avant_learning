import torch.nn as nn
import torch
from typing import List, Type


class MLP(nn.Module):
    def __init__(self, in_shape, dims, out_shape):
        super().__init__()

        latent_pi_net = create_mlp(in_shape, -1, dims)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        self.mu = nn.Linear(dims[-1], out_shape)
        self.log_std = nn.Linear(dims[-1], out_shape)

    def forward(self, x: torch.Tensor):
        x = self.latent_pi(x)
        return torch.tanh(self.mu(x))

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.Softplus,
    squash_output: bool = False,
    with_bias: bool = True,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules






def compute_terminal_cost(g, h, goal_pose):
    pass

    # A = jacobian(h, goal_pose)
    # B = jacobian(h, 0)
    
    # Q = diag()
    # R = diag()
