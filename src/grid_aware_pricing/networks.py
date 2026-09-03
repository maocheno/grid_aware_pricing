"""Independent Beta actors and a centralized value critic."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        layers.extend([nn.Linear(previous, size), nn.Tanh()])
        previous = size
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class BetaActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: list[int],
        price_min: float,
        price_max: float,
        beta_epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = _mlp(observation_dim, hidden_sizes, 2)
        self.price_min = float(price_min)
        self.price_max = float(price_max)
        self.beta_epsilon = float(beta_epsilon)

    def distribution(self, observations: torch.Tensor) -> Beta:
        parameters = self.network(observations)
        alpha = F.softplus(parameters[..., 0]) + self.beta_epsilon
        beta = F.softplus(parameters[..., 1]) + self.beta_epsilon
        return Beta(alpha, beta)

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        y = distribution.mean if deterministic else distribution.sample()
        y = y.clamp(1e-6, 1.0 - 1e-6)
        price = self.price_min + (self.price_max - self.price_min) * y
        return price, y, distribution.log_prob(y), distribution.entropy()

    def evaluate_actions(self, observations: torch.Tensor, normalized_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        actions = normalized_actions.clamp(1e-6, 1.0 - 1e-6)
        return distribution.log_prob(actions), distribution.entropy()


class CentralCritic(nn.Module):
    def __init__(self, global_state_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.network = _mlp(global_state_dim, hidden_sizes, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)


class LocalCritic(nn.Module):
    """Minimal local-observation value network used by IPPO."""

    def __init__(self, observation_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.network = _mlp(observation_dim, hidden_sizes, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).squeeze(-1)


def initialize_network(module: nn.Module, seed: int) -> None:
    torch.manual_seed(seed)
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(layer.bias)
