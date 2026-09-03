"""On-policy rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Legacy shared-reward MAPPO buffer.

    The public fields and ``add`` signature are intentionally unchanged.  Done
    masks are applied both to bootstrapping and to the recursive GAE term, so a
    buffer may safely contain several complete episodes.
    """

    observations: list[np.ndarray] = field(default_factory=list)
    global_states: list[np.ndarray] = field(default_factory=list)
    normalized_actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add(
        self,
        observations: np.ndarray,
        global_state: np.ndarray,
        normalized_actions: np.ndarray,
        log_probs: np.ndarray,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.observations.append(np.asarray(observations, dtype=np.float32))
        self.global_states.append(np.asarray(global_state, dtype=np.float32))
        self.normalized_actions.append(np.asarray(normalized_actions, dtype=np.float32))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = float(last_value)
        for step in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * next_value * nonterminal - values[step]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[step] = gae
            next_value = values[step]
        returns = advantages + values
        return advantages, returns

    def tensors(self, advantages: np.ndarray, returns: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "observations": torch.as_tensor(np.asarray(self.observations), device=device),
            "global_states": torch.as_tensor(np.asarray(self.global_states), device=device),
            "normalized_actions": torch.as_tensor(np.asarray(self.normalized_actions), device=device),
            "old_log_probs": torch.as_tensor(np.asarray(self.log_probs), device=device),
            "advantages": torch.as_tensor(normalized_advantages, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return len(self.rewards)


@dataclass
class AgentRolloutBuffer:
    """Per-agent reward/value storage for independent PPO critics."""

    observations: list[np.ndarray] = field(default_factory=list)
    global_states: list[np.ndarray] = field(default_factory=list)
    normalized_actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    rewards: list[np.ndarray] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add(
        self,
        observations: np.ndarray,
        global_state: np.ndarray,
        normalized_actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        done: bool,
    ) -> None:
        observations_array = np.asarray(observations, dtype=np.float32)
        n_agents = observations_array.shape[0]
        values_array = np.asarray(values, dtype=np.float32).reshape(-1)
        rewards_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
        if values_array.shape != (n_agents,) or rewards_array.shape != (n_agents,):
            raise ValueError("per-agent values and rewards must have shape [n_agents]")
        self.observations.append(observations_array)
        self.global_states.append(np.asarray(global_state, dtype=np.float32))
        self.normalized_actions.append(np.asarray(normalized_actions, dtype=np.float32))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.values.append(values_array)
        self.rewards.append(rewards_array)
        self.dones.append(bool(done))

    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        if rewards.ndim != 2:
            raise ValueError("agent rollout rewards must have shape [time, agents]")
        advantages = np.zeros_like(rewards)
        gae = np.zeros(rewards.shape[1], dtype=np.float32)
        next_values = np.asarray(last_values, dtype=np.float32).reshape(rewards.shape[1])
        for step in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * next_values * nonterminal - values[step]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[step] = gae
            next_values = values[step]
        returns = advantages + values
        return advantages, returns

    def tensors(self, advantages: np.ndarray, returns: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
        means = advantages.mean(axis=0, keepdims=True)
        stds = advantages.std(axis=0, keepdims=True)
        normalized_advantages = (advantages - means) / (stds + 1e-8)
        return {
            "observations": torch.as_tensor(np.asarray(self.observations), device=device),
            "global_states": torch.as_tensor(np.asarray(self.global_states), device=device),
            "normalized_actions": torch.as_tensor(np.asarray(self.normalized_actions), device=device),
            "old_log_probs": torch.as_tensor(np.asarray(self.log_probs), device=device),
            "advantages": torch.as_tensor(normalized_advantages, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return len(self.rewards)
