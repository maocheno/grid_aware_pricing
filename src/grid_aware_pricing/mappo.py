"""Multi-agent PPO with independent Beta actors and shared or local critics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import resolved_config
from .environment import GridAwarePricingEnv
from .networks import BetaActor, CentralCritic, LocalCritic, initialize_network
from .rollout import AgentRolloutBuffer, RolloutBuffer


@dataclass(frozen=True)
class ActionOutput:
    prices: np.ndarray
    normalized_actions: np.ndarray
    log_probs: np.ndarray
    value: float
    agent_values: np.ndarray | None = None


class MAPPO:
    """Independent actors with centralized MAPPO or independent IPPO values."""

    def __init__(
        self,
        config: dict[str, Any],
        observation_dim: int,
        global_state_dim: int,
        device: str = "cpu",
        method: str | None = None,
    ) -> None:
        self.config = config
        self.method = str(method or config.get("experiment", {}).get("method", "proposed"))
        self.independent = self.method == "ippo"
        self.device = torch.device(device)
        self.rng = np.random.default_rng(int(config["seed"]))
        self.n_hubs = int(config["system"]["n_hubs"])
        self.observation_dim = int(observation_dim)
        self.global_state_dim = int(global_state_dim)
        actor_hidden = list(config["network"]["hidden_sizes"])
        critic_hidden = list(config["network"].get("critic_hidden_sizes", actor_hidden))
        epsilon = float(config["network"]["beta_epsilon"])
        self.actors = nn.ModuleList([
            BetaActor(
                observation_dim,
                actor_hidden,
                config["price"]["min"][hub],
                config["price"]["max"][hub],
                epsilon,
            )
            for hub in range(self.n_hubs)
        ]).to(self.device)
        initialize_network(self.actors, int(config["seed"]))

        self.critic: CentralCritic | None
        self.local_critics: nn.ModuleList | None
        if self.independent:
            self.critic = None
            self.local_critics = nn.ModuleList([
                LocalCritic(observation_dim, critic_hidden) for _ in range(self.n_hubs)
            ]).to(self.device)
            initialize_network(self.local_critics, int(config["seed"]) + 1)
        else:
            self.critic = CentralCritic(global_state_dim, critic_hidden).to(self.device)
            self.local_critics = None
            initialize_network(self.critic, int(config["seed"]) + 1)

        training = config["training"]
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=float(training["actor_lr"]))
            for actor in self.actors
        ]
        if self.independent:
            assert self.local_critics is not None
            self.local_critic_optimizers = [
                torch.optim.Adam(critic.parameters(), lr=float(training["critic_lr"]))
                for critic in self.local_critics
            ]
            self.critic_optimizer = None
        else:
            assert self.critic is not None
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=float(training["critic_lr"])
            )
            self.local_critic_optimizers = []

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        global_state: np.ndarray,
        deterministic: bool = False,
    ) -> ActionOutput:
        observation_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        samples = [
            actor.sample(observation_tensor[hub], deterministic)
            for hub, actor in enumerate(self.actors)
        ]
        prices = torch.stack([sample[0] for sample in samples]).cpu().numpy()
        actions = torch.stack([sample[1] for sample in samples]).cpu().numpy()
        log_probs = torch.stack([sample[2] for sample in samples]).cpu().numpy()
        if self.independent:
            assert self.local_critics is not None
            agent_values = torch.stack([
                critic(observation_tensor[hub])
                for hub, critic in enumerate(self.local_critics)
            ]).cpu().numpy()
            return ActionOutput(
                prices, actions, log_probs, float(np.mean(agent_values)), agent_values
            )
        assert self.critic is not None
        state_tensor = torch.as_tensor(global_state, dtype=torch.float32, device=self.device)
        value = float(self.critic(state_tensor).cpu())
        return ActionOutput(prices, actions, log_probs, value)

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float | np.ndarray:
        if self.independent:
            observations = np.asarray(state, dtype=np.float32)
            if observations.ndim != 2 or observations.shape[0] != self.n_hubs:
                raise ValueError("IPPO value expects local observations with shape [agents, obs]")
            tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            assert self.local_critics is not None
            return torch.stack([
                critic(tensor[hub]) for hub, critic in enumerate(self.local_critics)
            ]).cpu().numpy()
        assert self.critic is not None
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return float(self.critic(tensor).cpu())

    def _local_rewards(self, info: dict[str, Any]) -> np.ndarray:
        wait_excess = np.asarray(
            info.get("wait_excess", info["wait_violation"]), dtype=float
        )
        rewards = (
            np.asarray(info["profit"], dtype=float)
            - float(self.config["reward"]["wait_penalty"]) * wait_excess
            - float(self.config["reward"]["unmet_penalty"])
            * np.asarray(info["unmet"], dtype=float)
        )
        return rewards * float(self.config["reward"].get("scale", 1.0))

    def collect_rollout(
        self,
        env: GridAwarePricingEnv,
        steps: int,
        observations: np.ndarray,
        global_state: np.ndarray,
    ) -> tuple[RolloutBuffer | AgentRolloutBuffer, np.ndarray, np.ndarray, list[dict]]:
        buffer: RolloutBuffer | AgentRolloutBuffer
        buffer = AgentRolloutBuffer() if self.independent else RolloutBuffer()
        infos: list[dict] = []
        for _ in range(steps):
            action = self.act(observations, global_state, deterministic=False)
            output = env.step(action.prices)
            if self.independent:
                assert isinstance(buffer, AgentRolloutBuffer)
                assert action.agent_values is not None
                buffer.add(
                    observations,
                    global_state,
                    action.normalized_actions,
                    action.log_probs,
                    action.agent_values,
                    self._local_rewards(output.info),
                    output.done,
                )
            else:
                assert isinstance(buffer, RolloutBuffer)
                buffer.add(
                    observations,
                    global_state,
                    action.normalized_actions,
                    action.log_probs,
                    action.value,
                    output.reward,
                    output.done,
                )
            infos.append(output.info)
            observations, global_state = output.observations, output.global_state
            if output.done:
                observations, global_state = env.reset(
                    reset_inference=bool(
                        self.config["inference"].get("reset_between_episodes", False)
                    )
                )
        return buffer, observations, global_state, infos

    def collect_episodes(
        self,
        env: GridAwarePricingEnv,
        n_episodes: int,
    ) -> tuple[RolloutBuffer | AgentRolloutBuffer, list[dict[str, float | int]], list[dict]]:
        """Collect complete episodes while preserving estimator state by default."""
        if n_episodes <= 0:
            raise ValueError("n_episodes must be positive")
        buffer: RolloutBuffer | AgentRolloutBuffer
        buffer = AgentRolloutBuffer() if self.independent else RolloutBuffer()
        summaries: list[dict[str, float | int]] = []
        all_infos: list[dict] = []
        reset_inference = bool(
            self.config["inference"].get("reset_between_episodes", False)
        )
        for episode_in_batch in range(n_episodes):
            observations, global_state = env.reset(reset_inference=reset_inference)
            episode_infos: list[dict] = []
            done = False
            while not done:
                action = self.act(observations, global_state, deterministic=False)
                output = env.step(action.prices)
                if self.independent:
                    assert isinstance(buffer, AgentRolloutBuffer)
                    assert action.agent_values is not None
                    buffer.add(
                        observations,
                        global_state,
                        action.normalized_actions,
                        action.log_probs,
                        action.agent_values,
                        self._local_rewards(output.info),
                        output.done,
                    )
                else:
                    assert isinstance(buffer, RolloutBuffer)
                    buffer.add(
                        observations,
                        global_state,
                        action.normalized_actions,
                        action.log_probs,
                        action.value,
                        output.reward,
                        output.done,
                    )
                episode_infos.append(output.info)
                all_infos.append(output.info)
                observations, global_state, done = (
                    output.observations,
                    output.global_state,
                    output.done,
                )
            summaries.append(self._episode_summary(episode_infos, episode_in_batch))
        return buffer, summaries, all_infos

    @staticmethod
    def _episode_summary(
        infos: list[dict[str, Any]], episode_in_batch: int
    ) -> dict[str, float | int]:
        summary: dict[str, float | int] = {
            "episode_in_batch": episode_in_batch,
            "episode_index": int(infos[0].get("episode_index", episode_in_batch)),
            "return": float(sum(float(info["reward"]) for info in infos)),
            "weighted_hub_profit_welfare": float(
                sum(float(info.get("weighted_hub_profit_welfare", info["welfare"])) for info in infos)
            ),
            "profit": float(sum(np.asarray(info["profit"], dtype=float).sum() for info in infos)),
            "wait_violation": float(
                sum(np.asarray(info["wait_violation"], dtype=float).sum() for info in infos)
            ),
            "unmet": float(sum(np.asarray(info["unmet"], dtype=float).sum() for info in infos)),
            "outside_cost_estimate": float(infos[-1]["outside_cost_estimate"]),
            "outside_nll": float(np.mean([
                info.get("inference_nll", info.get("inference_loss", np.nan)) for info in infos
            ])),
            "periods": len(infos),
        }
        if all("queued_energy_next_kwh" in info for info in infos):
            queue = np.asarray([
                np.asarray(info["queued_energy_next_kwh"], dtype=float).sum()
                for info in infos
            ])
            ratios = np.concatenate([
                np.asarray(info["admission_ratio"], dtype=float) for info in infos
            ])
            pending = np.concatenate([
                np.asarray(info["pending_vehicles"], dtype=float) for info in infos
            ])
            active = ratios[pending > 1e-12]
            summary.update({
                "peak_queued_energy_kwh": float(queue.max()),
                "mean_queued_energy_kwh": float(queue.mean()),
                "final_queued_energy_kwh": float(queue[-1]),
                "minimum_admission_ratio": float(active.min()) if len(active) else 1.0,
                "peak_admission_pressure": float(max(
                    np.asarray(info["admission_pressure"], dtype=float).max()
                    for info in infos
                )),
                "queue_cleared_by_end": int(queue[-1] <= 1e-8),
            })
        return summary

    def update(
        self,
        buffer: RolloutBuffer | AgentRolloutBuffer,
        last_value: float | np.ndarray = 0.0,
    ) -> dict[str, float]:
        if len(buffer) == 0:
            raise ValueError("cannot update from an empty rollout")
        training = self.config["training"]
        gamma = float(training["gamma"])
        gae_lambda = float(training["gae_lambda"])
        if self.independent:
            if not isinstance(buffer, AgentRolloutBuffer):
                raise TypeError("IPPO requires AgentRolloutBuffer")
            final_values = np.asarray(last_value, dtype=np.float32)
            if final_values.ndim == 0:
                final_values = np.full(self.n_hubs, float(final_values))
            advantages, returns = buffer.compute_gae(final_values, gamma, gae_lambda)
        else:
            if not isinstance(buffer, RolloutBuffer):
                raise TypeError("centralized MAPPO requires RolloutBuffer")
            advantages, returns = buffer.compute_gae(float(last_value), gamma, gae_lambda)
        data = buffer.tensors(advantages, returns, self.device)
        n_samples = len(buffer)
        minibatch_size = min(int(training["minibatch_size"]), n_samples)
        clip_ratio = float(training["clip_ratio"])
        entropy_coef = float(training["entropy_coef"])
        max_grad_norm = float(training["max_grad_norm"])
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clip_fractions: list[float] = []
        for _ in range(int(training["ppo_epochs"])):
            permutation = self.rng.permutation(n_samples)
            for start in range(0, n_samples, minibatch_size):
                indices = torch.as_tensor(
                    permutation[start:start + minibatch_size], device=self.device
                )
                actor_objectives: list[torch.Tensor] = []
                for hub, actor in enumerate(self.actors):
                    new_log_prob, entropy = actor.evaluate_actions(
                        data["observations"][indices, hub],
                        data["normalized_actions"][indices, hub],
                    )
                    old_log_prob = data["old_log_probs"][indices, hub]
                    advantage_batch = (
                        data["advantages"][indices, hub]
                        if self.independent
                        else data["advantages"][indices]
                    )
                    log_ratio = new_log_prob - old_log_prob
                    ratio = torch.exp(log_ratio)
                    unclipped = ratio * advantage_batch
                    clipped = torch.clamp(
                        ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
                    ) * advantage_batch
                    loss = -torch.min(unclipped, clipped).mean() - entropy_coef * entropy.mean()
                    actor_objectives.append(loss)
                    actor_losses.append(float(loss.detach().cpu()))
                    entropies.append(float(entropy.mean().detach().cpu()))
                    kls.append(float(((ratio - 1.0) - log_ratio).mean().detach().cpu()))
                    clip_fractions.append(float(
                        (torch.abs(ratio - 1.0) > clip_ratio).float().mean().detach().cpu()
                    ))
                for optimizer in self.actor_optimizers:
                    optimizer.zero_grad()
                torch.stack(actor_objectives).sum().backward()
                nn.utils.clip_grad_norm_(self.actors.parameters(), max_grad_norm)
                for optimizer in self.actor_optimizers:
                    optimizer.step()

                if self.independent:
                    assert self.local_critics is not None
                    for hub, (critic, optimizer) in enumerate(
                        zip(self.local_critics, self.local_critic_optimizers)
                    ):
                        predicted = critic(data["observations"][indices, hub])
                        critic_loss = torch.mean(
                            (predicted - data["returns"][indices, hub]) ** 2
                        )
                        optimizer.zero_grad()
                        (float(training["value_coef"]) * critic_loss).backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                        optimizer.step()
                        critic_losses.append(float(critic_loss.detach().cpu()))
                else:
                    assert self.critic is not None and self.critic_optimizer is not None
                    predicted = self.critic(data["global_states"][indices])
                    critic_loss = torch.mean((predicted - data["returns"][indices]) ** 2)
                    self.critic_optimizer.zero_grad()
                    (float(training["value_coef"]) * critic_loss).backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                    self.critic_optimizer.step()
                    critic_losses.append(float(critic_loss.detach().cpu()))

        with torch.no_grad():
            if self.independent:
                assert self.local_critics is not None
                predictions = torch.stack([
                    critic(data["observations"][:, hub])
                    for hub, critic in enumerate(self.local_critics)
                ], dim=1).cpu().numpy()
            else:
                assert self.critic is not None
                predictions = self.critic(data["global_states"]).cpu().numpy()
        target = np.asarray(returns)
        variance = float(np.var(target))
        explained_variance = (
            1.0 - float(np.var(target - predictions)) / variance
            if variance > 1e-12 else 0.0
        )
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "explained_variance": explained_variance,
            "advantage_mean": float(np.mean(advantages)),
            "return_mean": float(np.mean(returns)),
        }

    def save(
        self,
        path: str | Path,
        estimator_state: dict[str, float] | None = None,
    ) -> None:
        checkpoint: dict[str, Any] = {
            "method": self.method,
            "actors": self.actors.state_dict(),
            "actor_optimizers": [
                optimizer.state_dict() for optimizer in self.actor_optimizers
            ],
            "estimator": estimator_state,
            "config": resolved_config(self.config),
            "environment_metadata": {
                "data_mode": self.config.get("data", {}).get("mode", "synthetic"),
                "environment_schema_version": self.config.get("environment_schema_version"),
                "queue_semantics": self.config.get("queue_semantics"),
                "observation_dim": self.observation_dim,
                "global_state_dim": self.global_state_dim,
                "zip_sha256": self.config.get("data", {}).get("zip_sha256"),
            },
        }
        if self.independent:
            assert self.local_critics is not None
            checkpoint["local_critics"] = self.local_critics.state_dict()
            checkpoint["local_critic_optimizers"] = [
                optimizer.state_dict() for optimizer in self.local_critic_optimizers
            ]
        else:
            assert self.critic is not None and self.critic_optimizer is not None
            checkpoint["critic"] = self.critic.state_dict()
            checkpoint["critic_optimizer"] = self.critic_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str | Path, load_optimizers: bool = False) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        current_mode = self.config.get("data", {}).get("mode", "synthetic")
        metadata = checkpoint.get("environment_metadata")
        if current_mode == "york_zip":
            expected = {
                "data_mode": "york_zip",
                "environment_schema_version": self.config.get("environment_schema_version"),
                "queue_semantics": self.config.get("queue_semantics"),
                "observation_dim": self.observation_dim,
                "global_state_dim": self.global_state_dim,
                "zip_sha256": self.config.get("data", {}).get("zip_sha256"),
            }
            if not isinstance(metadata, dict):
                raise ValueError(
                    "legacy York checkpoint has no queue-v2 metadata; retraining is required"
                )
            mismatches = [
                key for key, value in expected.items()
                if metadata.get(key) != value
            ]
            if mismatches:
                raise ValueError(
                    "York checkpoint is incompatible with the current queue-v2 environment; "
                    "retraining is required (mismatched: " + ", ".join(mismatches) + ")"
                )
        elif isinstance(metadata, dict):
            if metadata.get("observation_dim") != self.observation_dim:
                raise ValueError("checkpoint observation dimension is incompatible")
            if metadata.get("global_state_dim") != self.global_state_dim:
                raise ValueError("checkpoint global state dimension is incompatible")
        saved_method = checkpoint.get("method")
        saved_independent = str(saved_method) == "ippo"
        if saved_method is None:
            if self.independent:
                raise ValueError("legacy centralized checkpoint is incompatible with method='ippo'")
        elif saved_independent != self.independent:
            raise ValueError(
                f"checkpoint method {saved_method!r} has incompatible critic architecture "
                f"for requested method {self.method!r}"
            )
        self.actors.load_state_dict(checkpoint["actors"])
        if self.independent:
            if "local_critics" not in checkpoint:
                raise ValueError("IPPO checkpoint is missing local critics")
            assert self.local_critics is not None
            self.local_critics.load_state_dict(checkpoint["local_critics"])
        else:
            if "critic" not in checkpoint:
                raise ValueError("centralized checkpoint is missing critic")
            assert self.critic is not None
            self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            for optimizer, state in zip(
                self.actor_optimizers, checkpoint.get("actor_optimizers", [])
            ):
                optimizer.load_state_dict(state)
            if self.independent:
                for optimizer, state in zip(
                    self.local_critic_optimizers,
                    checkpoint.get("local_critic_optimizers", []),
                ):
                    optimizer.load_state_dict(state)
            elif self.critic_optimizer is not None and "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return checkpoint
