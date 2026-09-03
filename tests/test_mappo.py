from pathlib import Path

import numpy as np
import pytest
import torch

from grid_aware_pricing.config import load_config
from grid_aware_pricing.environment import GridAwarePricingEnv
from grid_aware_pricing.mappo import MAPPO
from grid_aware_pricing.rollout import AgentRolloutBuffer, RolloutBuffer


ROOT = Path(__file__).resolve().parents[1]


def test_beta_actions_rollout_update_and_checkpoint(tmp_path):
    config = load_config(ROOT / "configs" / "smoke.yaml")
    env = GridAwarePricingEnv(config)
    algorithm = MAPPO(config, env.observation_dim, env.global_state_dim)
    observations, state = env.reset()
    action = algorithm.act(observations, state)
    assert np.all(action.prices >= env.price_min)
    assert np.all(action.prices <= env.price_max)
    assert np.all((action.normalized_actions > 0) & (action.normalized_actions < 1))
    buffer, observations, state, _ = algorithm.collect_rollout(env, 4, observations, state)
    diagnostics = algorithm.update(buffer, algorithm.value(state))
    for key in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance"):
        assert np.isfinite(diagnostics[key])
    checkpoint = tmp_path / "checkpoint.pt"
    algorithm.save(checkpoint, env.estimator.state_dict())
    restored = MAPPO(config, env.observation_dim, env.global_state_dim)
    data = restored.load(checkpoint)
    deterministic = restored.act(observations, state, deterministic=True)
    assert np.all(deterministic.prices >= env.price_min)
    assert data["estimator"] is not None


def test_york_queue_v2_checkpoint_round_trip_and_legacy_rejection(tmp_path):
    config = load_config(ROOT / "configs" / "york_smoke.yaml")
    env = GridAwarePricingEnv(config)
    algorithm = MAPPO(config, env.observation_dim, env.global_state_dim)
    checkpoint = tmp_path / "york_queue_v2.pt"
    algorithm.save(checkpoint, env.estimator.state_dict())
    restored = MAPPO(config, env.observation_dim, env.global_state_dim)
    data = restored.load(checkpoint)
    assert data["environment_metadata"] == {
        "data_mode": "york_zip",
        "environment_schema_version": 2,
        "queue_semantics": "dynamic_fluid_carryover_v1",
        "observation_dim": 16,
        "global_state_dim": 136,
        "zip_sha256": config["data"]["zip_sha256"],
    }

    legacy = tmp_path / "legacy_york.pt"
    data.pop("environment_metadata")
    torch.save(data, legacy)
    with pytest.raises(ValueError, match="retraining is required"):
        restored.load(legacy)


def test_synthetic_legacy_checkpoint_remains_loadable(tmp_path):
    config = load_config(ROOT / "configs" / "smoke.yaml")
    env = GridAwarePricingEnv(config)
    algorithm = MAPPO(config, env.observation_dim, env.global_state_dim)
    checkpoint = tmp_path / "synthetic.pt"
    algorithm.save(checkpoint)
    data = torch.load(checkpoint, weights_only=False)
    data.pop("environment_metadata")
    legacy = tmp_path / "synthetic_legacy.pt"
    torch.save(data, legacy)
    MAPPO(config, env.observation_dim, env.global_state_dim).load(legacy)


def test_gae_stops_at_episode_boundaries():
    buffer = RolloutBuffer()
    for reward, done in ((0.0, False), (1.0, True), (100.0, True)):
        buffer.add(
            np.zeros((2, 3)), np.zeros(6), np.zeros(2), np.zeros(2),
            0.0, reward, done,
        )
    advantages, _ = buffer.compute_gae(last_value=999.0, gamma=1.0, gae_lambda=1.0)
    assert np.allclose(advantages, [1.0, 1.0, 100.0])


def test_complete_episode_collection_and_ippo_update_are_finite():
    config = load_config(ROOT / "configs" / "york_smoke.yaml")
    config.setdefault("experiment", {})["outside_mode"] = "fixed"
    algorithm = MAPPO(
        config,
        GridAwarePricingEnv(config).observation_dim,
        GridAwarePricingEnv(config).global_state_dim,
        method="ippo",
    )
    env = GridAwarePricingEnv(config)
    buffer, summaries, infos = algorithm.collect_episodes(env, 2)
    assert isinstance(buffer, AgentRolloutBuffer)
    assert len(buffer) == 12
    assert np.flatnonzero(buffer.dones).tolist() == [5, 11]
    assert np.asarray(buffer.rewards).shape == (12, 8)
    assert [summary["periods"] for summary in summaries] == [6, 6]
    assert len(infos) == 12
    diagnostics = algorithm.update(buffer, np.zeros(8))
    assert all(np.isfinite(value) for value in diagnostics.values())
