from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from grid_aware_pricing.config import load_config, resolved_config
from grid_aware_pricing.environment import GridAwarePricingEnv


ROOT = Path(__file__).resolve().parents[1]


def test_environment_smoke_and_hidden_true_outside_state():
    config = load_config(ROOT / "configs" / "smoke.yaml")
    env = GridAwarePricingEnv(config)
    observations, global_state = env.reset()
    assert observations.shape == (2, 10)
    assert global_state.shape == (20,)
    true_cost = config["choice"]["true_outside_cost"]
    outside_feature = observations[:, -1]
    expected_feature = config["inference"]["initial_outside_cost"] / config["normalization"]["outside_cost"]
    assert np.allclose(outside_feature, expected_feature)
    assert not np.allclose(outside_feature, true_cost / config["normalization"]["outside_cost"])
    prices = (env.price_min + env.price_max) / 2
    output = env.step(prices)
    assert np.isfinite(output.reward)
    assert output.info["served_energy"].shape == (2,)
    assert output.info["outside_cost_estimate"] != output.info["true_outside_cost_for_evaluation_only"] or true_cost == config["inference"]["initial_outside_cost"]
    for result_unmet in output.info["unmet"]:
        assert result_unmet >= -1e-8


def test_reset_preserves_inference_and_advances_rng_unless_explicitly_seeded():
    config = load_config(ROOT / "configs" / "smoke.yaml")
    env = GridAwarePricingEnv(config)
    prices = (env.price_min + env.price_max) / 2
    first = env.step(prices).info["realized_demand"].copy()
    learned = env.estimator.cost
    env.reset()
    second = env.step(prices).info["realized_demand"].copy()
    assert env.estimator.cost != config["inference"]["initial_outside_cost"]
    assert learned != config["inference"]["initial_outside_cost"]
    assert not np.array_equal(first, second)
    env.reset(seed=config["seed"], reset_inference=True)
    replay = env.step(prices).info["realized_demand"]
    assert np.array_equal(first, replay)


def test_snapshot_restore_replays_candidate_exactly():
    config = load_config(ROOT / "configs" / "smoke.yaml")
    env = GridAwarePricingEnv(config)
    snapshot = env.snapshot()
    prices = (env.price_min + env.price_max) / 2
    first = env.step(prices)
    env.restore(snapshot)
    second = env.step(prices)
    assert first.reward == second.reward
    assert np.array_equal(first.info["realized_demand"], second.info["realized_demand"])


def test_environment_completes_episode_deterministically():
    config = load_config(ROOT / "configs" / "smoke.yaml")
    config["system"]["deterministic_demand"] = True
    env = GridAwarePricingEnv(config)
    observations, state = env.reset()
    done = False
    steps = 0
    while not done:
        output = env.step((env.price_min + env.price_max) / 2)
        observations, state, done = output.observations, output.global_state, output.done
        steps += 1
    assert steps == config["system"]["periods"]


def test_york_environment_deterministic_six_steps_and_hidden_cost_isolation():
    config = load_config(ROOT / "configs" / "york_smoke.yaml")
    visible = resolved_config(config)
    assert "16.5" not in repr(visible)
    env = GridAwarePricingEnv(config)
    observations, state = env.reset(reset_inference=True)
    assert observations.shape == (8, 16)
    assert state.shape == (136,)
    assert np.allclose(env.previous_price, config["price"]["initial"])
    assert np.allclose(env.previous_wait, 0.0)
    assert np.allclose(env.queued_energy_kwh, 0.0)
    assert env.demand_multiplier == 1.0
    assert np.allclose(
        observations[:, -1],
        config["inference"]["initial_outside_cost"] / config["normalization"]["outside_cost"],
    )
    assert not np.allclose(
        observations[:, -1], 16.5 / config["normalization"]["outside_cost"]
    )
    steps = 0
    done = False
    queue_totals = []
    while not done:
        output = env.step(np.asarray(config["price"]["initial"]), deterministic_demand=True)
        assert output.info["timestamp"] == config["_york_scenario"].timestamps[steps].isoformat()
        assert output.info["energy_balance_error"] < 1e-6
        assert np.max(np.abs(output.info["queue_vehicle_conservation_error"])) < 1e-10
        assert np.max(np.abs(output.info["queue_energy_conservation_error_kwh"])) < 1e-10
        assert np.allclose(
            output.info["pending_energy_kwh"],
            output.info["admitted_energy_kwh"] + output.info["queued_energy_next_kwh"],
        )
        assert np.isclose(output.reward, output.info["raw_reward"] * config["reward"]["scale"])
        queue_totals.append(output.info["queued_energy_next_kwh"].sum())
        done = output.done
        steps += 1
    assert steps == 6
    assert max(queue_totals) == pytest.approx(649.4582857916657)
    assert queue_totals[4:] == pytest.approx([0.0, 0.0])


def test_york_candidate_mask_is_strictly_enforced():
    config = load_config(ROOT / "configs" / "york_smoke.yaml")
    scenario = config["_york_scenario"]
    candidate_mask = scenario.candidate_mask.copy()
    candidate_mask[0, 0, 0] = False
    config["_york_scenario"] = replace(scenario, candidate_mask=candidate_mask)
    env = GridAwarePricingEnv(config)
    output = env.step(np.asarray(config["price"]["initial"]), deterministic_demand=True)
    assert output.info["hub_probabilities"][0, 0] == 0.0
    assert output.info["od_hub_realized_counts"][0, 0] == 0.0
    assert output.info["od_hub_realized_energy"][0, 0] == 0.0


def test_york_overload_carries_backlog_and_dispatches_only_admitted_energy():
    config = load_config(ROOT / "configs" / "york_smoke.yaml")
    scenario = config["_york_scenario"]
    expected = scenario.od_expected_demand.copy()
    expected[0] *= 30.0
    config["_york_scenario"] = replace(scenario, od_expected_demand=expected)
    env = GridAwarePricingEnv(config)
    first = env.step(np.asarray(config["price"]["initial"]), deterministic_demand=True)
    info = first.info
    assert info["queued_energy_next_kwh"].sum() > 0.0
    assert np.all(info["admitted_vehicles"] <= info["service_capacity"] + 1e-9)
    assert np.allclose(
        info["admitted_energy_kwh"],
        info["served_energy"] + info["unmet_energy"],
    )
    assert np.all(info["pending_full_service_ratio"] <= 1.0 + 1e-9)
    queued = info["queued_energy_next_kwh"].copy()
    wait = info["wait"].copy()
    second = env.step(np.asarray(config["price"]["initial"]), deterministic_demand=True)
    assert np.array_equal(second.info["queued_energy_start_kwh"], queued)
    assert np.allclose(
        second.info["historical_equivalent_vehicles"], env.service / env.dt * wait
    )


def test_york_snapshot_restore_replays_stochastic_demand_and_inference():
    config = load_config(ROOT / "configs" / "york.yaml")
    env = GridAwarePricingEnv(config)
    env.reset(seed=37, reset_inference=True)
    snapshot = env.snapshot()
    prices = np.asarray(config["price"]["initial"])
    first = env.step(prices)
    first_estimator_state = env.estimator.state_dict()
    env.restore(snapshot)
    second = env.step(prices)
    assert first.reward == second.reward
    for key in (
        "realized_od_demand",
        "od_hub_realized_counts",
        "od_hub_realized_energy",
        "outside_count_by_od",
        "admitted_energy_kwh",
        "queued_energy_next_kwh",
    ):
        assert np.array_equal(first.info[key], second.info[key])
    assert env.estimator.state_dict() == first_estimator_state
    assert env.demand_multiplier == snapshot["demand_multiplier"]


def test_york_observation_masks_do_not_change_physics_and_fixed_mode_does_not_update():
    base = load_config(ROOT / "configs" / "york_smoke.yaml")
    masked = load_config(ROOT / "configs" / "york_smoke.yaml")
    masked["experiment"] = {
        "observation_masks": {"no_traffic": True, "no_energy": True},
        "outside_mode": "fixed",
    }
    base_env = GridAwarePricingEnv(base)
    masked_env = GridAwarePricingEnv(masked)
    base_obs, _ = base_env.reset(seed=9, reset_inference=True)
    masked_obs, _ = masked_env.reset(seed=9, reset_inference=True)
    assert np.all(masked_obs[:, 3:7] == 0.0)
    assert np.all(masked_obs[:, 7:-1] == 0.0)
    initial_estimate = masked_env.estimator.cost
    prices = np.asarray(base["price"]["initial"])
    base_step = base_env.step(prices, deterministic_demand=True, update_inference=False)
    masked_step = masked_env.step(prices, deterministic_demand=True)
    assert np.array_equal(base_step.info["realized_demand"], masked_step.info["realized_demand"])
    assert np.array_equal(base_step.info["served_energy"], masked_step.info["served_energy"])
    assert masked_env.estimator.cost == initial_estimate
