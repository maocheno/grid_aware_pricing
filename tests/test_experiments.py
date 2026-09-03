from pathlib import Path

import numpy as np

from grid_aware_pricing.config import load_config
from grid_aware_pricing.environment import GridAwarePricingEnv
from grid_aware_pricing.mappo import MAPPO
from grid_aware_pricing.experiments import (
    FixedTariffPolicy,
    MyopicLocalPolicy,
    TrainedPolicy,
    approximate_unilateral_gain,
    centralized_coordinate_search_reference,
    evaluate_policy,
    method_config,
    scenario_seed_sequence,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_and_local_are_deterministic_and_share_scenario_seeds():
    config = load_config(ROOT / "configs" / "york_smoke.yaml")
    seeds = scenario_seed_sequence([31, 47], 3)
    fixed_config = method_config(config, "fixed_tariff")
    local_config = method_config(config, "myopic_local")
    fixed_a = evaluate_policy(fixed_config, FixedTariffPolicy(), seeds)
    fixed_b = evaluate_policy(fixed_config, FixedTariffPolicy(), seeds)
    local_a = evaluate_policy(
        local_config, MyopicLocalPolicy(local_config, grid_points=3), seeds
    )
    local_b = evaluate_policy(
        local_config, MyopicLocalPolicy(local_config, grid_points=3), seeds
    )
    assert seeds == scenario_seed_sequence([31, 47], 3)
    assert len(seeds) == len(set(seeds)) == 3
    assert all(0 <= seed <= np.iinfo(np.uint32).max for seed in seeds)
    assert scenario_seed_sequence([31, 47], 2) == [31, 47]
    assert fixed_a.episodes["scenario_seed"].tolist() == seeds
    assert local_a.episodes["scenario_seed"].tolist() == seeds
    assert np.allclose(fixed_a.period_hub["price"], 0.45)
    assert np.array_equal(fixed_a.period_hub.fillna(-1).to_numpy(), fixed_b.period_hub.fillna(-1).to_numpy())
    assert np.array_equal(local_a.period_hub.fillna(-1).to_numpy(), local_b.period_hub.fillna(-1).to_numpy())
    assert fixed_a.metadata["lower_layer_evaluation"] == "frozen"
    assert fixed_a.metadata["estimator_reset_each_episode"] is True
    assert fixed_a.period_hub.groupby("episode")["outside_cost_estimate"].first().tolist() == [13.5] * 3
    required_period = {
        "seed", "training_seed", "scenario_seed", "timestamp", "hub_id", "hub_index",
        "arrivals", "historical_equivalent_vehicles", "pending_vehicles",
        "admission_ratio", "admission_pressure", "admitted_vehicles",
        "wait_min", "wait_excess_min", "requested_energy_kwh",
        "queued_energy_start_kwh", "pending_energy_kwh", "admitted_energy_kwh",
        "queued_energy_next_kwh", "served_energy_kwh", "unmet_energy_kwh",
        "queue_vehicle_conservation_error", "queue_energy_conservation_error_kwh",
        "admitted_full_service_ratio", "pending_full_service_ratio",
        "pv_used_kwh", "battery_charge_kwh", "grid_import_kwh", "energy_cost_gbp",
        "access_mean_min", "outside_nll", "energy_balance_error_kwh",
    }
    required_episode = {
        "return", "profit_gbp", "welfare_gbp", "served_requests", "outside_requests",
        "pending_requests", "admitted_requests", "admission_ratio",
        "minimum_admission_ratio", "peak_admission_pressure",
        "admitted_full_service_ratio", "pending_full_service_ratio",
        "mean_wait_min", "p95_wait_min", "max_wait_min", "wait_violation_rate",
        "pending_energy_kwh", "admitted_energy_kwh", "unmet_energy_kwh",
        "peak_queued_energy_kwh", "mean_queued_energy_kwh", "final_queued_energy_kwh",
        "queue_cleared_by_end", "queue_clearance_period",
        "max_queue_vehicle_conservation_error",
        "max_queue_energy_conservation_error_kwh", "grid_energy_kwh", "pv_utilization",
        "battery_throughput_kwh", "mean_access_min", "outside_mae", "outside_nll",
        "approx_unilateral_gain", "centralized_reference_difference", "exact_oracle_gap",
    }
    assert required_period <= set(fixed_a.period_hub)
    assert {"accepted", "overflow", "accepted_energy_kwh", "full_service_ratio"}.isdisjoint(
        fixed_a.period_hub.columns
    )
    assert required_episode <= set(fixed_a.episodes)
    assert "full_service_ratio" not in fixed_a.episodes


def test_trained_policy_evaluation_is_frozen_and_resets_checkpoint_estimate(tmp_path):
    config = method_config(load_config(ROOT / "configs" / "york_smoke.yaml"), "proposed")
    env = GridAwarePricingEnv(config)
    checkpoint = tmp_path / "checkpoint.pt"
    algorithm = MAPPO(config, env.observation_dim, env.global_state_dim, method="proposed")
    algorithm.save(checkpoint, {"cost": 14.25, "loss_history": [9.0]})
    policy = TrainedPolicy(
        config, "proposed", checkpoint, env.observation_dim, env.global_state_dim
    )
    frozen = evaluate_policy(config, policy, [5, 6])
    starts = frozen.period_hub.groupby("episode")["outside_cost_estimate"].first()
    assert np.allclose(starts, 14.25)
    assert frozen.metadata["lower_layer_evaluation"] == "frozen"
    online = evaluate_policy(config, policy, [5], online_lower_layer=True)
    assert online.metadata["lower_layer_evaluation"] == "online"
    assert online.period_hub["outside_cost_estimate"].nunique() > 1


def test_myopic_candidate_simulation_uses_only_policy_outside_estimate():
    config = method_config(load_config(ROOT / "configs" / "york_smoke.yaml"), "myopic_local")
    env = GridAwarePricingEnv(config)
    observations, state = env.reset(seed=9, outside_estimate=13.5, reset_inference=True)
    policy = MyopicLocalPolicy(config, grid_points=2, outside_estimate=13.5)
    calls = []
    original = env.step

    def recorded_step(prices, **kwargs):
        calls.append(kwargs.copy())
        return original(prices, **kwargs)

    env.step = recorded_step
    policy.prices(observations, state, env)
    assert calls
    assert all(call["outside_cost"] == 13.5 for call in calls)
    assert all(call["update_inference"] is False for call in calls)


def test_reference_and_unilateral_search_respect_budgets_and_metadata():
    config = load_config(ROOT / "configs" / "york_smoke.yaml")
    reference = centralized_coordinate_search_reference(
        config, scenario_seed=9, budget=5, grid_points=2
    )
    report = reference["solver_report"]
    assert report["name"] == "centralized_coordinate_search_reference"
    assert report["evaluations"] <= report["budget"] == 5
    assert report["is_exact"] is False
    assert report["is_upper_bound"] is False
    assert report["uses_true_preference"] is True
    assert reference["trajectory"].shape == (6, 8)

    gain = approximate_unilateral_gain(
        config, reference["trajectory"], scenario_seed=9, budget=5, grid_points=2
    )
    assert gain["evaluations"] <= gain["budget"] == 5
    assert gain["lower_bound_on_exact_gain"] is True
    assert gain["found_maximum_gain"] >= 0.0
