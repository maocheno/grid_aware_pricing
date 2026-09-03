import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grid_aware_pricing.cli import (
    _sensitivity_runtime,
    ablation_or_sensitivity,
    build_parser,
    grid_oracle,
    report,
    train,
    validate_data,
)
from grid_aware_pricing.config import load_config, resolved_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "york_smoke.yaml"


def test_validate_data_writes_structural_sanity_and_provenance(tmp_path):
    output = validate_data(str(CONFIG), str(tmp_path / "validation"))
    report = json.loads((output / "validate_data.json").read_text())
    assert report["all_structural_checks_pass"] is True
    assert len(report["structural_checks"]) == 12
    assert len(report["zip_sha256"]) == 64
    assert report["limitations"]
    assert report["deterministic_fixed_tariff_sanity"]["total_demand"] == 382.83


def test_york_oracle_refuses_cartesian_search(tmp_path):
    with pytest.raises(ValueError, match="benchmark --reference-budget"):
        grid_oracle(str(CONFIG), str(tmp_path / "oracle"))


def test_tiny_york_training_writes_episode_metrics_and_checkpoints(tmp_path):
    output = train(
        str(CONFIG),
        str(tmp_path / "train"),
        method="proposed",
        seed=3,
        episodes=1,
        episodes_per_update=1,
    )
    required = (
        "training_metrics.csv",
        "period_hub.csv",
        "best_checkpoint.pt",
        "final_checkpoint.pt",
        "checkpoint.pt",
        "run_manifest.json",
    )
    assert all((output / name).exists() for name in required)
    metrics = pd.read_csv(output / "training_metrics.csv")
    assert len(metrics) == 1
    for column in (
        "return",
        "weighted_hub_profit_welfare",
        "profit",
        "wait_violation",
        "unmet",
        "peak_queued_energy_kwh",
        "mean_queued_energy_kwh",
        "final_queued_energy_kwh",
        "minimum_admission_ratio",
        "peak_admission_pressure",
        "queue_cleared_by_end",
        "outside_cost_estimate",
        "outside_nll",
        "update",
    ):
        assert column in metrics
    period = pd.read_csv(output / "period_hub.csv")
    assert {
        "training_seed", "scenario_seed", "timestamp", "hub_id", "arrivals",
        "historical_equivalent_vehicles", "pending_vehicles", "admission_ratio",
        "admission_pressure", "admitted_vehicles", "wait_min",
        "requested_energy_kwh", "queued_energy_start_kwh", "pending_energy_kwh",
        "admitted_energy_kwh", "queued_energy_next_kwh", "pv_used_kwh",
        "grid_import_kwh", "energy_cost_gbp", "admitted_full_service_ratio",
        "pending_full_service_ratio",
    } <= set(period)
    assert {"accepted", "overflow", "accepted_energy_kwh", "full_service_ratio"}.isdisjoint(
        period.columns
    )
    assert period["training_seed"].eq(3).all()
    assert period["scenario_seed"].isna().all()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["best_selection"] == "training_batch_mean_return"
    assert manifest["episodes"] == 1
    assert manifest["output_schema_version"] == "2.0"
    assert manifest["environment_schema_version"] == 2
    assert manifest["queue_semantics"] == "dynamic_fluid_carryover_v1"
    assert manifest["zip_sha256"] == load_config(CONFIG)["data"]["zip_sha256"]


def test_sensitivity_runtime_changes_physics_without_exposing_hidden_truth():
    config = load_config(CONFIG)
    scenario = config["_york_scenario"]
    demand = _sensitivity_runtime(config, "demand_multiplier", 1.15)
    grid = _sensitivity_runtime(config, "grid_cap_multiplier", 0.85)
    outside = _sensitivity_runtime(config, "true_outside_cost", 19.0)
    assert np.allclose(
        demand["_york_scenario"].od_expected_demand,
        scenario.od_expected_demand * 1.15,
    )
    assert np.allclose(
        grid["_york_scenario"].energy_parameters["grid_import_limit_kwh"],
        scenario.energy_parameters["grid_import_limit_kwh"] * 0.85,
    )
    assert "19.0" not in repr(resolved_config(outside))
    assert "true_hidden_cost" not in repr(resolved_config(outside))


def test_base_proposed_checkpoint_is_not_masqueraded_as_ablation(tmp_path):
    output = ablation_or_sensitivity(
        "ablation", str(CONFIG), str(tmp_path), "missing_base_proposed.pt",
        "proposed", 1, False, "cpu",
    )
    summary = json.loads((output / "ablation_summary.json").read_text())
    assert [row["method"] for row in summary["results"]] == [
        "no_traffic", "no_energy", "known_preference"
    ]
    assert all(row["status"] == "missing" for row in summary["results"])


def test_report_parser_supports_case_study_and_legacy_input(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "report", "--config", str(CONFIG), "--results-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "report"),
    ])
    assert args.config == str(CONFIG)
    assert args.results_dir == str(tmp_path)
    legacy = parser.parse_args(["plot", "--input", str(tmp_path / "legacy.csv")])
    assert legacy.input.endswith("legacy.csv")


def test_report_cli_generates_case_study_manifest(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame({"episode": [0], "return": [1.0]}).to_csv(
        results / "training_metrics.csv", index=False
    )
    output = report(
        output_dir=str(tmp_path / "figures"),
        config_path=str(CONFIG),
        results_dir=str(results),
    )
    assert (output / "figures_manifest.json").exists()
    assert (output / "fig3_study_area.png").exists()
