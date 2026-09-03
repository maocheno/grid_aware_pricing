from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grid_aware_pricing.config import load_config
from grid_aware_pricing.metrics import (
    _fig5,
    _method_seed_missing,
    generate_case_study_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "york_smoke.yaml"
METHODS = (
    "fixed_tariff", "myopic_local", "proposed", "mappo_no_inference", "ippo",
    "centralized_coordinate_search_reference",
)


def _has_red_watermark(path: Path) -> bool:
    image = plt.imread(path)
    rgb = image[..., :3]
    return bool(np.any((rgb[..., 0] > 0.45) & (rgb[..., 0] > rgb[..., 1] * 1.2)))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_complete_fixture(root: Path) -> None:
    training = root / "training"
    for seed in (11, 29, 47):
        directory = training / "proposed" / f"seed_{seed}"
        directory.mkdir(parents=True)
        x = np.arange(120)
        pd.DataFrame({
            "episode": x,
            "return": 1.0 + x / 100.0,
            "outside_cost_estimate": 13.5 + 3.0 * (1.0 - np.exp(-x / 30.0)),
            "outside_nll": 1.0 / (x + 1.0),
            "wait_violation": 0.2 / (x + 1.0),
            "unmet": np.zeros_like(x, dtype=float),
        }).to_csv(directory / "training_metrics.csv", index=False)
        (directory / "run_manifest.json").write_text(
            json.dumps({"seed": seed, "episodes": 120})
        )

    decoy = training / "ippo" / "seed_999"
    decoy.mkdir(parents=True)
    pd.DataFrame({"episode": [0], "return": [-99.0]}).to_csv(
        decoy / "training_metrics.csv", index=False
    )

    episodes = []
    periods = []
    for method_index, method in enumerate(METHODS):
        for seed in (101, 202, 303):
            episodes.append({
                "method": method,
                "training_seed": seed if method in {"proposed", "mappo_no_inference", "ippo"} else np.nan,
                "scenario_seed": seed,
                "return": 10.0 - method_index, "profit_gbp": 1000.0 - 20 * method_index,
                "welfare_gbp": 1000.0 - 20 * method_index,
                "weighted_hub_profit_welfare": 1000.0 - 20 * method_index,
                "served_requests": 80.0,
                "outside_requests": 10.0, "outside_share": 0.11,
                "pending_requests": 100.0, "admitted_requests": 85.0,
                "admission_ratio": 0.85, "minimum_admission_ratio": 0.65,
                "peak_admission_pressure": 1.5,
                "admitted_full_service_ratio": 80.0 / 85.0,
                "pending_full_service_ratio": 0.8,
                "mean_wait_min": 8.0 + method_index, "p95_wait_min": 14.0 + method_index,
                "max_wait_min": 18.0 + method_index, "wait_violation_rate": 0.1,
                "mean_wait_excess_min": 1.0, "pending_energy_kwh": 2400.0,
                "admitted_energy_kwh": 2040.0, "unmet_energy_kwh": 0.0,
                "peak_queued_energy_kwh": 360.0, "mean_queued_energy_kwh": 90.0,
                "final_queued_energy_kwh": 0.0, "queue_cleared_by_end": True,
                "queue_clearance_period": 4.0,
                "max_queue_vehicle_conservation_error": 0.0,
                "max_queue_energy_conservation_error_kwh": 0.0,
                "grid_energy_kwh": 500.0,
                "peak_grid_import_kwh": 110.0, "pv_utilization": 0.8,
                "battery_throughput_kwh": 100.0, "energy_cost_gbp": 60.0,
                "mean_access_min": 12.0, "p95_access_min": 18.0,
                "mean_detour_min": 3.0, "p95_detour_min": 6.0,
                "outside_mae": 0.5, "outside_nll": 0.2,
                "approx_unilateral_gain": 0.01,
                "centralized_reference_difference": float(method_index),
                "exact_oracle_gap": np.nan,
            })
            for period in range(6):
                for hub in range(8):
                    periods.append({
                        "method": method, "episode": 0, "scenario_seed": seed,
                        "period": period, "timestamp": f"2024-09-20T{16 + period:02d}:00:00",
                        "hub_id": f"H{hub + 1:02d}", "hub_index": hub,
                        "price": 0.35 + 0.01 * hub, "arrivals": 10.0 + hub,
                        "historical_equivalent_vehicles": 2.0,
                        "pending_vehicles": 12.0 + hub, "admission_ratio": 0.8,
                        "admission_pressure": 1.25, "admitted_vehicles": 9.6 + 0.8 * hub,
                        "outside_share": 0.1, "wait_min": 8.0 + hub,
                        "wait_excess_min": max(hub - 6, 0),
                        "requested_energy_kwh": 250.0, "queued_energy_start_kwh": 50.0,
                        "pending_energy_kwh": 300.0, "admitted_energy_kwh": 240.0,
                        "queued_energy_next_kwh": 0.0 if period >= 4 else 60.0,
                        "served_energy_kwh": 240.0, "unmet_energy_kwh": 0.0,
                        "queue_vehicle_conservation_error": 0.0,
                        "queue_energy_conservation_error_kwh": 0.0,
                        "admitted_full_service_ratio": 1.0,
                        "pending_full_service_ratio": 0.8,
                        "pv_used_kwh": 20.0,
                        "battery_charge_kwh": 4.0, "battery_discharge_kwh": 5.0,
                        "grid_import_kwh": 30.0, "grid_utilization": 0.5, "soc": 0.6,
                    })
    episode_path = root / "benchmark_episodes.csv"
    period_path = root / "benchmark_period_hub.csv"
    episode_frame = pd.DataFrame(episodes)
    period_frame = pd.DataFrame(periods)
    episode_frame.to_csv(episode_path, index=False)
    period_frame.to_csv(period_path, index=False)
    mechanism = period_frame[period_frame["method"] == "proposed"]
    mechanism.to_csv(root / "proposed_mechanism_period_hub.csv", index=False)
    config = load_config(CONFIG)
    (root / "benchmark_manifest.json").write_text(json.dumps({
        "output_schema_version": "2.0",
        "environment_schema_version": config["environment_schema_version"],
        "queue_semantics": config["queue_semantics"],
        "zip_sha256": config["data"]["zip_sha256"],
        "scenario_seeds": [101, 202, 303],
        "common_scenario_seeds_across_methods": True,
        "stochastic_evaluation": True,
        "evaluation_mode": "stochastic",
        "artifacts": {
            "benchmark_episodes.csv": {
                "sha256": _sha256(episode_path),
                "rows": len(episode_frame),
                "columns": list(episode_frame.columns),
            },
            "benchmark_period_hub.csv": {
                "sha256": _sha256(period_path),
                "rows": len(period_frame),
                "columns": list(period_frame.columns),
            },
        },
    }))


def test_fig3_complete_and_missing_benchmark_is_diagnostic(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame({"episode": [0, 1], "return": [0.0, 1.0]}).to_csv(
        results / "training_metrics.csv", index=False
    )
    output = tmp_path / "report"
    manifest = generate_case_study_report(CONFIG, results, output)
    by_number = {item["figure_number"]: item for item in manifest["figures"]}
    assert by_number["3"]["status"] == "complete"
    for number in ("4", "5", "6", "7", "8"):
        assert by_number[number]["status"] in {"partial", "blocked"}
        image = Path(by_number[number]["files"][0])
        assert image.exists()
        assert _has_red_watermark(image)
    assert all(item["section_status"] == "planned_extension_not_present_in_source_pdf" for item in manifest["figures"])
    assert any("17 sparse counters" in item for item in manifest["global_limitations"])
    assert any("single six-hour" in item for item in manifest["global_limitations"])
    assert manifest["reference_semantics"]["reference_is_upper_bound"] is False


def test_complete_small_fixture_generates_six_figures_table_and_manifest(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write_complete_fixture(results)
    output = tmp_path / "report"
    manifest = generate_case_study_report(CONFIG, results, output)
    expected = {
        "fig3_study_area.png", "fig4_learning_convergence.png",
        "fig5_dynamic_prices_allocation.png", "fig6_queue_service.png",
        "fig7_energy_grid.png", "fig8_benchmark_performance.png",
        "table1_main_results.csv", "figures_manifest.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    statuses = {item["figure_number"]: item["status"] for item in manifest["figures"]}
    assert statuses == {str(number): "complete" for number in range(3, 9)}
    figure4 = next(item for item in manifest["figures"] if item["figure_number"] == "4")
    assert len(figure4["sources"]) == 3
    assert all("training/proposed/" in source for source in figure4["sources"])
    table = pd.read_csv(output / "table1_main_results.csv")
    assert {"mean", "std", "n", "unit", "availability", "missing_reason"} <= set(table)
    assert {
        "pending_requests", "admitted_requests", "minimum_admission_ratio",
        "peak_admission_pressure", "admitted_full_service_ratio",
        "pending_full_service_ratio", "peak_queued_energy_kwh",
        "final_queued_energy_kwh", "queue_cleared_by_end",
    } <= set(table["metric"])
    assert "full_service_ratio" not in set(table["metric"])
    exact = table[table["metric"] == "exact_oracle_gap"]
    assert exact["availability"].eq("missing").all()
    stored = json.loads((output / "figures_manifest.json").read_text())
    assert stored["schema_version"] == "2.0"
    assert stored["environment_schema_version"] == 2
    assert stored["queue_semantics"] == "dynamic_fluid_carryover_v1"
    assert stored["benchmark_bundle"]["status"] == "compatible"
    assert stored["tables"][0]["table_number"] == "I"
    assert stored["figures"][5]["section"] == "Planned Section V-E"


def test_deterministic_benchmark_is_explicitly_partial(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write_complete_fixture(results)
    manifest_path = results / "benchmark_manifest.json"
    benchmark_manifest = json.loads(manifest_path.read_text())
    benchmark_manifest.update({
        "stochastic_evaluation": False,
        "evaluation_mode": "deterministic_expected_demand",
    })
    manifest_path.write_text(json.dumps(benchmark_manifest))
    manifest = generate_case_study_report(CONFIG, results, tmp_path / "report")
    by_number = {item["figure_number"]: item for item in manifest["figures"]}
    for number in ("6", "7", "8"):
        assert by_number[number]["status"] == "partial"
        assert any("deterministic expected demand" in item for item in by_number[number]["missing_items"])
    assert manifest["benchmark_evaluation"]["stochastic"] is False
    assert manifest["tables"][0]["status"] == "partial"


def test_incompatible_benchmark_bundle_is_blocked(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write_complete_fixture(results)
    episodes = pd.read_csv(results / "benchmark_episodes.csv")
    episodes.loc[0, "return"] += 1.0
    episodes.to_csv(results / "benchmark_episodes.csv", index=False)

    manifest = generate_case_study_report(CONFIG, results, tmp_path / "report")

    assert manifest["benchmark_bundle"]["status"] == "incompatible"
    assert any(
        "artifact hash mismatch" in error
        for error in manifest["benchmark_bundle"]["validation_errors"]
    )
    by_number = {item["figure_number"]: item for item in manifest["figures"]}
    for number in ("6", "7", "8"):
        assert by_number[number]["status"] in {"partial", "blocked"}
    assert manifest["tables"][0]["status"] == "partial"


def test_method_seed_requirements_distinguish_training_and_scenario_seeds():
    rows = []
    trained = {"proposed", "mappo_no_inference", "ippo"}
    for method in METHODS:
        for scenario_seed in (11, 29, 47):
            rows.append({
                "method": method,
                "scenario_seed": scenario_seed,
                "training_seed": 5 if method in trained else np.nan,
            })
    missing = _method_seed_missing(pd.DataFrame(rows))
    assert missing == [
        "proposed has fewer than 3 training seeds",
        "mappo_no_inference has fewer than 3 training seeds",
        "ippo has fewer than 3 training seeds",
    ]


def test_fig5_marks_undertrained_checkpoint_as_mechanism_only(tmp_path):
    results = tmp_path / "results"
    checkpoint = results / "training" / "proposed" / "seed_5"
    checkpoint.mkdir(parents=True)
    (checkpoint / "run_manifest.json").write_text(
        json.dumps({"seed": 5, "episodes": 2})
    )
    rows = []
    for period in range(6):
        for hub in range(8):
            rows.append({
                "period": period,
                "timestamp": f"2024-09-20T{16 + period:02d}:00:00",
                "hub_id": f"H{hub + 1:02d}",
                "price": 0.4,
                "arrivals": 10.0,
                "outside_share": 0.1,
            })
    pd.DataFrame(rows).to_csv(
        results / "proposed_mechanism_period_hub.csv", index=False
    )
    status, missing, sources = _fig5(
        results, pd.DataFrame(), tmp_path / "fig5.png"
    )
    assert status == "partial"
    assert any("undertrained smoke model (2 episodes)" in item for item in missing)
    assert checkpoint / "run_manifest.json" in sources
