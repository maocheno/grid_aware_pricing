"""Command-line workflows for training, evaluation, benchmarks, and plotting."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import itertools
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import load_config, save_resolved_config
from .environment import GridAwarePricingEnv
from .experiments import (
    FixedTariffPolicy,
    MyopicLocalPolicy,
    OUTPUT_SCHEMA_VERSION,
    TRAINED_METHODS,
    TrainedPolicy,
    aggregate_episode_rows,
    approximate_unilateral_gain,
    centralized_coordinate_search_reference,
    combine_evaluations,
    evaluation_from_infos,
    evaluate_policy,
    method_config,
    period_hub_rows,
    scenario_seed_sequence,
)
from .mappo import MAPPO
from .metrics import generate_case_study_report, plot_training_metrics, plot_trajectory


_METHODS = (
    "proposed",
    "mappo_no_inference",
    "ippo",
    "known_preference",
    "no_traffic",
    "no_energy",
)


def _output_dir(config: dict[str, Any], explicit: str | None) -> Path:
    path = Path(explicit or config["output"]["directory"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def _artifact_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "environment_schema_version": config.get("environment_schema_version"),
        "queue_semantics": config.get("queue_semantics"),
        "zip_sha256": config.get("data", {}).get("zip_sha256"),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_overrides(
    method: str,
    seed: int | None,
    episodes: int | None,
    episodes_per_update: int | None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {"experiment": {"method": method}}
    if seed is not None:
        overrides["seed"] = int(seed)
    if episodes is not None or episodes_per_update is not None:
        overrides["training"] = {}
        if episodes is not None:
            overrides["training"]["episodes_per_seed"] = int(episodes)
        if episodes_per_update is not None:
            overrides["training"]["episodes_per_update"] = int(episodes_per_update)
    return overrides


def _training_period_hub_rows(
    infos: list[dict[str, Any]],
    update: int,
    method: str,
    training_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episode_ids = list(dict.fromkeys(int(info.get("episode_index", -1)) for info in infos))
    for episode_id in episode_ids:
        episode_infos = [
            info for info in infos if int(info.get("episode_index", -1)) == episode_id
        ]
        episode_rows = period_hub_rows(
            method,
            episode_id,
            None,
            episode_infos,
            training_seed=training_seed,
        )
        for row in episode_rows:
            row["training_update"] = update
            row["episode_index"] = episode_id
        rows.extend(episode_rows)
    return rows


def train(
    config_path: str,
    output_dir: str | None = None,
    device: str = "cpu",
    method: str = "proposed",
    seed: int | None = None,
    episodes: int | None = None,
    episodes_per_update: int | None = None,
) -> Path:
    config = load_config(
        config_path,
        overrides=_training_overrides(
            method, seed, episodes, episodes_per_update
        ),
    )
    config = method_config(config, method)
    output = _output_dir(config, output_dir)
    save_resolved_config(config, output / "resolved_config.yaml")
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    env = GridAwarePricingEnv(config)
    algorithm = MAPPO(
        config, env.observation_dim, env.global_state_dim, device, method=method
    )
    training = config["training"]
    total_episodes = int(
        episodes
        if episodes is not None
        else training.get("episodes_per_seed", training.get("updates", 1))
    )
    requested_batch = int(
        episodes_per_update
        if episodes_per_update is not None
        else training.get("episodes_per_update", 20)
    )
    if total_episodes <= 0 or requested_batch <= 0:
        raise ValueError("episodes and episodes-per-update must be positive")
    batch_size = min(requested_batch, total_episodes)
    training_rows: list[dict[str, Any]] = []
    period_hub_path = output / "period_hub.csv"
    period_hub_path.unlink(missing_ok=True)
    wrote_period_header = False
    episodes_done = 0
    update = 0
    best_return = -np.inf
    started = time.perf_counter()
    while episodes_done < total_episodes:
        current_batch = min(batch_size, total_episodes - episodes_done)
        buffer, summaries, infos = algorithm.collect_episodes(env, current_batch)
        diagnostics = algorithm.update(buffer, 0.0)
        batch_mean_return = float(np.mean([row["return"] for row in summaries]))
        for summary in summaries:
            row = dict(summary)
            row["episode"] = episodes_done + int(summary["episode_in_batch"])
            row["update"] = update
            row["episodes_in_update"] = current_batch
            row["training_batch_mean_return"] = batch_mean_return
            row.update(diagnostics)
            training_rows.append(row)
        period_rows = _training_period_hub_rows(
            infos, update, method, int(config["seed"])
        )
        pd.DataFrame(period_rows).to_csv(
            period_hub_path,
            mode="a",
            header=not wrote_period_header,
            index=False,
        )
        wrote_period_header = True
        episodes_done += current_batch
        if batch_mean_return > best_return:
            best_return = batch_mean_return
            algorithm.save(output / "best_checkpoint.pt", env.estimator.state_dict())
        update += 1

    algorithm.save(output / "final_checkpoint.pt", env.estimator.state_dict())
    shutil.copyfile(output / "final_checkpoint.pt", output / "checkpoint.pt")
    training_frame = pd.DataFrame(training_rows)
    training_frame.to_csv(output / "training_metrics.csv", index=False)
    training_frame.to_csv(output / "train_metrics.csv", index=False)
    manifest = {
        "command": "train",
        "method": method,
        "seed": int(config["seed"]),
        "episodes": total_episodes,
        "episodes_per_update": batch_size,
        "updates": update,
        "best_selection": "training_batch_mean_return",
        "best_training_batch_mean_return": float(best_return),
        "duration_seconds": time.perf_counter() - started,
        "checkpoint_compatibility_copy": "checkpoint.pt contains final_checkpoint.pt",
        "hidden_preference_in_config": False if env.is_york else "synthetic_config_only",
        **_artifact_metadata(config),
    }
    _write_json(output / "run_manifest.json", manifest)
    plot_training_metrics(training_frame, output)
    return output


def _load_policy(
    config: dict[str, Any],
    method: str,
    checkpoint: str | None,
    device: str,
):
    runtime = method_config(config, method)
    env = GridAwarePricingEnv(runtime)
    if method == "fixed_tariff":
        return runtime, FixedTariffPolicy()
    if method == "myopic_local":
        return runtime, MyopicLocalPolicy(runtime)
    if method not in TRAINED_METHODS:
        raise ValueError(f"unknown evaluation method: {method}")
    if not checkpoint:
        raise ValueError(f"method {method!r} requires --checkpoint")
    return runtime, TrainedPolicy(
        runtime,
        method,
        checkpoint,
        env.observation_dim,
        env.global_state_dim,
        device,
    )


def _write_evaluation(output: Path, result, prefix: str = "evaluation") -> None:
    result.period_hub.to_csv(output / f"{prefix}_period_hub.csv", index=False)
    result.episodes.to_csv(output / f"{prefix}_episodes.csv", index=False)
    result.aggregate.to_csv(output / f"{prefix}_aggregate.csv", index=False)
    _write_json(output / f"{prefix}_summary.json", result.metadata)
    if prefix == "evaluation":
        result.period_hub.to_csv(output / "period_hub.csv", index=False)
        result.episodes.to_csv(output / "episodes.csv", index=False)
        result.aggregate.to_csv(output / "aggregate.csv", index=False)


def evaluate(
    config_path: str,
    checkpoint: str | None,
    output_dir: str | None = None,
    device: str = "cpu",
    method: str = "proposed",
    episodes: int = 1,
    stochastic: bool = False,
    seeds: list[int] | None = None,
    online_lower_layer: bool = False,
) -> Path:
    config = load_config(config_path)
    output = _output_dir(config, output_dir)
    runtime, policy = _load_policy(config, method, checkpoint, device)
    save_resolved_config(runtime, output / "resolved_config.yaml")
    seed_values = scenario_seed_sequence(
        seeds or list(runtime["training"].get("seeds", [runtime["seed"]])),
        episodes,
    )
    result = evaluate_policy(
        runtime,
        policy,
        seed_values,
        stochastic=stochastic,
        online_lower_layer=online_lower_layer,
    )
    _write_evaluation(output, result)
    return output


def validate_data(config_path: str, output_dir: str | None = None) -> Path:
    config = load_config(config_path)
    output = _output_dir(config, output_dir)
    if config.get("data", {}).get("mode") == "york_zip":
        from .york_data import deterministic_fixed_tariff_sanity, structural_validation

        scenario = config["_york_scenario"]
        structural = structural_validation(scenario)
        sanity = deterministic_fixed_tariff_sanity(scenario)
        result = {
            "mode": "york_zip",
            "structural_checks": structural,
            "all_structural_checks_pass": bool(all(structural.values())),
            "deterministic_fixed_tariff_sanity": sanity,
            "zip_sha256": scenario.zip_sha256,
            "environment_schema_version": config["environment_schema_version"],
            "queue_semantics": config["queue_semantics"],
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "limitations": list(scenario.data_limitations),
        }
    else:
        env = GridAwarePricingEnv(config)
        result = {
            "mode": "synthetic",
            "structural_checks": {
                "positive_periods": env.periods > 0,
                "price_vectors_match_hubs": len(env.price_min) == env.n_hubs,
                "finite_route_profiles": bool(np.isfinite(env.route_time_profiles).all()),
            },
            "all_structural_checks_pass": True,
            "deterministic_fixed_tariff_sanity": None,
            "sha256": None,
            "limitations": ["Synthetic scenario has no packaged source-data checksum."],
        }
    _write_json(output / "validate_data.json", result)
    return output


def _parse_checkpoint_items(items: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--checkpoint must use METHOD=PATH")
        method, path = item.split("=", 1)
        if not method or not path:
            raise ValueError("--checkpoint must use METHOD=PATH")
        result[method] = path
    return result


def _policy_trajectory(result, episode: int = 0) -> np.ndarray:
    rows = result.period_hub[result.period_hub["episode"] == episode]
    return rows.pivot(index="period", columns="hub", values="price").sort_index().to_numpy()


def benchmark(
    config_path: str,
    output_dir: str | None,
    checkpoints: dict[str, str],
    seeds: list[int],
    eval_episodes: int,
    reference_budget: int,
    unilateral_budget: int,
    device: str = "cpu",
    stochastic: bool = False,
) -> Path:
    config = load_config(config_path)
    output = _output_dir(config, output_dir)
    seed_values = scenario_seed_sequence(seeds or [int(config["seed"])], eval_episodes)
    evaluations = []
    missing: list[dict[str, str]] = []
    policies: list[tuple[dict[str, Any], Any]] = [
        (method_config(config, "fixed_tariff"), FixedTariffPolicy()),
        (method_config(config, "myopic_local"), MyopicLocalPolicy(method_config(config, "myopic_local"))),
    ]
    requested_trained = list(dict.fromkeys([
        "proposed", "mappo_no_inference", "ippo", *checkpoints.keys()
    ]))
    for method in requested_trained:
        checkpoint = checkpoints.get(method)
        if method not in TRAINED_METHODS:
            missing.append({"method": method, "reason": "unsupported_method"})
            continue
        if not checkpoint:
            missing.append({"method": method, "reason": "checkpoint_not_provided"})
            continue
        if not Path(checkpoint).exists():
            missing.append({"method": method, "reason": f"checkpoint_not_found:{checkpoint}"})
            continue
        runtime = method_config(config, method)
        env = GridAwarePricingEnv(runtime)
        try:
            policy = TrainedPolicy(
                runtime, method, checkpoint, env.observation_dim, env.global_state_dim, device
            )
        except (ValueError, RuntimeError, KeyError) as error:
            missing.append({"method": method, "reason": f"checkpoint_incompatible:{error}"})
            continue
        policies.append((runtime, policy))
    for runtime, policy in policies:
        evaluations.append(
            evaluate_policy(runtime, policy, seed_values, stochastic=stochastic)
        )
        if policy.name == "proposed" and isinstance(policy, TrainedPolicy):
            mechanism = evaluate_policy(
                runtime,
                policy,
                [seed_values[0]],
                stochastic=False,
                online_lower_layer=True,
            )
            mechanism.period_hub.to_csv(
                output / "proposed_mechanism_period_hub.csv", index=False
            )
            mechanism.episodes.to_csv(
                output / "proposed_mechanism_episodes.csv", index=False
            )
            _write_json(
                output / "proposed_mechanism_summary.json", mechanism.metadata
            )
    combined = combine_evaluations(evaluations)
    unilateral_reports: list[dict[str, Any]] = []
    if unilateral_budget > 0:
        for evaluation in evaluations:
            method_name = str(evaluation.metadata["method"])
            for episode, seed in enumerate(seed_values):
                trajectory = _policy_trajectory(evaluation, episode)
                gain = approximate_unilateral_gain(config, trajectory, seed, unilateral_budget)
                found_gain = float(gain["found_maximum_gain"])
                mask = (
                    (combined.episodes["method"] == method_name)
                    & (combined.episodes["episode"] == episode)
                    & (combined.episodes["scenario_seed"] == seed)
                )
                combined.episodes.loc[mask, "approx_unilateral_gain"] = found_gain
                unilateral_reports.append({
                    "method": method_name, "episode": episode, "scenario_seed": seed,
                    "found_maximum_gain": found_gain, "gain_by_hub": gain["gain_by_hub"],
                    "lower_bound_on_exact_gain": True, "budget": gain["budget"],
                    "evaluations": gain["evaluations"], "uses_true_preference": True,
                    "hidden_preference_access": gain["hidden_preference_access"],
                })

    reference_reports: list[dict[str, Any]] = []
    reference_episodes: list[tuple[int, list[dict[str, Any]]]] = []
    for episode, seed in enumerate(seed_values):
        reference = centralized_coordinate_search_reference(config, seed, reference_budget)
        reference_episodes.append((int(seed), reference["infos"]))
        report = dict(reference["solver_report"])
        report.update({"episode": episode, "scenario_seed": seed})
        reference_reports.append(report)
    reference_evaluation = evaluation_from_infos(
        "centralized_coordinate_search_reference", reference_episodes,
        {"uses_true_preference": True, "is_exact": False, "is_upper_bound": False},
    )
    reference_frame = reference_evaluation.episodes.copy()
    reference_frame["centralized_reference_difference"] = 0.0
    reference_frame["centralized_reference_gap"] = 0.0
    if not combined.episodes.empty:
        reference_by_seed = reference_frame.set_index("scenario_seed")["return"]
        differences = [
            float(reference_by_seed.loc[int(seed)] - value)
            for seed, value in zip(combined.episodes["scenario_seed"], combined.episodes["return"])
        ]
        combined.episodes["centralized_reference_difference"] = differences
        combined.episodes["centralized_reference_gap"] = differences
    all_episodes = pd.concat([combined.episodes, reference_frame], ignore_index=True)
    all_period_hub = pd.concat(
        [combined.period_hub, reference_evaluation.period_hub], ignore_index=True
    )
    period_path = output / "benchmark_period_hub.csv"
    episode_path = output / "benchmark_episodes.csv"
    aggregate_path = output / "benchmark_aggregate.csv"
    all_period_hub.to_csv(period_path, index=False)
    all_episodes.to_csv(episode_path, index=False)
    aggregate_episode_rows(all_episodes).to_csv(aggregate_path, index=False)
    _write_json(output / "reference_solver_reports.json", reference_reports)
    _write_json(output / "unilateral_gain_reports.json", unilateral_reports)
    _write_json(
        output / "benchmark_manifest.json",
        {
            "scenario_seeds": seed_values,
            "common_scenario_seeds_across_methods": True,
            "stochastic_evaluation": bool(stochastic),
            "evaluation_mode": "stochastic" if stochastic else "deterministic_expected_demand",
            "available_methods": [policy.name for _, policy in policies],
            "missing": missing,
            "reference_budget_per_episode": reference_budget,
            "unilateral_budget": unilateral_budget,
            "unilateral_gain_status": "not_run" if unilateral_budget <= 0 else "completed",
            "evaluation_lower_layer": "frozen",
            "estimator_reset_each_episode": True,
            "exact_oracle_gap": None,
            "centralized_reference_difference_computed": True,
            "centralized_reference_gap": "deprecated alias of signed centralized_reference_difference; the reference is neither an oracle nor an upper bound",
            "reference_is_exact": False,
            "reference_is_upper_bound": False,
            "hidden_preference_fields_are_evaluation_only": True,
            **_artifact_metadata(config),
            "artifacts": {
                "benchmark_period_hub.csv": {
                    "sha256": _file_sha256(period_path),
                    "rows": int(len(all_period_hub)),
                    "columns": list(all_period_hub.columns),
                },
                "benchmark_episodes.csv": {
                    "sha256": _file_sha256(episode_path),
                    "rows": int(len(all_episodes)),
                    "columns": list(all_episodes.columns),
                },
                "benchmark_aggregate.csv": {
                    "sha256": _file_sha256(aggregate_path),
                    "rows": int(len(aggregate_episode_rows(all_episodes))),
                },
            },
        },
    )
    return output


def _checkpoint_mapping(
    checkpoint: str | dict[str, str] | None, method: str | None
) -> dict[str, str]:
    if isinstance(checkpoint, dict):
        return checkpoint
    return {method: checkpoint} if checkpoint and method else {}


def _sensitivity_runtime(
    config: dict[str, Any], axis: str, level: float
) -> dict[str, Any]:
    runtime = deepcopy(config)
    if runtime.get("data", {}).get("mode") != "york_zip":
        if axis == "demand_multiplier":
            runtime["demand"]["base_od_counts"] = (
                np.asarray(runtime["demand"]["base_od_counts"], dtype=float) * level
            ).tolist()
        elif axis == "grid_cap_multiplier":
            runtime["hubs"]["grid_cap_kw"] = (
                np.asarray(runtime["hubs"]["grid_cap_kw"], dtype=float) * level
            ).tolist()
        elif axis == "true_outside_cost":
            runtime["choice"]["true_outside_cost"] = float(level)
        elif axis == "inverse_cost_sensitivity":
            runtime["choice"]["inverse_cost_sensitivity"] = float(level)
        return runtime
    scenario = runtime["_york_scenario"]
    if axis == "demand_multiplier":
        scenario = replace(
            scenario, od_expected_demand=scenario.od_expected_demand * float(level)
        )
    elif axis == "grid_cap_multiplier":
        energy = {key: np.asarray(value).copy() for key, value in scenario.energy_parameters.items()}
        energy["grid_import_limit_kwh"] *= float(level)
        scenario = replace(scenario, energy_parameters=energy)
    elif axis == "true_outside_cost":
        package = deepcopy(scenario.package_config)
        package["user_choice"]["outside_option"]["true_hidden_cost_gbp"] = float(level)
        scenario = replace(scenario, package_config=package)
    elif axis == "inverse_cost_sensitivity":
        runtime["choice"]["inverse_cost_sensitivity"] = float(level)
        package = deepcopy(scenario.package_config)
        package["user_choice"]["inverse_cost_sensitivity_lambda_per_gbp"] = float(level)
        scenario = replace(scenario, package_config=package)
    runtime["_york_scenario"] = scenario
    return runtime


def ablation_or_sensitivity(
    command: str,
    config_path: str,
    output_dir: str | None,
    checkpoint: str | dict[str, str] | None,
    method: str | None,
    episodes: int,
    stochastic: bool,
    device: str,
) -> Path:
    config = load_config(config_path)
    output = _output_dir(config, output_dir)
    checkpoint_map = _checkpoint_mapping(checkpoint, method)
    seeds = scenario_seed_sequence(
        list(config["training"].get("seeds", [config["seed"]])), episodes
    )
    results = []
    statuses: list[dict[str, Any]] = []
    if command == "ablation":
        for ablation_method in ("no_traffic", "no_energy", "known_preference"):
            path = checkpoint_map.get(ablation_method)
            if not path:
                statuses.append({"method": ablation_method, "status": "missing", "reason": "checkpoint_not_provided"})
                continue
            if not Path(path).exists():
                statuses.append({"method": ablation_method, "status": "missing", "reason": f"checkpoint_not_found:{path}"})
                continue
            runtime = method_config(config, ablation_method)
            env = GridAwarePricingEnv(runtime)
            try:
                policy = TrainedPolicy(runtime, ablation_method, path, env.observation_dim, env.global_state_dim, device)
            except (ValueError, RuntimeError, KeyError) as error:
                statuses.append({"method": ablation_method, "status": "incompatible", "reason": str(error)})
                continue
            results.append(evaluate_policy(runtime, policy, seeds, stochastic=stochastic))
            statuses.append({"method": ablation_method, "status": "completed", "checkpoint": path})
    else:
        path = checkpoint_map.get("proposed") or (checkpoint if isinstance(checkpoint, str) else None)
        if not path or not Path(path).exists():
            statuses.append({"method": "proposed", "status": "missing", "reason": "checkpoint_not_provided_or_found"})
        else:
            levels = {
                "demand_multiplier": (0.85, 1.0, 1.15),
                "grid_cap_multiplier": (0.85, 1.0, 1.15),
                "inverse_cost_sensitivity": (0.22, 0.30, 0.55),
                "true_outside_cost": (14.0, 16.5, 19.0),
            }
            for axis, axis_levels in levels.items():
                for level in axis_levels:
                    runtime = method_config(_sensitivity_runtime(config, axis, level), "proposed")
                    env = GridAwarePricingEnv(runtime)
                    try:
                        policy = TrainedPolicy(runtime, "proposed", path, env.observation_dim, env.global_state_dim, device)
                    except (ValueError, RuntimeError, KeyError) as error:
                        statuses.append({"method": "proposed", "axis": axis, "level": level, "status": "incompatible", "reason": str(error)})
                        continue
                    result = evaluate_policy(runtime, policy, seeds, stochastic=stochastic)
                    result.period_hub["axis"] = axis
                    result.period_hub["level"] = level
                    result.episodes["axis"] = axis
                    result.episodes["level"] = level
                    results.append(result)
                    statuses.append({"method": "proposed", "axis": axis, "level": level, "status": "completed"})
    combined = combine_evaluations(results)
    if results:
        period = pd.concat([result.period_hub for result in results], ignore_index=True)
        episode_frame = pd.concat([result.episodes for result in results], ignore_index=True)
        combined = type(combined)(period, episode_frame, aggregate_episode_rows(episode_frame), combined.metadata)
        _write_evaluation(output, combined, prefix=command)
    _write_json(
        output / f"{command}_summary.json",
        {
            "command": command,
            "scenario_seeds": seeds,
            "common_scenario_seeds": True,
            "results": statuses,
            **_artifact_metadata(config),
        },
    )
    return output


def grid_oracle(config_path: str, output_dir: str | None = None) -> Path:
    config = load_config(config_path)
    if config.get("data", {}).get("mode") == "york_zip" and int(config["system"]["n_hubs"]) > 3:
        raise ValueError(
            "York oracle refuses Cartesian exhaustive search for n_hubs > 3; "
            "use benchmark --reference-budget for the budgeted centralized reference."
        )
    output = _output_dir(config, output_dir)
    save_resolved_config(config, output / "resolved_config.yaml")
    env = GridAwarePricingEnv(config)
    points = int(config["oracle"]["points_per_hub"])
    grids = [
        np.linspace(lo, hi, points)
        for lo, hi in zip(config["price"]["min"], config["price"]["max"])
    ]
    rows: list[dict[str, Any]] = []
    deterministic_demand = bool(config["oracle"]["deterministic_expected_demand"])
    for period in range(env.periods):
        snapshot = env.snapshot()
        best_prices: np.ndarray | None = None
        best_reward = -np.inf
        best_info: dict | None = None
        true_cost = float(config["choice"]["true_outside_cost"])
        for candidate in itertools.product(*grids):
            env.restore(snapshot)
            result = env.step(
                np.asarray(candidate),
                deterministic_demand=deterministic_demand,
                update_inference=False,
                outside_cost=true_cost,
            )
            if result.reward > best_reward + 1e-12:
                best_reward = result.reward
                best_prices = np.asarray(candidate)
                best_info = result.info
        assert best_prices is not None and best_info is not None
        env.restore(snapshot)
        chosen = env.step(
            best_prices,
            deterministic_demand=deterministic_demand,
            update_inference=False,
            outside_cost=true_cost,
        )
        row = {
            "period": period,
            "grid_oracle_reward": chosen.reward,
            "weighted_hub_profit_welfare": chosen.info["weighted_hub_profit_welfare"],
            "grid_oracle_not_exact_gne_proof": True,
        }
        for hub, price in enumerate(best_prices):
            row[f"price_hub_{hub}"] = float(price)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "grid_oracle_trajectory.csv", index=False)
    _write_json(
        output / "grid_oracle_summary.json",
        {
            "oracle_type": "centralized_cartesian_price_grid_reference",
            "is_exact_gne_proof": False,
            "uses_true_outside_cost_for_evaluation_only": True,
            "uses_deterministic_expected_demand": deterministic_demand,
            "total_reward": float(frame["grid_oracle_reward"].sum()),
            "weighted_hub_profit_welfare": float(
                frame["weighted_hub_profit_welfare"].sum()
            ),
        },
    )
    return output


def report(
    path: str | None = None,
    output_dir: str | None = None,
    *,
    config_path: str | None = None,
    results_dir: str | None = None,
) -> Path:
    if config_path and results_dir:
        output = Path(output_dir) if output_dir else Path(results_dir) / "report"
        manifest = generate_case_study_report(config_path, results_dir, output)
        for item in manifest["figures"]:
            missing = "、".join(item["missing_items"]) if item["missing_items"] else "无"
            print(
                f"{item['section']}，Fig.{item['figure_number']}，状态={item['status']}，"
                f"路径={','.join(item['files']) or '未生成'}，缺失={missing}"
            )
        return output
    if not path:
        raise ValueError("report requires --config and --results-dir, or legacy --input")
    source = Path(path)
    output = Path(output_dir) if output_dir else source.parent / "report"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    if "update" in frame.columns:
        plot_training_metrics(frame, output)
    else:
        plot_trajectory(frame, output)
    return output


def plot(
    path: str | None = None,
    output_dir: str | None = None,
    *,
    config_path: str | None = None,
    results_dir: str | None = None,
) -> Path:
    return report(path, output_dir, config_path=config_path, results_dir=results_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grid-pricing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-data")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--output-dir")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--method", choices=_METHODS, default="proposed")
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--episodes", type=int)
    train_parser.add_argument("--episodes-per-update", type=int)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint")
    evaluate_parser.add_argument(
        "--method", choices=("fixed_tariff", "myopic_local", *_METHODS), default="proposed"
    )
    evaluate_parser.add_argument("--episodes", type=int, default=1)
    evaluate_parser.add_argument("--stochastic", action="store_true")
    evaluate_parser.add_argument("--online-lower-layer", action="store_true")
    evaluate_parser.add_argument("--seeds", nargs="+", type=int)
    evaluate_parser.add_argument("--output-dir")
    evaluate_parser.add_argument("--device", default="cpu")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--config", required=True)
    benchmark_parser.add_argument("--output-dir", required=True)
    benchmark_parser.add_argument("--checkpoint", action="append", default=[])
    benchmark_parser.add_argument("--seeds", nargs="+", type=int, required=True)
    benchmark_parser.add_argument("--eval-episodes", type=int, default=1)
    benchmark_parser.add_argument("--reference-budget", type=int, default=100)
    benchmark_parser.add_argument("--unilateral-budget", type=int, default=0)
    benchmark_parser.add_argument("--stochastic", action="store_true")
    benchmark_parser.add_argument("--device", default="cpu")

    for name in ("ablation", "sensitivity"):
        experiment_parser = subparsers.add_parser(name)
        experiment_parser.add_argument("--config", required=True)
        experiment_parser.add_argument("--checkpoint", action="append", default=[])
        experiment_parser.add_argument("--method", choices=_METHODS)
        experiment_parser.add_argument("--episodes", type=int, default=1)
        experiment_parser.add_argument("--stochastic", action="store_true")
        experiment_parser.add_argument("--output-dir")
        experiment_parser.add_argument("--device", default="cpu")

    oracle_parser = subparsers.add_parser("oracle")
    oracle_parser.add_argument("--config", required=True)
    oracle_parser.add_argument("--output-dir")

    for name in ("report", "plot"):
        plot_parser = subparsers.add_parser(name)
        plot_parser.add_argument("--config")
        plot_parser.add_argument("--results-dir")
        plot_parser.add_argument("--input")
        plot_parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        output = validate_data(args.config, args.output_dir)
    elif args.command == "train":
        output = train(
            args.config,
            args.output_dir,
            args.device,
            args.method,
            args.seed,
            args.episodes,
            args.episodes_per_update,
        )
    elif args.command == "evaluate":
        output = evaluate(
            args.config,
            args.checkpoint,
            args.output_dir,
            args.device,
            args.method,
            args.episodes,
            args.stochastic,
            args.seeds,
            args.online_lower_layer,
        )
    elif args.command == "benchmark":
        output = benchmark(
            args.config,
            args.output_dir,
            _parse_checkpoint_items(args.checkpoint),
            args.seeds,
            args.eval_episodes,
            args.reference_budget,
            args.unilateral_budget,
            args.device,
            args.stochastic,
        )
    elif args.command in {"ablation", "sensitivity"}:
        output = ablation_or_sensitivity(
            args.command,
            args.config,
            args.output_dir,
            (
                _parse_checkpoint_items(args.checkpoint)
                if any("=" in item for item in args.checkpoint)
                else (args.checkpoint[0] if args.checkpoint else None)
            ),
            args.method,
            args.episodes,
            args.stochastic,
            args.device,
        )
    elif args.command == "oracle":
        output = grid_oracle(args.config, args.output_dir)
    else:
        if not args.input and not (args.config and args.results_dir):
            raise ValueError("report/plot requires --config and --results-dir, or --input")
        output = report(
            args.input,
            args.output_dir,
            config_path=args.config,
            results_dir=args.results_dir,
        )
    print(output.resolve())


if __name__ == "__main__":
    main()
