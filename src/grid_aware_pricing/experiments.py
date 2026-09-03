"""Unified policies, evaluation, and budgeted reference experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .environment import GridAwarePricingEnv
from .mappo import MAPPO


OUTPUT_SCHEMA_VERSION = "2.0"


TRAINED_METHODS = {
    "proposed",
    "mappo_no_inference",
    "ippo",
    "known_preference",
    "no_traffic",
    "no_energy",
}


def method_config(config: dict[str, Any], method: str) -> dict[str, Any]:
    """Return a runtime config for an experimental method without exposing hidden cost."""
    result = deepcopy(config)
    experiment = result.setdefault("experiment", {})
    experiment["method"] = method
    masks = experiment.setdefault("observation_masks", {})
    if method in {"fixed_tariff", "myopic_local", "mappo_no_inference", "ippo"}:
        experiment["outside_mode"] = "fixed"
    elif method == "known_preference":
        experiment["outside_mode"] = "known"
    else:
        experiment["outside_mode"] = "inferred"
    if method == "no_traffic":
        masks["no_traffic"] = True
    if method == "no_energy":
        masks["no_energy"] = True
    return result


def local_immediate_reward(config: dict[str, Any], info: dict[str, Any]) -> np.ndarray:
    wait = np.asarray(info.get("wait_excess", info["wait_violation"]), dtype=float)
    value = (
        np.asarray(info["profit"], dtype=float)
        - float(config["reward"]["wait_penalty"]) * wait
        - float(config["reward"]["unmet_penalty"]) * np.asarray(info["unmet"], dtype=float)
    )
    return value * float(config["reward"].get("scale", 1.0))


class Policy(Protocol):
    name: str
    online_lower_layer: bool

    def prices(
        self,
        observations: np.ndarray,
        global_state: np.ndarray,
        env: GridAwarePricingEnv,
    ) -> np.ndarray:
        ...


@dataclass
class FixedTariffPolicy:
    tariff: float = 0.45
    name: str = "fixed_tariff"
    online_lower_layer: bool = False

    def prices(
        self,
        observations: np.ndarray,
        global_state: np.ndarray,
        env: GridAwarePricingEnv,
    ) -> np.ndarray:
        del observations, global_state
        return np.clip(np.full(env.n_hubs, self.tariff), env.price_min, env.price_max)


@dataclass
class MyopicLocalPolicy:
    """Jacobi local best responses from one common period snapshot."""

    config: dict[str, Any]
    grid_points: int | None = None
    outside_estimate: float = 13.5
    name: str = "myopic_local"
    online_lower_layer: bool = False

    def _grid(self, env: GridAwarePricingEnv, hub: int) -> np.ndarray:
        configured = self.config.get("experiment", {}).get("myopic_price_grid")
        if configured is not None:
            values = np.asarray(configured, dtype=float)
            return values[(values >= env.price_min[hub]) & (values <= env.price_max[hub])]
        points = int(
            self.grid_points
            or self.config.get("experiment", {}).get("myopic_grid_points", 7)
        )
        return np.linspace(env.price_min[hub], env.price_max[hub], max(points, 2))

    def prices(
        self,
        observations: np.ndarray,
        global_state: np.ndarray,
        env: GridAwarePricingEnv,
    ) -> np.ndarray:
        del observations, global_state
        env.estimator.cost = float(self.outside_estimate)
        snapshot = env.snapshot()
        previous = np.asarray(env.previous_price, dtype=float).copy()
        chosen = previous.copy()
        for hub in range(env.n_hubs):
            best_value = -np.inf
            best_price = previous[hub]
            for price in self._grid(env, hub):
                env.restore(snapshot)
                candidate = previous.copy()
                candidate[hub] = price
                result = env.step(
                    candidate,
                    deterministic_demand=True,
                    update_inference=False,
                    outside_cost=float(self.outside_estimate),
                )
                objective = float(local_immediate_reward(self.config, result.info)[hub])
                if objective > best_value + 1e-12:
                    best_value = objective
                    best_price = float(price)
            chosen[hub] = best_price
        env.restore(snapshot)
        return chosen


class TrainedPolicy:
    """Checkpoint policy evaluated with deterministic Beta means."""

    def __init__(
        self,
        config: dict[str, Any],
        method: str,
        checkpoint: str | Path,
        observation_dim: int,
        global_state_dim: int,
        device: str = "cpu",
    ) -> None:
        if method not in TRAINED_METHODS:
            raise ValueError(f"unsupported trained method: {method}")
        self.name = method
        self.online_lower_layer = False
        self.supports_online_lower_layer = method in {
            "proposed", "no_traffic", "no_energy"
        }
        self.algorithm = MAPPO(
            config, observation_dim, global_state_dim, device=device, method=method
        )
        self.checkpoint = self.algorithm.load(checkpoint)
        saved_method = self.checkpoint.get("method")
        if saved_method is not None and str(saved_method) != method:
            raise ValueError(
                f"checkpoint method {saved_method!r} is incompatible with requested "
                f"method {method!r}"
            )
        estimator = self.checkpoint.get("estimator") or {}
        self.initial_outside_estimate = estimator.get("cost")
        checkpoint_config = self.checkpoint.get("config") or {}
        self.training_seed = checkpoint_config.get("seed")

    def prices(
        self,
        observations: np.ndarray,
        global_state: np.ndarray,
        env: GridAwarePricingEnv,
    ) -> np.ndarray:
        del env
        return self.algorithm.act(
            observations, global_state, deterministic=True
        ).prices


@dataclass
class EvaluationResult:
    period_hub: pd.DataFrame
    episodes: pd.DataFrame
    aggregate: pd.DataFrame
    metadata: dict[str, Any]


def scenario_seed_sequence(seeds: list[int] | tuple[int, ...], episodes: int) -> list[int]:
    """Return reproducible, unique uint32 scenario seeds derived from base seeds."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not seeds:
        raise ValueError("at least one scenario seed is required")
    base = [int(seed) for seed in seeds]
    if episodes <= len(base) and len(set(base[:episodes])) == episodes:
        return base[:episodes]
    children = np.random.SeedSequence(base).spawn(episodes)
    result: list[int] = []
    used: set[int] = set()
    for index, child in enumerate(children):
        value = int(child.generate_state(1, dtype=np.uint32)[0])
        while value in used:
            value = int(np.random.SeedSequence([*base, index, value]).generate_state(1, dtype=np.uint32)[0])
        result.append(value)
        used.add(value)
    return result


def _vector(info: dict[str, Any], key: str, n_hubs: int, fallback: str | None = None) -> np.ndarray:
    value = info.get(key, info.get(fallback) if fallback else None)
    if value is None:
        return np.full(n_hubs, np.nan)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(n_hubs, float(array[0]))
    if array.size != n_hubs:
        return np.full(n_hubs, np.nan)
    return array


def _scalar(info: dict[str, Any], key: str, fallback: str | None = None) -> float:
    value = info.get(key, info.get(fallback, np.nan) if fallback else np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def period_hub_rows(
    method: str,
    episode: int,
    scenario_seed: int | None,
    infos: list[dict[str, Any]],
    *,
    training_seed: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for info in infos:
        prices = np.asarray(info["prices"], dtype=float)
        n_hubs = len(prices)
        hub_ids = tuple(info.get("hub_ids", tuple(str(index) for index in range(n_hubs))))
        queue_v2 = "pending_vehicles" in info
        arrays = {
            "arrivals": _vector(info, "realized_arrivals", n_hubs, "realized_demand"),
            "accepted": _vector(info, "accepted_arrivals", n_hubs, "realized_demand"),
            "historical": _vector(info, "historical_equivalent_vehicles", n_hubs),
            "pending": _vector(info, "pending_vehicles", n_hubs),
            "admission_ratio": _vector(info, "admission_ratio", n_hubs),
            "admission_pressure": _vector(info, "admission_pressure", n_hubs),
            "admitted": _vector(info, "admitted_vehicles", n_hubs),
            "wait": _vector(info, "wait", n_hubs),
            "wait_excess": _vector(info, "wait_excess", n_hubs, "wait_violation"),
            "overflow": _vector(info, "overflow", n_hubs),
            "requested_energy": _vector(info, "requested_energy", n_hubs),
            "accepted_energy": _vector(info, "accepted_energy", n_hubs, "requested_energy"),
            "queued_energy_start": _vector(info, "queued_energy_start_kwh", n_hubs),
            "pending_energy": _vector(info, "pending_energy_kwh", n_hubs),
            "admitted_energy": _vector(info, "admitted_energy_kwh", n_hubs),
            "queued_energy_next": _vector(info, "queued_energy_next_kwh", n_hubs),
            "queue_vehicle_error": _vector(info, "queue_vehicle_conservation_error", n_hubs),
            "queue_energy_error": _vector(info, "queue_energy_conservation_error_kwh", n_hubs),
            "served_energy": _vector(info, "served_energy", n_hubs),
            "unmet": _vector(info, "unmet_energy", n_hubs, "unmet"),
            "pv_used": _vector(info, "pv_used", n_hubs),
            "pv_to_ev": _vector(info, "pv_to_ev", n_hubs),
            "battery_charge": _vector(info, "battery_charge", n_hubs),
            "battery_discharge": _vector(info, "battery_discharge", n_hubs, "battery_to_ev"),
            "soc": _vector(info, "soc", n_hubs),
            "grid_import": _vector(info, "grid_import", n_hubs),
            "grid_limit": _vector(info, "grid_limit", n_hubs),
            "grid_utilization": _vector(info, "grid_utilization", n_hubs),
            "grid_price": _vector(info, "grid_prices", n_hubs, "grid_price"),
            "revenue": _vector(info, "revenue", n_hubs),
            "energy_cost": _vector(info, "energy_cost", n_hubs),
            "operating_cost": _vector(info, "operating_cost", n_hubs),
            "profit": _vector(info, "profit", n_hubs),
            "access": _vector(info, "access_time_weighted_by_hub_hours", n_hubs),
            "detour": _vector(info, "detour_time_weighted_by_hub_hours", n_hubs),
            "full_service": _vector(info, "full_service_request_ratio", n_hubs),
            "admitted_full_service": _vector(info, "admitted_full_service_ratio", n_hubs),
            "pending_full_service": _vector(info, "pending_full_service_ratio", n_hubs),
            "energy_balance": _vector(info, "energy_balance_error_by_hub", n_hubs),
        }
        if np.isnan(arrays["pv_used"]).all():
            arrays["pv_used"] = _vector(info, "pv_to_ev", n_hubs) + _vector(
                info, "pv_to_battery", n_hubs
            )
        seed_value = int(scenario_seed) if scenario_seed is not None else np.nan
        for hub in range(n_hubs):
            row = {
                "seed": seed_value,
                "training_seed": training_seed,
                "scenario_seed": seed_value,
                "episode": int(episode),
                "timestamp": info.get("timestamp", np.nan),
                "method": method,
                "hub_id": str(hub_ids[hub]),
                "hub_index": hub,
                "hub": hub,
                "period": int(info["period"]),
                "price": float(prices[hub]),
                "arrivals": float(arrays["arrivals"][hub]),
                "accepted": float(arrays["accepted"][hub]),
                "outside_share": _scalar(info, "outside_share"),
                "wait_min": float(arrays["wait"][hub] * 60.0),
                "wait_excess_min": float(arrays["wait_excess"][hub] * 60.0),
                "overflow": float(arrays["overflow"][hub]),
                "requested_energy_kwh": float(arrays["requested_energy"][hub]),
                "accepted_energy_kwh": float(arrays["accepted_energy"][hub]),
                "served_energy_kwh": float(arrays["served_energy"][hub]),
                "unmet_energy_kwh": float(arrays["unmet"][hub]),
                "pv_used_kwh": float(arrays["pv_used"][hub]),
                "pv_to_ev_kwh": float(arrays["pv_to_ev"][hub]),
                "battery_charge_kwh": float(arrays["battery_charge"][hub]),
                "battery_discharge_kwh": float(arrays["battery_discharge"][hub]),
                "soc": float(arrays["soc"][hub]),
                "grid_import_kwh": float(arrays["grid_import"][hub]),
                "grid_limit_kwh": float(arrays["grid_limit"][hub]),
                "grid_utilization": float(arrays["grid_utilization"][hub]),
                "grid_price_gbp_per_kwh": float(arrays["grid_price"][hub]),
                "revenue_gbp": float(arrays["revenue"][hub]),
                "energy_cost_gbp": float(arrays["energy_cost"][hub]),
                "operating_cost_gbp": float(arrays["operating_cost"][hub]),
                "profit_gbp": float(arrays["profit"][hub]),
                "access_mean_min": float(arrays["access"][hub] * 60.0),
                "route_mean_min": float(arrays["access"][hub] * 60.0),
                "detour_mean_min": float(arrays["detour"][hub] * 60.0),
                "outside_cost_estimate": _scalar(info, "outside_cost_estimate"),
                "outside_nll": _scalar(info, "inference_nll", "inference_loss"),
                "reward": _scalar(info, "reward"),
                "welfare": _scalar(info, "weighted_hub_profit_welfare", "welfare"),
                "full_service_ratio": float(arrays["full_service"][hub]),
                "energy_balance_error_kwh": float(arrays["energy_balance"][hub]),
                "outside_cost_error_for_evaluation_only": _scalar(
                    info, "outside_cost_error_for_evaluation_only"
                ),
            }
            row.update({
                "requested_energy": row["requested_energy_kwh"],
                "accepted_energy": row["accepted_energy_kwh"],
                "served_energy": row["served_energy_kwh"],
                "unmet_energy": row["unmet_energy_kwh"],
                "pv_used": row["pv_used_kwh"],
                "pv_to_ev": row["pv_to_ev_kwh"],
                "battery_charge": row["battery_charge_kwh"],
                "battery_discharge": row["battery_discharge_kwh"],
                "grid_import": row["grid_import_kwh"],
                "grid_limit": row["grid_limit_kwh"],
                "grid_limit_utilization": row["grid_utilization"],
                "grid_price": row["grid_price_gbp_per_kwh"],
                "revenue": row["revenue_gbp"],
                "energy_cost": row["energy_cost_gbp"],
                "operating_cost": row["operating_cost_gbp"],
                "profit": row["profit_gbp"],
                "full_service": row["full_service_ratio"],
                "energy_balance": row["energy_balance_error_kwh"],
                "wait_violation": row["wait_excess_min"] / 60.0,
                "unmet": row["unmet_energy_kwh"],
                "realized_demand": row["arrivals"],
                "weighted_hub_profit_welfare": row["welfare"],
            })
            if queue_v2:
                for legacy in (
                    "accepted", "overflow", "accepted_energy_kwh", "accepted_energy",
                    "full_service_ratio", "full_service",
                ):
                    row.pop(legacy, None)
                row.update({
                    "historical_equivalent_vehicles": float(arrays["historical"][hub]),
                    "pending_vehicles": float(arrays["pending"][hub]),
                    "admission_ratio": float(arrays["admission_ratio"][hub]),
                    "admission_pressure": float(arrays["admission_pressure"][hub]),
                    "admitted_vehicles": float(arrays["admitted"][hub]),
                    "queued_energy_start_kwh": float(arrays["queued_energy_start"][hub]),
                    "pending_energy_kwh": float(arrays["pending_energy"][hub]),
                    "admitted_energy_kwh": float(arrays["admitted_energy"][hub]),
                    "queued_energy_next_kwh": float(arrays["queued_energy_next"][hub]),
                    "queue_vehicle_conservation_error": float(arrays["queue_vehicle_error"][hub]),
                    "queue_energy_conservation_error_kwh": float(arrays["queue_energy_error"][hub]),
                    "admitted_full_service_ratio": float(arrays["admitted_full_service"][hub]),
                    "pending_full_service_ratio": float(arrays["pending_full_service"][hub]),
                })
            rows.append(row)
    return rows


def _episode_row(
    method: str,
    episode: int,
    scenario_seed: int,
    infos: list[dict[str, Any]],
    *,
    training_seed: int | None = None,
) -> dict[str, Any]:
    queue_v2 = all("pending_vehicles" in info for info in infos)
    arrivals = np.concatenate([
        _vector(info, "realized_arrivals", len(info["prices"]), "realized_demand")
        for info in infos
    ])
    waits = np.concatenate([
        _vector(info, "wait", len(info["prices"])) for info in infos
    ]) * 60.0
    excess = np.concatenate([
        _vector(info, "wait_excess", len(info["prices"]), "wait_violation")
        for info in infos
    ]) * 60.0
    wait_weights = (
        np.concatenate([
            _vector(info, "pending_vehicles", len(info["prices"])) for info in infos
        ])
        if queue_v2 else arrivals
    )
    outside_requests = float(sum(_scalar(info, "outside_count") for info in infos))
    hub_requests = float(np.nansum(arrivals))
    total_requests = hub_requests + outside_requests
    pv_used = float(sum(
        np.nansum(_vector(info, "pv_used", len(info["prices"])))
        if "pv_used" in info else np.nansum(
            _vector(info, "pv_to_ev", len(info["prices"]))
            + _vector(info, "pv_to_battery", len(info["prices"]))
        )
        for info in infos
    ))
    pv_available = float(sum(
        np.nansum(_vector(info, "pv_available", len(info["prices"]))) for info in infos
    ))
    true_errors = np.asarray([
        _scalar(info, "outside_cost_error_for_evaluation_only") for info in infos
    ])
    grid_by_period = [
        float(np.nansum(_vector(info, "grid_import", len(info["prices"]))))
        for info in infos
    ]
    served_requests = float(sum(
        np.nansum(_vector(info, "served_count", len(info["prices"]))) for info in infos
    ))
    profit = float(sum(
        np.nansum(_vector(info, "profit", len(info["prices"]))) for info in infos
    ))
    welfare = float(sum(
        _scalar(info, "weighted_hub_profit_welfare", "welfare") for info in infos
    ))
    unmet = float(sum(
        np.nansum(_vector(info, "unmet_energy", len(info["prices"]), "unmet"))
        for info in infos
    ))
    row = {
        "seed": int(scenario_seed),
        "training_seed": training_seed,
        "method": method,
        "episode": int(episode),
        "scenario_seed": int(scenario_seed),
        "return": float(sum(_scalar(info, "reward") for info in infos)),
        "profit_gbp": profit,
        "profit": profit,
        "welfare_gbp": welfare,
        "welfare": welfare,
        "weighted_hub_profit_welfare": welfare,
        "served_requests": served_requests,
        "outside_requests": outside_requests,
        "outside_share": outside_requests / total_requests if total_requests > 0.0 else np.nan,
        "mean_wait_min": float(np.average(waits, weights=wait_weights)) if np.nansum(wait_weights) > 0 else float(np.nanmean(waits)),
        "p95_wait_min": float(np.nanpercentile(waits, 95)),
        "max_wait_min": float(np.nanmax(waits)),
        "wait_violation_rate": float(np.mean(excess > 1e-12)),
        "mean_wait_excess_min": float(np.average(excess, weights=wait_weights)) if np.nansum(wait_weights) > 0 else float(np.nanmean(excess)),
        "wait_violation": float(np.nansum(excess) / 60.0),
        "unmet_energy_kwh": unmet,
        "unmet": unmet,
        "grid_energy_kwh": float(np.nansum(grid_by_period)),
        "peak_grid_import_kwh": float(np.nanmax(grid_by_period)),
        "pv_utilization": pv_used / pv_available if pv_available > 0.0 else np.nan,
        "battery_throughput_kwh": float(sum(np.nansum(_vector(info, "battery_throughput", len(info["prices"]))) for info in infos)),
        "energy_cost_gbp": float(sum(np.nansum(_vector(info, "energy_cost", len(info["prices"]))) for info in infos)),
        "mean_access_min": float(np.nanmean([_scalar(info, "access_time_weighted_hours") * 60.0 for info in infos])),
        "p95_access_min": float(np.nanpercentile([_scalar(info, "access_time_weighted_hours") * 60.0 for info in infos], 95)),
        "mean_detour_min": float(np.nanmean([_scalar(info, "detour_time_weighted_hours") * 60.0 for info in infos])),
        "p95_detour_min": float(np.nanpercentile([_scalar(info, "detour_time_weighted_hours") * 60.0 for info in infos], 95)),
        "outside_cost_estimate": _scalar(infos[-1], "outside_cost_estimate"),
        "outside_mae": float(np.nanmean(np.abs(true_errors))) if np.isfinite(true_errors).any() else np.nan,
        "outside_nll": float(np.nanmean([_scalar(info, "inference_nll", "inference_loss") for info in infos])),
        "approx_unilateral_gain": np.nan,
        "centralized_reference_difference": np.nan,
        "centralized_reference_gap": np.nan,
        "exact_oracle_gap": np.nan,
    }
    if not queue_v2:
        row["full_service_ratio"] = (
            served_requests / hub_requests if hub_requests > 0.0 else 1.0
        )
        return row

    pending = np.concatenate([
        _vector(info, "pending_vehicles", len(info["prices"])) for info in infos
    ])
    admitted = np.concatenate([
        _vector(info, "admitted_vehicles", len(info["prices"])) for info in infos
    ])
    ratios = np.concatenate([
        _vector(info, "admission_ratio", len(info["prices"])) for info in infos
    ])
    pressures = np.concatenate([
        _vector(info, "admission_pressure", len(info["prices"])) for info in infos
    ])
    queued_by_period = np.asarray([
        np.nansum(_vector(info, "queued_energy_next_kwh", len(info["prices"])))
        for info in infos
    ])
    pending_energy = float(sum(
        np.nansum(_vector(info, "pending_energy_kwh", len(info["prices"])))
        for info in infos
    ))
    admitted_energy = float(sum(
        np.nansum(_vector(info, "admitted_energy_kwh", len(info["prices"])))
        for info in infos
    ))
    active_ratios = ratios[pending > 1e-12]
    clearance = next((
        index for index in range(len(queued_by_period))
        if np.all(queued_by_period[index:] <= 1e-8)
    ), None)
    row.update({
        "pending_requests": float(np.nansum(pending)),
        "admitted_requests": float(np.nansum(admitted)),
        "admission_ratio": float(np.nansum(admitted) / np.nansum(pending)) if np.nansum(pending) > 0.0 else 1.0,
        "minimum_admission_ratio": float(np.nanmin(active_ratios)) if len(active_ratios) else 1.0,
        "peak_admission_pressure": float(np.nanmax(pressures)),
        "admitted_full_service_ratio": float(served_requests / np.nansum(admitted)) if np.nansum(admitted) > 0.0 else 1.0,
        "pending_full_service_ratio": float(served_requests / np.nansum(pending)) if np.nansum(pending) > 0.0 else 1.0,
        "pending_energy_kwh": pending_energy,
        "admitted_energy_kwh": admitted_energy,
        "peak_queued_energy_kwh": float(np.nanmax(queued_by_period)),
        "mean_queued_energy_kwh": float(np.nanmean(queued_by_period)),
        "final_queued_energy_kwh": float(queued_by_period[-1]),
        "queue_cleared_by_end": bool(queued_by_period[-1] <= 1e-8),
        "queue_clearance_period": float(clearance) if clearance is not None else np.nan,
        "max_queue_vehicle_conservation_error": float(max(
            np.nanmax(np.abs(_vector(info, "queue_vehicle_conservation_error", len(info["prices"]))))
            for info in infos
        )),
        "max_queue_energy_conservation_error_kwh": float(max(
            np.nanmax(np.abs(_vector(info, "queue_energy_conservation_error_kwh", len(info["prices"]))))
            for info in infos
        )),
    })
    return row


_METRIC_UNITS = {
    "return": "scaled_reward", "profit_gbp": "GBP", "welfare_gbp": "GBP_weighted_hub_profit",
    "weighted_hub_profit_welfare": "GBP_weighted_hub_profit", "profit": "GBP",
    "served_requests": "requests", "outside_requests": "requests", "outside_share": "fraction",
    "mean_wait_min": "min", "p95_wait_min": "min", "max_wait_min": "min",
    "wait_violation_rate": "fraction_of_hub_periods", "mean_wait_excess_min": "min",
    "wait_violation": "hub_hours", "unmet_energy_kwh": "kWh", "unmet": "kWh",
    "full_service_ratio": "fraction", "pending_requests": "requests", "admitted_requests": "requests",
    "admission_ratio": "fraction", "minimum_admission_ratio": "fraction",
    "peak_admission_pressure": "pending_over_capacity",
    "admitted_full_service_ratio": "fraction", "pending_full_service_ratio": "fraction",
    "pending_energy_kwh": "kWh", "admitted_energy_kwh": "kWh",
    "peak_queued_energy_kwh": "kWh", "mean_queued_energy_kwh": "kWh",
    "final_queued_energy_kwh": "kWh", "queue_cleared_by_end": "boolean",
    "queue_clearance_period": "period_index", "max_queue_vehicle_conservation_error": "hours",
    "max_queue_energy_conservation_error_kwh": "kWh",
    "grid_energy_kwh": "kWh", "peak_grid_import_kwh": "kWh_per_period",
    "pv_utilization": "fraction", "battery_throughput_kwh": "kWh", "energy_cost_gbp": "GBP",
    "mean_access_min": "min", "p95_access_min": "min", "mean_detour_min": "min", "p95_detour_min": "min",
    "outside_cost_estimate": "GBP", "outside_mae": "GBP", "outside_nll": "nll",
    "approx_unilateral_gain": "scaled_local_reward", "centralized_reference_difference": "scaled_reward",
    "centralized_reference_gap": "scaled_reward_deprecated_non_oracle_alias", "exact_oracle_gap": "scaled_reward",
}


def aggregate_episode_rows(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if episodes.empty:
        return pd.DataFrame(columns=["method", "metric", "mean", "std", "n", "unit", "availability"])
    metrics = [name for name in _METRIC_UNITS if name in episodes.columns]
    group_columns = [column for column in ("method", "axis", "level") if column in episodes.columns]
    for keys, group in episodes.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append({
                **identity,
                "metric": metric,
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
                "unit": _METRIC_UNITS[metric],
                "availability": "available" if len(values) else "missing",
            })
    return pd.DataFrame(rows)


def evaluate_policy(
    config: dict[str, Any],
    policy: Policy,
    scenario_seeds: list[int],
    *,
    stochastic: bool = False,
    initial_outside_estimate: float | None = None,
    online_lower_layer: bool = False,
) -> EvaluationResult:
    """Evaluate independent episodes with common seeds and frozen inference by default."""
    env = GridAwarePricingEnv(config)
    training_seed = getattr(policy, "training_seed", None)
    if isinstance(policy, (FixedTariffPolicy, MyopicLocalPolicy)):
        initial_outside_estimate = float(getattr(policy, "outside_estimate", 13.5))
    elif isinstance(policy, TrainedPolicy) and initial_outside_estimate is None:
        initial_outside_estimate = policy.initial_outside_estimate
    if initial_outside_estimate is None:
        initial_outside_estimate = float(config["inference"]["initial_outside_cost"])
    online_enabled = bool(
        online_lower_layer
        and isinstance(policy, TrainedPolicy)
        and policy.supports_online_lower_layer
    )
    period_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for episode, seed in enumerate(scenario_seeds):
        observations, global_state = env.reset(
            seed=int(seed), outside_estimate=initial_outside_estimate, reset_inference=True
        )
        infos: list[dict[str, Any]] = []
        done = False
        while not done:
            prices = policy.prices(observations, global_state, env)
            result = env.step(
                prices,
                deterministic_demand=not stochastic,
                update_inference=online_enabled,
            )
            infos.append(result.info)
            observations, global_state, done = result.observations, result.global_state, result.done
        period_rows.extend(period_hub_rows(policy.name, episode, int(seed), infos, training_seed=training_seed))
        episode_rows.append(_episode_row(policy.name, episode, int(seed), infos, training_seed=training_seed))
    episodes_frame = pd.DataFrame(episode_rows)
    return EvaluationResult(
        period_hub=pd.DataFrame(period_rows), episodes=episodes_frame,
        aggregate=aggregate_episode_rows(episodes_frame),
        metadata={
            "method": policy.name,
            "evaluation_mode": "stochastic_demand_beta_mean_actions" if stochastic else "deterministic_demand_beta_mean_or_fixed_actions",
            "scenario_seeds": [int(seed) for seed in scenario_seeds],
            "lower_layer_evaluation": "online" if online_enabled else "frozen",
            "online_lower_layer_inference": online_enabled,
            "estimator_reset_each_episode": True,
            "initial_outside_estimate": float(initial_outside_estimate),
            "training_seed": training_seed,
            "hidden_preference_fields_are_evaluation_only": True,
            "welfare_label": "weighted_hub_profit_welfare",
            "exact_oracle_gap": None,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "environment_schema_version": config.get("environment_schema_version"),
            "queue_semantics": config.get("queue_semantics"),
            "zip_sha256": config.get("data", {}).get("zip_sha256"),
        },
    )


def _known_evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result.setdefault("experiment", {})["outside_mode"] = "known"
    return result


def _evaluate_trajectory(
    env: GridAwarePricingEnv,
    trajectory: np.ndarray,
    scenario_seed: int,
    *,
    objective_hub: int | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    observations, global_state = env.reset(seed=scenario_seed, reset_inference=True)
    del observations, global_state
    infos: list[dict[str, Any]] = []
    for prices in trajectory:
        result = env.step(
            prices,
            deterministic_demand=True,
            update_inference=False,
        )
        infos.append(result.info)
    if objective_hub is None:
        value = float(sum(float(info["reward"]) for info in infos))
    else:
        value = float(sum(local_immediate_reward(env.config, info)[objective_hub] for info in infos))
    return value, infos


def centralized_coordinate_search_reference(
    config: dict[str, Any],
    scenario_seed: int,
    budget: int,
    *,
    initial_policy_trajectory: np.ndarray | None = None,
    grid_points: int | None = None,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Budgeted deterministic coordinate search over the full period-by-hub trajectory."""
    if budget <= 0:
        raise ValueError("reference budget must be positive")
    env = GridAwarePricingEnv(_known_evaluation_config(config))
    points = int(grid_points or config.get("oracle", {}).get("points_per_hub", 7))
    fixed = np.clip(
        np.full((env.periods, env.n_hubs), 0.45), env.price_min, env.price_max
    )
    starts = [fixed]
    if initial_policy_trajectory is not None:
        candidate = np.asarray(initial_policy_trajectory, dtype=float)
        if candidate.shape != fixed.shape:
            raise ValueError("initial policy trajectory has the wrong shape")
        starts.append(np.clip(candidate, env.price_min, env.price_max))
    evaluations = 0
    best_value = -np.inf
    best_trajectory = fixed.copy()
    best_infos: list[dict[str, Any]] = []
    for start in starts:
        if evaluations >= budget:
            break
        value, infos = _evaluate_trajectory(env, start, int(scenario_seed))
        evaluations += 1
        if value > best_value + tolerance:
            best_value, best_trajectory, best_infos = value, start.copy(), infos
    improved = True
    while improved and evaluations < budget:
        improved = False
        for period in range(env.periods):
            for hub in range(env.n_hubs):
                for price in np.linspace(env.price_min[hub], env.price_max[hub], max(points, 2)):
                    if evaluations >= budget:
                        break
                    if abs(price - best_trajectory[period, hub]) <= 1e-15:
                        continue
                    candidate = best_trajectory.copy()
                    candidate[period, hub] = price
                    value, infos = _evaluate_trajectory(env, candidate, int(scenario_seed))
                    evaluations += 1
                    if value > best_value + tolerance:
                        best_value = value
                        best_trajectory = candidate
                        best_infos = infos
                        improved = True
                if evaluations >= budget:
                    break
            if evaluations >= budget:
                break
    return {
        "trajectory": best_trajectory,
        "objective": float(best_value),
        "infos": best_infos,
        "solver_report": {
            "name": "centralized_coordinate_search_reference",
            "is_exact": False,
            "is_upper_bound": False,
            "uses_true_preference": True,
            "hidden_preference_access": "evaluation_only_environment_known_mode",
            "budget": int(budget),
            "evaluations": int(evaluations),
            "tolerance": float(tolerance),
        },
    }


def approximate_unilateral_gain(
    config: dict[str, Any],
    trajectory: np.ndarray,
    scenario_seed: int,
    budget: int,
    *,
    grid_points: int | None = None,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Find budget-limited profitable unilateral trajectory deviations."""
    if budget <= 0:
        raise ValueError("unilateral budget must be positive")
    env = GridAwarePricingEnv(_known_evaluation_config(config))
    base = np.asarray(trajectory, dtype=float)
    if base.shape != (env.periods, env.n_hubs):
        raise ValueError("trajectory must have shape [periods, hubs]")
    points = int(grid_points or config.get("oracle", {}).get("points_per_hub", 7))
    evaluations = 0
    gains: list[float] = []
    best_trajectories: list[np.ndarray] = []
    for hub in range(env.n_hubs):
        if evaluations >= budget:
            gains.append(0.0)
            best_trajectories.append(base.copy())
            continue
        base_value, _ = _evaluate_trajectory(
            env, base, int(scenario_seed), objective_hub=hub
        )
        evaluations += 1
        best_value = base_value
        best = base.copy()
        improved = True
        while improved and evaluations < budget:
            improved = False
            for period in range(env.periods):
                for price in np.linspace(env.price_min[hub], env.price_max[hub], max(points, 2)):
                    if evaluations >= budget:
                        break
                    if abs(price - best[period, hub]) <= 1e-15:
                        continue
                    candidate = best.copy()
                    candidate[period, hub] = price
                    value, _ = _evaluate_trajectory(
                        env, candidate, int(scenario_seed), objective_hub=hub
                    )
                    evaluations += 1
                    if value > best_value + tolerance:
                        best_value, best = value, candidate
                        improved = True
                if evaluations >= budget:
                    break
        gains.append(max(float(best_value - base_value), 0.0))
        best_trajectories.append(best)
    return {
        "gain_by_hub": gains,
        "found_maximum_gain": float(max(gains, default=0.0)),
        "best_trajectories": best_trajectories,
        "lower_bound_on_exact_gain": True,
        "budget": int(budget),
        "evaluations": int(evaluations),
        "tolerance": float(tolerance),
        "uses_true_preference": True,
        "hidden_preference_access": "evaluation_only_environment_known_mode",
    }


def evaluation_from_infos(
    method: str,
    episodes_with_seeds: list[tuple[int, list[dict[str, Any]]]],
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Build the standard tables from already evaluated episode trajectories."""
    period_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for episode, (seed, infos) in enumerate(episodes_with_seeds):
        period_rows.extend(period_hub_rows(method, episode, int(seed), infos))
        episode_rows.append(_episode_row(method, episode, int(seed), infos))
    episode_frame = pd.DataFrame(episode_rows)
    details = {"method": method}
    if metadata:
        details.update(metadata)
    return EvaluationResult(
        period_hub=pd.DataFrame(period_rows),
        episodes=episode_frame,
        aggregate=aggregate_episode_rows(episode_frame),
        metadata=details,
    )


def combine_evaluations(results: list[EvaluationResult]) -> EvaluationResult:
    period = pd.concat([result.period_hub for result in results], ignore_index=True) if results else pd.DataFrame()
    episodes = pd.concat([result.episodes for result in results], ignore_index=True) if results else pd.DataFrame()
    return EvaluationResult(
        period_hub=period,
        episodes=episodes,
        aggregate=aggregate_episode_rows(episodes),
        metadata={"methods": [result.metadata.get("method") for result in results]},
    )
