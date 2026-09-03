"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_york_config(
    config_path: Path,
    user_config: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    from .york_data import load_york_scenario, york_config_from_scenario

    override_config = overrides or {}
    data_options = _merge(user_config.get("data", {}), override_config.get("data", {}))
    zip_value = data_options.get("zip_path", data_options.get("path"))
    if not zip_value:
        raise ValueError("data.zip_path is required for york_zip mode")
    zip_path = Path(zip_value).expanduser()
    if not zip_path.is_absolute():
        zip_path = config_path.parent / zip_path
    route_mode = str(data_options.get("route_mode", "network_bpr"))
    scenario = load_york_scenario(zip_path)
    config = york_config_from_scenario(scenario, route_mode)

    user_explicit = {key: value for key, value in user_config.items() if key != "data"}
    override_explicit = {key: value for key, value in override_config.items() if key != "data"}
    config = _merge(config, user_explicit)
    config = _merge(config, override_explicit)
    data_metadata = {
        key: value
        for key, value in data_options.items()
        if key not in {"mode", "zip_path", "path", "zip_sha256", "route_mode"}
    }
    config["data"] = _merge(config["data"], data_metadata)
    config["data"]["route_mode"] = route_mode
    config["traffic"]["route_mode"] = route_mode
    config["_york_scenario"] = scenario
    config["_york_zip_path"] = str(zip_path.resolve())
    return config


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    override_mode = (overrides or {}).get("data", {}).get("mode")
    mode = override_mode or config.get("data", {}).get("mode")
    if mode == "york_zip":
        config = _load_york_config(config_path, config, overrides)
    elif overrides:
        config = _merge(config, overrides)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def _validate_common_vectors(config: dict[str, Any], n_hubs: int, n_ods: int) -> None:
    hub_vectors = (
        "chargers", "service_time_hours", "max_wait_hours", "pv_peak_kwh",
        "battery_capacity_kwh", "initial_soc", "min_soc", "max_soc",
        "charge_limit_kw", "discharge_limit_kw", "grid_cap_kw", "eta_charge",
        "eta_discharge", "battery_cost_per_kwh", "pv_cost_per_kwh",
        "operating_cost_per_request", "welfare_weights",
    )
    for name in hub_vectors:
        if len(config["hubs"][name]) != n_hubs:
            raise ValueError(f"hubs.{name} must have length n_hubs")
    for name in ("min", "max"):
        if len(config["price"][name]) != n_hubs:
            raise ValueError(f"price.{name} must have length n_hubs")
    if len(config["demand"]["energy_kwh"]) != n_ods:
        raise ValueError("demand.energy_kwh must have length n_ods")
    if len(config["demand"]["base_od_counts"]) != n_ods:
        raise ValueError("demand.base_od_counts must have length n_ods")
    if any(lo >= hi for lo, hi in zip(config["price"]["min"], config["price"]["max"])):
        raise ValueError("each minimum price must be below its maximum")


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(key in forbidden or _contains_key(item, forbidden) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def validate_config(config: dict[str, Any]) -> None:
    n_hubs = int(config["system"]["n_hubs"])
    n_ods = int(config["system"]["n_ods"])
    _validate_common_vectors(config, n_hubs, n_ods)
    if int(config["system"]["periods"]) <= 0 or float(config["system"]["dt_hours"]) <= 0:
        raise ValueError("system periods and dt_hours must be positive")
    if int(config["oracle"]["points_per_hub"]) < 2:
        raise ValueError("oracle.points_per_hub must be at least 2")

    if config.get("data", {}).get("mode") == "york_zip":
        from .york_data import YorkScenario, structural_validation

        scenario = config.get("_york_scenario")
        if not isinstance(scenario, YorkScenario):
            raise ValueError("York config is missing its runtime scenario")
        structural_validation(scenario)
        if n_hubs != len(scenario.hub_ids) or n_ods != len(scenario.od_ids):
            raise ValueError("York config dimensions must match the packaged scenario")
        if int(config["system"]["periods"]) != len(scenario.timestamps):
            raise ValueError("York periods must match packaged timestamps")
        route_mode = config["data"].get("route_mode")
        if route_mode not in {"packaged_freeflow", "network_bpr"}:
            raise ValueError("York route mode must be packaged_freeflow or network_bpr")
        if config["traffic"].get("route_mode") != route_mode:
            raise ValueError("data.route_mode and traffic.route_mode must match")
        if config.get("environment_schema_version") != 2:
            raise ValueError("York environment_schema_version must be 2")
        if config.get("queue_semantics") != "dynamic_fluid_carryover_v1":
            raise ValueError("York queue_semantics must be dynamic_fluid_carryover_v1")
        queue = config.get("queue", {})
        if queue.get("carry_over_between_periods") is not True:
            raise ValueError("York queue carryover must be enabled")
        if queue.get("finite_waiting_space_enforced") is not False:
            raise ValueError("York base case cannot enforce finite waiting space")
        visible = resolved_config(config)
        if _contains_key(visible, {"true_hidden_cost_gbp", "true_outside_cost"}):
            raise ValueError("hidden York outside-option cost cannot be actor-visible")
        return

    routes = config["traffic"]["routes"]
    if len(routes) != n_ods or any(len(row) != n_hubs for row in routes):
        raise ValueError("traffic.routes must have shape [n_ods][n_hubs]")
    profiles = config["profiles"]
    if len(profiles["traffic_peak_hours"]) != len(profiles["traffic_peak_width_hours"]):
        raise ValueError("traffic peak hours and widths must have equal lengths")
    if len(profiles["grid_peak_shift_hours"]) != len(profiles["grid_wave_weights"]):
        raise ValueError("grid peak shifts and weights must have equal lengths")
    if profiles["solar_end_hour"] <= profiles["solar_start_hour"]:
        raise ValueError("solar_end_hour must be after solar_start_hour")


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config(config), handle, sort_keys=False)
