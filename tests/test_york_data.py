from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from grid_aware_pricing.config import load_config, resolved_config
from grid_aware_pricing.york_data import (
    deterministic_fixed_tariff_sanity,
    load_york_scenario,
    structural_validation,
)


ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "york_ev_case_study_runnable.zip"


@pytest.fixture(scope="module")
def scenario():
    return load_york_scenario(ZIP_PATH)


def _contains_key(value, forbidden):
    if isinstance(value, dict):
        return any(key in forbidden or _contains_key(item, forbidden) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _contains_runtime_object(value):
    if isinstance(value, (pd.DataFrame, np.ndarray)):
        return True
    if isinstance(value, dict):
        return any(_contains_runtime_object(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_runtime_object(item) for item in value)
    return False


def test_loads_package_in_memory_with_typed_fields(scenario):
    assert scenario.timestamps == tuple(pd.date_range("2024-09-20 16:00:00", periods=6, freq="h"))
    assert scenario.hub_ids == tuple(f"H{index:02d}" for index in range(1, 9))
    assert scenario.od_ids == tuple(f"OD{index:02d}" for index in range(1, 7))
    assert scenario.zip_sha256 and len(scenario.zip_sha256) == 64
    assert isinstance(scenario.network, nx.DiGraph) and scenario.network.is_directed()
    assert scenario.road_edges["speed_kph"].gt(0).all()
    assert scenario.road_edges["lanes"].ge(1).all()
    assert all(len(geometry) >= 2 for geometry in scenario.road_edges["geometry"])
    assert all(isinstance(value, str) for value in scenario.road_nodes["node_id"].head())
    assert all(isinstance(value, dict) for value in scenario.od_demand["route_freeflow_time_min"])


def test_all_twelve_packaged_structural_checks_pass(scenario):
    checks = structural_validation(scenario)
    assert len(checks) == 12
    assert all(checks.values())


def test_deterministic_fixed_tariff_reference_values(scenario):
    assert deterministic_fixed_tariff_sanity(scenario) == {
        "total_demand": 382.830,
        "peak_hourly_demand": 103.800,
        "max_wait_minutes": 75.996,
        "peak_queued_energy_kwh": 649.458,
        "min_supply_margin_kwh": 152.400,
        "queue_cleared_from": "2024-09-20 20:00:00",
    }


def test_route_and_energy_matrices_are_finite_and_bpr_is_not_faster(scenario):
    route_shape = (6, 6, 8)
    assert scenario.od_expected_demand.shape == (6, 6)
    assert scenario.candidate_mask.shape == route_shape
    assert scenario.candidate_mask.all()
    for matrix in (
        scenario.packaged_route_times_hours,
        scenario.packaged_detours_hours,
        scenario.network_free_flow_route_times_hours,
        scenario.network_free_flow_detours_hours,
        scenario.network_route_times_hours,
        scenario.network_detours_hours,
    ):
        assert matrix.shape == route_shape
        assert np.isfinite(matrix).all()
        assert np.all(matrix >= 0.0)
    assert scenario.network_edge_bpr_hours.shape == (6, len(scenario.road_edges))
    assert np.all(
        scenario.network_edge_bpr_hours
        >= scenario.network_edge_free_flow_hours[None, :] - 1e-12
    )
    assert np.all(
        scenario.network_route_times_hours
        >= scenario.network_free_flow_route_times_hours - 1e-12
    )
    assert np.all(
        scenario.network_edge_flows / scenario.network_edge_capacities[None, :] <= 1.2 + 1e-12
    )


def test_york_config_resolution_and_hidden_cost_privacy():
    formal = load_config(ROOT / "configs" / "york.yaml")
    smoke = load_config(ROOT / "configs" / "york_smoke.yaml")
    assert formal["data"]["route_mode"] == "network_bpr"
    assert formal["training"]["episodes_per_seed"] == 3000
    assert formal["training"]["seeds"] == [11, 29, 47]
    assert smoke["data"]["route_mode"] == "packaged_freeflow"
    assert smoke["training"]["updates"] == 3
    assert formal["_york_scenario"].package_config["user_choice"]["outside_option"][
        "true_hidden_cost_gbp"
    ] == 16.5

    visible = resolved_config(formal)
    assert visible["environment_schema_version"] == 2
    assert visible["queue_semantics"] == "dynamic_fluid_carryover_v1"
    assert visible["queue"] == {
        "discipline": "dynamic_fluid_carryover",
        "semantics": "dynamic_fluid_carryover_v1",
        "carry_over_between_periods": True,
        "finite_waiting_space_enforced": False,
        "queue_capacity_vehicles_usage": "optional_extension_not_used_in_base_case",
        "initial_residual_wait_hours": 0.0,
        "initial_queued_energy_kwh": 0.0,
    }
    assert visible["data"]["zip_sha256"] == formal["_york_scenario"].zip_sha256
    assert visible["data"]["assumptions"]
    assert not _contains_key(visible, {"true_hidden_cost_gbp", "true_outside_cost"})
    assert not _contains_runtime_object(visible)


def test_york_user_overrides_are_merged_after_package_defaults():
    config = load_config(
        ROOT / "configs" / "york_smoke.yaml",
        overrides={
            "training": {"episodes_per_seed": 7},
            "reward": {"wait_penalty": 99.0},
            "data": {"route_mode": "network_bpr"},
        },
    )
    assert config["training"]["episodes_per_seed"] == 7
    assert config["reward"]["wait_penalty"] == 99.0
    assert config["data"]["route_mode"] == "network_bpr"
    assert config["traffic"]["route_mode"] == "network_bpr"
