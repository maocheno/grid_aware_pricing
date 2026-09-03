"""In-memory loading and routing for the packaged York scenario."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import re
from typing import Any
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from .system_model import (
    bpr_travel_times,
    multinomial_logit,
    service_capacity,
    transition_fluid_queue,
)


_RUNTIME_TABLES = (
    "road_nodes.csv",
    "road_edges.csv",
    "traffic_flow.csv",
    "charging_hubs.csv",
    "od_demand.csv",
    "energy_profiles.csv",
)
_ROUTE_MODES = {"packaged_freeflow", "network_bpr"}
_YORK_ENVIRONMENT_SCHEMA_VERSION = 2
_YORK_QUEUE_SEMANTICS = "dynamic_fluid_carryover_v1"
_QUEUE_MODEL_CONTRACT = {
    "queue_discipline": "dynamic_fluid_carryover",
    "waiting_time_state_transition": "paper_equation_10",
    "historical_equivalent_vehicle_formula": "paper_equation_13",
    "service_admission_formula": "paper_equation_14",
    "queued_energy_state_transition": "paper_equation_19",
    "carry_over_between_periods": True,
    "finite_waiting_space_enforced": False,
    "queue_capacity_vehicles_usage": "optional_extension_not_used_in_base_case",
}
_STRUCTURAL_CHECK_NAMES = (
    "36 demand rows",
    "6 OD groups",
    "6 hourly demand periods",
    "48 energy rows",
    "8 hubs",
    "energy hub IDs match",
    "edge nodes exist",
    "hub nodes exist",
    "counter nodes exist",
    "OD nodes exist",
    "route JSON hub keys match",
    "YAML dimensions match",
)


@dataclass(frozen=True)
class YorkScenario:
    timestamps: tuple[pd.Timestamp, ...]
    hub_ids: tuple[str, ...]
    od_ids: tuple[str, ...]
    package_config: dict[str, Any]
    hub_parameters: dict[str, np.ndarray]
    energy_parameters: dict[str, np.ndarray]
    od_expected_demand: np.ndarray
    od_energy_parameters: dict[str, np.ndarray]
    candidate_mask: np.ndarray
    packaged_route_times_hours: np.ndarray
    packaged_detours_hours: np.ndarray
    network_route_times_hours: np.ndarray
    network_detours_hours: np.ndarray
    network_free_flow_route_times_hours: np.ndarray
    network_free_flow_detours_hours: np.ndarray
    network_edge_flows: np.ndarray
    network_edge_free_flow_hours: np.ndarray
    network_edge_bpr_hours: np.ndarray
    network_edge_capacities: np.ndarray
    counter_edge_mapping: dict[tuple[str, str, str], str]
    network: nx.DiGraph
    road_nodes: pd.DataFrame
    road_edges: pd.DataFrame
    traffic_flow: pd.DataFrame
    charging_hubs: pd.DataFrame
    od_demand: pd.DataFrame
    energy_profiles: pd.DataFrame
    data_limitations: tuple[str, ...]
    zip_sha256: str

    def route_profiles(self, route_mode: str) -> tuple[np.ndarray, np.ndarray]:
        if route_mode == "packaged_freeflow":
            return self.packaged_route_times_hours, self.packaged_detours_hours
        if route_mode == "network_bpr":
            return self.network_route_times_hours, self.network_detours_hours
        raise ValueError(f"unknown York route mode: {route_mode}")

    @property
    def map_dataframes(self) -> dict[str, pd.DataFrame]:
        return {
            "road_nodes": self.road_nodes,
            "road_edges": self.road_edges,
            "traffic_flow": self.traffic_flow,
            "charging_hubs": self.charging_hubs,
        }


@dataclass(frozen=True)
class _NetworkProfiles:
    graph: nx.DiGraph
    edge_flows: np.ndarray
    edge_free_flow_hours: np.ndarray
    edge_bpr_hours: np.ndarray
    edge_capacities: np.ndarray
    free_flow_route_times_hours: np.ndarray
    free_flow_detours_hours: np.ndarray
    route_times_hours: np.ndarray
    detours_hours: np.ndarray
    counter_mapping: dict[tuple[str, str, str], str]


def _zip_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_root(archive: ZipFile) -> str:
    matches = [name for name in archive.namelist() if name.endswith("/data/scenario_config.yaml")]
    if len(matches) != 1:
        raise ValueError("York ZIP must contain exactly one data/scenario_config.yaml")
    return matches[0].rsplit("/", 1)[0]


def _read_csv(archive: ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as handle:
        return pd.read_csv(handle, dtype=str, keep_default_na=False)


def _validate_queue_model(package_config: dict[str, Any]) -> None:
    queue = package_config.get("queue_model")
    if not isinstance(queue, dict):
        raise ValueError("York scenario_config.yaml must define queue_model")
    mismatches = [
        key for key, expected in _QUEUE_MODEL_CONTRACT.items()
        if queue.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "York queue_model does not match dynamic fluid carryover contract: "
            + ", ".join(mismatches)
        )
    for key in (
        "average_service_time_hours",
        "maximum_wait_hours",
        "initial_residual_wait_hours",
        "initial_queued_energy_kwh",
    ):
        if key not in queue:
            raise ValueError(f"York queue_model is missing {key}")
    if float(queue["average_service_time_hours"]) <= 0.0:
        raise ValueError("average_service_time_hours must be positive")
    if float(queue["maximum_wait_hours"]) < 0.0:
        raise ValueError("maximum_wait_hours must be nonnegative")
    if float(queue["initial_residual_wait_hours"]) < 0.0:
        raise ValueError("initial_residual_wait_hours must be nonnegative")
    if float(queue["initial_queued_energy_kwh"]) < 0.0:
        raise ValueError("initial_queued_energy_kwh must be nonnegative")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")


def _timestamp(frame: pd.DataFrame) -> None:
    frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"], errors="raise")


def _parse_speed_kph(value: str, default_kph: float) -> float:
    text = str(value).strip().lower()
    if not text:
        return float(default_kph)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return float(default_kph)
    speed = float(match.group(1))
    if "mph" in text:
        speed *= 1.609344
    if speed <= 0:
        raise ValueError(f"invalid maxspeed value: {value!r}")
    return speed


def _parse_lanes(value: str) -> int:
    match = re.search(r"\d+", str(value))
    return max(int(match.group(0)), 1) if match else 1


def _parse_linestring(value: str) -> tuple[tuple[float, float], ...]:
    match = re.fullmatch(r"\s*LINESTRING\s*\((.+)\)\s*", str(value), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"unsupported or invalid WKT geometry: {value!r}")
    coordinates = []
    for point in match.group(1).split(","):
        parts = point.strip().split()
        if len(parts) < 2:
            raise ValueError(f"invalid WKT point: {point!r}")
        coordinates.append((float(parts[0]), float(parts[1])))
    if len(coordinates) < 2:
        raise ValueError("LINESTRING must contain at least two points")
    return tuple(coordinates)


def _parse_json_times(value: str) -> dict[str, float]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("route-time JSON must be an object")
    return {str(key): float(item) for key, item in parsed.items()}


def _prepare_tables(
    package_config: dict[str, Any], tables: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    nodes = tables["road_nodes.csv"]
    edges = tables["road_edges.csv"]
    traffic = tables["traffic_flow.csv"]
    hubs = tables["charging_hubs.csv"]
    demand = tables["od_demand.csv"]
    energy = tables["energy_profiles.csv"]

    for frame, columns in (
        (nodes, ("node_id",)),
        (edges, ("edge_id", "u", "v", "osm_way_id")),
        (traffic, ("site_number", "channel_id", "nearest_road_node_id")),
        (hubs, ("hub_id", "nearest_road_node_id")),
        (demand, ("od_id", "origin_node_id", "destination_node_id")),
        (energy, ("hub_id",)),
    ):
        for column in columns:
            frame[column] = frame[column].astype(str)

    _numeric(nodes, ("longitude", "latitude"))
    _numeric(edges, ("length_m",))
    _numeric(
        traffic,
        (
            "easting_m", "northing_m", "latitude", "longitude",
            "nearest_road_node_distance_m", "motor_vehicle_count",
        ),
    )
    _numeric(
        hubs,
        (
            "easting_m", "northing_m", "latitude", "longitude", "num_chargers",
            "charger_power_kw", "queue_capacity_vehicles", "pv_capacity_kw",
            "battery_capacity_kwh", "battery_power_kw", "grid_import_limit_kw",
            "min_price_gbp_per_kwh", "max_price_gbp_per_kwh",
            "initial_price_gbp_per_kwh", "nearest_road_node_distance_m",
        ),
    )
    _numeric(
        demand,
        (
            "source_car_count", "traffic_to_request_ratio", "potential_charging_requests",
            "mean_requested_energy_kwh", "std_requested_energy_kwh",
            "min_requested_energy_kwh", "max_requested_energy_kwh",
            "direct_freeflow_time_min",
        ),
    )
    _numeric(
        energy,
        (
            "pv_availability_factor", "pv_capacity_kw", "pv_available_kwh",
            "grid_price_gbp_per_kwh", "grid_import_limit_kwh",
            "battery_continuation_value_gbp_per_kwh", "initial_battery_soc_fraction",
        ),
    )
    _timestamp(traffic)
    _timestamp(demand)
    _timestamp(energy)

    defaults = package_config["traffic_model"]["default_speed_kph"]
    edges["speed_kph"] = [
        _parse_speed_kph(value, float(defaults[highway]))
        for value, highway in zip(edges["maxspeed_tag"], edges["highway"])
    ]
    edges["lanes"] = edges["lanes_tag"].map(_parse_lanes).astype(int)
    edges["geometry"] = edges["geometry_wkt"].map(_parse_linestring)
    edges["free_flow_time_hours"] = edges["length_m"] / (edges["speed_kph"] * 1000.0)

    demand["candidate_hubs"] = demand["candidate_hub_ids"].map(
        lambda value: tuple(item for item in value.split(";") if item)
    )
    demand["route_freeflow_time_min"] = demand["route_freeflow_time_min_json"].map(
        _parse_json_times
    )
    demand["detour_freeflow_time_min"] = demand["detour_freeflow_time_min_json"].map(
        _parse_json_times
    )
    return tables


def _ordered_matrix(
    frame: pd.DataFrame,
    value: str,
    timestamps: tuple[pd.Timestamp, ...],
    item_ids: tuple[str, ...],
    item_column: str,
) -> np.ndarray:
    matrix = frame.pivot(index="timestamp_local", columns=item_column, values=value)
    matrix = matrix.reindex(index=list(timestamps), columns=list(item_ids))
    result = matrix.to_numpy(dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{value} does not form a complete finite time-by-item matrix")
    return result


def _packaged_route_matrices(
    demand: pd.DataFrame,
    timestamps: tuple[pd.Timestamp, ...],
    od_ids: tuple[str, ...],
    hub_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(timestamps), len(od_ids), len(hub_ids))
    routes = np.empty(shape, dtype=float)
    detours = np.empty(shape, dtype=float)
    candidates = np.zeros(shape, dtype=bool)
    time_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    od_index = {od_id: index for index, od_id in enumerate(od_ids)}
    hub_index = {hub_id: index for index, hub_id in enumerate(hub_ids)}
    for row in demand.itertuples(index=False):
        t = time_index[row.timestamp_local]
        od = od_index[row.od_id]
        candidate_set = set(row.candidate_hubs)
        for hub_id in hub_ids:
            hub = hub_index[hub_id]
            routes[t, od, hub] = row.route_freeflow_time_min[hub_id] / 60.0
            detours[t, od, hub] = row.detour_freeflow_time_min[hub_id] / 60.0
            candidates[t, od, hub] = hub_id in candidate_set
    if not np.isfinite(routes).all() or not np.isfinite(detours).all():
        raise ValueError("packaged route matrices contain non-finite values")
    return routes, detours, candidates


def _direction_vector(direction: str) -> np.ndarray | None:
    text = re.sub(r"\b(bus|vehicles?|cycles?)\b", "", direction.lower()).strip()
    text = re.sub(r"bound$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    aliases = {
        "north": (0.0, 1.0),
        "south": (0.0, -1.0),
        "east": (1.0, 0.0),
        "west": (-1.0, 0.0),
        "northeast": (1.0, 1.0),
        "north east": (1.0, 1.0),
        "southwest": (-1.0, -1.0),
        "south west": (-1.0, -1.0),
        "northwest": (-1.0, 1.0),
        "north west": (-1.0, 1.0),
        "southeast": (1.0, -1.0),
        "south east": (1.0, -1.0),
    }
    vector = aliases.get(text)
    if vector is None:
        return None
    result = np.asarray(vector, dtype=float)
    return result / np.linalg.norm(result)


def _counter_mapping(
    graph: nx.DiGraph, nodes: pd.DataFrame, edges: pd.DataFrame, traffic: pd.DataFrame
) -> dict[tuple[str, str, str], str]:
    coordinates = nodes.set_index("node_id")[["longitude", "latitude"]].to_dict("index")
    edge_lookup = edges.set_index("edge_id")
    mapping: dict[tuple[str, str, str], str] = {}
    unique = traffic[["site_number", "channel_id", "direction", "nearest_road_node_id"]].drop_duplicates()
    for row in unique.itertuples(index=False):
        key = (row.site_number, row.channel_id, row.direction)
        node = row.nearest_road_node_id
        adjacent: set[str] = set()
        if node in graph:
            adjacent.update(str(data["edge_id"]) for _, _, data in graph.out_edges(node, data=True))
            adjacent.update(str(data["edge_id"]) for _, _, data in graph.in_edges(node, data=True))
        if not adjacent:
            raise ValueError(f"counter node {node} has no adjacent directed edge")
        target = _direction_vector(row.direction)
        scored: list[tuple[float, str]] = []
        for edge_id in adjacent:
            edge = edge_lookup.loc[edge_id]
            start = coordinates[str(edge["u"])]
            end = coordinates[str(edge["v"])]
            vector = np.asarray(
                [end["longitude"] - start["longitude"], end["latitude"] - start["latitude"]],
                dtype=float,
            )
            norm = np.linalg.norm(vector)
            score = float(np.dot(vector / norm, target)) if target is not None and norm > 0 else 0.0
            scored.append((score, edge_id))
        mapping[key] = max(scored, key=lambda item: (item[0], item[1]))[1]
    return mapping


def _network_profiles(
    package_config: dict[str, Any],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    traffic: pd.DataFrame,
    hubs: pd.DataFrame,
    demand: pd.DataFrame,
    timestamps: tuple[pd.Timestamp, ...],
    hub_ids: tuple[str, ...],
    od_ids: tuple[str, ...],
) -> _NetworkProfiles:
    capacities_by_class = package_config["traffic_model"]["capacity_veh_per_hour_per_direction"]
    capacities = edges["highway"].map(capacities_by_class).to_numpy(dtype=float)
    if not np.isfinite(capacities).all() or np.any(capacities <= 0):
        raise ValueError("every road edge must map to a positive YAML capacity")

    graph = nx.DiGraph()
    for row in nodes.itertuples(index=False):
        graph.add_node(row.node_id, longitude=float(row.longitude), latitude=float(row.latitude))
    for index, row in enumerate(edges.itertuples(index=False)):
        graph.add_edge(
            row.u,
            row.v,
            edge_id=row.edge_id,
            edge_index=index,
            highway=row.highway,
            free_flow_time_hours=float(row.free_flow_time_hours),
            capacity=float(capacities[index]),
        )
    mapping = _counter_mapping(graph, nodes, edges, traffic)
    edge_index = {edge_id: index for index, edge_id in enumerate(edges["edge_id"])}
    highways = edges["highway"].to_numpy(dtype=str)
    alpha = float(package_config["traffic_model"]["bpr_alpha"])
    beta = float(package_config["traffic_model"]["bpr_beta"])
    free_flow = edges["free_flow_time_hours"].to_numpy(dtype=float)
    flows = np.empty((len(timestamps), len(edges)), dtype=float)
    bpr_times = np.empty_like(flows)

    for time_index, timestamp in enumerate(timestamps):
        hourly = traffic.loc[traffic["timestamp_local"] == timestamp]
        observed: dict[int, float] = {}
        for row in hourly.itertuples(index=False):
            key = (row.site_number, row.channel_id, row.direction)
            index = edge_index[mapping[key]]
            observed[index] = observed.get(index, 0.0) + float(row.motor_vehicle_count)
        ratios_by_class: dict[str, list[float]] = {}
        for index, flow in observed.items():
            ratios_by_class.setdefault(highways[index], []).append(flow / capacities[index])
        all_ratios = [ratio for values in ratios_by_class.values() for ratio in values]
        fallback = float(np.median(all_ratios)) if all_ratios else 0.0
        class_medians = {key: float(np.median(values)) for key, values in ratios_by_class.items()}
        ratios = np.asarray([class_medians.get(highway, fallback) for highway in highways], dtype=float)
        for index, flow in observed.items():
            ratios[index] = flow / capacities[index]
        ratios = np.clip(ratios, 0.0, 1.2)
        flows[time_index] = ratios * capacities
        bpr_times[time_index] = bpr_travel_times(
            free_flow, flows[time_index], capacities, alpha, beta
        )

    hub_nodes = hubs.set_index("hub_id")["nearest_road_node_id"].to_dict()
    od_nodes = (
        demand.drop_duplicates("od_id").set_index("od_id")[["origin_node_id", "destination_node_id"]]
    )

    def shortest_route_profiles(edge_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        route_profiles = np.empty((len(edge_times), len(od_ids), len(hub_ids)), dtype=float)
        detour_profiles = np.empty_like(route_profiles)
        for time_index, hourly_edge_times in enumerate(edge_times):
            for _, _, data in graph.edges(data=True):
                data["travel_time_hours"] = float(hourly_edge_times[int(data["edge_index"])])
            origin_distances = {
                od_id: nx.single_source_dijkstra_path_length(
                    graph, str(od_nodes.loc[od_id, "origin_node_id"]), weight="travel_time_hours"
                )
                for od_id in od_ids
            }
            hub_distances = {
                hub_id: nx.single_source_dijkstra_path_length(
                    graph, str(hub_nodes[hub_id]), weight="travel_time_hours"
                )
                for hub_id in hub_ids
            }
            for od_index, od_id in enumerate(od_ids):
                destination = str(od_nodes.loc[od_id, "destination_node_id"])
                direct = origin_distances[od_id].get(destination)
                if direct is None:
                    raise ValueError(f"no directed path for {od_id} origin-to-destination")
                for hub_index, hub_id in enumerate(hub_ids):
                    hub_node = str(hub_nodes[hub_id])
                    first = origin_distances[od_id].get(hub_node)
                    second = hub_distances[hub_id].get(destination)
                    if first is None or second is None:
                        raise ValueError(f"no directed path for {od_id} through {hub_id}")
                    route = float(first + second)
                    route_profiles[time_index, od_index, hub_index] = route
                    detour_profiles[time_index, od_index, hub_index] = max(route - float(direct), 0.0)
        return route_profiles, detour_profiles

    free_flow_routes, free_flow_detours = shortest_route_profiles(free_flow[None, :])
    route_profiles, detour_profiles = shortest_route_profiles(bpr_times)
    return _NetworkProfiles(
        graph=graph,
        edge_flows=flows,
        edge_free_flow_hours=free_flow,
        edge_bpr_hours=bpr_times,
        edge_capacities=capacities,
        free_flow_route_times_hours=np.repeat(free_flow_routes, len(timestamps), axis=0),
        free_flow_detours_hours=np.repeat(free_flow_detours, len(timestamps), axis=0),
        route_times_hours=route_profiles,
        detours_hours=detour_profiles,
        counter_mapping=mapping,
    )


def load_york_scenario(zip_path: str | Path) -> YorkScenario:
    path = Path(zip_path).expanduser().resolve()
    with ZipFile(path) as archive:
        root = _data_root(archive)
        with archive.open(f"{root}/scenario_config.yaml") as handle:
            package_config = yaml.safe_load(handle)
        if not isinstance(package_config, dict):
            raise ValueError("York scenario_config.yaml root must be a mapping")
        _validate_queue_model(package_config)
        configured_paths = package_config["scenario"]["data_paths"]
        expected = {f"{root}/{name}" for name in configured_paths.values()}
        runtime_members = {f"{root}/{name}" for name in _RUNTIME_TABLES}
        if expected != runtime_members:
            raise ValueError("York YAML must reference the six packaged runtime CSV files")
        tables = {
            name: _read_csv(archive, f"{root}/{name}")
            for name in _RUNTIME_TABLES
        }
    tables = _prepare_tables(package_config, tables)

    nodes = tables["road_nodes.csv"]
    edges = tables["road_edges.csv"]
    traffic = tables["traffic_flow.csv"]
    hubs = tables["charging_hubs.csv"].sort_values("hub_id").reset_index(drop=True)
    demand = tables["od_demand.csv"].sort_values(["timestamp_local", "od_id"]).reset_index(drop=True)
    energy = tables["energy_profiles.csv"].sort_values(["timestamp_local", "hub_id"]).reset_index(drop=True)
    timestamps = tuple(pd.Timestamp(value) for value in sorted(demand["timestamp_local"].unique()))
    hub_ids = tuple(sorted(hubs["hub_id"].unique()))
    od_ids = tuple(sorted(demand["od_id"].unique()))

    packaged_routes, packaged_detours, candidates = _packaged_route_matrices(
        demand, timestamps, od_ids, hub_ids
    )
    hub_parameters = {
        column: hubs[column].to_numpy(dtype=float)
        for column in (
            "num_chargers", "charger_power_kw", "queue_capacity_vehicles", "pv_capacity_kw",
            "battery_capacity_kwh", "battery_power_kw", "grid_import_limit_kw",
            "min_price_gbp_per_kwh", "max_price_gbp_per_kwh", "initial_price_gbp_per_kwh",
        )
    }
    energy_parameters = {
        column: _ordered_matrix(energy, column, timestamps, hub_ids, "hub_id")
        for column in (
            "pv_availability_factor", "pv_capacity_kw", "pv_available_kwh",
            "grid_price_gbp_per_kwh", "grid_import_limit_kwh",
            "battery_continuation_value_gbp_per_kwh", "initial_battery_soc_fraction",
        )
    }
    od_expected_demand = _ordered_matrix(
        demand, "potential_charging_requests", timestamps, od_ids, "od_id"
    )
    od_energy_parameters = {
        column: _ordered_matrix(demand, column, timestamps, od_ids, "od_id")
        for column in (
            "mean_requested_energy_kwh", "std_requested_energy_kwh",
            "min_requested_energy_kwh", "max_requested_energy_kwh",
        )
    }
    network = _network_profiles(
        package_config, nodes, edges, traffic, hubs, demand, timestamps, hub_ids, od_ids
    )
    limitations = (
        "Traffic observations cover 17 sparse ATC counters; unobserved edges use mapped-edge highway median v/c.",
        "traffic_flow.road_class is a counter road label and is not joined to OSM highway classes.",
        "Hub locations are York Open Data, while facility and energy parameters are provisional assumptions.",
        "Potential demand is ATC car count scaled by the packaged 0.015 ratio and is not observed charging demand.",
        "The road network is a prepared OSM snapshot and packaged free-flow routes are an alternative route model.",
    )
    scenario = YorkScenario(
        timestamps=timestamps,
        hub_ids=hub_ids,
        od_ids=od_ids,
        package_config=package_config,
        hub_parameters=hub_parameters,
        energy_parameters=energy_parameters,
        od_expected_demand=od_expected_demand,
        od_energy_parameters=od_energy_parameters,
        candidate_mask=candidates,
        packaged_route_times_hours=packaged_routes,
        packaged_detours_hours=packaged_detours,
        network_route_times_hours=network.route_times_hours,
        network_detours_hours=network.detours_hours,
        network_free_flow_route_times_hours=network.free_flow_route_times_hours,
        network_free_flow_detours_hours=network.free_flow_detours_hours,
        network_edge_flows=network.edge_flows,
        network_edge_free_flow_hours=network.edge_free_flow_hours,
        network_edge_bpr_hours=network.edge_bpr_hours,
        network_edge_capacities=network.edge_capacities,
        counter_edge_mapping=network.counter_mapping,
        network=network.graph,
        road_nodes=nodes,
        road_edges=edges,
        traffic_flow=traffic,
        charging_hubs=hubs,
        od_demand=demand,
        energy_profiles=energy,
        data_limitations=limitations,
        zip_sha256=_zip_digest(path),
    )
    structural_validation(scenario)
    return scenario


def structural_validation(scenario: YorkScenario) -> dict[str, bool]:
    nodes = set(scenario.road_nodes["node_id"])
    hub_ids = set(scenario.hub_ids)
    demand = scenario.od_demand
    checks = dict(zip(
        _STRUCTURAL_CHECK_NAMES,
        (
            len(demand) == 36,
            demand["od_id"].nunique() == 6,
            demand["timestamp_local"].nunique() == 6,
            len(scenario.energy_profiles) == 48,
            len(scenario.hub_ids) == 8,
            set(scenario.energy_profiles["hub_id"]) == hub_ids,
            all(u in nodes and v in nodes for u, v in zip(scenario.road_edges["u"], scenario.road_edges["v"])),
            all(node in nodes for node in scenario.charging_hubs["nearest_road_node_id"]),
            all(node in nodes for node in scenario.traffic_flow["nearest_road_node_id"]),
            all(
                origin in nodes and destination in nodes
                for origin, destination in zip(demand["origin_node_id"], demand["destination_node_id"])
            ),
            all(set(route) == hub_ids for route in demand["route_freeflow_time_min"]),
            int(scenario.package_config["scenario"]["number_of_hubs"]) == 8
            and int(scenario.package_config["scenario"]["number_of_od_groups"]) == 6,
        ),
    ))
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise ValueError("York structural validation failed: " + ", ".join(failures))
    return checks


def deterministic_fixed_tariff_sanity(
    scenario: YorkScenario,
    route_mode: str = "packaged_freeflow",
    *,
    validate: bool = True,
) -> dict[str, Any]:
    if route_mode != "packaged_freeflow":
        raise ValueError("deterministic reference values require packaged_freeflow routes")
    config = scenario.package_config
    sensitivity = float(config["user_choice"]["inverse_cost_sensitivity_lambda_per_gbp"])
    value_of_time = float(config["user_choice"]["value_of_time_gbp_per_hour"])
    outside_cost = float(config["user_choice"]["outside_option"]["true_hidden_cost_gbp"])
    queue = config["queue_model"]
    service_time = float(queue["average_service_time_hours"])
    dt = float(config["scenario"]["time_step_hours"])
    price = float(config["pricing"]["initial_reference_price_gbp_per_kwh"])
    chargers = scenario.hub_parameters["num_chargers"]
    service_times = np.full(len(scenario.hub_ids), service_time, dtype=float)
    hourly_service = service_capacity(chargers, dt, service_times)
    battery_power = scenario.hub_parameters["battery_power_kw"]
    previous_wait = np.full(
        len(scenario.hub_ids),
        float(queue["initial_residual_wait_hours"]),
        dtype=float,
    )
    queued_energy = np.full(
        len(scenario.hub_ids),
        float(queue["initial_queued_energy_kwh"]),
        dtype=float,
    )
    hourly_demand: list[float] = []
    hourly_wait: list[float] = []
    hourly_queue: list[float] = []
    hourly_margin: list[float] = []
    queue_states: list[tuple[np.ndarray, np.ndarray]] = []

    for time_index in range(len(scenario.timestamps)):
        energy = scenario.od_energy_parameters["mean_requested_energy_kwh"][time_index]
        costs = price * energy[:, None] + value_of_time * (
            scenario.packaged_route_times_hours[time_index] + previous_wait[None, :]
        )
        hub_probabilities, _ = multinomial_logit(costs, outside_cost, sensitivity)
        od_demand = scenario.od_expected_demand[time_index]
        hub_requests = od_demand @ hub_probabilities
        requested_energy = (od_demand * energy) @ hub_probabilities
        transition = transition_fluid_queue(
            previous_wait,
            queued_energy,
            hub_requests,
            requested_energy,
            hourly_service,
            dt,
        )
        maximum_supply = (
            scenario.energy_parameters["pv_available_kwh"][time_index]
            + scenario.energy_parameters["grid_import_limit_kwh"][time_index]
            + battery_power * dt
        )
        margins = maximum_supply - transition.admitted_energy_kwh
        previous_wait = transition.residual_wait_next_hours
        queued_energy = transition.queued_energy_next_kwh
        queue_states.append((previous_wait.copy(), queued_energy.copy()))
        hourly_demand.append(float(od_demand.sum()))
        hourly_wait.append(float(previous_wait.max() * 60.0))
        hourly_queue.append(float(queued_energy.sum()))
        hourly_margin.append(float(margins.min()))

    clearance_index = next(
        (
            index for index in range(len(queue_states))
            if all(
                np.allclose(wait, 0.0, atol=1e-10)
                and np.allclose(energy, 0.0, atol=1e-8)
                for wait, energy in queue_states[index:]
            )
        ),
        None,
    )
    result = {
        "total_demand": round(sum(hourly_demand), 3),
        "peak_hourly_demand": round(max(hourly_demand), 3),
        "max_wait_minutes": round(max(hourly_wait), 3),
        "peak_queued_energy_kwh": round(max(hourly_queue), 3),
        "min_supply_margin_kwh": round(min(hourly_margin), 3),
        "queue_cleared_from": (
            str(scenario.timestamps[clearance_index])
            if clearance_index is not None
            else None
        ),
    }
    if validate:
        expected = {
            "total_demand": 382.830,
            "peak_hourly_demand": 103.800,
            "max_wait_minutes": 75.996,
            "peak_queued_energy_kwh": 649.458,
            "min_supply_margin_kwh": 152.400,
        }
        tolerances = {
            "total_demand": 0.002,
            "peak_hourly_demand": 0.002,
            "max_wait_minutes": 0.010,
            "peak_queued_energy_kwh": 0.020,
            "min_supply_margin_kwh": 0.020,
        }
        failures = [
            key for key in expected
            if not np.isclose(result[key], expected[key], rtol=0.0, atol=tolerances[key])
        ]
        if result["queue_cleared_from"] != "2024-09-20 20:00:00":
            failures.append("queue_cleared_from")
        if failures:
            raise AssertionError(f"York deterministic sanity mismatch: {failures}; got {result}")
    return result


def york_config_from_scenario(scenario: YorkScenario, route_mode: str) -> dict[str, Any]:
    if route_mode not in _ROUTE_MODES:
        raise ValueError(f"data.route_mode must be one of {sorted(_ROUTE_MODES)}")
    package = scenario.package_config
    queue = package["queue_model"]
    dispatch = package["energy_dispatch"]
    choice = package["user_choice"]
    outside = choice["outside_option"]
    reward = package["shared_reward"]
    mappo = package["mappo"]
    hubs = scenario.hub_parameters
    n_hubs = len(scenario.hub_ids)
    mean_energy = scenario.od_energy_parameters["mean_requested_energy_kwh"].mean(axis=0)
    return {
        "seed": int(mappo["training_seeds"][0]),
        "environment_schema_version": _YORK_ENVIRONMENT_SCHEMA_VERSION,
        "queue_semantics": _YORK_QUEUE_SEMANTICS,
        "data": {
            "mode": "york_zip",
            "zip_sha256": scenario.zip_sha256,
            "route_mode": route_mode,
            "assumptions": list(scenario.data_limitations),
        },
        "system": {
            "periods": len(scenario.timestamps),
            "dt_hours": float(package["scenario"]["time_step_hours"]),
            "n_hubs": n_hubs,
            "n_ods": len(scenario.od_ids),
            "deterministic_demand": False,
        },
        "traffic": {
            "route_mode": route_mode,
            "bpr_a": float(package["traffic_model"]["bpr_alpha"]),
            "bpr_b": float(package["traffic_model"]["bpr_beta"]),
        },
        "demand": {
            "base_od_counts": scenario.od_expected_demand[0].tolist(),
            "energy_kwh": mean_energy.tolist(),
            "episode_multiplier_mean": float(package["demand_generation"]["episode_demand_multiplier_mean"]),
            "episode_multiplier_std": float(package["demand_generation"]["episode_demand_multiplier_std"]),
            "episode_multiplier_min": float(package["demand_generation"]["episode_demand_multiplier_min"]),
            "episode_multiplier_max": float(package["demand_generation"]["episode_demand_multiplier_max"]),
            "noise_fraction": float(package["demand_generation"]["observed_demand_noise_std_fraction"]),
        },
        "choice": {
            "inverse_cost_sensitivity": float(choice["inverse_cost_sensitivity_lambda_per_gbp"]),
            "value_of_time_per_hour": float(choice["value_of_time_gbp_per_hour"]),
            "inaccessible_hub_cost": float(choice["inaccessible_hub_cost_gbp"]),
        },
        "queue": {
            "discipline": queue["queue_discipline"],
            "semantics": _YORK_QUEUE_SEMANTICS,
            "carry_over_between_periods": bool(queue["carry_over_between_periods"]),
            "finite_waiting_space_enforced": bool(queue["finite_waiting_space_enforced"]),
            "queue_capacity_vehicles_usage": queue["queue_capacity_vehicles_usage"],
            "initial_residual_wait_hours": float(queue["initial_residual_wait_hours"]),
            "initial_queued_energy_kwh": float(queue["initial_queued_energy_kwh"]),
        },
        "hubs": {
            "chargers": hubs["num_chargers"].tolist(),
            "service_time_hours": [float(queue["average_service_time_hours"])] * n_hubs,
            "max_wait_hours": [float(queue["maximum_wait_hours"])] * n_hubs,
            "pv_peak_kwh": hubs["pv_capacity_kw"].tolist(),
            "battery_capacity_kwh": hubs["battery_capacity_kwh"].tolist(),
            "initial_soc": scenario.energy_parameters["initial_battery_soc_fraction"][0].tolist(),
            "min_soc": [float(dispatch["minimum_soc_fraction"])] * n_hubs,
            "max_soc": [float(dispatch["maximum_soc_fraction"])] * n_hubs,
            "charge_limit_kw": hubs["battery_power_kw"].tolist(),
            "discharge_limit_kw": hubs["battery_power_kw"].tolist(),
            "grid_cap_kw": hubs["grid_import_limit_kw"].tolist(),
            "eta_charge": [float(dispatch["charge_efficiency"])] * n_hubs,
            "eta_discharge": [float(dispatch["discharge_efficiency"])] * n_hubs,
            "battery_cost_per_kwh": [float(dispatch["battery_degradation_cost_gbp_per_kwh"])] * n_hubs,
            "pv_cost_per_kwh": [float(dispatch["pv_om_cost_gbp_per_kwh"])] * n_hubs,
            "operating_cost_per_request": [float(dispatch["operating_cost_gbp_per_served_vehicle"])] * n_hubs,
            "welfare_weights": [float(value) for value in reward["hub_profit_weights"]],
            "queue_capacity_vehicles": hubs["queue_capacity_vehicles"].tolist(),
            "charger_power_kw": hubs["charger_power_kw"].tolist(),
        },
        "price": {
            "min": hubs["min_price_gbp_per_kwh"].tolist(),
            "max": hubs["max_price_gbp_per_kwh"].tolist(),
            "initial": hubs["initial_price_gbp_per_kwh"].tolist(),
        },
        "dispatch": {
            "unmet_penalty_per_kwh": float(dispatch["dispatch_unmet_energy_priority_penalty_gbp_per_kwh"]),
            "future_battery_value_mode": "profile",
            "future_battery_value_constant": 0.0,
            "feasibility_tolerance": float(package["evaluation"]["feasibility_tolerance_unmet_kwh"]),
        },
        "inference": {
            "initial_outside_cost": float(outside["initial_estimate_gbp"]),
            "learning_rate": float(outside["inference_learning_rate"]),
            "min_outside_cost": float(outside["estimate_min_gbp"]),
            "max_outside_cost": float(outside["estimate_max_gbp"]),
            "rolling_loss_window_periods": int(outside["rolling_loss_window_periods"]),
            "reset_between_episodes": bool(outside["reset_estimate_between_episodes"]),
        },
        "reward": {
            "wait_penalty": float(reward["wait_excess_penalty_gbp_per_hour"]),
            "unmet_penalty": float(reward["unmet_energy_penalty_gbp_per_kwh"]),
            "scale": float(reward["reward_scale"]),
        },
        "normalization": {
            "demand": float(np.ceil(scenario.od_expected_demand.sum(axis=1).max())),
            "wait_hours": 1.0,
            "queued_energy_kwh": float(np.ceil(
                np.max(
                    scenario.od_expected_demand
                    * scenario.od_energy_parameters["mean_requested_energy_kwh"]
                )
            )),
            "backlog_vehicles": float(np.ceil(scenario.od_expected_demand.sum(axis=1).max())),
            "pv_kwh": float(np.ceil(scenario.energy_parameters["pv_available_kwh"].max())),
            "battery_kwh": float(np.ceil(hubs["battery_capacity_kwh"].max())),
            "grid_price": float(np.ceil(scenario.energy_parameters["grid_price_gbp_per_kwh"].max() * 10.0) / 10.0),
            "grid_cap_kwh": float(np.ceil(hubs["grid_import_limit_kw"].max())),
            "route_time_hours": float(np.ceil(max(
                scenario.packaged_route_times_hours.max(), scenario.network_route_times_hours.max()
            ))),
            "outside_cost": float(outside["estimate_max_gbp"]),
        },
        "network": {
            "hidden_sizes": [int(value) for value in mappo["actor_hidden_layers"]],
            "critic_hidden_sizes": [int(value) for value in mappo["critic_hidden_layers"]],
            "beta_epsilon": 0.1,
        },
        "training": {
            "updates": int(mappo["episodes_per_seed"]),
            "episodes_per_seed": int(mappo["episodes_per_seed"]),
            "seeds": [int(value) for value in mappo["training_seeds"]],
            "rollout_steps": int(mappo["rollout_horizon"]),
            "ppo_epochs": int(mappo["update_epochs_per_rollout"]),
            "minibatch_size": int(mappo["minibatch_size"]),
            "actor_lr": float(mappo["actor_learning_rate"]),
            "critic_lr": float(mappo["critic_learning_rate"]),
            "gamma": float(mappo["discount_factor_gamma"]),
            "gae_lambda": float(mappo["gae_lambda"]),
            "clip_ratio": float(mappo["ppo_clip_epsilon"]),
            "entropy_coef": float(mappo["entropy_coefficient_start"]),
            "entropy_coef_end": float(mappo["entropy_coefficient_end"]),
            "value_coef": float(mappo["value_loss_coefficient"]),
            "max_grad_norm": float(mappo["max_gradient_norm"]),
        },
        "oracle": {
            "points_per_hub": int(package["evaluation"]["centralized_oracle_price_grid_levels"]),
            "deterministic_expected_demand": True,
        },
        "output": {"directory": "outputs/york"},
    }
