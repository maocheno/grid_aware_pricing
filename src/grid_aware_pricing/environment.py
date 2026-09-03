"""Dual-layer EV charging pricing environment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import truncnorm

from .dispatch import DispatchInputs, DispatchResult, dispatch_energy
from .inference import OutsideOptionEstimator
from .synthetic import SyntheticProfiles, generate_profiles
from .system_model import (
    bpr_travel_times,
    choice_result,
    generalized_costs,
    multinomial_logit,
    profit_components,
    realize_demand,
    realize_york_demand,
    route_times,
    service_capacity,
    transition_fluid_queue,
    waiting_time,
)


@dataclass(frozen=True)
class StepOutput:
    observations: np.ndarray
    global_state: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class GridAwarePricingEnv:
    """Finite-horizon simulator with decentralized observations and shared reward."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.is_york = config.get("data", {}).get("mode") == "york_zip"
        self.n_hubs = int(config["system"]["n_hubs"])
        self.n_ods = int(config["system"]["n_ods"])
        self.periods = int(config["system"]["periods"])
        self.dt = float(config["system"]["dt_hours"])
        self.seed = int(config["seed"])
        self.rng = np.random.default_rng(self.seed)
        self.episode_index = -1

        inference = config["inference"]
        objective = str(inference.get("objective", "outside_nll" if self.is_york else "squared_hub_counts"))
        self.estimator = OutsideOptionEstimator(
            inference["initial_outside_cost"],
            inference["learning_rate"],
            inference["min_outside_cost"],
            inference["max_outside_cost"],
            config["choice"]["inverse_cost_sensitivity"],
            objective=objective,
            rolling_loss_window=int(inference.get("rolling_loss_window_periods", 1)),
        )
        experiment = config.get("experiment", {})
        self.outside_mode = str(
            experiment.get("outside_mode", experiment.get("outside_option_mode", inference.get("mode", "inferred")))
        )
        if self.outside_mode not in {"inferred", "fixed", "known"}:
            raise ValueError("outside mode must be inferred, fixed, or known")
        masks = experiment.get(
            "observation_masks", experiment.get("observation_mask", experiment.get("observation", {}))
        )
        if isinstance(masks, str):
            mask_names = {masks}
        elif isinstance(masks, (list, tuple, set)):
            mask_names = set(masks)
        elif isinstance(masks, dict):
            mask_names = {name for name, enabled in masks.items() if enabled}
        else:
            mask_names = set()
        self.mask_no_traffic = bool(experiment.get("no_traffic", "no_traffic" in mask_names))
        self.mask_no_energy = bool(experiment.get("no_energy", "no_energy" in mask_names))

        self.price_min = np.asarray(config["price"]["min"], dtype=float)
        self.price_max = np.asarray(config["price"]["max"], dtype=float)
        hubs = config["hubs"]
        self.battery_capacity = np.asarray(hubs["battery_capacity_kwh"], dtype=float)
        self.service_time = np.asarray(hubs["service_time_hours"], dtype=float)
        self.service = service_capacity(np.asarray(hubs["chargers"], dtype=float), self.dt, self.service_time)
        self.queue_capacity = np.asarray(hubs.get("queue_capacity_vehicles", np.zeros(self.n_hubs)), dtype=float)

        if self.is_york:
            self._initialize_york()
        else:
            self._initialize_synthetic()
        self.observation_dim = (10 if self.is_york else 8) + self.n_ods
        self.global_state_dim = self.observation_dim * self.n_hubs + (
            self.n_ods + 2 if self.is_york else 0
        )
        self.reset(reset_inference=True)

    def _initialize_synthetic(self) -> None:
        config = self.config
        self.profiles: SyntheticProfiles | None = generate_profiles(config, np.random.default_rng(self.seed))
        self.scenario = None
        self.hub_ids = tuple(str(index) for index in range(self.n_hubs))
        self.od_ids = tuple(str(index) for index in range(self.n_ods))
        self.energy_per_od = np.asarray(config["demand"]["energy_kwh"], dtype=float)
        hubs = config["hubs"]
        self.grid_cap = np.asarray(hubs["grid_cap_kw"], dtype=float) * self.dt
        self.initial_battery_energy = np.asarray(hubs["initial_soc"], dtype=float) * self.battery_capacity
        traffic = config["traffic"]
        free_flow = np.asarray(traffic["free_flow_times"], dtype=float)
        capacities = np.asarray(traffic["link_capacities"], dtype=float)
        assert self.profiles is not None
        self.route_time_profiles = np.asarray([
            route_times(
                bpr_travel_times(
                    free_flow,
                    flows,
                    capacities,
                    float(traffic["bpr_a"]),
                    float(traffic["bpr_b"]),
                ),
                traffic["routes"],
            )
            for flows in self.profiles.link_flows
        ])
        self.detour_profiles = np.zeros_like(self.route_time_profiles)
        self.candidate_masks = np.ones_like(self.route_time_profiles, dtype=bool)
        self._true_outside_cost = float(config["choice"]["true_outside_cost"])

    def _initialize_york(self) -> None:
        scenario = self.config.get("_york_scenario")
        if scenario is None:
            raise ValueError("York mode requires config['_york_scenario']")
        self.scenario = scenario
        self.profiles = None
        self.hub_ids = tuple(scenario.hub_ids)
        self.od_ids = tuple(scenario.od_ids)
        self.route_time_profiles, self.detour_profiles = scenario.route_profiles(
            self.config["data"]["route_mode"]
        )
        self.candidate_masks = np.asarray(scenario.candidate_mask, dtype=bool)
        self.energy_per_od = np.asarray(
            scenario.od_energy_parameters["mean_requested_energy_kwh"], dtype=float
        ).mean(axis=0)
        self.grid_cap = np.asarray(scenario.energy_parameters["grid_import_limit_kwh"], dtype=float)
        initial_soc = np.asarray(
            scenario.energy_parameters["initial_battery_soc_fraction"][0], dtype=float
        )
        self.initial_battery_energy = initial_soc * self.battery_capacity
        self._true_outside_cost = float(
            scenario.package_config["user_choice"]["outside_option"]["true_hidden_cost_gbp"]
        )
        if self.outside_mode == "known":
            self.estimator.cost = self._true_outside_cost

    def _sample_demand_multiplier(self) -> float:
        if not self.is_york or bool(self.config["system"]["deterministic_demand"]):
            return 1.0
        demand = self.config["demand"]
        mean = float(demand.get("episode_multiplier_mean", 1.0))
        std = float(demand.get("episode_multiplier_std", 0.0))
        lower = float(demand.get("episode_multiplier_min", mean))
        upper = float(demand.get("episode_multiplier_max", mean))
        if std <= 0.0:
            return float(np.clip(mean, lower, upper))
        distribution = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
        return float(distribution.rvs(random_state=self.rng))

    def reset(
        self,
        seed: int | None = None,
        outside_estimate: float | None = None,
        *,
        reset_inference: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_index += 1
        self.t = 0
        self.previous_demand = np.zeros(self.n_hubs, dtype=float)
        if self.is_york:
            queue = self.config["queue"]
            self.previous_wait = np.full(
                self.n_hubs, float(queue["initial_residual_wait_hours"]), dtype=float
            )
            self.queued_energy_kwh = np.full(
                self.n_hubs, float(queue["initial_queued_energy_kwh"]), dtype=float
            )
        else:
            self.previous_wait = np.zeros(self.n_hubs, dtype=float)
            self.previous_queue = np.zeros(self.n_hubs, dtype=float)
        if self.is_york:
            self.previous_price = np.asarray(self.config["price"]["initial"], dtype=float).copy()
        else:
            self.previous_price = (self.price_min + self.price_max) / 2.0
        self.battery_energy = self.initial_battery_energy.copy()
        self.demand_multiplier = self._sample_demand_multiplier()
        if reset_inference:
            self.estimator.loss_history = []
        if self.outside_mode == "known":
            self.estimator.cost = self._true_outside_cost
        elif outside_estimate is not None:
            self.estimator.cost = float(
                np.clip(outside_estimate, self.estimator.min_cost, self.estimator.max_cost)
            )
        elif reset_inference:
            initial = float(self.config["inference"]["initial_outside_cost"])
            self.estimator.cost = float(
                np.clip(initial, self.estimator.min_cost, self.estimator.max_cost)
            )
        observations = self._observations()
        return observations, self._global_state(observations)

    def snapshot(self) -> dict[str, Any]:
        state = {
            "t": self.t,
            "episode_index": self.episode_index,
            "demand_multiplier": self.demand_multiplier,
            "previous_demand": self.previous_demand.copy(),
            "previous_wait": self.previous_wait.copy(),
            "previous_price": self.previous_price.copy(),
            "battery_energy": self.battery_energy.copy(),
            "estimator": deepcopy(self.estimator.state_dict()),
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }
        if self.is_york:
            state["queued_energy_kwh"] = self.queued_energy_kwh.copy()
        else:
            state["previous_queue"] = self.previous_queue.copy()
        return state

    def restore(self, state: dict[str, Any]) -> None:
        self.t = int(state["t"])
        self.episode_index = int(state.get("episode_index", self.episode_index))
        self.demand_multiplier = float(state.get("demand_multiplier", 1.0))
        self.previous_demand = np.asarray(state["previous_demand"], dtype=float).copy()
        self.previous_wait = np.asarray(state["previous_wait"], dtype=float).copy()
        if self.is_york:
            if "queued_energy_kwh" not in state:
                raise ValueError("York queue-v2 snapshot is missing queued_energy_kwh")
            self.queued_energy_kwh = np.asarray(
                state["queued_energy_kwh"], dtype=float
            ).copy()
        else:
            self.previous_queue = np.asarray(
                state.get("previous_queue", np.zeros(self.n_hubs)), dtype=float
            ).copy()
        self.previous_price = np.asarray(state["previous_price"], dtype=float).copy()
        self.battery_energy = np.asarray(state["battery_energy"], dtype=float).copy()
        if "estimator" in state:
            self.estimator.load_state_dict(state["estimator"])
        else:
            self.estimator.cost = float(state["outside_cost_estimate"])
        self.rng.bit_generator.state = deepcopy(state["rng_state"])

    def _time_index(self) -> int:
        return min(self.t, self.periods - 1)

    def current_route_times(self) -> np.ndarray:
        return np.asarray(self.route_time_profiles[self._time_index()], dtype=float)

    def _current_energy_profiles(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.is_york:
            assert self.scenario is not None
            energy = self.scenario.energy_parameters
            return (
                np.asarray(energy["pv_available_kwh"][index], dtype=float),
                np.asarray(energy["grid_price_gbp_per_kwh"][index], dtype=float),
                np.asarray(energy["grid_import_limit_kwh"][index], dtype=float),
            )
        assert self.profiles is not None
        return (
            np.asarray(self.profiles.pv_kwh[index], dtype=float),
            np.full(self.n_hubs, float(self.profiles.grid_prices[index])),
            np.asarray(self.grid_cap, dtype=float),
        )

    def _observations(self) -> np.ndarray:
        index = self._time_index()
        normalization = self.config["normalization"]
        routes = self.current_route_times()
        pv, grid_prices, grid_caps = self._current_energy_profiles(index)
        soc = self.battery_energy / self.battery_capacity
        visible_routes = np.zeros_like(routes) if self.is_york and self.mask_no_traffic else routes
        if self.is_york and self.mask_no_energy:
            pv = np.zeros_like(pv)
            grid_prices = np.zeros_like(grid_prices)
            grid_caps = np.zeros_like(grid_caps)
            soc = np.zeros_like(soc)
        observations = []
        historical_backlog = self.service / self.dt * self.previous_wait
        for hub in range(self.n_hubs):
            local_state = [
                self.previous_demand[hub] / float(normalization["demand"]),
                (self.previous_price[hub] - self.price_min[hub])
                / (self.price_max[hub] - self.price_min[hub]),
                self.previous_wait[hub] / float(normalization["wait_hours"]),
            ]
            if self.is_york:
                local_state.extend([
                    self.queued_energy_kwh[hub]
                    / float(normalization["queued_energy_kwh"]),
                    historical_backlog[hub]
                    / float(normalization["backlog_vehicles"]),
                ])
            local_state.extend([
                pv[hub] / float(normalization["pv_kwh"]),
                soc[hub],
                grid_prices[hub] / float(normalization["grid_price"]),
                grid_caps[hub] / float(normalization["grid_cap_kwh"]),
            ])
            raw = np.concatenate([
                local_state,
                visible_routes[:, hub] / float(normalization["route_time_hours"]),
                [self.estimator.cost / float(normalization["outside_cost"])],
            ])
            observations.append(raw)
        return np.asarray(observations, dtype=np.float32)

    def _global_state(self, observations: np.ndarray) -> np.ndarray:
        if not self.is_york:
            return observations.reshape(-1)
        assert self.scenario is not None
        current_demand = np.asarray(
            self.scenario.od_expected_demand[self._time_index()], dtype=float
        ) * self.demand_multiplier
        normalization = self.config["normalization"]
        normalized = current_demand / float(normalization["demand"])
        queue_state = np.asarray([
            self.queued_energy_kwh.sum()
            / (float(normalization["queued_energy_kwh"]) * self.n_hubs),
            (self.service / self.dt * self.previous_wait).sum()
            / (float(normalization["backlog_vehicles"]) * self.n_hubs),
        ])
        return np.concatenate([
            observations.reshape(-1), normalized, queue_state
        ]).astype(np.float32)

    def _future_value(self, hub: int, grid_price: float, index: int) -> float:
        dispatch = self.config["dispatch"]
        mode = dispatch["future_battery_value_mode"]
        if mode == "zero":
            return 0.0
        if mode == "constant":
            return float(dispatch["future_battery_value_constant"])
        if mode == "spread":
            eta = float(self.config["hubs"]["eta_discharge"][hub])
            cost = float(self.config["hubs"]["battery_cost_per_kwh"][hub])
            return eta * max(grid_price - cost, 0.0)
        if mode == "profile" and self.is_york:
            assert self.scenario is not None
            return float(
                self.scenario.energy_parameters["battery_continuation_value_gbp_per_kwh"][index, hub]
            )
        raise ValueError(f"unknown future_battery_value_mode: {mode}")

    def _dispatch(
        self,
        requested_energy: np.ndarray,
        pv_available: np.ndarray,
        grid_prices: np.ndarray,
        grid_caps: np.ndarray,
        index: int,
    ) -> list[DispatchResult]:
        hubs = self.config["hubs"]
        results = []
        for hub in range(self.n_hubs):
            dispatch_input = DispatchInputs(
                requested_energy=float(requested_energy[hub]),
                pv_available=float(pv_available[hub]),
                battery_energy=float(self.battery_energy[hub]),
                battery_capacity=float(self.battery_capacity[hub]),
                min_soc=float(hubs["min_soc"][hub]),
                max_soc=float(hubs["max_soc"][hub]),
                charge_limit=float(hubs["charge_limit_kw"][hub]) * self.dt,
                discharge_limit=float(hubs["discharge_limit_kw"][hub]) * self.dt,
                grid_cap=float(grid_caps[hub]),
                eta_charge=float(hubs["eta_charge"][hub]),
                eta_discharge=float(hubs["eta_discharge"][hub]),
                grid_price=float(grid_prices[hub]),
                battery_cost=float(hubs["battery_cost_per_kwh"][hub]),
                pv_cost=float(hubs["pv_cost_per_kwh"][hub]),
                unmet_penalty=float(self.config["dispatch"]["unmet_penalty_per_kwh"]),
                future_battery_value=self._future_value(hub, float(grid_prices[hub]), index),
            )
            results.append(
                dispatch_energy(
                    dispatch_input,
                    feasibility_tolerance=float(self.config["dispatch"]["feasibility_tolerance"]),
                )
            )
        return results

    def step(
        self,
        prices: np.ndarray,
        *,
        deterministic_demand: bool | None = None,
        update_inference: bool = True,
        outside_cost: float | None = None,
    ) -> StepOutput:
        if self.is_york:
            return self._step_york(prices, deterministic_demand, update_inference, outside_cost)
        return self._step_synthetic(prices, deterministic_demand, update_inference, outside_cost)

    def _step_synthetic(
        self,
        prices: np.ndarray,
        deterministic_demand: bool | None,
        update_inference: bool,
        outside_cost: float | None,
    ) -> StepOutput:
        if self.t >= self.periods:
            raise RuntimeError("episode is done; call reset")
        assert self.profiles is not None
        index = self.t
        prices = np.clip(np.asarray(prices, dtype=float), self.price_min, self.price_max)
        routes = self.current_route_times()
        costs = generalized_costs(
            prices,
            self.energy_per_od,
            routes,
            self.previous_wait,
            float(self.config["choice"]["value_of_time_per_hour"]),
        )
        true_outside = self._true_outside_cost if outside_cost is None else float(outside_cost)
        od_counts = self.profiles.od_counts[index]
        expected = choice_result(
            costs,
            true_outside,
            float(self.config["choice"]["inverse_cost_sensitivity"]),
            od_counts,
            self.energy_per_od,
        )
        deterministic = (
            self.config["system"]["deterministic_demand"]
            if deterministic_demand is None
            else deterministic_demand
        )
        realized_counts, realized_energy, outside_count = realize_demand(
            od_counts,
            self.energy_per_od,
            expected.hub_probabilities,
            expected.outside_probabilities,
            self.rng,
            bool(deterministic),
        )
        waits = waiting_time(realized_counts, self.service, self.service_time)
        pv, grid_prices, grid_caps = self._current_energy_profiles(index)
        dispatches = self._dispatch(realized_energy, pv, grid_prices, grid_caps, index)
        revenue = np.zeros(self.n_hubs)
        operating_cost = np.zeros(self.n_hubs)
        served_counts = np.zeros(self.n_hubs)
        profits = np.zeros(self.n_hubs)
        for hub, result in enumerate(dispatches):
            revenue[hub], operating_cost[hub], served_counts[hub], profits[hub] = profit_components(
                prices[hub],
                realized_energy[hub],
                realized_counts[hub],
                result.served_energy,
                result.energy_cost,
                float(self.config["hubs"]["operating_cost_per_request"][hub]),
            )
        wait_limits = np.asarray(self.config["hubs"]["max_wait_hours"], dtype=float)
        wait_violations = np.maximum(waits - wait_limits, 0.0)
        unmet = np.asarray([result.unmet_energy for result in dispatches])
        weights = np.asarray(self.config["hubs"]["welfare_weights"], dtype=float)
        welfare = float(weights @ profits)
        raw_reward = welfare - float(self.config["reward"]["wait_penalty"]) * float(
            wait_violations.sum()
        ) - float(self.config["reward"]["unmet_penalty"]) * float(unmet.sum())
        reward = raw_reward * float(self.config["reward"].get("scale", 1.0))
        if update_inference and self.outside_mode == "inferred":
            inference = self.estimator.update(costs, od_counts, realized_counts)
            inference_loss = inference.loss
            predicted_counts = inference.predicted_counts
        else:
            inference_loss, _, predicted_counts = self.estimator.loss_and_gradient(
                costs, od_counts, realized_counts
            )
        total_potential = float(np.sum(od_counts))
        outside_share = float(outside_count) / total_potential if total_potential > 0 else 0.0
        next_battery = np.asarray([result.next_battery_energy for result in dispatches])
        info: dict[str, Any] = {
            "period": index,
            "prices": prices.copy(),
            "route_times": routes.copy(),
            "expected_demand": expected.expected_counts.copy(),
            "expected_energy": expected.expected_energy.copy(),
            "realized_demand": realized_counts.copy(),
            "requested_energy": realized_energy.copy(),
            "outside_count": float(outside_count),
            "outside_share": outside_share,
            "hub_probabilities": expected.hub_probabilities.copy(),
            "outside_probabilities": expected.outside_probabilities.copy(),
            "wait": waits.copy(),
            "wait_violation": wait_violations.copy(),
            "service_capacity": self.service.copy(),
            "served_energy": np.asarray([result.served_energy for result in dispatches]),
            "unmet": unmet,
            "pv_to_ev": np.asarray([result.pv_to_ev for result in dispatches]),
            "pv_to_battery": np.asarray([result.pv_to_battery for result in dispatches]),
            "pv_curtailed": np.asarray([result.pv_curtailed for result in dispatches]),
            "battery_to_ev": np.asarray([result.battery_to_ev for result in dispatches]),
            "grid_to_ev": np.asarray([result.grid_to_ev for result in dispatches]),
            "grid_to_battery": np.asarray([result.grid_to_battery for result in dispatches]),
            "dispatch_mode": [result.mode for result in dispatches],
            "energy_cost": np.asarray([result.energy_cost for result in dispatches]),
            "revenue": revenue,
            "operating_cost": operating_cost,
            "served_count": served_counts,
            "profit": profits,
            "welfare": welfare,
            "weighted_hub_profit_welfare": welfare,
            "raw_reward": raw_reward,
            "scaled_reward": reward,
            "reward": reward,
            "grid_price": float(grid_prices[0]),
            "inference_loss": inference_loss,
            "predicted_demand_under_estimate": predicted_counts,
            "outside_cost_estimate": self.estimator.cost,
            "outside_cost_error": self.estimator.cost - true_outside,
            "outside_cost_error_for_evaluation_only": self.estimator.cost - true_outside,
            "outside_cost_error_is_evaluation_only": True,
            "true_outside_cost_for_evaluation_only": true_outside,
            "soc": next_battery / self.battery_capacity,
        }
        self.previous_demand = realized_counts.copy()
        self.previous_wait = waits.copy()
        self.previous_price = prices.copy()
        self.battery_energy = next_battery
        self.t += 1
        done = self.t >= self.periods
        observations = self._observations()
        return StepOutput(observations, self._global_state(observations), reward, done, info)

    def _step_york(
        self,
        prices: np.ndarray,
        deterministic_demand: bool | None,
        update_inference: bool,
        outside_cost: float | None,
    ) -> StepOutput:
        if self.t >= self.periods:
            raise RuntimeError("episode is done; call reset")
        assert self.scenario is not None
        index = self.t
        prices = np.clip(np.asarray(prices, dtype=float), self.price_min, self.price_max)
        routes = np.asarray(self.route_time_profiles[index], dtype=float)
        detours = np.asarray(self.detour_profiles[index], dtype=float)
        candidate_mask = np.asarray(self.candidate_masks[index], dtype=bool)
        energy_parameters = self.scenario.od_energy_parameters
        mean_energy = np.asarray(energy_parameters["mean_requested_energy_kwh"][index], dtype=float)
        costs = generalized_costs(
            prices,
            mean_energy,
            routes,
            self.previous_wait,
            float(self.config["choice"]["value_of_time_per_hour"]),
        )
        costs = np.where(
            candidate_mask,
            costs,
            float(self.config["choice"]["inaccessible_hub_cost"]),
        )
        true_outside = self._true_outside_cost if outside_cost is None else float(outside_cost)
        hub_probabilities, outside_probabilities = multinomial_logit(
            costs,
            true_outside,
            float(self.config["choice"]["inverse_cost_sensitivity"]),
            candidate_mask,
        )
        deterministic = bool(
            self.config["system"]["deterministic_demand"]
            if deterministic_demand is None
            else deterministic_demand
        )
        multiplier = 1.0 if deterministic else self.demand_multiplier
        expected_od = np.asarray(self.scenario.od_expected_demand[index], dtype=float)
        demand = realize_york_demand(
            expected_od,
            multiplier,
            mean_energy,
            np.asarray(energy_parameters["std_requested_energy_kwh"][index], dtype=float),
            np.asarray(energy_parameters["min_requested_energy_kwh"][index], dtype=float),
            np.asarray(energy_parameters["max_requested_energy_kwh"][index], dtype=float),
            hub_probabilities,
            outside_probabilities,
            self.rng,
            deterministic,
        )
        expected_hub_counts = (expected_od * multiplier) @ hub_probabilities
        expected_hub_energy = (expected_od * multiplier * mean_energy) @ hub_probabilities
        arrivals = demand.hub_counts
        requested_energy = demand.hub_energy
        queue_transition = transition_fluid_queue(
            self.previous_wait,
            self.queued_energy_kwh,
            arrivals,
            requested_energy,
            self.service,
            self.dt,
        )
        waits = queue_transition.residual_wait_next_hours
        admitted_vehicles = queue_transition.admitted_vehicles
        admitted_energy = queue_transition.admitted_energy_kwh
        admission_pressure = np.divide(
            queue_transition.total_pending_vehicles,
            self.service,
            out=np.zeros_like(admitted_vehicles),
            where=self.service > 0.0,
        )
        service_utilization = np.divide(
            admitted_vehicles,
            self.service,
            out=np.zeros_like(admitted_vehicles),
            where=self.service > 0.0,
        )

        pv_available, grid_prices, grid_caps = self._current_energy_profiles(index)
        dispatches = self._dispatch(admitted_energy, pv_available, grid_prices, grid_caps, index)
        served_energy = np.asarray([result.served_energy for result in dispatches])
        unmet = np.asarray([result.unmet_energy for result in dispatches])
        next_battery = np.asarray([result.next_battery_energy for result in dispatches])
        revenue = np.zeros(self.n_hubs)
        operating_cost = np.zeros(self.n_hubs)
        served_counts = np.zeros(self.n_hubs)
        profits = np.zeros(self.n_hubs)
        for hub, result in enumerate(dispatches):
            revenue[hub], operating_cost[hub], served_counts[hub], profits[hub] = profit_components(
                prices[hub],
                admitted_energy[hub],
                admitted_vehicles[hub],
                result.served_energy,
                result.energy_cost,
                float(self.config["hubs"]["operating_cost_per_request"][hub]),
            )
        admitted_full_service_ratio = np.divide(
            served_counts,
            admitted_vehicles,
            out=np.ones_like(served_counts),
            where=admitted_vehicles > 0.0,
        )
        pending_full_service_ratio = np.divide(
            served_counts,
            queue_transition.total_pending_vehicles,
            out=np.ones_like(served_counts),
            where=queue_transition.total_pending_vehicles > 0.0,
        )
        total_pending = float(queue_transition.total_pending_vehicles.sum())
        full_service_ratio_total = (
            float(served_counts.sum()) / total_pending if total_pending > 0.0 else 1.0
        )

        wait_limits = np.asarray(self.config["hubs"]["max_wait_hours"], dtype=float)
        wait_excess = np.maximum(waits - wait_limits, 0.0)
        weights = np.asarray(self.config["hubs"]["welfare_weights"], dtype=float)
        weighted_welfare = float(weights @ profits)
        raw_reward = weighted_welfare - float(self.config["reward"]["wait_penalty"]) * float(
            wait_excess.sum()
        ) - float(self.config["reward"]["unmet_penalty"]) * float(unmet.sum())
        reward = raw_reward * float(self.config["reward"].get("scale", 0.01))

        inference_kwargs = {
            "observed_outside_counts": demand.od_outside_counts,
            "candidate_mask": candidate_mask,
        }
        can_update = update_inference and self.outside_mode == "inferred"
        if can_update:
            inference = self.estimator.update(
                costs,
                demand.od_total_counts,
                arrivals,
                **inference_kwargs,
            )
            inference_loss = inference.loss
            inference_instantaneous_loss = inference.instantaneous_loss
            inference_gradient = inference.gradient
            predicted_counts = inference.predicted_counts
            predicted_outside = inference.predicted_outside_counts
        else:
            inference_instantaneous_loss, inference_gradient, predicted_counts = self.estimator.loss_and_gradient(
                costs,
                demand.od_total_counts,
                arrivals,
                **inference_kwargs,
            )
            inference_loss = inference_instantaneous_loss
            _, _, estimate_outside_probabilities = self.estimator.predict(
                costs, demand.od_total_counts, candidate_mask=candidate_mask
            )
            predicted_outside = demand.od_total_counts * estimate_outside_probabilities

        total_realized = float(demand.od_total_counts.sum())
        outside_share = demand.outside_count / total_realized if total_realized > 0.0 else 0.0
        route_weight = demand.od_hub_counts
        total_hub_arrivals = float(route_weight.sum())
        weighted_access = (
            float(np.sum(route_weight * routes) / total_hub_arrivals)
            if total_hub_arrivals > 0.0
            else 0.0
        )
        weighted_detour = (
            float(np.sum(route_weight * detours) / total_hub_arrivals)
            if total_hub_arrivals > 0.0
            else 0.0
        )
        access_by_hub = np.divide(
            np.sum(route_weight * routes, axis=0),
            arrivals,
            out=np.zeros(self.n_hubs),
            where=arrivals > 0.0,
        )
        detour_by_hub = np.divide(
            np.sum(route_weight * detours, axis=0),
            arrivals,
            out=np.zeros(self.n_hubs),
            where=arrivals > 0.0,
        )

        pv_to_ev = np.asarray([result.pv_to_ev for result in dispatches])
        pv_to_battery = np.asarray([result.pv_to_battery for result in dispatches])
        pv_curtailed = np.asarray([result.pv_curtailed for result in dispatches])
        battery_to_ev = np.asarray([result.battery_to_ev for result in dispatches])
        grid_to_ev = np.asarray([result.grid_to_ev for result in dispatches])
        grid_to_battery = np.asarray([result.grid_to_battery for result in dispatches])
        battery_charge = pv_to_battery + grid_to_battery
        battery_throughput = battery_charge + battery_to_ev
        grid_import = grid_to_ev + grid_to_battery
        grid_utilization = np.divide(
            grid_import,
            grid_caps,
            out=np.zeros_like(grid_import),
            where=grid_caps > 0.0,
        )
        pv_used = pv_to_ev + pv_to_battery
        pv_utilization = np.divide(
            pv_used,
            pv_available,
            out=np.zeros_like(pv_used),
            where=pv_available > 0.0,
        )
        energy_balance_by_hub = np.asarray([
            max(
                abs(pv_available[hub] - pv_to_ev[hub] - pv_to_battery[hub] - pv_curtailed[hub]),
                abs(admitted_energy[hub] - served_energy[hub] - unmet[hub]),
                abs(served_energy[hub] - pv_to_ev[hub] - battery_to_ev[hub] - grid_to_ev[hub]),
                abs(
                    next_battery[hub]
                    - self.battery_energy[hub]
                    - float(self.config["hubs"]["eta_charge"][hub]) * battery_charge[hub]
                    + battery_to_ev[hub] / float(self.config["hubs"]["eta_discharge"][hub])
                ),
            )
            for hub in range(self.n_hubs)
        ])

        outside_error = self.estimator.cost - true_outside
        info: dict[str, Any] = {
            "period": index,
            "timestamp": self.scenario.timestamps[index].isoformat(),
            "hub_ids": self.hub_ids,
            "od_ids": self.od_ids,
            "episode_index": self.episode_index,
            "demand_multiplier": multiplier,
            "prices": prices.copy(),
            "expected_od_demand": expected_od * multiplier,
            "realized_od_demand": demand.od_total_counts.copy(),
            "expected": expected_hub_counts.copy(),
            "expected_demand": expected_hub_counts,
            "expected_energy": expected_hub_energy,
            "realized": arrivals.copy(),
            "realized_demand": arrivals.copy(),
            "realized_arrivals": arrivals.copy(),
            "historical_equivalent_vehicles": queue_transition.historical_equivalent_vehicles.copy(),
            "pending_vehicles": queue_transition.total_pending_vehicles.copy(),
            "admission_ratio": queue_transition.admission_ratio.copy(),
            "admission_pressure": admission_pressure,
            "admitted_vehicles": admitted_vehicles.copy(),
            "requested_energy": requested_energy.copy(),
            "queued_energy_start_kwh": queue_transition.queued_energy_start_kwh.copy(),
            "pending_energy_kwh": queue_transition.pending_energy_kwh.copy(),
            "admitted_energy_kwh": admitted_energy.copy(),
            "queued_energy_next_kwh": queue_transition.queued_energy_next_kwh.copy(),
            "queue_vehicle_conservation_error": queue_transition.vehicle_conservation_error.copy(),
            "queue_energy_conservation_error_kwh": queue_transition.energy_conservation_error_kwh.copy(),
            "served_energy": served_energy,
            "unmet": unmet,
            "unmet_energy": unmet.copy(),
            "outside_count": float(demand.outside_count),
            "outside_count_by_od": demand.od_outside_counts.copy(),
            "outside_share": float(outside_share),
            "hub_probabilities": hub_probabilities.copy(),
            "outside_probabilities": outside_probabilities.copy(),
            "od_hub_probabilities": hub_probabilities.copy(),
            "od_hub_realized_counts": demand.od_hub_counts.copy(),
            "od_hub_realized_energy": demand.od_hub_energy.copy(),
            "candidate_mask": candidate_mask.copy(),
            "wait": waits.copy(),
            "wait_violation": wait_excess.copy(),
            "wait_excess": wait_excess.copy(),
            "service_capacity": self.service.copy(),
            "service_utilization": service_utilization,
            "admitted_full_service_ratio": admitted_full_service_ratio,
            "pending_full_service_ratio": pending_full_service_ratio,
            "full_service_request_ratio_total": full_service_ratio_total,
            "route_times": routes.copy(),
            "detours": detours.copy(),
            "access_time_weighted_hours": weighted_access,
            "detour_time_weighted_hours": weighted_detour,
            "access_time_weighted_by_hub_hours": access_by_hub,
            "detour_time_weighted_by_hub_hours": detour_by_hub,
            "pv_available": pv_available.copy(),
            "pv_to_ev": pv_to_ev,
            "pv_to_battery": pv_to_battery,
            "pv_curtailed": pv_curtailed,
            "pv_utilization": pv_utilization,
            "battery_to_ev": battery_to_ev,
            "grid_to_ev": grid_to_ev,
            "grid_to_battery": grid_to_battery,
            "grid_import": grid_import,
            "grid_limit": grid_caps.copy(),
            "grid_utilization": grid_utilization,
            "grid_price": float(np.mean(grid_prices)),
            "grid_prices": grid_prices.copy(),
            "battery_energy_before": self.battery_energy.copy(),
            "battery_energy": next_battery.copy(),
            "soc": next_battery / self.battery_capacity,
            "battery_charge": battery_charge,
            "battery_discharge": battery_to_ev.copy(),
            "battery_throughput": battery_throughput,
            "dispatch_mode": [result.mode for result in dispatches],
            "dispatch_objective": np.asarray([result.objective for result in dispatches]),
            "energy_cost": np.asarray([result.energy_cost for result in dispatches]),
            "energy_balance_error_by_hub": energy_balance_by_hub,
            "energy_balance_error": float(np.max(energy_balance_by_hub)),
            "revenue": revenue,
            "operating_cost": operating_cost,
            "served_count": served_counts,
            "profit": profits,
            "weighted_hub_profit_welfare": weighted_welfare,
            "welfare": weighted_welfare,
            "raw_reward": raw_reward,
            "scaled_reward": reward,
            "reward": reward,
            "inference_objective": self.estimator.objective,
            "inference_loss": inference_loss,
            "inference_nll": inference_loss,
            "inference_instantaneous_loss": inference_instantaneous_loss,
            "inference_gradient": inference_gradient,
            "predicted_demand_under_estimate": predicted_counts,
            "predicted_outside_under_estimate": predicted_outside,
            "outside_cost_estimate": self.estimator.cost,
            "outside_cost_error": outside_error,
            "outside_cost_error_for_evaluation_only": outside_error,
            "outside_cost_error_is_evaluation_only": True,
            "true_outside_cost_for_evaluation_only": true_outside,
            "outside_mode": self.outside_mode,
        }
        self.previous_demand = arrivals.copy()
        self.previous_wait = waits.copy()
        self.queued_energy_kwh = queue_transition.queued_energy_next_kwh.copy()
        self.previous_price = prices.copy()
        self.battery_energy = next_battery
        self.t += 1
        done = self.t >= self.periods
        observations = self._observations()
        return StepOutput(observations, self._global_state(observations), reward, done, info)
