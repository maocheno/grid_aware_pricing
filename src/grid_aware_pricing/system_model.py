"""Traffic, user choice, queueing, and profit equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import truncnorm


@dataclass(frozen=True)
class ChoiceResult:
    hub_probabilities: np.ndarray
    outside_probabilities: np.ndarray
    expected_counts: np.ndarray
    expected_energy: np.ndarray


@dataclass(frozen=True)
class YorkDemandRealization:
    hub_counts: np.ndarray
    hub_energy: np.ndarray
    outside_count: float
    od_hub_counts: np.ndarray
    od_hub_energy: np.ndarray
    od_outside_counts: np.ndarray
    od_total_counts: np.ndarray


@dataclass(frozen=True)
class FluidQueueTransition:
    historical_equivalent_vehicles: np.ndarray
    total_pending_vehicles: np.ndarray
    admission_ratio: np.ndarray
    admitted_vehicles: np.ndarray
    requested_energy_kwh: np.ndarray
    queued_energy_start_kwh: np.ndarray
    pending_energy_kwh: np.ndarray
    admitted_energy_kwh: np.ndarray
    queued_energy_next_kwh: np.ndarray
    residual_wait_next_hours: np.ndarray
    vehicle_conservation_error: np.ndarray
    energy_conservation_error_kwh: np.ndarray


def bpr_travel_times(
    free_flow_times: np.ndarray,
    flows: np.ndarray,
    capacities: np.ndarray,
    a: float = 0.15,
    b: float = 4.0,
) -> np.ndarray:
    free_flow_times = np.asarray(free_flow_times, dtype=float)
    flows = np.asarray(flows, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    if np.any(capacities <= 0):
        raise ValueError("link capacities must be positive")
    return free_flow_times * (1.0 + a * np.power(np.maximum(flows, 0.0) / capacities, b))


def route_times(link_times: np.ndarray, routes: list[list[list[int]]]) -> np.ndarray:
    link_times = np.asarray(link_times, dtype=float)
    return np.asarray(
        [[float(np.sum(link_times[np.asarray(route, dtype=int)])) for route in od_routes] for od_routes in routes],
        dtype=float,
    )


def generalized_costs(
    prices: np.ndarray,
    energy_per_od: np.ndarray,
    route_time_matrix: np.ndarray,
    previous_wait: np.ndarray,
    value_of_time: float,
) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    energy_per_od = np.asarray(energy_per_od, dtype=float)
    route_time_matrix = np.asarray(route_time_matrix, dtype=float)
    previous_wait = np.asarray(previous_wait, dtype=float)
    return energy_per_od[:, None] * prices[None, :] + value_of_time * (
        route_time_matrix + previous_wait[None, :]
    )


def multinomial_logit(
    costs: np.ndarray,
    outside_cost: float,
    inverse_sensitivity: float,
    candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.asarray(costs, dtype=float)
    hub_logits = -inverse_sensitivity * costs
    if candidate_mask is not None:
        mask = np.asarray(candidate_mask, dtype=bool)
        if mask.shape != costs.shape:
            raise ValueError("candidate_mask must match costs")
        hub_logits = np.where(mask, hub_logits, -np.inf)
    outside_logits = np.full((costs.shape[0], 1), -inverse_sensitivity * outside_cost)
    logits = np.concatenate([outside_logits, hub_logits], axis=1)
    log_denominator = logsumexp(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits - log_denominator)
    hub_probabilities = probabilities[:, 1:]
    if candidate_mask is not None:
        hub_probabilities = np.where(mask, hub_probabilities, 0.0)
    return hub_probabilities, probabilities[:, 0]


def expected_demand(
    od_counts: np.ndarray,
    energy_per_od: np.ndarray,
    hub_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(od_counts, dtype=float)
    energy = np.asarray(energy_per_od, dtype=float)
    probabilities = np.asarray(hub_probabilities, dtype=float)
    return counts @ probabilities, (counts * energy) @ probabilities


def choice_result(
    costs: np.ndarray,
    outside_cost: float,
    inverse_sensitivity: float,
    od_counts: np.ndarray,
    energy_per_od: np.ndarray,
) -> ChoiceResult:
    hub_probabilities, outside_probabilities = multinomial_logit(costs, outside_cost, inverse_sensitivity)
    counts, energy = expected_demand(od_counts, energy_per_od, hub_probabilities)
    return ChoiceResult(hub_probabilities, outside_probabilities, counts, energy)


def realize_demand(
    od_counts: np.ndarray,
    energy_per_od: np.ndarray,
    hub_probabilities: np.ndarray,
    outside_probabilities: np.ndarray,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if deterministic:
        counts, energy = expected_demand(od_counts, energy_per_od, hub_probabilities)
        outside = float(np.asarray(od_counts, dtype=float) @ outside_probabilities)
        return counts, energy, np.asarray(outside)
    n_hubs = hub_probabilities.shape[1]
    realized_counts = np.zeros(n_hubs, dtype=float)
    realized_energy = np.zeros(n_hubs, dtype=float)
    outside_count = 0.0
    for count, energy, hub_probs, outside_prob in zip(
        od_counts, energy_per_od, hub_probabilities, outside_probabilities
    ):
        probabilities = np.concatenate([[outside_prob], hub_probs])
        probabilities = probabilities / probabilities.sum()
        sampled = rng.multinomial(int(count), probabilities)
        outside_count += sampled[0]
        realized_counts += sampled[1:]
        realized_energy += sampled[1:] * energy
    return realized_counts, realized_energy, np.asarray(outside_count)


def realize_york_demand(
    expected_od_counts: np.ndarray,
    demand_multiplier: float,
    energy_mean: np.ndarray,
    energy_std: np.ndarray,
    energy_min: np.ndarray,
    energy_max: np.ndarray,
    hub_probabilities: np.ndarray,
    outside_probabilities: np.ndarray,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> YorkDemandRealization:
    expected = np.asarray(expected_od_counts, dtype=float) * float(demand_multiplier)
    means = np.asarray(energy_mean, dtype=float)
    stds = np.asarray(energy_std, dtype=float)
    minimums = np.asarray(energy_min, dtype=float)
    maximums = np.asarray(energy_max, dtype=float)
    hub_probs = np.asarray(hub_probabilities, dtype=float)
    outside_probs = np.asarray(outside_probabilities, dtype=float)
    n_ods, n_hubs = hub_probs.shape
    if any(array.shape != (n_ods,) for array in (expected, means, stds, minimums, maximums, outside_probs)):
        raise ValueError("York OD demand and energy vectors must match probabilities")

    od_hub_counts = np.zeros((n_ods, n_hubs), dtype=float)
    od_hub_energy = np.zeros_like(od_hub_counts)
    od_outside_counts = np.zeros(n_ods, dtype=float)
    if deterministic:
        od_hub_counts = expected[:, None] * hub_probs
        od_hub_energy = od_hub_counts * means[:, None]
        od_outside_counts = expected * outside_probs
        od_total_counts = expected
    else:
        od_total_counts = rng.poisson(np.maximum(expected, 0.0)).astype(float)
        for od in range(n_ods):
            probabilities = np.concatenate([[outside_probs[od]], hub_probs[od]])
            probabilities = probabilities / probabilities.sum()
            allocated = rng.multinomial(int(od_total_counts[od]), probabilities)
            od_outside_counts[od] = allocated[0]
            od_hub_counts[od] = allocated[1:]
            if stds[od] <= 0.0:
                sampled_energy = None
            else:
                lower = (minimums[od] - means[od]) / stds[od]
                upper = (maximums[od] - means[od]) / stds[od]
                sampled_energy = truncnorm(lower, upper, loc=means[od], scale=stds[od])
            for hub, count in enumerate(allocated[1:]):
                if count <= 0:
                    continue
                if sampled_energy is None:
                    energies = np.full(int(count), np.clip(means[od], minimums[od], maximums[od]))
                else:
                    energies = sampled_energy.rvs(size=int(count), random_state=rng)
                od_hub_energy[od, hub] = float(np.sum(energies))
    return YorkDemandRealization(
        hub_counts=od_hub_counts.sum(axis=0),
        hub_energy=od_hub_energy.sum(axis=0),
        outside_count=float(od_outside_counts.sum()),
        od_hub_counts=od_hub_counts,
        od_hub_energy=od_hub_energy,
        od_outside_counts=od_outside_counts,
        od_total_counts=od_total_counts,
    )


def service_capacity(chargers: np.ndarray, dt_hours: float, service_time_hours: np.ndarray) -> np.ndarray:
    return np.asarray(chargers, dtype=float) * dt_hours / np.asarray(service_time_hours, dtype=float)


def waiting_time(arrivals: np.ndarray, service: np.ndarray, service_time_hours: np.ndarray) -> np.ndarray:
    arrivals = np.asarray(arrivals, dtype=float)
    service = np.asarray(service, dtype=float)
    tau = np.asarray(service_time_hours, dtype=float)
    if np.any(service <= 0):
        raise ValueError("service capacities must be positive")
    return tau * np.maximum(arrivals / service - 1.0, 0.0)


def transition_fluid_queue(
    previous_wait_hours: np.ndarray,
    queued_energy_kwh: np.ndarray,
    arrivals_vehicles: np.ndarray,
    requested_energy_kwh: np.ndarray,
    service_capacity_vehicles: np.ndarray,
    period_hours: float,
) -> FluidQueueTransition:
    previous_wait, queued_energy, arrivals, requested_energy, service = np.broadcast_arrays(
        np.asarray(previous_wait_hours, dtype=float),
        np.asarray(queued_energy_kwh, dtype=float),
        np.asarray(arrivals_vehicles, dtype=float),
        np.asarray(requested_energy_kwh, dtype=float),
        np.asarray(service_capacity_vehicles, dtype=float),
    )
    if period_hours <= 0.0:
        raise ValueError("period_hours must be positive")
    if np.any(service <= 0.0):
        raise ValueError("service capacities must be positive")
    if np.any(previous_wait < 0.0):
        raise ValueError("previous waits must be nonnegative")
    if np.any(queued_energy < 0.0):
        raise ValueError("queued energy must be nonnegative")
    if np.any(arrivals < 0.0):
        raise ValueError("arrivals must be nonnegative")
    if np.any(requested_energy < 0.0):
        raise ValueError("requested energy must be nonnegative")

    historical = service / period_hours * previous_wait
    pending_vehicles = historical + arrivals
    admission_ratio = np.divide(
        service,
        pending_vehicles,
        out=np.zeros_like(pending_vehicles),
        where=pending_vehicles > 0.0,
    )
    admission_ratio = np.minimum(admission_ratio, 1.0)
    admitted_vehicles = admission_ratio * pending_vehicles

    pending_energy = queued_energy + requested_energy
    admitted_energy = admission_ratio * pending_energy
    queued_energy_next = pending_energy - admitted_energy
    residual_wait_next = np.maximum(
        previous_wait + period_hours * (arrivals / service - 1.0),
        0.0,
    )

    leftover_vehicles = np.maximum(pending_vehicles - service, 0.0)
    equivalent_wait = period_hours / service * leftover_vehicles
    return FluidQueueTransition(
        historical_equivalent_vehicles=historical,
        total_pending_vehicles=pending_vehicles,
        admission_ratio=admission_ratio,
        admitted_vehicles=admitted_vehicles,
        requested_energy_kwh=requested_energy,
        queued_energy_start_kwh=queued_energy,
        pending_energy_kwh=pending_energy,
        admitted_energy_kwh=admitted_energy,
        queued_energy_next_kwh=queued_energy_next,
        residual_wait_next_hours=residual_wait_next,
        vehicle_conservation_error=residual_wait_next - equivalent_wait,
        energy_conservation_error_kwh=(
            pending_energy - admitted_energy - queued_energy_next
        ),
    )


def profit_components(
    price: float,
    requested_energy: float,
    requested_count: float,
    served_energy: float,
    energy_cost: float,
    operating_cost_per_request: float,
) -> tuple[float, float, float, float]:
    served_fraction = served_energy / requested_energy if requested_energy > 0.0 else 1.0
    served_count = requested_count * served_fraction
    revenue = price * served_energy
    operating_cost = operating_cost_per_request * served_count
    profit = revenue - energy_cost - operating_cost
    return revenue, operating_cost, served_count, profit
