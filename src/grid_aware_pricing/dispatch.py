"""Linear-program dispatch enforcing the paper's PV, storage, and grid constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class DispatchInputs:
    requested_energy: float
    pv_available: float
    battery_energy: float
    battery_capacity: float
    min_soc: float
    max_soc: float
    charge_limit: float
    discharge_limit: float
    grid_cap: float
    eta_charge: float
    eta_discharge: float
    grid_price: float
    battery_cost: float
    pv_cost: float
    unmet_penalty: float
    future_battery_value: float


@dataclass(frozen=True)
class DispatchResult:
    pv_to_ev: float
    pv_to_battery: float
    pv_curtailed: float
    battery_to_ev: float
    grid_to_ev: float
    grid_to_battery: float
    served_energy: float
    unmet_energy: float
    next_battery_energy: float
    energy_cost: float
    objective: float
    mode: str
    success: bool

    def validate(self, inputs: DispatchInputs, tolerance: float = 1e-6) -> None:
        values = np.asarray([
            self.pv_to_ev, self.pv_to_battery, self.pv_curtailed,
            self.battery_to_ev, self.grid_to_ev, self.grid_to_battery,
            self.served_energy, self.unmet_energy,
        ])
        if np.any(values < -tolerance):
            raise ValueError("dispatch contains negative flow")
        if abs(inputs.pv_available - self.pv_to_ev - self.pv_to_battery - self.pv_curtailed) > tolerance:
            raise ValueError("PV balance violated")
        if abs(inputs.requested_energy - self.served_energy - self.unmet_energy) > tolerance:
            raise ValueError("requested-energy balance violated")
        if abs(self.served_energy - self.pv_to_ev - self.battery_to_ev - self.grid_to_ev) > tolerance:
            raise ValueError("served-energy balance violated")
        expected_battery = inputs.battery_energy + inputs.eta_charge * (
            self.pv_to_battery + self.grid_to_battery
        ) - self.battery_to_ev / inputs.eta_discharge
        if abs(expected_battery - self.next_battery_energy) > tolerance:
            raise ValueError("battery dynamics violated")
        if not inputs.min_soc * inputs.battery_capacity - tolerance <= self.next_battery_energy <= inputs.max_soc * inputs.battery_capacity + tolerance:
            raise ValueError("battery bounds violated")
        if self.pv_to_battery + self.grid_to_battery > inputs.charge_limit + tolerance:
            raise ValueError("charge limit violated")
        if self.battery_to_ev > inputs.discharge_limit + tolerance:
            raise ValueError("discharge limit violated")
        if self.grid_to_ev + self.grid_to_battery > inputs.grid_cap + tolerance:
            raise ValueError("grid cap violated")
        if (self.pv_to_battery + self.grid_to_battery) * self.battery_to_ev > tolerance:
            raise ValueError("simultaneous charging and discharging")


# pv_ev, pv_b, curtail, b_ev, grid_ev, grid_b, served, unmet, next_b
_N = 9


def _solve_mode(inputs: DispatchInputs, mode: str, feasibility_tolerance: float) -> DispatchResult:
    c = np.asarray([
        inputs.pv_cost,
        inputs.pv_cost,
        0.0,
        inputs.battery_cost,
        inputs.grid_price,
        inputs.grid_price,
        0.0,
        inputs.unmet_penalty,
        -inputs.future_battery_value,
    ])
    a_eq = np.zeros((4, _N), dtype=float)
    b_eq = np.asarray([inputs.pv_available, inputs.requested_energy, 0.0, -inputs.battery_energy])
    a_eq[0, [0, 1, 2]] = 1.0
    a_eq[1, [6, 7]] = 1.0
    a_eq[2, [0, 3, 4, 6]] = [1.0, 1.0, 1.0, -1.0]
    a_eq[3, [1, 3, 5, 8]] = [inputs.eta_charge, -1.0 / inputs.eta_discharge, inputs.eta_charge, -1.0]

    a_ub = np.zeros((3, _N), dtype=float)
    b_ub = np.asarray([inputs.charge_limit, inputs.discharge_limit, inputs.grid_cap])
    a_ub[0, [1, 5]] = 1.0
    a_ub[1, 3] = 1.0
    a_ub[2, [4, 5]] = 1.0

    upper = [None] * _N
    if mode == "charge":
        upper[3] = 0.0
    elif mode == "discharge":
        upper[1] = 0.0
        upper[5] = 0.0
    else:
        raise ValueError(f"unknown dispatch mode {mode}")
    bounds = [(0.0, upper[index]) for index in range(_N)]
    bounds[8] = (inputs.min_soc * inputs.battery_capacity, inputs.max_soc * inputs.battery_capacity)

    solution = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not solution.success:
        raise RuntimeError(f"dispatch {mode} mode failed: {solution.message}")
    x = np.maximum(solution.x, 0.0)
    energy_cost = (
        inputs.grid_price * (x[4] + x[5])
        + inputs.battery_cost * x[3]
        + inputs.pv_cost * (x[0] + x[1])
    )
    result = DispatchResult(
        pv_to_ev=x[0], pv_to_battery=x[1], pv_curtailed=x[2], battery_to_ev=x[3],
        grid_to_ev=x[4], grid_to_battery=x[5], served_energy=x[6], unmet_energy=x[7],
        next_battery_energy=x[8], energy_cost=float(energy_cost), objective=float(solution.fun),
        mode=mode, success=True,
    )
    result.validate(inputs, feasibility_tolerance)
    return result


def _tie_key(result: DispatchResult) -> tuple[float, float, float, float, int]:
    return (
        result.unmet_energy,
        result.grid_to_ev + result.grid_to_battery,
        result.pv_to_battery + result.grid_to_battery + result.battery_to_ev,
        result.pv_curtailed,
        0 if result.mode == "charge" else 1,
    )


def dispatch_energy(
    inputs: DispatchInputs,
    tie_tolerance: float = 1e-9,
    feasibility_tolerance: float = 1e-6,
) -> DispatchResult:
    charge = _solve_mode(inputs, "charge", feasibility_tolerance)
    discharge = _solve_mode(inputs, "discharge", feasibility_tolerance)
    if discharge.objective < charge.objective - tie_tolerance:
        return discharge
    if charge.objective < discharge.objective - tie_tolerance:
        return charge
    return min((charge, discharge), key=_tie_key)
