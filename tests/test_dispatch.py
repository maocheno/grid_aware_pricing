import numpy as np

from grid_aware_pricing.dispatch import DispatchInputs, dispatch_energy


def make_inputs(**overrides):
    values = dict(
        requested_energy=80.0, pv_available=30.0, battery_energy=50.0,
        battery_capacity=100.0, min_soc=0.1, max_soc=0.9,
        charge_limit=40.0, discharge_limit=40.0, grid_cap=60.0,
        eta_charge=0.95, eta_discharge=0.9, grid_price=0.3,
        battery_cost=0.02, pv_cost=0.01, unmet_penalty=10.0,
        future_battery_value=0.0,
    )
    values.update(overrides)
    return DispatchInputs(**values)


def test_dispatch_balances_bounds_and_discharge_mode():
    inputs = make_inputs()
    result = dispatch_energy(inputs)
    result.validate(inputs)
    assert result.mode == "discharge"
    assert result.unmet_energy < 1e-7
    assert result.battery_to_ev > 0.0
    assert result.pv_to_battery + result.grid_to_battery < 1e-7


def test_dispatch_charge_mode_and_grid_cap_shortfall():
    inputs = make_inputs(
        requested_energy=10.0, pv_available=60.0, battery_energy=20.0,
        grid_cap=5.0, future_battery_value=1.0,
    )
    result = dispatch_energy(inputs)
    result.validate(inputs)
    assert result.mode == "charge"
    assert result.pv_to_battery > 0.0
    assert result.battery_to_ev < 1e-7

    constrained = make_inputs(
        requested_energy=200.0, pv_available=0.0, battery_energy=10.0,
        grid_cap=20.0, discharge_limit=0.0,
    )
    shortfall = dispatch_energy(constrained)
    shortfall.validate(constrained)
    assert np.isclose(shortfall.grid_to_ev, 20.0)
    assert np.isclose(shortfall.unmet_energy, 180.0)
