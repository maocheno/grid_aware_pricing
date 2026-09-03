import numpy as np
import pytest

from grid_aware_pricing.system_model import transition_fluid_queue


def test_overload_carries_wait_and_energy_forward():
    transition = transition_fluid_queue(
        previous_wait_hours=np.array([0.0]),
        queued_energy_kwh=np.array([0.0]),
        arrivals_vehicles=np.array([12.0]),
        requested_energy_kwh=np.array([300.0]),
        service_capacity_vehicles=np.array([8.0]),
        period_hours=1.0,
    )
    assert transition.historical_equivalent_vehicles[0] == 0.0
    assert transition.total_pending_vehicles[0] == 12.0
    assert transition.admission_ratio[0] == pytest.approx(2.0 / 3.0)
    assert transition.admitted_vehicles[0] == pytest.approx(8.0)
    assert transition.admitted_energy_kwh[0] == pytest.approx(200.0)
    assert transition.queued_energy_next_kwh[0] == pytest.approx(100.0)
    assert transition.residual_wait_next_hours[0] == pytest.approx(0.5)


def test_partial_recovery_serves_historical_backlog_and_new_arrivals():
    transition = transition_fluid_queue(
        previous_wait_hours=np.array([0.5]),
        queued_energy_kwh=np.array([100.0]),
        arrivals_vehicles=np.array([2.0]),
        requested_energy_kwh=np.array([50.0]),
        service_capacity_vehicles=np.array([8.0]),
        period_hours=1.0,
    )
    assert transition.historical_equivalent_vehicles[0] == pytest.approx(4.0)
    assert transition.total_pending_vehicles[0] == pytest.approx(6.0)
    assert transition.admission_ratio[0] == 1.0
    assert transition.admitted_energy_kwh[0] == pytest.approx(150.0)
    assert transition.queued_energy_next_kwh[0] == 0.0
    assert transition.residual_wait_next_hours[0] == 0.0


def test_empty_queue_has_zero_admission_and_zero_next_state():
    transition = transition_fluid_queue(
        np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.array([8.0, 12.0]), 1.0
    )
    assert np.all(transition.admission_ratio == 0.0)
    assert np.all(transition.admitted_energy_kwh == 0.0)
    assert np.all(transition.queued_energy_next_kwh == 0.0)
    assert np.all(transition.residual_wait_next_hours == 0.0)


def test_multi_hub_transition_preserves_vehicle_equivalence_and_energy():
    transition = transition_fluid_queue(
        previous_wait_hours=np.array([0.5, 0.0, 0.25]),
        queued_energy_kwh=np.array([100.0, 0.0, 60.0]),
        arrivals_vehicles=np.array([12.0, 4.0, 2.0]),
        requested_energy_kwh=np.array([300.0, 100.0, 40.0]),
        service_capacity_vehicles=np.array([8.0, 8.0, 12.0]),
        period_hours=1.0,
    )
    assert np.allclose(transition.vehicle_conservation_error, 0.0)
    assert np.allclose(transition.energy_conservation_error_kwh, 0.0)
    assert np.allclose(
        transition.pending_energy_kwh,
        transition.admitted_energy_kwh + transition.queued_energy_next_kwh,
    )
    leftover = np.maximum(
        transition.total_pending_vehicles - np.array([8.0, 8.0, 12.0]), 0.0
    )
    assert np.allclose(
        transition.residual_wait_next_hours,
        leftover / np.array([8.0, 8.0, 12.0]),
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("previous_wait_hours", -0.1, "previous waits"),
        ("queued_energy_kwh", -1.0, "queued energy"),
        ("arrivals_vehicles", -1.0, "arrivals"),
        ("requested_energy_kwh", -1.0, "requested energy"),
        ("service_capacity_vehicles", 0.0, "service capacities"),
    ],
)
def test_invalid_queue_inputs_are_rejected(field, value, error):
    values = {
        "previous_wait_hours": np.array([0.0]),
        "queued_energy_kwh": np.array([0.0]),
        "arrivals_vehicles": np.array([0.0]),
        "requested_energy_kwh": np.array([0.0]),
        "service_capacity_vehicles": np.array([8.0]),
    }
    values[field] = np.array([value])
    with pytest.raises(ValueError, match=error):
        transition_fluid_queue(period_hours=1.0, **values)


def test_nonpositive_period_is_rejected():
    with pytest.raises(ValueError, match="period_hours"):
        transition_fluid_queue(
            np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1), 0.0
        )
