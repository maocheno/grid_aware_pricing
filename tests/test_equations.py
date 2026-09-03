import numpy as np

from grid_aware_pricing.system_model import (
    bpr_travel_times,
    generalized_costs,
    multinomial_logit,
    profit_components,
    route_times,
    service_capacity,
    waiting_time,
)


def test_bpr_routes_costs_and_queue_equations():
    links = bpr_travel_times(np.array([1.0, 2.0]), np.array([50.0, 100.0]), np.array([100.0, 100.0]), 0.15, 4.0)
    assert np.allclose(links, [1.009375, 2.3])
    routes = route_times(links, [[[0], [0, 1]], [[1], [0]]])
    assert np.allclose(routes, [[links[0], links.sum()], [links[1], links[0]]])
    costs = generalized_costs(np.array([0.2, 0.3]), np.array([10.0, 20.0]), routes, np.array([0.1, 0.2]), 5.0)
    assert np.isclose(costs[0, 0], 0.2 * 10 + 5 * (routes[0, 0] + 0.1))
    service = service_capacity(np.array([4, 2]), 1.0, np.array([0.5, 1.0]))
    assert np.allclose(service, [8, 2])
    waits = waiting_time(np.array([12, 1]), service, np.array([0.5, 1.0]))
    assert np.allclose(waits, [0.25, 0.0])


def test_stable_logit_and_delivered_energy_profit():
    hub, outside = multinomial_logit(np.array([[10000.0, 10001.0], [-10000.0, -9999.0]]), 10002.0, 20.0)
    assert np.all(np.isfinite(hub))
    assert np.all(np.isfinite(outside))
    assert np.allclose(hub.sum(axis=1) + outside, 1.0)
    revenue, operating, served_count, profit = profit_components(0.5, 100.0, 10.0, 40.0, 8.0, 2.0)
    assert revenue == 20.0
    assert operating == 8.0
    assert served_count == 4.0
    assert profit == 4.0
