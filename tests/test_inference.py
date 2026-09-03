import numpy as np

from grid_aware_pricing.inference import OutsideOptionEstimator
from grid_aware_pricing.system_model import multinomial_logit


def test_analytical_gradient_matches_finite_difference():
    estimator = OutsideOptionEstimator(8.0, 0.1, 1.0, 20.0, 0.4)
    costs = np.array([[5.0, 7.0], [9.0, 6.0], [8.0, 8.5]])
    od_counts = np.array([12.0, 8.0, 10.0])
    observed = np.array([9.0, 7.0])
    _, gradient, _ = estimator.loss_and_gradient(costs, od_counts, observed)
    epsilon = 1e-5
    plus = estimator.loss_and_gradient(costs, od_counts, observed, cost=estimator.cost + epsilon)[0]
    minus = estimator.loss_and_gradient(costs, od_counts, observed, cost=estimator.cost - epsilon)[0]
    finite_difference = (plus - minus) / (2 * epsilon)
    assert np.isclose(gradient, finite_difference, rtol=1e-5, atol=1e-6)


def test_higher_outside_cost_increases_hub_share():
    costs = np.array([[5.0, 7.0], [9.0, 6.0]])
    low, _ = multinomial_logit(costs, 3.0, 0.5)
    high, _ = multinomial_logit(costs, 10.0, 0.5)
    assert np.all(high.sum(axis=1) > low.sum(axis=1))


def test_update_direction_and_projection():
    estimator = OutsideOptionEstimator(5.0, 10.0, 2.0, 6.0, 0.5)
    costs = np.full((2, 2), 5.0)
    counts = np.array([20.0, 20.0])
    hub, _ = multinomial_logit(costs, 10.0, 0.5)
    observed = counts @ hub
    result = estimator.update(costs, counts, observed)
    assert result.estimate_after == 6.0


def test_outside_nll_gradient_matches_finite_difference():
    estimator = OutsideOptionEstimator(
        13.5, 0.04, 10.0, 22.0, 0.55, objective="outside_nll", rolling_loss_window=3
    )
    costs = np.array([[10.0, 12.0, 14.0], [11.0, 15.0, 13.0]])
    od_counts = np.array([35.0, 22.0])
    observed_hubs = np.array([18.0, 12.0, 9.0])
    observed_outside = np.array([10.0, 8.0])
    _, gradient, _ = estimator.loss_and_gradient(
        costs,
        od_counts,
        observed_hubs,
        observed_outside_counts=observed_outside,
    )
    epsilon = 1e-5
    plus = estimator.loss_and_gradient(
        costs,
        od_counts,
        observed_hubs,
        cost=estimator.cost + epsilon,
        observed_outside_counts=observed_outside,
    )[0]
    minus = estimator.loss_and_gradient(
        costs,
        od_counts,
        observed_hubs,
        cost=estimator.cost - epsilon,
        observed_outside_counts=observed_outside,
    )[0]
    assert np.isclose(gradient, (plus - minus) / (2 * epsilon), rtol=1e-5, atol=1e-7)


def test_outside_nll_updates_from_13_5_toward_16_5():
    estimator = OutsideOptionEstimator(13.5, 0.04, 10.0, 22.0, 0.55, objective="outside_nll")
    costs = np.array([[12.0, 14.0], [13.0, 15.0], [11.0, 16.0]])
    od_counts = np.array([40.0, 30.0, 20.0])
    true_hub, true_outside = multinomial_logit(costs, 16.5, 0.55)
    result = estimator.update(
        costs,
        od_counts,
        od_counts @ true_hub,
        observed_outside_counts=od_counts * true_outside,
    )
    assert 13.5 < result.estimate_after < 16.5
    assert result.predicted_outside_counts.shape == od_counts.shape
