"""Projected analytical inference for the shared hidden outside-option cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .system_model import expected_demand, multinomial_logit


@dataclass(frozen=True)
class InferenceResult:
    estimate_before: float
    estimate_after: float
    loss: float
    gradient: float
    predicted_counts: np.ndarray
    predicted_outside_counts: np.ndarray
    outside_probabilities: np.ndarray
    instantaneous_loss: float


class OutsideOptionEstimator:
    def __init__(
        self,
        initial_cost: float,
        learning_rate: float,
        min_cost: float,
        max_cost: float,
        inverse_sensitivity: float,
        objective: str = "squared_hub_counts",
        rolling_loss_window: int = 1,
    ) -> None:
        if min_cost >= max_cost:
            raise ValueError("outside-option bounds are invalid")
        if objective not in {"squared_hub_counts", "outside_nll"}:
            raise ValueError(f"unknown inference objective: {objective}")
        if rolling_loss_window < 1:
            raise ValueError("rolling_loss_window must be positive")
        self.cost = float(np.clip(initial_cost, min_cost, max_cost))
        self.learning_rate = float(learning_rate)
        self.min_cost = float(min_cost)
        self.max_cost = float(max_cost)
        self.inverse_sensitivity = float(inverse_sensitivity)
        self.objective = objective
        self.rolling_loss_window = int(rolling_loss_window)
        self.loss_history: list[float] = []

    def predict(
        self,
        costs: np.ndarray,
        od_counts: np.ndarray,
        cost: float | None = None,
        candidate_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        estimate = self.cost if cost is None else float(cost)
        hub_probabilities, outside_probabilities = multinomial_logit(
            costs, estimate, self.inverse_sensitivity, candidate_mask
        )
        predicted_counts, _ = expected_demand(
            od_counts,
            np.ones_like(np.asarray(od_counts, dtype=float)),
            hub_probabilities,
        )
        return predicted_counts, hub_probabilities, outside_probabilities

    def loss_and_gradient(
        self,
        costs: np.ndarray,
        od_counts: np.ndarray,
        observed_counts: np.ndarray,
        cost: float | None = None,
        *,
        observed_outside_counts: np.ndarray | float | None = None,
        candidate_mask: np.ndarray | None = None,
    ) -> tuple[float, float, np.ndarray]:
        predicted, hub_probabilities, outside_probabilities = self.predict(
            costs, od_counts, cost, candidate_mask
        )
        counts = np.asarray(od_counts, dtype=float)
        if self.objective == "squared_hub_counts":
            observed = np.asarray(observed_counts, dtype=float)
            errors = observed - predicted
            n_hubs = predicted.size
            loss = float(np.sum(errors**2) / (2.0 * n_hubs))
            derivatives = self.inverse_sensitivity * np.sum(
                counts[:, None] * hub_probabilities * outside_probabilities[:, None],
                axis=0,
            )
            gradient = float(-np.sum(errors * derivatives) / n_hubs)
            return loss, gradient, predicted

        if observed_outside_counts is None:
            raise ValueError("outside_nll requires observed outside counts")
        total = float(np.sum(counts))
        if total <= 0.0:
            return 0.0, 0.0, predicted
        observed_outside = np.asarray(observed_outside_counts, dtype=float)
        epsilon = np.finfo(float).eps
        if observed_outside.ndim == 0 or observed_outside.size == 1:
            observed_total = float(observed_outside.reshape(-1)[0])
            probability = float(counts @ outside_probabilities / total)
            probability = float(np.clip(probability, epsilon, 1.0 - epsilon))
            loss = -(
                observed_total * np.log(probability)
                + (total - observed_total) * np.log1p(-probability)
            ) / total
            probability_derivative = float(
                counts @ (-self.inverse_sensitivity * outside_probabilities * (1.0 - outside_probabilities))
                / total
            )
            gradient = -(
                observed_total / probability
                - (total - observed_total) / (1.0 - probability)
            ) * probability_derivative / total
            return float(loss), float(gradient), predicted

        observed_outside = observed_outside.reshape(-1)
        if observed_outside.shape != counts.shape:
            raise ValueError("per-OD outside counts must match od_counts")
        probabilities = np.clip(outside_probabilities, epsilon, 1.0 - epsilon)
        loss = -np.sum(
            observed_outside * np.log(probabilities)
            + (counts - observed_outside) * np.log1p(-probabilities)
        ) / total
        gradient = self.inverse_sensitivity * np.sum(
            observed_outside - counts * outside_probabilities
        ) / total
        return float(loss), float(gradient), predicted

    def update(
        self,
        costs: np.ndarray,
        od_counts: np.ndarray,
        observed_counts: np.ndarray,
        *,
        observed_outside_counts: np.ndarray | float | None = None,
        candidate_mask: np.ndarray | None = None,
    ) -> InferenceResult:
        before = self.cost
        instantaneous_loss, gradient, predicted = self.loss_and_gradient(
            costs,
            od_counts,
            observed_counts,
            observed_outside_counts=observed_outside_counts,
            candidate_mask=candidate_mask,
        )
        self.cost = float(np.clip(before - self.learning_rate * gradient, self.min_cost, self.max_cost))
        self.loss_history.append(float(instantaneous_loss))
        self.loss_history = self.loss_history[-self.rolling_loss_window :]
        rolling_loss = float(np.mean(self.loss_history))
        _, _, outside_probabilities = self.predict(costs, od_counts, before, candidate_mask)
        predicted_outside = np.asarray(od_counts, dtype=float) * outside_probabilities
        return InferenceResult(
            before,
            self.cost,
            rolling_loss,
            gradient,
            predicted,
            predicted_outside,
            outside_probabilities,
            float(instantaneous_loss),
        )

    def state_dict(self) -> dict[str, Any]:
        return {"cost": self.cost, "loss_history": list(self.loss_history)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.cost = float(np.clip(state["cost"], self.min_cost, self.max_cost))
        self.loss_history = [float(value) for value in state.get("loss_history", [])]
        self.loss_history = self.loss_history[-self.rolling_loss_window :]
