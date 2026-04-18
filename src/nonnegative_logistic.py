from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin


class NonNegativeLogisticRegression(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
        self,
        *,
        max_iter: int = 5000,
        class_weight: str | dict[int, float] | None = None,
        random_state: int | None = None,
        C: float = 1.0,
        min_coef: float = 1e-6,
    ) -> None:
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.C = C
        self.min_coef = min_coef

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    def _build_sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones_like(y, dtype=float)

        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            total = float(len(y))
            weight_map = {
                int(label): total / (len(classes) * float(count))
                for label, count in zip(classes, counts)
            }
            return np.array([weight_map[int(label)] for label in y], dtype=float)

        if isinstance(self.class_weight, dict):
            return np.array([float(self.class_weight.get(int(label), 1.0)) for label in y], dtype=float)

        raise ValueError(f"Unsupported class_weight: {self.class_weight}")

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if X_array.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        if set(np.unique(y_array)) - {0.0, 1.0}:
            raise ValueError("y must be binary with labels 0 and 1.")

        n_samples, n_features = X_array.shape
        sample_weights = self._build_sample_weights(y_array)
        weight_sum = float(sample_weights.sum())
        regularization = 1.0 / max(float(self.C), 1e-8)

        positive_rate = float(np.clip(np.average(y_array, weights=sample_weights), 1e-5, 1 - 1e-5))
        initial_intercept = float(np.log(positive_rate / (1.0 - positive_rate)))
        initial_params = np.concatenate(
            [[initial_intercept], np.full(n_features, max(self.min_coef, 0.01), dtype=float)]
        )

        def objective_and_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = params[0]
            coef = params[1:]
            linear = intercept + X_array @ coef
            probabilities = np.clip(self._sigmoid(linear), 1e-9, 1.0 - 1e-9)

            loss_terms = -(
                y_array * np.log(probabilities) + (1.0 - y_array) * np.log(1.0 - probabilities)
            )
            weighted_loss = float(np.dot(sample_weights, loss_terms) / weight_sum)
            penalty = 0.5 * regularization * float(np.dot(coef, coef)) / n_samples
            loss = weighted_loss + penalty

            error = sample_weights * (probabilities - y_array) / weight_sum
            intercept_grad = float(error.sum())
            coef_grad = X_array.T @ error + (regularization / n_samples) * coef
            gradient = np.concatenate([[intercept_grad], coef_grad])
            return loss, gradient

        bounds = [(None, None)] + [(self.min_coef, None)] * n_features
        result = minimize(
            fun=lambda params: objective_and_gradient(params)[0],
            x0=initial_params,
            jac=lambda params: objective_and_gradient(params)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(self.max_iter)},
        )

        if not result.success:
            raise RuntimeError(f"Non-negative logistic regression failed to converge: {result.message}")

        self.intercept_ = np.array([float(result.x[0])], dtype=float)
        self.coef_ = np.array([result.x[1:]], dtype=float)
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = n_features
        return self

    def decision_function(self, X) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        return self.intercept_[0] + X_array @ self.coef_[0]

    def predict_proba(self, X) -> np.ndarray:
        linear = self.decision_function(X)
        probabilities = self._sigmoid(linear)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
