from typing import Self

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class QuantileEstimator(RegressorMixin, BaseEstimator):
    """Adapts one target to independent or joint quantile estimators.

    Attributes:
        estimator (BaseEstimator): Unfitted quantile-regression prototype.
        confidence_level (float): Desired coverage level.
        gamma (float): Miscoverage fraction assigned to the lower tail.
        estimators_ (list[BaseEstimator]): Fitted backend quantile estimators.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        confidence_level: float,
        gamma: float,
    ):
        """Initializes the backend adapter.

        Args:
            estimator (BaseEstimator): Quantile-regression estimator to clone.
            confidence_level (float): Desired coverage level.
            gamma (float): Miscoverage fraction assigned to the lower tail.
        """
        self.estimator = estimator
        self.confidence_level = confidence_level
        self.gamma = gamma

    def _configure_estimators(self) -> list[BaseEstimator]:
        """Clones and configures the backend quantile estimators.

        Returns:
            list[BaseEstimator]: Independent bound estimators or one joint-quantile
                estimator.

        Raises:
            TypeError: If the estimator backend is unsupported.
        """
        alpha = 1 - self.confidence_level
        quantiles = [
            quantile
            for quantile in (alpha * self.gamma, 1 - alpha * (1 - self.gamma))
            if quantile not in (0, 1)
        ]
        module = type(self.estimator).__module__
        name = type(self.estimator).__name__

        if module.startswith("sklearn.") and name == "HistGradientBoostingRegressor":
            return [
                clone(self.estimator).set_params(loss="quantile", quantile=quantile)
                for quantile in quantiles
            ]

        if module.startswith("lightgbm.") and name == "LGBMRegressor":
            return [
                clone(self.estimator).set_params(objective="quantile", alpha=quantile)
                for quantile in quantiles
            ]

        if module.startswith("xgboost.") and name == "XGBRegressor":
            return [
                clone(self.estimator).set_params(
                    objective="reg:quantileerror",
                    quantile_alpha=(
                        quantiles[0] if len(quantiles) == 1 else np.asarray(quantiles)
                    ),
                )
            ]

        if module.startswith("catboost.") and name == "CatBoostRegressor":
            loss = (
                f"Quantile:alpha={quantiles[0]}"
                if len(quantiles) == 1
                else f"MultiQuantile:alpha={','.join(map(str, quantiles))}"
            )
            return [clone(self.estimator).set_params(loss_function=loss)]

        raise TypeError(
            "estimator must be a `HistGradientBoosting`, `LightGBM`, `XGBoost`, or "
            "`CatBoost` regressor instance"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fits the configured quantile models for one target.

        Args:
            X (np.ndarray): Training features with shape
                ``(n_samples, n_features)``.
            y (np.ndarray): Training target with shape ``(n_samples,)``.

        Returns:
            Self: The fitted adapter.
        """
        self.estimators_ = self._configure_estimators()
        for estimator in self.estimators_:
            estimator.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts ordered lower and upper bounds for one target.

        Args:
            X (np.ndarray): Features with shape ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Bounds with shape ``(n_samples, 2)``.
        """
        predictions = np.column_stack(
            [np.asarray(estimator.predict(X)) for estimator in self.estimators_]
        )
        if self.gamma == 0:
            predictions = np.column_stack((np.full(len(X), -np.inf), predictions))
        elif self.gamma == 1:
            predictions = np.column_stack((predictions, np.full(len(X), np.inf)))

        return np.maximum.accumulate(predictions, axis=1)
