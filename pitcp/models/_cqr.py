from numbers import Real
from typing import ClassVar, Self

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._param_validation import HasMethods, Interval, validate_params
from sklearn.utils.validation import check_is_fitted, validate_data

from ..utils._quantile import QuantileEstimator
from ..utils._utils import collapse
from ._scp import SCP


class CQR(RegressorMixin, SCP):
    """Fits conformalized quantile intervals with a boosting estimator.

    The estimator can be a scikit-learn histogram gradient booster or an optional
    LightGBM, XGBoost, or CatBoost regressor. Backend-specific quantile models are
    fitted independently for every target output.

    Attributes:
        estimator (BaseEstimator): Unfitted quantile-regression prototype.
        confidence_level (float): Desired marginal coverage level.
        gamma (float): Miscoverage fraction assigned to the lower tail.
        estimators_ (list[QuantileEstimator]): Fitted quantile adapter for each
            target output.
        n_outputs_ (int): Number of fitted target outputs.
        scores_ (np.ndarray): Joint calibration scores.
        correction_ (float): Finite-sample conformal correction.

    Examples:
        >>> from sklearn.ensemble import HistGradientBoostingRegressor
        >>> from pitcp import CQR
        >>> model = CQR(HistGradientBoostingRegressor(), confidence_level=0.9)
        >>> model.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
        CQR(...)
        >>> model.conformalize([[3.0], [4.0]], [3.0, 4.0])
        CQR(...)
        >>> model.predict([[5.0]]).shape
        (1, 2)
    """

    _parameter_constraints: ClassVar[dict] = {
        "estimator": [HasMethods(["fit", "predict", "get_params", "set_params"])],
        "confidence_level": [Interval(Real, 0, 1, closed="neither")],
        "gamma": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        confidence_level: float = 0.9,
        gamma: float = 0.5,
    ):
        """Initializes the conformalized quantile estimator.

        Args:
            estimator (BaseEstimator): HistGradientBoosting, LightGBM, XGBoost, or
                CatBoost regressor.
            confidence_level (float, optional): Coverage level. Defaults to 0.9.
            gamma (float, optional): Miscoverage assigned to the lower tail. Defaults
                to 0.5.
        """
        self.estimator = estimator
        self.confidence_level = confidence_level
        self.gamma = gamma

    @validate_params(
        {"X": ["array-like"], "y": ["array-like"]},
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike) -> Self:
        """Fits cloned quantile estimators.

        Args:
            X (np.typing.ArrayLike): Training features with shape
                ``(n_samples, n_features)``.
            y (np.typing.ArrayLike): Targets with shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``.

        Returns:
            Self: The fitted estimator.

        Raises:
            TypeError: If the estimator backend is unsupported.
        """
        self._validate_params()
        X, y = [
            np.asarray(value) for value in validate_data(self, X, y, multi_output=True)
        ]
        if y.ndim == 1:
            y = y[:, None]

        self.n_outputs_ = y.shape[1]
        self.estimators_ = [
            QuantileEstimator(self.estimator, self.confidence_level, self.gamma).fit(
                X, target
            )
            for target in y.T
        ]

        return self

    def _bounds(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts uncalibrated lower and upper bounds.

        Args:
            X (np.ndarray): Features with shape ``(n_samples, n_features)``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Lower and upper bounds, each with shape
                ``(n_samples, n_outputs)``.
        """
        values = np.stack(
            [estimator.predict(X) for estimator in self.estimators_],
            axis=-1,
        )
        lower, upper = values[:, 0], values[:, 1]

        return np.minimum(lower, upper), np.maximum(lower, upper)

    @validate_params(
        {"X": ["array-like"], "y": ["array-like"]},
        prefer_skip_nested_validation=True,
    )
    def conformalize(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike) -> Self:
        """Calibrates a joint correction on held-out targets.

        Args:
            X (np.typing.ArrayLike): Calibration features with shape
                ``(n_samples, n_features)``.
            y (np.typing.ArrayLike): Calibration targets with shape
                ``(n_samples,)`` or ``(n_samples, n_outputs)``.

        Returns:
            Self: The calibrated estimator.
        """
        self._validate_params()
        check_is_fitted(self, "estimators_")
        X, y = validate_data(self, X, y, reset=False, multi_output=True)
        y = np.asarray(y).reshape(len(X), -1)

        lower, upper = self._bounds(np.asarray(X))
        self.scores_ = np.maximum(lower - y, y - upper).max(axis=1)
        self.correction_ = self.thresholds(self.confidence_level)[0]

        return self

    @validate_params({"X": ["array-like"]}, prefer_skip_nested_validation=True)
    def predict(self, X: np.typing.ArrayLike) -> np.ndarray:
        """Predicts calibrated lower and upper bounds.

        Args:
            X (np.typing.ArrayLike): Features with shape
                ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Bounds with shape ``(n_samples, 2, n_outputs)``, with a
                singleton output axis collapsed.
        """
        check_is_fitted(self, "correction_")
        X = np.asarray(validate_data(self, X, reset=False))
        lower, upper = self._bounds(X)
        bounds = np.stack((lower - self.correction_, upper + self.correction_), axis=1)

        return collapse(bounds)

    @validate_params(
        {"X": ["array-like"], "y": ["array-like"]}, prefer_skip_nested_validation=True
    )
    def contains(
        self,
        X: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
    ) -> np.ndarray:
        """Tests whether targets lie inside every output interval.

        Args:
            X (np.typing.ArrayLike): Features with shape
                ``(n_samples, n_features)``.
            y (np.typing.ArrayLike): Targets with shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``.

        Returns:
            np.ndarray: One boolean per sample.
        """
        X, y = [
            np.asarray(value)
            for value in validate_data(self, X, y, reset=False, multi_output=True)
        ]
        bounds = self.predict(X)
        bounds = bounds.reshape(len(bounds), 2, -1)
        y = np.asarray(y).reshape(len(y), 1, -1)

        return ((y >= bounds[:, :1]) & (y <= bounds[:, 1:])).all(axis=(1, 2))  # type: ignore
