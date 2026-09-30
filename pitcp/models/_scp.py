from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.utils._param_validation import validate_params
from sklearn.utils.validation import check_array

from ..utils._utils import collapse, confidence_levels


class SCP(BaseEstimator):
    """Calibrates scalar nonconformity scores by split conformal prediction.

    Calibration settings:
        - ``s``: Held-out scalar nonconformity scores used to compute the finite-sample
          corrected empirical quantiles.

    Prediction settings:
        - ``X``: Test features. Only the number of samples is used because split
          conformal thresholds do not depend on the feature values.
        - ``confidence_level``: Target marginal coverage level or sequence of levels in
          the open interval from zero to one. Defaults to 0.9.

    Attributes:
        scores_ (np.ndarray): Held-out calibration scores with shape ``(n_samples,)``.
    """

    @validate_params({"s": ["array-like"]}, prefer_skip_nested_validation=True)
    def conformalize(self, s: np.typing.ArrayLike):
        """Stores held-out calibration scores.

        Args:
            s (np.typing.ArrayLike): Scores with shape ``(n_samples,)``.

        Returns:
            Self: The calibrated estimator.

        """
        self.scores_ = np.asarray(check_array(s, ensure_2d=False)).reshape(-1)

        return self

    fit = conformalize

    @validate_params(
        {"confidence_level": [float, Sequence]}, prefer_skip_nested_validation=True
    )
    def thresholds(self, confidence_level: float | Sequence[float] = 0.9) -> np.ndarray:
        """Computes finite-sample corrected score thresholds.

        Args:
            confidence_level (float | Sequence[float], optional): Target marginal
                coverage level or levels. Defaults to 0.9.

        Returns:
            np.ndarray: One threshold for each requested confidence level.
        """
        check_is_fitted(self, "scores_")

        n = self.scores_.size

        levels = confidence_levels(confidence_level)
        ranks = np.ceil(levels * (n + 1))
        thresholds = np.empty(levels.size, dtype=float)
        finite = ranks <= self.scores_.size

        thresholds[~finite] = np.inf
        if np.any(finite):
            thresholds[finite] = np.quantile(
                self.scores_, ranks[finite] / n, method="higher"
            )

        return thresholds

    @validate_params(
        {"X": ["array-like"], "confidence_level": [float, Sequence]},
        prefer_skip_nested_validation=True,
    )
    def predict(
        self,
        X: np.typing.ArrayLike,
        *,
        confidence_level: float | Sequence[float] = 0.9,
    ) -> np.ndarray:
        """Returns score-space thresholds for test features.

        Args:
            X (np.typing.ArrayLike): Test features with shape ``(n_samples,
                n_features)``.
            confidence_level (float | Sequence[float], optional): Target marginal
                coverage level or levels. Defaults to 0.9.

        Returns:
            np.ndarray: Score thresholds with shape ```(n_samples,)`` or ``(n_samples,
                n_levels)``.
        """
        thresholds = self.thresholds(confidence_level)

        return collapse(np.broadcast_to(thresholds, (len(X), len(thresholds))))

    @validate_params(
        {
            "s": ["array-like"],
            "confidence_level": [float, Sequence],
        },
        prefer_skip_nested_validation=True,
    )
    def contains(
        self,
        s: np.typing.ArrayLike,
        *,
        confidence_level: float | Sequence[float] = 0.9,
    ) -> np.ndarray:
        """Tests whether scores lie inside calibrated regions.

        Args:
            s (np.typing.ArrayLike): Test scores with shape ``(n_samples,)``.
            confidence_level (float | Sequence[float], optional): Requested coverage
                levels. Defaults to 0.9.

        Returns:
            np.ndarray: Coverage indicators with shape ``(n_samples,)`` or ``(n_samples,
                n_levels)``.
        """
        check_is_fitted(self, "scores_")

        s = np.asarray(check_array(s, ensure_2d=False)).reshape(-1, 1)
        covered = s <= self.thresholds(confidence_levels(confidence_level))

        return collapse(covered)
