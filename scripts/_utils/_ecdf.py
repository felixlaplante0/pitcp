from collections.abc import Callable, Sequence
from numbers import Integral
from typing import ClassVar, Self

import numpy as np
import torch
from pitcp import SCP
from pitcp.utils._utils import collapse
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_array, check_is_fitted
from zuko.flows import Flow
from zuko.mixtures import GMM


class ECDF(SCP):
    """Calibrates empirical conditional CDF scores by Monte Carlo sampling.

    A fitted conditional ``zuko`` distribution of the targets generates samples whose
    base scores define an empirical conditional CDF, following the ECDF score of Dheur
    et al. (2025). Observed base scores are mapped to their empirical ranks, which are
    then calibrated by the shared split-conformal threshold.

    Sampling settings:
        - ``estimator``: Fitted conditional ``zuko`` flow or Gaussian mixture model of
          the targets given the features.
        - ``score``: Base nonconformity score mapping targets with shape ``(n_samples,
          n_outputs)`` and predictions with the same shape to scores with shape
          ``(n_samples,)``.
        - ``n_samples``: Positive number of Monte Carlo draws used to estimate each
          empirical CDF. Defaults to 100.
        - ``batch_size``: Positive mini-batch size used during sampling. ``None`` uses
          the full dataset. Defaults to ``None``.

    Attributes:
        estimator (Flow | GMM): Fitted conditional target-density estimator.
        score (Callable[[np.ndarray, np.ndarray], np.ndarray]): Base nonconformity
            score.
        n_samples (int): Monte Carlo sample count used for empirical CDFs.
        batch_size (int | None): Sampling batch size or ``None`` for full batches.
        scores_ (np.ndarray): Empirical CDF calibration scores with shape
            ``(n_samples,)``.
    """

    _parameter_constraints: ClassVar[dict] = {
        "estimator": [Flow, GMM],
        "score": [callable],
        "n_samples": [Interval(Integral, 1, None, closed="left")],
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
    }

    def __init__(
        self,
        estimator: Flow | GMM,
        score: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        n_samples: int = 100,
        batch_size: int | None = None,
    ):
        """Initializes the ECDF conformal regressor.

        Args:
            estimator (Flow | GMM): Fitted conditional target-density estimator.
            score (Callable[[np.ndarray, np.ndarray], np.ndarray]): Base nonconformity
                score.
            n_samples (int, optional): Monte Carlo sample count. Defaults to 100.
            batch_size (int | None, optional): Sampling batch size. ``None`` uses full
                batches. Defaults to None.
        """
        self.estimator = estimator
        self.score = score
        self.n_samples = n_samples
        self.batch_size = batch_size

    @torch.no_grad()
    def _sample_scores(self, X: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes sorted base scores of sampled targets.

        Args:
            X (np.ndarray): Features with shape ``(n_samples, n_features)``.
            y_pred (np.ndarray): Predictions with shape ``(n_samples, n_outputs)``.

        Returns:
            np.ndarray: Sorted sampled scores with shape ``(n_samples,
                self.n_samples)``.
        """
        dtype = next(self.estimator.parameters()).dtype
        device = next(self.estimator.parameters()).device
        X = torch.tensor(X, dtype=dtype)

        self.estimator.eval()
        y = torch.cat(
            [
                self.estimator(xb.to(device)).sample((self.n_samples,)).cpu()
                for xb in X.split(self.batch_size or len(X))
            ],
            dim=1,
        ).numpy()
        scores = self.score(
            y.reshape(-1, y.shape[-1]),
            np.tile(y_pred, (self.n_samples, 1)),
        )

        return np.sort(scores.reshape(self.n_samples, -1).T, axis=1)

    def _rank(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes empirical conditional CDF scores.

        Args:
            X (np.ndarray): Features with shape ``(n_samples, n_features)``.
            y (np.ndarray): Targets with shape ``(n_samples, n_outputs)``.
            y_pred (np.ndarray): Predictions with shape ``(n_samples, n_outputs)``.

        Returns:
            np.ndarray: Empirical CDF scores with shape ``(n_samples,)``.
        """
        sampled = self._sample_scores(X, y_pred)

        return (sampled <= self.score(y, y_pred)[:, None]).mean(axis=1)

    @validate_params(
        {"X": ["array-like"], "y": ["array-like"], "y_pred": ["array-like"]},
        prefer_skip_nested_validation=True,
    )
    def conformalize(
        self,
        X: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        y_pred: np.typing.ArrayLike,
    ) -> Self:
        """Stores held-out empirical CDF calibration scores.

        Args:
            X (np.typing.ArrayLike): Calibration features with shape ``(n_samples,
                n_features)``.
            y (np.typing.ArrayLike): Targets with shape ``(n_samples, n_outputs)``.
            y_pred (np.typing.ArrayLike): Predictions with shape ``(n_samples,
                n_outputs)``.

        Returns:
            Self: The calibrated estimator.
        """
        self._validate_params()
        self.scores_ = self._rank(check_array(X), check_array(y), check_array(y_pred))

        return self

    @validate_params(
        {
            "X": ["array-like"],
            "y_pred": ["array-like"],
            "confidence_level": [float, Sequence],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(
        self,
        X: np.typing.ArrayLike,
        y_pred: np.typing.ArrayLike,
        *,
        confidence_level: float | Sequence[float] = 0.9,
    ) -> np.ndarray:
        """Predicts base-score radii of the calibrated regions.

        Args:
            X (np.typing.ArrayLike): Test features with shape ``(n_samples,
                n_features)``.
            y_pred (np.typing.ArrayLike): Predictions with shape ``(n_samples,
                n_outputs)``.
            confidence_level (float | Sequence[float], optional): Requested coverage
                levels. Defaults to 0.9.

        Returns:
            np.ndarray: Base-score radii with shape ``(n_samples,)`` or ``(n_samples,
                n_levels)``.
        """
        check_is_fitted(self, "scores_")

        sampled = self._sample_scores(check_array(X), check_array(y_pred))
        ranks = np.minimum(
            np.round(self.thresholds(confidence_level) * self.n_samples),
            self.n_samples,
        ).astype(int)

        return collapse(
            np.pad(sampled, ((0, 0), (0, 1)), constant_values=np.inf)[:, ranks]
        )

    @validate_params(
        {
            "X": ["array-like"],
            "y": ["array-like"],
            "y_pred": ["array-like"],
            "confidence_level": [float, Sequence],
        },
        prefer_skip_nested_validation=True,
    )
    def contains(
        self,
        X: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        y_pred: np.typing.ArrayLike,
        *,
        confidence_level: float | Sequence[float] = 0.9,
    ) -> np.ndarray:
        """Tests whether targets lie inside calibrated empirical CDF sets.

        Args:
            X (np.typing.ArrayLike): Test features with shape ``(n_samples,
                n_features)``.
            y (np.typing.ArrayLike): Test targets with shape ``(n_samples,
                n_outputs)``.
            y_pred (np.typing.ArrayLike): Predictions with shape ``(n_samples,
                n_outputs)``.
            confidence_level (float | Sequence[float], optional): Requested coverage
                levels. Defaults to 0.9.

        Returns:
            np.ndarray: Coverage indicators with shape ``(n_samples,)`` or ``(n_samples,
                n_levels)``.
        """
        check_is_fitted(self, "scores_")

        return super().contains(
            self._rank(check_array(X), check_array(y), check_array(y_pred)),
            confidence_level=confidence_level,
        )
