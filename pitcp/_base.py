from collections.abc import Callable
from numbers import Integral
from typing import Self

import numpy as np
import torch
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    assert_all_finite,  # type: ignore
    check_consistent_length,  # type: ignore
    check_is_fitted,  # type: ignore
)
from torch import nn
from torch.distributions import Normal
from tqdm import trange
from zuko.lazy import LazyDistribution  # type: ignore

from ._defs import CPResult
from ._utils import is_flow, is_mixture


class PITCP(BaseEstimator, nn.Module):
    """PIT conformal predictor using a normalizing flow or mixture density estimator.

    Attributes:
        estimator (LazyDistribution): Conditional density estimator.
        optimizer (torch.optim.Optimizer): Optimizer for training the estimator.
        n_epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loading.
        verbose (bool | int): Whether to display a progress bar during training.
        estimator_type_ (str): Either ``"flow"`` or ``"mixture"``.
        scores_ (torch.Tensor): Calibration PIT scores stored after conformalization.
    """

    estimator: LazyDistribution
    optimizer: torch.optim.Optimizer
    n_epochs: int
    batch_size: int
    verbose: bool | int
    estimator_type_: str
    scores_: torch.Tensor

    @validate_params(
        {
            "estimator": [LazyDistribution],
            "optimizer": [torch.optim.Optimizer],
            "n_epochs": [Interval(Integral, 1, None, closed="left")],
            "verbose": ["verbose"],
            "batch_size": [Interval(Integral, 1, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        base_score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        estimator: LazyDistribution,
        optimizer: torch.optim.Optimizer,
        *,
        n_epochs: int = 10,
        batch_size: int = 128,
        verbose: bool | int = True,
    ):
        """Initializes the PITCP instance.

        Args:
            base_score (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): Function
                computing nonconformity scores from features and labels.
            estimator (LazyDistribution): Conditional density estimator.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            n_epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Batch size for data loading. Defaults to 128.
            verbose (bool | int, optional): Whether to show a training progress bar.
                Defaults to True.
        """
        super().__init__()

        self.base_score = base_score
        self.estimator = estimator
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        if is_flow(self.estimator):
            self.estimator_type_ = "flow"
        elif is_mixture(self.estimator):
            self.estimator_type_ = "mixture"
        else:
            raise ValueError(
                "Estimator must be either a `zuko.flows` or `zuko.mixtures` submodule"
            )

    def _compute_base_scores(
        self,
        X: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes base nonconformity scores from features and labels.

        Args:
            X (np.typing.ArrayLike): Input features.
            y (np.typing.ArrayLike): Target labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Feature and flattened score tensors.
        """
        assert_all_finite(X, input_name="X")
        assert_all_finite(y, input_name="y")
        check_consistent_length(X, y)

        X = torch.as_tensor(X)
        y = torch.as_tensor(y)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return X, torch.cat([self.base_score(xb, yb).flatten() for xb, yb in loader])

    @torch.no_grad()
    def transform(self, x: torch.Tensor, s: torch.Tensor):
        """Maps nonconformity scores to PIT values via the learned conditional CDF.

        Args:
            x (torch.Tensor): Input features.
            s (torch.Tensor): Nonconformity scores.

        Returns:
            torch.Tensor: PIT-corrected non-conformity scores.
        """

        def _transform_flow(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            dist = self.estimator(x)
            return dist.transform(s).flatten()

        def _transform_mixture(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            dist = self.estimator(x)
            weights = dist.logits.softmax(dim=-1).reshape(-1, 1)
            means = dist.base.loc.reshape(-1, 1)
            stds = dist.base.covariance_matrix.reshape(-1, 1).sqrt()
            return (weights * Normal(means, stds).cdf(s)).sum(dim=-2)

        if self.estimator_type_ == "flow":
            return _transform_flow(x, s)
        return _transform_mixture(x, s)

    @validate_params(
        {
            "X": ["array-like"],
            "y": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike) -> Self:
        """Fits the conditional density estimator on nonconformity scores.

        Args:
            X (np.typing.ArrayLike): Input features.
            y (np.typing.ArrayLike): Target labels.

        Returns:
            PITCP: The fitted estimator.
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        X, s = self._compute_base_scores(X, y)
        X = X.to(dtype=dtype, device=device)  # type: ignore
        s = s.to(dtype=dtype, device=device)

        dataset = torch.utils.data.TensorDataset(X, s)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.train()

        pbar = trange(self.n_epochs, disable=not self.verbose)

        for epoch in pbar:
            epoch_loss = 0.0
            n_seen = 0

            for xb, sb in loader:
                self.optimizer.zero_grad()

                dist = self.estimator(xb)
                loss = -dist.log_prob(sb).mean()

                loss.backward()
                self.optimizer.step()

                batch_size = xb.shape[0]
                epoch_loss += loss.item() * batch_size
                n_seen += batch_size

            epoch_loss /= n_seen

            if self.verbose:
                pbar.set_description(f"Epoch: {epoch}, NLL: {epoch_loss:.4f}")

        return self

    @validate_params(
        {
            "X": ["array-like"],
            "y": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def conformalize(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike) -> Self:
        """Computes and stores calibration PIT scores from a held-out dataset.

        Args:
            X (np.typing.ArrayLike): Calibration features.
            y (np.typing.ArrayLike): Calibration labels.

        Returns:
            PITCP: The updated estimator.
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        X, s = self._compute_base_scores(X, y)
        X = X.to(dtype=dtype, device=device)  # type: ignore
        s = s.to(dtype=dtype, device=device)

        dataset = torch.utils.data.TensorDataset(X, s)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.eval()

        self.scores_ = torch.cat([self.transform(xb, sb) for xb, sb in loader])

        return self

    @validate_params(
        {
            "X": ["array-like"],
            "y": ["array-like"],
            "quantile": [float],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(
        self,
        X: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        *,
        quantile: float = 0.9,
    ) -> CPResult:
        """Predicts conformal coverage for test points.

        Args:
            X (np.typing.ArrayLike): Test features.
            y (np.typing.ArrayLike): Test labels.
            quantile (float, optional): Target coverage level. Defaults to 0.9.

        Returns:
            CPResult: Coverage indicators and conformal quantile.
        """
        check_is_fitted(self, "scores_")

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        n = self.scores_.numel()
        quantile = min(quantile * (1 + 1 / (n + 1)), 1)
        if quantile < n / (n + 1):
            q = torch.quantile(self.scores_, quantile).item()
        else:
            q = float("inf")

        X, s = self._compute_base_scores(X, y)
        X = X.to(dtype=dtype, device=device)  # type: ignore
        s = s.to(dtype=dtype, device=device)

        self.eval()

        u = self.transform(X, s)

        return CPResult(is_covered=u <= q, quantile=q)
