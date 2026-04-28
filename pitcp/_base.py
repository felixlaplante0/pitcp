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

        return X, torch.cat([self.base_score(xb, yb) for xb, yb in loader])

    @torch.no_grad()
    def transform(self, x: torch.Tensor, s: torch.Tensor):
        def _transform_flow(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            dist = super().estimator(x)
            return dist.transform(s)

        def _transform_mixture(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            dist = super().estimator(x)
            weights = dist.logits.softmax(dim=-1).reshape(-1, 1)
            means = dist.base.loc.reshape(-1, 1)
            stds = dist.base.covariance_matrix.reshape(-1, 1).sqrt()
            return (weights * Normal(means, stds).cdf(s)).sum(dim=-2)

        if self.type == "flow":
            return self._transform_flow(x, s)
        return self._transform_mixture(x, s)

    @validate_params(
        {
            "X": ["array-like"],
            "y": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike) -> Self:
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

        self.scores_ = torch.cat([self._transform(xb, sb) for xb, sb in loader])

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
        check_is_fitted(self, "scores_")

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        q = torch.quantile(self.scores_, quantile).item()

        X, s = self._compute_base_scores(X, y)
        X = X.to(dtype=dtype, device=device)  # type: ignore
        s = s.to(dtype=dtype, device=device)

        self.eval()

        u = self.transform(X, s)
        is_covered = u <= q

        return CPResult(is_covered=is_covered, quantile=q)
