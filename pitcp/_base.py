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
    column_or_1d,  # type: ignore
)
from torch import nn
from tqdm import trange
from zuko.lazy import LazyDistribution  # type: ignore

from ._utils import is_flow, is_mixture


class PITCP(BaseEstimator, nn.Module):
    estimator: LazyDistribution
    optimizer: torch.optim.Optimizer
    n_epochs: int
    batch_size: int
    verbose: bool | int
    estimator_type_: str

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
        base_score: Callable[
            [np.typing.ArrayLike, np.typing.ArrayLike], np.typing.ArrayLike
        ],
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

        s = column_or_1d(self.base_score(X, y), warn=True)
        assert_all_finite(X, input_name="X")
        assert_all_finite(s, input_name="s")
        check_consistent_length(X, s)

        X = torch.as_tensor(X, dtype=dtype, device=device)
        s = torch.as_tensor(s, dtype=dtype, device=device).reshape(-1)

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
    @torch.no_grad()  # type: ignore
    def conformalize(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike) -> Self:
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        s = column_or_1d(self.base_score(X, y), warn=True)
        assert_all_finite(X, input_name="X")
        assert_all_finite(s, input_name="s")
        check_consistent_length(X, s)

        X = torch.as_tensor(X, dtype=dtype, device=device)
        s = torch.as_tensor(s, dtype=dtype, device=device).reshape(-1)

        dataset = torch.utils.data.TensorDataset(X, s)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.eval()

        if self.type == "flow":
            def transform(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor: 
                dist = super().estimator(x)
                return dist.transform(s)
        else:
            pass

        for xb, sb in loader:
            self.optimizer.zero_grad()

            dist = self.estimator(xb)
            loss = -dist.log_prob(sb).mean()

            loss.backward()
            self.optimizer.step()

            batch_size = xb.shape[0]
            epoch_loss += loss.item() * batch_size
            n_seen += batch_size

        return self

    def conformalize(self, )
