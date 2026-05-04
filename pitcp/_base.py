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
from zuko.flows import Flow  # type: ignore
from zuko.mixtures import GMM  # type: ignore

from ._utils import is_flow, is_mixture


class PITCP(BaseEstimator, nn.Module):
    """PIT conformal predictor using a normalizing flow or mixture density estimator.

    This class implements probability integral transform (PIT) conformal prediction.
    Given arbitrary base nonconformity scores, it fits a conditional density estimator
    on the score distribution over a train set, then uses the learned conditional CDF to
    map raw scores to PIT values. Conformal coverage guarantees are obtained by
    comparing test PIT values against a calibration quantile.

    The estimator must be a `zuko` lazy distribution from either `zuko.flows` (a
    normalizing flow) or `zuko.mixtures` (a mixture density network). The class
    internally detects which family is used and applies the appropriate CDF computation.

    Density estimation settings:
        - `estimator`: A `zuko` lazy distribution instance conditioned on features, used
          to model the score distribution. Must be from `zuko.flows` or `zuko.mixtures`.
        - `optimizer`: Optimizer used to train the density estimator via maximum
          likelihood (negative log-likelihood/forward KL divergence minimization).

    Train settings:
        - `n_epochs`: Number of full passes over the Train data.
        - `batch_size`: Mini-batch size used during both Train and inference.
        - `verbose`: Whether to display a `tqdm` progress bar during `fit`.

    Attributes:
        estimator (Flow | GMM): Conditional density estimator from
            `zuko.flows` or `zuko.mixtures`.
        optimizer (torch.optim.Optimizer): Optimizer for training the estimator.
        n_epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loading.
        verbose (bool | int): Whether to display a progress bar during training.
        estimator_type_ (str): Either `flow` or `mixture`, set at initialization based
            on the type of `estimator`.
        scores_ (torch.Tensor): Calibration PIT scores stored after calling
            `conformalize`.
    """

    estimator: Flow | GMM
    optimizer: torch.optim.Optimizer
    n_epochs: int
    batch_size: int
    verbose: bool | int
    estimator_type_: str
    scores_: torch.Tensor

    @validate_params(
        {
            "estimator": [Flow, GMM],
            "optimizer": [torch.optim.Optimizer],
            "n_epochs": [Interval(Integral, 1, None, closed="left")],
            "verbose": ["verbose"],
            "batch_size": [Interval(Integral, 1, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        estimator: Flow | GMM,
        optimizer: torch.optim.Optimizer,
        *,
        n_epochs: int = 10,
        batch_size: int = 128,
        verbose: bool | int = True,
    ):
        """Initializes the PITCP instance.

        Args:
            estimator (Flow | GMM): Conditional density estimator.
            optimizer (torch.optim.Optimizer): Optimizer for Train.
            n_epochs (int, optional): Number of Train epochs. Defaults to 10.
            batch_size (int, optional): Batch size for data loading. Defaults to 128.
            verbose (bool | int, optional): Whether to show a Train progress bar.
                Defaults to True.
        """
        super().__init__()

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

    def _validate_X_s(
        self,
        X: np.typing.ArrayLike,
        s: np.typing.ArrayLike,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validates input and converts features and scores to tensors.

        Args:
            X (np.typing.ArrayLike): Input features.
            s (np.typing.ArrayLike): Target scores.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Feature and score tensors.
        """
        assert_all_finite(X, input_name="X")
        assert_all_finite(s, input_name="s")
        check_consistent_length(X, s)

        return torch.as_tensor(X), torch.as_tensor(s)

    @torch.no_grad()
    def _correct(self, X: torch.Tensor, s: torch.Tensor):
        """Maps nonconformity scores to PIT values via the learned conditional CDF.

        Args:
            x (torch.Tensor): Input features.
            s (torch.Tensor): Nonconformity scores.

        Returns:
            torch.Tensor: PIT-corrected nonconformity scores.
        """

        def _correct_flow(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            dist = self.estimator(x)
            return dist.transform(s)

        def _correct_mixture(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            dist = self.estimator(x)
            weights = dist.logits.softmax(dim=-1)

            if hasattr(dist.base, "base_dist"):
                means = dist.base.base_dist.loc.squeeze(-1)
                stds = dist.base.base_dist.scale.squeeze(-1).sqrt()
            else:
                means = dist.base.loc.squeeze(-1)
                stds = dist.base.covariance_matrix.squeeze((-2, -1)).sqrt()

            return (weights * Normal(means, stds).cdf(s)).sum(dim=-1, keepdim=True)

        _correct = _correct_flow if self.estimator_type_ == "flow" else _correct_mixture

        device = next(self.parameters()).device

        dataset = torch.utils.data.TensorDataset(X, s)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return torch.cat(
            [_correct(xb.to(device), sb.to(device)).cpu() for xb, sb in loader]
        )

    @validate_params(
        {
            "X": ["array-like"],
            "s": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: np.typing.ArrayLike, s: np.typing.ArrayLike) -> Self:
        """Fits the conditional density estimator on nonconformity scores.

        Args:
            X (np.typing.ArrayLike): Train features.
            s (np.typing.ArrayLike): Train scores.

        Returns:
            Self: The fitted estimator.
        """
        device = next(self.parameters()).device

        X, s = self._validate_X_s(X, s)  # type: ignore

        dataset = torch.utils.data.TensorDataset(X, s)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.train()

        pbar = trange(self.n_epochs, disable=not self.verbose, unit="epoch")
        for _ in pbar:
            epoch_loss = 0.0

            for xb, sb in loader:
                self.optimizer.zero_grad()

                dist = self.estimator(xb.to(device))
                loss = -dist.log_prob(sb.to(device)).mean()

                loss.backward()
                self.optimizer.step()

                batch_size = xb.size(0)
                epoch_loss += loss.item() * batch_size

            epoch_loss /= len(loader.dataset)  # type: ignore
            pbar.set_postfix({"NLL": f"{epoch_loss:.4f}"})  # type: ignore

        return self

    @validate_params(
        {
            "X": ["array-like"],
            "s": ["array-like"],
        },
        prefer_skip_nested_validation=True,
    )
    def conformalize(self, X: np.typing.ArrayLike, s: np.typing.ArrayLike) -> Self:
        """Computes and stores calibration PIT scores from a held-out dataset.

        Args:
            X (np.typing.ArrayLike): Calibration features.
            s (np.typing.ArrayLike): Calibration scores.

        Returns:
            Self: The updated estimator.
        """
        X, s = self._validate_X_s(X, s)  # type: ignore

        self.eval()

        self.scores_ = self._correct(X, s)

        return self

    @validate_params(
        {
            "X": ["array-like"],
            "s": ["array-like"],
            "quantile": [float, torch.Tensor],
            "return_threshold": [bool],
        },
        prefer_skip_nested_validation=True,
    )
    def predict(
        self,
        X: np.typing.ArrayLike,
        s: np.typing.ArrayLike,
        *,
        quantile: float | torch.Tensor = 0.9,
    ) -> torch.Tensor:
        """Predicts conformal coverage for test points.

        Args:
            X (np.typing.ArrayLike): Test features.
            s (np.typing.ArrayLike): Test scores.
            quantile (float | torch.Tensor, optional): Target coverage level. Defaults
                to 0.9.

        Returns:
            torch.Tensor: Coverage indicators.
        """
        check_is_fitted(self, "scores_")

        n = self.scores_.numel()
        k = torch.ceil(torch.as_tensor(quantile) * (n + 1))
        level = (k / n).clamp(max=1.0)
        threshold = torch.quantile(self.scores_, level)

        X, s = self._validate_X_s(X, s)  # type: ignore

        self.eval()

        u = self._correct(X, s)
        covered = u.le(threshold)
        covered[k > n] = True

        return covered
