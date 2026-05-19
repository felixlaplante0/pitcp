from collections.abc import Sequence
from numbers import Integral
from typing import Self, cast

import numpy as np
import torch
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    check_is_fitted,  # type: ignore
    validate_data,  # type: ignore
)
from torch import nn
from tqdm import trange
from zuko.flows import Flow  # type: ignore
from zuko.mixtures import GMM  # type: ignore

from ._utils import correct_mixture, invert_mixture, is_flow, is_mixture, to_1d


class PITCP(BaseEstimator, nn.Module):
    """PIT conformal predictor using a normalizing flow or mixture density estimator.

    This class implements probability integral transform (PIT) conformal prediction.
    Given a potentially black-box nonconformity scores, it fits a conditional density
    estimator on the score distribution over a training set, then uses the learned
    conditional CDF to map raw scores to PIT values. Conformal coverage guarantees are
    obtained by comparing test PIT values against a calibration quantile.

    The estimator must be a `zuko` subclass, coming from either `zuko.flows.Flow` (a
    normalizing flow) or `zuko.mixtures.GMM` (a mixture density network). The class
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
        batch_size (int | None): Batch size for data loading. None means full-batch
            training.
        verbose (bool | int): Whether to display a progress bar during training.
        estimator_type_ (str): Either `flow` or `mixture`, set at initialization based
            on the type of `estimator`.
        scores_ (torch.Tensor | None): Calibration PIT scores stored after calling
            `conformalize`.
    """

    estimator: Flow | GMM
    optimizer: torch.optim.Optimizer
    n_epochs: int
    batch_size: int | None
    verbose: bool | int
    estimator_type_: str
    scores_: np.ndarray

    @validate_params(
        {
            "estimator": [Flow, GMM],
            "optimizer": [torch.optim.Optimizer],
            "n_epochs": [Interval(Integral, 1, None, closed="left")],
            "verbose": ["verbose"],
            "batch_size": [Interval(Integral, 1, None, closed="left"), None],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        estimator: Flow | GMM,
        optimizer: torch.optim.Optimizer,
        *,
        n_epochs: int = 10,
        batch_size: int | None = None,
        verbose: bool | int = True,
    ):
        """Initializes the PITCP instance.

        Args:
            estimator (Flow | GMM): Conditional density estimator.
            optimizer (torch.optim.Optimizer): Optimizer for Train.
            n_epochs (int, optional): Number of Train epochs. Defaults to 10.
            batch_size (int | None, optional): Batch size for data loading. Defaults to
                None.
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

    def _validate(
        self,
        X: np.typing.ArrayLike,
        s: np.typing.ArrayLike | None = None,
        *,
        reset: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Validates input and converts features and scores to tensors.

        Args:
            X (np.typing.ArrayLike): Input features.
            s (np.typing.ArrayLike | None, optional): Target scores or None. Defaults to
                None.
            reset (bool, optional): Whether to set the reset attribute. Deaults to True.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Feature and optionally
                score tensors.
        """
        dtype = next(self.parameters()).dtype or torch.get_default_dtype()
        if s is None:
            X = validate_data(self, X, reset=reset)  # type: ignore
            return torch.tensor(X, dtype=dtype)

        X, s = validate_data(self, X, s, reset=reset)  # type: ignore
        return torch.tensor(X, dtype=dtype), torch.tensor(s, dtype=dtype).reshape(-1, 1)

    @torch.no_grad()
    def _correct(self, X: torch.Tensor, s: torch.Tensor) -> np.ndarray:
        """Maps nonconformity scores to PIT values via the learned conditional CDF.

        Args:
            X (torch.Tensor): Input features.
            s (torch.Tensor): Input scores.

        Returns:
            np.ndarray: PIT-corrected nonconformity scores.
        """

        def _correct_flow(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return self.estimator(x).transform(s)

        def _correct_mixture(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return correct_mixture(cast(GMM, self.estimator), x, s)

        _correct = _correct_flow if self.estimator_type_ == "flow" else _correct_mixture

        dataset = torch.utils.data.TensorDataset(X, s)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=False,
        )

        device = next(self.parameters()).device or torch.get_default_device()
        return torch.cat(
            [_correct(xb.to(device), sb.to(device)).cpu() for xb, sb in loader]
        ).numpy()

    @torch.no_grad()
    def _invert(self, X: torch.Tensor, quantile: float | Sequence[float]) -> np.ndarray:
        """Inverts PIT-corrected nonconformity scores via the learned conditional CDF.

        Args:
            X (torch.Tensor): Input features.
            quantile (float | Sequence[float], optional): Target coverage level(s).

        Returns:
            np.ndarray: Inverted PIT-corrected nonconformity scores.
        """
        dtype = next(self.parameters()).dtype or torch.get_default_dtype()
        threshold = torch.tensor(self.threshold(quantile), dtype=dtype)

        def _invert_flow(x: torch.Tensor) -> torch.Tensor:
            return self.estimator(x).transform.inv(threshold)

        def _invert_mixture(x: torch.Tensor) -> torch.Tensor:
            return invert_mixture(cast(GMM, self.estimator), x, threshold)  # type: ignore

        _invert = _invert_flow if self.estimator_type_ == "flow" else _invert_mixture

        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=False,
        )

        device = next(self.parameters()).device or torch.get_default_device()
        return torch.cat([_invert(xb.to(device)).cpu() for (xb,) in loader]).numpy()

    @validate_params(
        {"X": ["array-like"], "s": ["array-like"]}, prefer_skip_nested_validation=True
    )
    def fit(self, X: np.typing.ArrayLike, s: np.typing.ArrayLike) -> Self:
        """Fits the conditional density estimator on nonconformity scores.

        Args:
            X (np.typing.ArrayLike): Train features.
            s (np.typing.ArrayLike): Train scores.

        Returns:
            Self: The fitted estimator.
        """
        X, s = self._validate(X, s)  # type: ignore

        dataset = torch.utils.data.TensorDataset(X, s)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=True,
        )

        self.train()

        device = next(self.parameters()).device or torch.get_default_device()
        pbar = trange(self.n_epochs, disable=not self.verbose, unit="epoch")
        for _ in pbar:
            epoch_loss = 0.0

            for xb, sb in loader:
                self.optimizer.zero_grad()

                dist = self.estimator(xb.to(device))
                loss = -dist.log_prob(sb.to(device)).mean()

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)  # type: ignore
            pbar.set_postfix({"NLL": f"{epoch_loss:.4f}"})  # type: ignore

        return self

    @validate_params(
        {"X": ["array-like"], "s": ["array-like"]}, prefer_skip_nested_validation=True
    )
    def conformalize(self, X: np.typing.ArrayLike, s: np.typing.ArrayLike) -> Self:
        """Computes and stores calibration PIT scores from a held-out dataset.

        Args:
            X (np.typing.ArrayLike): Calibration features.
            s (np.typing.ArrayLike): Calibration scores.

        Returns:
            Self: The updated estimator.
        """
        self.eval()

        X, s = self._validate(X, s, reset=False)  # type: ignore
        self.scores_ = self._correct(X, s)

        return self

    @validate_params(
        {"quantile": [float, Sequence]}, prefer_skip_nested_validation=True
    )
    def threshold(self, quantile: float | Sequence[float] = 0.9) -> np.ndarray:
        """Computes the PIT threshold for a given quantile.

        Args:
            quantile (float | Sequence[float], optional): Target coverage level(s).
                Defaults to 0.9.

        Returns:
            np.ndarray: PIT threshold values.
        """
        check_is_fitted(self, "scores_")

        n = self.scores_.size
        quantile_1d = np.atleast_1d(np.asarray(quantile))
        k = np.ceil(quantile_1d * (n + 1))
        level = np.minimum(k / n, 1.0)
        threshold = np.quantile(self.scores_, level)
        threshold[k > n] = np.inf

        return threshold

    @validate_params(
        {"X": ["array-like"], "quantile": [float, Sequence]},
        prefer_skip_nested_validation=True,
    )
    def predict(
        self, X: np.typing.ArrayLike, *, quantile: float | Sequence[float] = 0.9
    ) -> np.ndarray:
        """Predicts conformal regions for test points.

        Args:
            X (np.typing.ArrayLike): Test features.
            quantile (float | Sequence[float], optional): Target coverage level(s).
                Defaults to 0.9.

        Returns:
            np.ndarray: Maximum base score threshold values.
        """
        check_is_fitted(self, "scores_")
        X = self._validate(X, reset=False)  # type: ignore

        self.eval()

        return to_1d(self._invert(X, quantile))  # type: ignore

    @validate_params(
        {"X": ["array-like"], "s": ["array-like"], "quantile": [float, Sequence]},
        prefer_skip_nested_validation=True,
    )
    def predict_coverage(
        self,
        X: np.typing.ArrayLike,
        s: np.typing.ArrayLike,
        *,
        quantile: float | Sequence[float] = 0.9,
    ) -> np.ndarray:
        """Predicts conformal coverage for test points.

        Args:
            X (np.typing.ArrayLike): Test features.
            s (np.typing.ArrayLike): Test scores.
            quantile (float | Sequence[float], optional): Target coverage level(s).
                Defaults to 0.9.

        Returns:
            np.ndarray: Coverage indicators.
        """
        self.eval()

        threshold = self.threshold(quantile)
        X, s = self._validate(X, s, reset=False)  # type: ignore

        return to_1d(self._correct(X, s) <= threshold)
