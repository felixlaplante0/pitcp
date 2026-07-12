from collections.abc import Sequence
from functools import cached_property
from numbers import Integral
from typing import ClassVar, Self

import numpy as np
import torch
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from sklearn.utils.validation import (  # type: ignore
    check_is_fitted,  # type: ignore
    validate_data,  # type: ignore
)
from torch import nn
from torch.distributions import Normal
from tqdm import trange
from zuko.flows import Flow  # type: ignore
from zuko.mixtures import GMM  # type: ignore

from ..utils._utils import collapse
from ._scp import SCP

# Constants
_MAX_ITER_BISECT = 50


class PITCP(SCP, nn.Module):
    """PIT conformal predictor using a normalizing flow or mixture density estimator.

    This class implements probability integral transform (PIT) conformal prediction.
    Given a potentially black-box nonconformity scores, it fits a conditional density
    estimator on the score distribution over a training set, then uses the learned
    conditional CDF to map raw scores to PIT values. Conformal coverage guarantees are
    obtained by comparing test PIT values against a calibration confidence_level.

    The estimator must be a ``zuko`` subclass, coming from either ``zuko.flows.Flow`` (a
    normalizing flow) or ``zuko.mixtures.GMM`` (a mixture density network). The class
    internally detects which family is used and applies the appropriate CDF computation.

    Density estimation settings:
        - ``estimator``: A ``zuko`` lazy distribution instance conditioned on features,
          used to model the score distribution. Must be from ``zuko.flows`` or
          ``zuko.mixtures``.
        - ``optimizer``: Optimizer used to train the density estimator via maximum
          likelihood (negative log-likelihood/forward KL divergence minimization).

    Training settings:
        - ``n_epochs``: Number of full passes over the training data.
        - ``batch_size``: Mini-batch size used during both Train and inference.
        - ``verbose``: Whether to display a ``tqdm`` progress bar during ``fit``.
        - ``random_state``: Seed used to shuffle mini-batches during ``fit``.

    Attributes:
        estimator (Flow | GMM): Conditional density estimator from ``zuko.flows`` or
            ``zuko.mixtures``.
        optimizer (torch.optim.Optimizer): Optimizer for training the estimator.
        n_epochs (int): Number of training epochs.
        batch_size (int | None): Batch size for data loading. None means full-batch
            training.
        verbose (bool | int): Whether to display a progress bar during training.
        random_state (int | None): Seed used to shuffle mini-batches during ``fit``.
        estimator_type_ (str): Either ``flow`` or ``mixture``, set during ``fit`` based
            on the type of ``estimator``.
        scores_ (torch.Tensor | None): Calibration PIT scores stored after calling
            ``conformalize``.

    Examples:
        >>> import torch
        >>> import zuko
        >>> from pitcp import PITCP
        >>> X = torch.linspace(-1.0, 1.0, 32).reshape(-1, 1)
        >>> s = torch.abs(torch.sin(3.0 * X)) + 0.1
        >>> estimator = zuko.flows.NSF(features=1, context=1, bins=4)
        >>> optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-2)
        >>> model = PITCP(estimator, optimizer, n_epochs=1, verbose=False)
        >>> model.fit(X, s)
        >>> model.conformalize(X, s)
        >>> model.predict(X, confidence_level=0.9)
    """

    estimator: Flow | GMM
    optimizer: torch.optim.Optimizer
    n_epochs: int
    batch_size: int | None
    verbose: bool | int
    random_state: int | None
    scores_: np.ndarray

    _parameter_constraints: ClassVar[dict] = {
        "estimator": [Flow, GMM],
        "optimizer": [torch.optim.Optimizer],
        "n_epochs": [Interval(Integral, 1, None, closed="left")],
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator: Flow | GMM,
        optimizer: torch.optim.Optimizer,
        *,
        n_epochs: int = 10,
        batch_size: int | None = None,
        verbose: bool | int = True,
        random_state: int | None = None,
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
            random_state (int | None, optional): Seed used to shuffle mini-batches
                during ``fit``. Defaults to None.
        """
        super().__init__()

        self.estimator = estimator
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

    @cached_property
    def estimator_type_(self) -> str:
        """Detects the type of the estimator.

        Raises:
            ValueError: If the estimator is not a subclass of ``zuko.flows.Flow`` or
                ``zuko.mixtures.GMM``.

        Returns:
            str: Either ``flow`` or ``mixture``.
        """
        if issubclass(type(self.estimator), Flow):
            return "flow"
        if issubclass(type(self.estimator), GMM):
            return "mixture"

        raise ValueError(
            f"Unsupported estimator type: {type(self.estimator)}. Must be a subclass of"
            "`zuko.flows.Flow` or `zuko.mixtures.GMM`."
        )

    def _to_tensor(
        self,
        X: np.typing.ArrayLike,
        s: np.typing.ArrayLike | None = None,
        *,
        reset: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Validates input and converts features and scores to tensors.

        Args:
            X (np.typing.ArrayLike): Input features with shape
                ``(n_samples, n_features)``.
            s (np.typing.ArrayLike | None, optional): Target scores with shape
                ``(n_samples,)`` or ``None``. Defaults to None.
            reset (bool, optional): Whether to set the reset attribute. Deaults to True.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Feature tensor with
                shape ``(n_samples, n_features)`` and, when supplied, score tensor
                with shape ``(n_samples, 1)``.
        """
        dtype = next(self.parameters()).dtype
        if s is None:
            X = validate_data(self, X, reset=False)  # type: ignore
            return torch.tensor(X, dtype=dtype)

        X, s = validate_data(self, X, s, reset=reset)  # type: ignore
        return torch.tensor(X, dtype=dtype), torch.tensor(s, dtype=dtype).reshape(-1, 1)

    @torch.no_grad()
    def _correct(self, X: torch.Tensor, s: torch.Tensor) -> np.ndarray:
        """Maps nonconformity scores to PIT values via the learned conditional CDF.

        Args:
            X (torch.Tensor): Input features with shape
                ``(n_samples, n_features)``.
            s (torch.Tensor): Input scores with shape ``(n_samples, 1)``.

        Returns:
            np.ndarray: PIT-corrected scores with shape ``(n_samples, 1)``.
        """

        def _correct_flow(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return self.estimator(x).transform(s)

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

        dataset = torch.utils.data.TensorDataset(X, s)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=False,
        )

        device = next(self.parameters()).device
        return torch.cat(
            [_correct(xb.to(device), sb.to(device)).cpu() for xb, sb in loader]
        ).numpy()

    @torch.no_grad()
    def _invert(
        self, X: torch.Tensor, confidence_level: float | Sequence[float]
    ) -> np.ndarray:
        """Inverts PIT-corrected nonconformity scores via the learned conditional CDF.

        Args:
            X (torch.Tensor): Input features with shape
                ``(n_samples, n_features)``.
            confidence_level (float | Sequence[float], optional): Target coverage
                level(s). Defaults to 0.9.

        Returns:
            np.ndarray: Inverted scores with shape ``(n_samples, n_levels)``.
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        thresholds = torch.tensor(
            self.thresholds(confidence_level), dtype=dtype, device=device
        )

        def _invert_flow(x: torch.Tensor) -> torch.Tensor:
            return self.estimator(x).transform.inv(thresholds)

        def _invert_mixture(x: torch.Tensor) -> torch.Tensor:
            dist = self.estimator(x)
            weights = dist.logits.softmax(dim=-1).unsqueeze(-1)

            if hasattr(dist.base, "base_dist"):
                means = dist.base.base_dist.loc
                stds = dist.base.base_dist.scale.sqrt()
            else:
                means = dist.base.loc
                stds = dist.base.covariance_matrix.squeeze(-1).sqrt()

            lo = (means - 10 * stds).min(dim=-2).values
            hi = (means + 10 * stds).max(dim=-2).values

            normal = Normal(means, stds)

            def _cdf(u: torch.Tensor):
                return (weights * normal.cdf(u.unsqueeze(-2))).sum(dim=-2)

            for _ in range(_MAX_ITER_BISECT):
                mid = 0.5 * (lo + hi)
                val = _cdf(mid)
                lo = torch.where(val < thresholds, mid, lo)
                hi = torch.where(val >= thresholds, mid, hi)
                if torch.allclose(lo, hi, equal_nan=True):
                    break

            return lo

        _invert = _invert_flow if self.estimator_type_ == "flow" else _invert_mixture

        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=False,
        )

        return torch.cat([_invert(xb.to(device)).cpu() for (xb,) in loader]).numpy()

    @validate_params(
        {"X": ["array-like"], "s": ["array-like"]}, prefer_skip_nested_validation=True
    )
    def fit(self, X: np.typing.ArrayLike, s: np.typing.ArrayLike) -> Self:
        """Fits the conditional density estimator on nonconformity scores.

        Args:
            X (np.typing.ArrayLike): Training features with shape
                ``(n_samples, n_features)``.
            s (np.typing.ArrayLike): Training scores with shape ``(n_samples,)``.

        Returns:
            Self: The fitted estimator.
        """
        self._validate_params()
        X, s = self._to_tensor(X, s)  # type: ignore

        dataset = torch.utils.data.TensorDataset(X, s)
        generator = (
            None
            if self.random_state is None
            else torch.Generator().manual_seed(self.random_state)
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=True,
            generator=generator,
        )

        self.train()

        device = next(self.parameters()).device
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
            X (np.typing.ArrayLike): Calibration features with shape
                ``(n_samples, n_features)``.
            s (np.typing.ArrayLike): Calibration scores with shape ``(n_samples,)``.

        Returns:
            Self: The updated estimator.
        """
        self._validate_params()
        self.eval()

        X, s = self._to_tensor(X, s, reset=False)  # type: ignore
        self.scores_ = self._correct(X, s)

        return self

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
        """Predicts conformal regions for test points.

        Args:
            X (np.typing.ArrayLike): Test features with shape
                ``(n_samples, n_features)``.
            confidence_level (float | Sequence[float], optional): Target coverage
                level(s). Defaults to 0.9.

        Returns:
            np.ndarray: Score limits with shape ``(n_samples,)`` or
                ``(n_samples, n_levels)``.
        """
        check_is_fitted(self, "scores_")
        X = self._to_tensor(X)  # type: ignore

        self.eval()

        return collapse(self._invert(X, confidence_level))  # type: ignore

    @validate_params(
        {
            "X": ["array-like"],
            "s": ["array-like"],
            "confidence_level": [float, Sequence],
        },
        prefer_skip_nested_validation=True,
    )
    def contains(
        self,
        X: np.typing.ArrayLike,
        s: np.typing.ArrayLike,
        *,
        confidence_level: float | Sequence[float] = 0.9,
    ) -> np.ndarray:
        """Predicts conformal coverage for test points.

        Args:
            X (np.typing.ArrayLike): Test features with shape
                ``(n_samples, n_features)``.
            s (np.typing.ArrayLike): Test scores with shape ``(n_samples,)``.
            confidence_level (float | Sequence[float], optional): Target coverage
                level(s). Defaults to 0.9.

        Returns:
            np.ndarray: Coverage indicators with shape ``(n_samples,)`` or
                ``(n_samples, n_levels)``.
        """
        check_is_fitted(self, "scores_")
        X, s = self._to_tensor(X, s, reset=False)  # type: ignore

        self.eval()

        return super().contains(self._correct(X, s), confidence_level=confidence_level)
