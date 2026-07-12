from collections.abc import Sequence
from numbers import Integral
from typing import ClassVar, Self

import numpy as np
import torch
import tqdm
from sklearn.utils._param_validation import HasMethods, Interval, validate_params
from sklearn.utils.validation import check_is_fitted, validate_data
from torch import nn
from zuko.flows import Flow
from zuko.mixtures import GMM

from ..utils._utils import collapse
from ._scp import SCP


class HPD(SCP, nn.Module):
    """Calibrates conditional highest-density scores by Monte Carlo sampling.

    A conditional Zuko distribution models raw scalar scores. Monte Carlo density
    ranks turn observations into highest-predictive-density nonconformity scores,
    which are then calibrated by the shared split-conformal threshold.

    Attributes:
        estimator (Flow | GMM): Conditional score-density estimator.
        optimizer (torch.optim.Optimizer): Optimizer used for density training.
        n_epochs (int): Number of training epochs.
        n_samples (int): Monte Carlo sample count used for density ranks.
        batch_size (int | None): Training batch size or ``None`` for full batches.
        verbose (bool | int): Whether to display the training progress bar.
        random_state (int | None): Mini-batch shuffling seed.
        scores_ (np.ndarray): HPD calibration scores with shape ``(n_samples,)``.

    Examples:
        >>> import torch
        >>> import zuko
        >>> from pitcp import HPD
        >>> density = zuko.mixtures.GMM(features=1, context=1, components=2)
        >>> optimizer = torch.optim.Adam(density.parameters())
        >>> model = HPD(density, optimizer, n_epochs=1, verbose=False)
    """

    _parameter_constraints: ClassVar[dict] = {
        "estimator": [Flow, GMM],
        "optimizer": [HasMethods(["zero_grad", "step"])],
        "n_epochs": [Interval(Integral, 1, None, closed="left")],
        "n_samples": [Interval(Integral, 1, None, closed="left")],
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
        n_samples: int = 1000,
        batch_size: int | None = None,
        verbose: bool | int = True,
        random_state: int | None = None,
    ):
        """Initializes the HPD conformal regressor.

        Args:
            estimator (Flow | GMM): Conditional score-density estimator.
            optimizer (torch.optim.Optimizer): Optimizer for density training.
            n_epochs (int, optional): Number of training epochs. Defaults to 10.
            n_samples (int, optional): Monte Carlo density-rank sample count.
                Defaults to 1000.
            batch_size (int | None, optional): Mini-batch size. ``None`` uses full
                batches. Defaults to None.
            verbose (bool | int, optional): Whether to show a progress bar. Defaults
                to True.
            random_state (int | None, optional): Mini-batch shuffling seed. Defaults
                to None.
        """
        super().__init__()

        self.estimator = estimator
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

    def _to_tensor(
        self,
        X: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        *,
        reset: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validates inputs and converts them to estimator-compatible tensors.

        Args:
            X (np.typing.ArrayLike): Input features with shape ``(n_samples,
                n_features)``.
            y (np.typing.ArrayLike): Targets with shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``.
            reset (bool, optional): Whether to reset fitted feature metadata.
                Defaults to True.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Feature tensor and,
                when supplied, target tensor.
        """
        dtype = next(self.estimator.parameters()).dtype
        X, y = validate_data(self, X, y, reset=reset, multi_output=True)

        return torch.tensor(X, dtype=dtype), torch.tensor(y, dtype=dtype).reshape(
            len(y), -1
        )

    @validate_params(
        {"X": ["array-like"], "y": ["array-like"]},
        prefer_skip_nested_validation=True,
    )
    def fit(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike) -> Self:
        """Fits the conditional density estimator to raw scores.

        Args:
            X (np.typing.ArrayLike): Score-training features with shape
                ``(n_samples, n_features)``.
            y (np.typing.ArrayLike): Targets with shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``.

        Returns:
            Self: The fitted estimator.
        """
        self._validate_params()
        X, y = self._to_tensor(X, y)

        dataset = torch.utils.data.TensorDataset(X, y)
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

        self.estimator.train()
        device = next(self.estimator.parameters()).device
        pbar = tqdm.trange(self.n_epochs, disable=not self.verbose, unit="epoch")
        for _ in pbar:
            epoch_loss = 0.0

            for xb, yb in loader:
                self.optimizer.zero_grad()

                loss = -self.estimator(xb.to(device)).log_prob(yb.to(device)).mean()

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            pbar.set_postfix({"NLL": f"{epoch_loss:.4f}"})

        self.is_fitted_ = True

        return self

    @torch.no_grad()
    def _score(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Computes Monte Carlo highest-density rank scores.

        Args:
            X (torch.Tensor): Features with shape ``(n_samples, n_features)``.
            y (torch.Tensor): Targets with shape ``(n_samples, 1)``.

        Returns:
            np.ndarray: Highest-density rank scores with shape ``(n_samples,)``.
        """

        def _score_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            dist = self.estimator(x)
            observed = -dist.log_prob(y)
            sampled = -dist.log_prob(dist.sample((self.n_samples,)))

            return (sampled <= observed).float().mean(dim=0).reshape(-1)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=False,
        )

        self.estimator.eval()
        device = next(self.estimator.parameters()).device

        return torch.cat(
            [_score_batch(xb.to(device), sb.to(device)).cpu() for xb, sb in loader]
        ).numpy()

    @validate_params(
        {"X": ["array-like"], "y": ["array-like"]},
        prefer_skip_nested_validation=True,
    )
    def conformalize(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike) -> Self:
        """Stores held-out HPD-rank calibration scores.

        Args:
            X (np.typing.ArrayLike): Calibration features with shape
                ``(n_samples, n_features)``.
            y (np.typing.ArrayLike): Targets with shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``.

        Returns:
            Self: The calibrated estimator.
        """
        self._validate_params()
        check_is_fitted(self, "is_fitted_")

        X, y = self._to_tensor(X, y, reset=False)
        self.scores_ = self._score(X, y)

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
        """Predicts calibrated highest-density rank thresholds.

        Args:
            X (np.typing.ArrayLike): Test features with shape
                ``(n_samples, n_features)``.
            confidence_level (float | Sequence[float], optional): Target marginal
                coverage level or levels. Defaults to 0.9.

        Returns:
            np.ndarray: Thresholds, with the level axis omitted for one level.
        """
        check_is_fitted(self, "scores_")
        X = self._to_tensor(X)

        thresholds = self.thresholds(confidence_level)

        return collapse(np.broadcast_to(thresholds, (len(X), len(thresholds))))

    @validate_params(
        {
            "X": ["array-like"],
            "y": ["array-like"],
            "confidence_level": [float, Sequence],
        },
        prefer_skip_nested_validation=True,
    )
    def contains(
        self,
        X: np.typing.ArrayLike,
        y: np.typing.ArrayLike,
        *,
        confidence_level: float | Sequence[float] = 0.9,
    ) -> np.ndarray:
        """Tests whether scores lie inside calibrated HPD sets.

        Args:
            X (np.typing.ArrayLike): Test features with shape
                ``(n_samples, n_features)``.
            y (np.typing.ArrayLike): Test targets with shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``.
            confidence_level (float | Sequence[float], optional): Requested coverage
                levels. Defaults to 0.9.

        Returns:
            np.ndarray: Coverage indicators with shape ``(n_samples,)`` or
                ``(n_samples, n_levels)``.
        """
        check_is_fitted(self, "scores_")
        X, y = self._to_tensor(X, y, reset=False)

        self.eval()

        return super().contains(self._score(X, y), confidence_level=confidence_level)
