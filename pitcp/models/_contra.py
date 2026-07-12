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

from ._scp import SCP


class CONTRA(SCP, nn.Module):
    r"""Fits inverse-flow images calibrated by the latent Euclidean norm.

    CONTRA maps targets into a conditional flow's latent coordinates and calibrates the
    values :math:`\lVert z \rVert_2`, where ``z`` is the transformed target.

    Density estimation settings:
        - ``estimator``: Conditional ``zuko`` flow mapping targets to latent
          coordinates. Although construction accepts a Gaussian mixture model for
          scikit-learn compatibility, ``fit`` rejects it because it has no invertible
          transform.
        - ``optimizer``: PyTorch optimizer bound to ``estimator.parameters()`` and used
          to minimize the negative conditional log-likelihood.

    Training settings:
        - ``n_epochs``: Positive number of full passes over the training data. Defaults
          to 10.
        - ``batch_size``: Positive mini-batch size used during training and scoring.
          ``None`` uses the full dataset. Defaults to ``None``.
        - ``verbose``: Boolean or integer controlling the ``tqdm`` training progress
          bar. Defaults to ``True``.
        - ``random_state``: Seed controlling mini-batch shuffling during ``fit``.
          ``None`` uses PyTorch's current random state. Defaults to ``None``.

    Attributes:
        estimator (Flow): Conditional normalizing flow.
        optimizer (Optimizer): Optimizer used for density training.
        n_epochs (int): Number of training epochs.
        batch_size (int | None): Training batch size or ``None`` for full batches.
        verbose (bool | int): Whether to display training progress.
        random_state (int | None): Mini-batch shuffling seed.
        scores_ (np.ndarray): Calibrated values of :math:`\lVert z \rVert_2`.

    Examples:
        >>> import torch
        >>> import zuko
        >>> from pitcp import CONTRA
        >>> flow = zuko.flows.SOSPF(features=1, context=1, hidden_features=(4, 4))
        >>> optimizer = torch.optim.Adam(flow.parameters())
        >>> model = CONTRA(flow, optimizer, n_epochs=1, verbose=False)
    """

    _parameter_constraints: ClassVar[dict] = {
        "estimator": [Flow, GMM],
        "optimizer": [HasMethods(["zero_grad", "step"])],
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
        """Initializes CONTRA.

        Args:
            estimator (Flow | GMM): Conditional density estimator. GMM instances are
                rejected by ``fit`` because they have no invertible transform.
            optimizer (Optimizer): Torch optimizer for density training.
            n_epochs (int, optional): Training epochs. Defaults to 10.
            batch_size (int | None, optional): Training batch size. Defaults to None.
            verbose (bool | int, optional): Whether to show training progress. Defaults
                to True.
            random_state (int | None, optional): DataLoader seed. Defaults to None.
        """
        super().__init__()

        self.estimator = estimator
        self.optimizer = optimizer
        self.n_epochs = n_epochs
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
            reset (bool, optional): Whether to reset fitted feature metadata. Defaults
                to True.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Feature tensor and, when
                supplied, target tensor.
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
        """Fits the conditional flow to targets.

        Args:
            X (np.typing.ArrayLike): Training features with shape ``(n_samples,
                n_features)``.
            y (np.typing.ArrayLike): Training targets with shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``.

        Returns:
            Self: The fitted regressor.
        """
        self._validate_params()
        if not isinstance(self.estimator, Flow):
            raise TypeError("CONTRA requires a zuko Flow estimator")

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
        device = next(self.estimator.parameters()).device

        self.estimator.train()
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
    def _score(self, X, y) -> np.ndarray:
        """Computes latent Euclidean norms.

        Args:
            X (torch.Tensor): Conditioning features with shape ``(n_samples,
                n_features)``.
            y (torch.Tensor): Targets with shape ``(n_samples, n_outputs)``.

        Returns:
            np.ndarray: One latent norm per sample.
        """

        def _score_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            latent = self.estimator(x).transform(y)

            return torch.linalg.vector_norm(latent, dim=-1)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=False,
        )

        self.estimator.eval()
        device = next(self.estimator.parameters()).device
        scores = torch.cat(
            [_score_batch(xb.to(device), yb.to(device)).cpu() for xb, yb in loader]
        )

        return scores.numpy()

    @validate_params(
        {"X": ["array-like"], "y": ["array-like"]},
        prefer_skip_nested_validation=True,
    )
    def conformalize(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike) -> Self:
        """Calibrates latent Euclidean norms using a fitted density estimator.

        Args:
            X (np.typing.ArrayLike): Calibration features with shape ``(n_samples,
                n_features)``.
            y (np.typing.ArrayLike): Targets in original coordinates with shape
                ``(n_samples,)`` or ``(n_samples, n_outputs)``.

        Returns:
            Self: The calibrated regressor.
        """
        self._validate_params()
        X, y = self._to_tensor(X, y, reset=not hasattr(self, "n_features_in_"))
        self.scores_ = self._score(X, y)

        return self

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
        """Tests whether targets lie inside calibrated latent balls.

        Args:
            X (np.typing.ArrayLike): Test features with shape ``(n_samples,
                n_features)``.
            y (np.typing.ArrayLike): Test targets with shape ``(n_samples,)`` or
                ``(n_samples, n_outputs)``.
            confidence_level (float | Sequence[float], optional): Requested coverage
                levels. Defaults to 0.9.

        Returns:
            np.ndarray: Coverage indicators with shape ``(n_samples,)`` or ``(n_samples,
                n_levels)``.
        """
        check_is_fitted(self, "scores_")
        X, y = self._to_tensor(X, y, reset=False)

        self.eval()

        return super().contains(self._score(X, y), confidence_level=confidence_level)
