from typing import Self

import numpy as np
import torch
from sklearn.base import BaseEstimator  # type: ignore
from torch import nn
from tqdm import trange
from zuko.flows import Flow  # type: ignore
from zuko.mixtures import GMM  # type: ignore


class HPD(BaseEstimator, nn.Module):
    estimator: Flow | GMM
    optimizer: torch.optim.Optimizer
    n_epochs: int
    n_samples: int
    batch_size: int | None
    verbose: bool | int
    scores_: np.ndarray

    def __init__(
        self,
        estimator: Flow | GMM,
        optimizer: torch.optim.Optimizer,
        *,
        n_epochs: int,
        n_samples: int = 1000,
        batch_size: int | None = None,
        verbose: bool | int = True,
    ):
        super().__init__()

        self.estimator = estimator
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.verbose = verbose

    @torch.no_grad()
    def _score(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        def _score_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            dist = self.estimator(x)
            nlog_prob = -dist.log_prob(y)
            nlog_prob_samples = -dist.log_prob(dist.sample((self.n_samples,)))
            return (nlog_prob_samples <= nlog_prob).float().mean(dim=0).reshape(-1, 1)

        dataset = torch.utils.data.TensorDataset(X, y)  # type: ignore
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=False,
        )

        return torch.cat([_score_batch(xb, yb) for xb, yb in loader]).numpy()

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        dtype = next(self.parameters()).dtype

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=dtype), torch.tensor(y, dtype=dtype)
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size or len(dataset),
            shuffle=True,
        )

        self.train()

        pbar = trange(self.n_epochs, disable=not self.verbose, unit="epoch")
        for _ in pbar:
            epoch_loss = 0.0

            for xb, yb in loader:
                self.optimizer.zero_grad()

                dist = self.estimator(xb)
                loss = -dist.log_prob(yb).mean()

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)  # type: ignore
            pbar.set_postfix({"NLL": f"{epoch_loss:.4f}"})  # type: ignore

        return self

    def conformalize(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.eval()

        dtype = next(self.parameters()).dtype
        X, y = torch.tensor(X, dtype=dtype), torch.tensor(y, dtype=dtype)  # type: ignore
        self.scores_ = self._score(X, y)

        return self

    def threshold(self, quantile: float) -> np.ndarray:
        n = self.scores_.size
        quantile_1d = np.atleast_1d(np.asarray(quantile))
        k = np.ceil(quantile_1d * (n + 1))
        level = np.minimum(k / n, 1.0)
        threshold = np.quantile(self.scores_, level)
        threshold[k > n] = np.inf

        return threshold

    def predict_coverage(self, X: np.ndarray, y: np.ndarray, *, quantile: float):
        dtype = next(self.parameters()).dtype
        X, y = torch.tensor(X, dtype=dtype), torch.tensor(y, dtype=dtype)  # type: ignore

        self.eval()

        return self._score(X, y) <= self.threshold(quantile)
