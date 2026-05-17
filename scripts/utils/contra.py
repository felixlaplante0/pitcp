from typing import Self

import numpy as np
import torch
from sklearn.base import BaseEstimator  # type: ignore
from torch import nn
from zuko.flows import Flow  # type: ignore


class CONTRA(BaseEstimator, nn.Module):
    estimator: Flow
    scores_: np.ndarray

    def __init__(
        self,
        estimator: Flow,
    ):
        super().__init__()
        self.estimator = estimator

    @torch.no_grad()
    def _score(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        def _score_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            dist = self.estimator(x)
            z = dist.transform(y)
            return torch.norm(z, dim=-1, keepdim=True)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
        )

        return torch.cat([_score_batch(xb, yb) for xb, yb in loader]).numpy()

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
