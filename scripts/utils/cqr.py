from typing import Self

import numpy as np
from catboost import CatBoostRegressor


class CQR:
    alpha: float
    model: CatBoostRegressor
    threshold_: float

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.model = CatBoostRegressor(
            loss_function=f"MultiQuantile:alpha={alpha / 2},{1 - alpha / 2}",
            verbose=False,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.model.fit(X, y)

        return self

    def _predict_raw(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = self.model.predict(X)
        return preds[:, 0], preds[:, 1]

    def conformalize(self, X: np.ndarray, y: np.ndarray) -> Self:
        lo, hi = self._predict_raw(X)
        scores = np.maximum(lo - y, y - hi)

        n = scores.size
        k = np.ceil((n + 1) * (1 - self.alpha))
        level = np.minimum(k / n, 1.0)
        self.threshold_ = np.quantile(scores, level)

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo_raw, hi_raw = self._predict_raw(X)
        lo = lo_raw - self.threshold_
        hi = hi_raw + self.threshold_

        return np.minimum(lo, hi), np.maximum(lo, hi)


class CQRHyperRectangle:
    alpha: float
    models: list[CQR]
    threshold_: float

    def __init__(self, alpha: float):
        self.alpha: float = alpha
        self.models: list[CQR] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.models = [CQR(alpha=self.alpha).fit(X, y[:, j]) for j in range(y.shape[1])]
        return self

    def conformalize(self, X: np.ndarray, y: np.ndarray) -> Self:
        scores = np.zeros(y.shape[0])
        for j, cqr in enumerate(self.models):
            lo, hi = cqr._predict_raw(X)
            scores = np.maximum(scores, np.maximum(lo - y[:, j], y[:, j] - hi))

        n = scores.size
        k = np.ceil((n + 1) * (1 - self.alpha))
        level = np.minimum(k / n, 1.0)
        self.threshold_ = np.quantile(scores, level)

        return self

    def predict(self, X: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        bounds: list[tuple[np.ndarray, np.ndarray]] = []
        for cqr in self.models:
            lo_raw, hi_raw = cqr._predict_raw(X)
            lo = lo_raw - self.threshold_
            hi = hi_raw + self.threshold_
            bounds.append((lo, hi))

        return bounds

    def predict_coverage(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        inside = np.ones(y.shape[0], dtype=bool)
        for j, cqr in enumerate(self.models):
            lo_raw, hi_raw = cqr._predict_raw(X)
            lo = lo_raw - self.threshold_
            hi = hi_raw + self.threshold_
            inside &= (y[:, j] >= lo) & (y[:, j] <= hi)

        return inside.reshape(-1, 1)
