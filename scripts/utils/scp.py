from typing import Self

import numpy as np


class SCP:
    threshold_: float

    def __init__(self, alpha: float):
        self.alpha = alpha

    def conformalize(self, X: np.ndarray, s: np.ndarray) -> Self:
        n = s.size
        k = np.ceil((n + 1) * (1 - self.alpha))
        level = np.minimum(k / n, 1.0)
        self.threshold_ = np.quantile(s, level)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.threshold_)

    def predict_coverage(self, X: np.ndarray, s: np.ndarray) -> np.ndarray:
        return s.flatten() <= self.threshold_
