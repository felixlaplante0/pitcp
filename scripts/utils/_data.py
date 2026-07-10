import numpy as np
from scipy.stats import chi2, norm


def std(x: np.ndarray) -> np.ndarray:
    return np.abs(1 - 2 * x**2) + 0.1


def gen_data(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.rand(n) * 2 - 1
    return x, np.random.randn(n) * std(x)


def score_abs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(y)


def inv_score_abs(x: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return -s, s


def oracle_score_abs(x: np.ndarray, q: float) -> np.ndarray:
    return std(x) * norm.ppf((q + 1) / 2)


def score_hpd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    v = std(x) ** 2
    return 0.5 * (np.log(2 * np.pi) + np.log(v) + y**2 / v)


def inv_score_hpd(x: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = std(x) ** 2
    y = np.sqrt(np.maximum((2 * s - np.log(2 * np.pi) - np.log(v)) * v, 0))
    return -y, y


def oracle_score_hpd(x: np.ndarray, q: float) -> np.ndarray:
    v = std(x) ** 2
    return 0.5 * (np.log(2 * np.pi) + np.log(v) + chi2.ppf(q, df=1))


def score_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y


def inv_score_y(x: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.full_like(s, -10), s


def oracle_score_y(x: np.ndarray, q: float) -> np.ndarray:
    return std(x) * norm.ppf(q)
