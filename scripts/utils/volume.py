import numpy as np
from pitcp import PITCP
from sklearn.preprocessing import StandardScaler

from .cqr import CQRHyperRectangle
from .scp import SCP


def vol_base(
    scp: SCP, s_scaler: StandardScaler, r_scaler: StandardScaler
) -> tuple[float, float, float]:
    d = r_scaler.scale_.size
    s = s_scaler.inverse_transform([[scp.threshold_]])[0, 0]
    vol = (2 * s) ** d * np.prod(r_scaler.scale_)

    return vol, vol, vol


def vol_cqr(
    cqr: CQRHyperRectangle, X: np.ndarray, y_scaler: StandardScaler
) -> tuple[float, float, float]:
    pred = cqr.predict(X)
    vols = np.ones(X.shape[0])

    for lo, hi in pred:
        vols *= np.abs(hi - lo)

    vols *= np.prod(y_scaler.scale_)

    return np.quantile(vols, [0.25, 0.5, 0.75]).tolist()


def vol_pit(
    pit: PITCP,
    X: np.ndarray,
    s_scaler: StandardScaler,
    r_scaler: StandardScaler,
    quantile: float,
) -> tuple[float, float, float]:
    d = r_scaler.scale_.size
    s = s_scaler.inverse_transform(pit.predict(X, quantile=quantile)[:, None])
    vols = (2 * s.squeeze()) ** d * np.prod(r_scaler.scale_)

    return np.quantile(vols, [0.25, 0.5, 0.75]).tolist()
