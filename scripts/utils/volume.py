import numpy as np
import torch
from pitcp import PITCP
from scipy.special import gamma
from sklearn.preprocessing import StandardScaler

from .contra import CONTRA
from .cqr import CQRHyperRectangle
from .hpd import HPD
from .scp import SCP


def _ball_vol(d: int, r: float) -> float:
    return (np.pi ** (d / 2) / gamma(d / 2 + 1)) * (r**d)


def _sample_ball(d: int, r: float, n: int) -> torch.Tensor:
    direction = torch.randn(n, d)
    direction /= torch.norm(direction, dim=1, keepdim=True)
    u = torch.rand(n, 1)
    return direction * r * (u ** (1 / d))


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
    vols = np.prod([np.abs(hi - lo) for lo, hi in pred], axis=0)
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
    vols = (2 * s.flatten()) ** d * np.prod(r_scaler.scale_)

    return np.quantile(vols, [0.25, 0.5, 0.75]).tolist()


def vol_hpd(
    hpd: HPD,
    X: np.ndarray,
    y_scaler: StandardScaler,
    quantile: float,
    n_samples: int = 1000,
) -> tuple[float, float, float]:
    dtype = next(hpd.parameters()).dtype

    threshold = hpd.threshold(quantile).item()
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=dtype))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hpd.batch_size or len(dataset),
        shuffle=False,
    )

    @torch.no_grad()
    def _vol_batch(xb: torch.Tensor):
        dist = hpd.estimator(xb)
        nlog_probs = hpd._sample_nlog_prob(dist, xb, n_samples)
        tau = torch.quantile(nlog_probs, threshold, dim=0)
        covered = nlog_probs <= tau
        return (nlog_probs.exp() * covered).mean(dim=0)

    hpd.eval()

    vols = torch.cat([_vol_batch(xb) for (xb,) in loader]).numpy()
    vols *= np.prod(y_scaler.scale_)

    return np.quantile(vols, [0.25, 0.5, 0.75]).tolist()


def vol_contra(
    contra: CONTRA,
    X: np.ndarray,
    y_scaler: StandardScaler,
    quantile: float,
    n_samples: int = 1000,
) -> tuple[float, float, float]:
    dtype = next(contra.parameters()).dtype

    d = y_scaler.scale_.size
    r = contra.threshold(quantile).item()
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=dtype))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=contra.batch_size or len(dataset),
        shuffle=False,
    )

    @torch.no_grad()
    def _vol_batch(xb: torch.Tensor):
        dist = contra.estimator(xb)
        z = _sample_ball(d, r, n_samples).unsqueeze(1)
        y = dist.transform.inv(z)
        log_det = dist.transform.inv.log_abs_det_jacobian(z, y)
        return torch.exp(log_det).mean(dim=0)

    contra.eval()

    vols = torch.cat([_vol_batch(xb) for (xb,) in loader]).numpy()
    vols *= np.prod(y_scaler.scale_) * _ball_vol(d, r)

    return np.quantile(vols, [0.25, 0.5, 0.75]).tolist()
