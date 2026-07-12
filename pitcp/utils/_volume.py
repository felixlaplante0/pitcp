from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.special import gamma
from sklearn.utils._param_validation import (
    Interval,
    validate_parameter_constraints,
    validate_params,
)
from sklearn.utils.validation import validate_data

from ._utils import collapse

if TYPE_CHECKING:
    from ..models import CONTRA, CQR, HPD, PITCP, SCP


@validate_params({"X": ["array-like"]}, prefer_skip_nested_validation=True)
def cqr_volume(cqr: CQR, X: np.typing.ArrayLike) -> np.ndarray:
    """Computes volumes of conformalized quantile hyperrectangles.

    Args:
        cqr (CQR): Calibrated conformalized quantile estimator.
        X (np.typing.ArrayLike): Features with shape ``(n_samples, n_features)``.

    Returns:
        np.ndarray: Hyperrectangle volumes with shape ``(n_samples,)``.
    """
    from ..models._cqr import CQR

    validate_parameter_constraints({"cqr": [CQR]}, {"cqr": cqr}, "cqr_volume")
    bounds = cqr.predict(X).reshape(len(X), 2, -1)

    return np.prod(bounds[:, 1] - bounds[:, 0], axis=1)


@validate_params(
    {
        "X": ["array-like"],
        "d": [Interval(Integral, 1, None, closed="left")],
        "confidence_level": [float, Sequence],
        "ord": [Interval(Real, 1, None, closed="both")],
    },
    prefer_skip_nested_validation=True,
)
def lp_volume(
    estimator: SCP | PITCP,
    X: np.typing.ArrayLike,
    d: int,
    *,
    confidence_level: float | Sequence[float] = 0.9,
    ord: float = 2.0,
) -> np.ndarray:
    """Computes volumes of Lp-ball regions predicted by a conformal estimator.

    Args:
        estimator (SCP | PITCP): Calibrated radius estimator.
        X (np.typing.ArrayLike): Features with shape ``(n_samples, n_features)``.
        d (int): Ambient dimension.
        confidence_level (float | Sequence[float], optional): Coverage level or
            levels. Defaults to 0.9.
        ord (float, optional): Norm order in ``[1, inf]``. Defaults to 2.

    Returns:
        np.ndarray: Volumes with shape ``(n_samples,)`` or ``(n_samples,
            n_levels)``.
    """
    from ..models._scp import SCP

    validate_parameter_constraints(
        {"estimator": [SCP]}, {"estimator": estimator}, "lp_volume"
    )
    radii = estimator.predict(X, confidence_level=confidence_level)
    unit_volume = (
        2.0**d
        if np.isinf(ord)
        else (2.0 * gamma(1.0 + 1.0 / ord)) ** d / gamma(1.0 + d / ord)
    )

    return unit_volume * radii**d


@validate_params(
    {
        "X": ["array-like"],
        "confidence_level": [float, Sequence],
        "n_samples": [Interval(Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
@torch.no_grad()
def hpd_volume(
    hpd: HPD,
    X: np.typing.ArrayLike,
    confidence_level: float | Sequence[float] = 0.9,
    n_samples: int = 1000,
) -> np.ndarray:
    """Estimates conditional HPD-set volumes by Monte Carlo integration.

    Args:
        hpd (HPD): Calibrated HPD estimator.
        X (np.typing.ArrayLike): Features with shape ``(n_samples, n_features)``.
        confidence_level (float | Sequence[float], optional): Coverage level or
            levels. Defaults to 0.9.
        n_samples (int, optional): Monte Carlo samples per feature row. Defaults to
            1000.

    Returns:
        np.ndarray: Estimated volumes with shape ``(n_samples,)`` or ``(n_samples,
            n_levels)``.
    """
    from ..models._hpd import HPD

    validate_parameter_constraints({"hpd": [HPD]}, {"hpd": hpd}, "hpd_volume")
    dtype = next(hpd.estimator.parameters()).dtype
    device = next(hpd.estimator.parameters()).device

    X = torch.tensor(validate_data(hpd, X, reset=False), dtype=dtype)
    levels = torch.tensor(hpd.thresholds(confidence_level), dtype=dtype, device=device)

    def _volume_batch(x: torch.Tensor) -> torch.Tensor:
        dist = hpd.estimator(x)
        nlog_probs = -dist.log_prob(dist.sample((n_samples,)))
        cutoffs = torch.quantile(nlog_probs, levels, dim=0)

        return (
            (nlog_probs[:, None].exp() * (nlog_probs[:, None] <= cutoffs)).mean(dim=0).T
        )

    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hpd.batch_size or len(dataset),
        shuffle=False,
    )

    hpd.eval()
    volumes = torch.cat([_volume_batch(xb.to(device)).cpu() for (xb,) in loader])

    return collapse(volumes.numpy())


@validate_params(
    {
        "X": ["array-like"],
        "confidence_level": [float, Sequence],
        "n_samples": [Interval(Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
@torch.no_grad()
def contra_volume(
    contra: CONTRA,
    X: np.typing.ArrayLike,
    confidence_level: float | Sequence[float] = 0.9,
    n_samples: int = 1000,
) -> np.ndarray:
    """Estimates conditional CONTRA-set volumes by Monte Carlo integration.

    Args:
        contra (CONTRA): Calibrated CONTRA estimator.
        X (np.typing.ArrayLike): Features with shape ``(n_samples, n_features)``.
        confidence_level (float | Sequence[float], optional): Coverage level or
            levels. Defaults to 0.9.
        n_samples (int, optional): Monte Carlo samples per feature row. Defaults to
            1000.

    Returns:
        np.ndarray: Estimated volumes with shape ``(n_samples,)`` or ``(n_samples,
            n_levels)``.
    """
    from ..models._contra import CONTRA

    validate_parameter_constraints(
        {"contra": [CONTRA]}, {"contra": contra}, "contra_volume"
    )
    dtype = next(contra.estimator.parameters()).dtype
    device = next(contra.estimator.parameters()).device

    X = torch.tensor(validate_data(contra, X, reset=False), dtype=dtype)

    d = contra.estimator(X[:1].to(device)).sample().shape[-1]
    radii = torch.tensor(
        contra.thresholds(confidence_level), dtype=dtype, device=device
    )

    def _volume_batch(x: torch.Tensor) -> torch.Tensor:
        dist = contra.estimator(x)
        z = torch.randn(n_samples, 1, len(x), d, device=device, dtype=dtype)
        z /= torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        z *= torch.rand(n_samples, 1, 1, 1, device=device, dtype=dtype) ** (1 / d)
        z = z * radii[None, :, None, None]
        y = dist.transform.inv(z)

        return dist.transform.inv.log_abs_det_jacobian(z, y).exp().mean(dim=0).T

    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=contra.batch_size or len(dataset),
        shuffle=False,
    )

    contra.eval()
    volumes = torch.cat(
        [_volume_batch(xb.to(device)).cpu() for (xb,) in loader]
    ).numpy()
    volumes *= np.pi ** (d / 2) / gamma(1 + d / 2) * radii.cpu().numpy() ** d

    return collapse(volumes)
