"""Tests for the pitcp.utils submodule."""

import numpy as np
import pytest
import torch
import zuko
from catboost import CatBoostRegressor

from pitcp import CONTRA, CQR, HPD, PITCP
from pitcp.utils import (
    contra_volume,
    coverage_gap,
    cqr_volume,
    hpd_volume,
    lp_volume,
)


def test_coverage_gap():
    """Computes scalar and multi-level coverage gaps."""
    labels = [0, 0, 1, 1]
    covered = [[True, True], [True, False], [False, False], [True, False]]
    assert np.isclose(coverage_gap(labels, np.asarray(covered)[:, 0]), 0.5)
    assert np.allclose(coverage_gap(labels, covered), [0.5, 0.5])
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        coverage_gap([0, 1], [True])


def test_cqr_volume():
    """Computes conformal interval lengths."""
    X = np.arange(18, dtype=float).reshape(-1, 1)
    y = X[:, 0]
    model = CQR(
        CatBoostRegressor(iterations=2, depth=1, verbose=False), confidence_level=0.8
    )
    model.fit(X[:8], y[:8]).conformalize(X[8:13], y[8:13])
    bounds = model.predict(X[13:])

    assert np.allclose(cqr_volume(model, X[13:]), bounds[:, 1] - bounds[:, 0])


def test_lp_volume():
    """Computes Lp-ball volumes."""
    torch.manual_seed(42)
    X = np.arange(8, dtype=float).reshape(-1, 1)
    scores = np.abs(np.sin(X)).ravel() + 0.1
    density = zuko.mixtures.GMM(features=1, context=1, components=2)
    model = PITCP(
        density,
        torch.optim.Adam(density.parameters()),
        n_epochs=1,
        batch_size=2,
        verbose=False,
    )
    model.fit(X[:4], scores[:4]).conformalize(X[4:], scores[4:])
    levels = [0.5, 0.75]

    assert np.allclose(
        lp_volume(model, X, 2, confidence_level=levels),
        np.pi * model.predict(X, confidence_level=levels) ** 2,
    )


def test_hpd_volume():
    """Estimates HPD-set volumes."""
    torch.manual_seed(42)
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.sin(X).ravel()
    density = zuko.mixtures.GMM(features=1, context=1, components=2)
    model = HPD(
        density,
        torch.optim.Adam(density.parameters()),
        n_epochs=1,
        n_samples=8,
        batch_size=2,
        verbose=False,
    )
    model.fit(X[:4], y[:4]).conformalize(X[4:], y[4:])
    volumes = hpd_volume(model, X, [0.5, 0.75], n_samples=8)

    assert volumes.shape == (8, 2)
    assert np.isfinite(volumes).all()


def test_contra_volume():
    """Estimates CONTRA-set volumes."""
    torch.manual_seed(42)
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.column_stack((np.sin(X).ravel(), np.cos(X).ravel()))
    density = zuko.flows.RealNVP(features=2, context=1, hidden_features=(4, 4))
    model = CONTRA(
        density,
        torch.optim.Adam(density.parameters()),
        n_epochs=1,
        batch_size=2,
        verbose=False,
    )
    model.fit(X[:4], y[:4]).conformalize(X[4:], y[4:])
    volumes = contra_volume(model, X, [0.5, 0.75], n_samples=8)

    assert volumes.shape == (8, 2)
    assert np.isfinite(volumes).all()
