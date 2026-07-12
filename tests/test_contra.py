"""Tests for CONTRA conformal prediction."""

import numpy as np
import pytest
import torch
import zuko

from pitcp import CONTRA


def test_contra():
    """Runs fitting, calibration, and containment."""
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
        random_state=42,
    )

    assert model.fit(X[:4], y[:4]) is model
    assert model.conformalize(X[4:], y[4:]) is model
    assert model.contains(X, y, confidence_level=[0.5, 0.75]).shape == (8, 2)


def test_gmm():
    """Rejects a mixture without an invertible transform."""
    density = zuko.mixtures.GMM(features=1, context=1, components=2)
    model = CONTRA(
        density,
        torch.optim.Adam(density.parameters()),
        n_epochs=1,
        verbose=False,
    )
    with pytest.raises(TypeError, match="Flow"):
        model.fit([[0], [1]], [0, 1])
