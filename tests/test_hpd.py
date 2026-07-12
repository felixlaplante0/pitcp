"""Tests for highest-predictive-density conformal prediction."""

import numpy as np
import pytest
import torch
import zuko
from sklearn.exceptions import NotFittedError

from pitcp import HPD


def test_hpd():
    """Runs fitting, calibration, prediction, and containment."""
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
        random_state=42,
    )

    with pytest.raises(NotFittedError):
        model.predict(X)
    assert model.fit(X[:4], y[:4]) is model
    assert model.conformalize(X[4:], y[4:]) is model
    assert model.predict(X, confidence_level=[0.5, 0.75]).shape == (8, 2)
    assert model.contains(X, y, confidence_level=[0.5, 0.75]).shape == (8, 2)
