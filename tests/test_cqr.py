"""Tests for conformalized quantile regression."""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor

from pitcp import CQR


def test_clone_and_nested_params():
    """Preserves nested estimator parameters while cloning."""
    estimator = CQR(GradientBoostingRegressor(n_estimators=2), confidence_level=0.8)
    copied = clone(estimator)

    assert copied.estimator.n_estimators == estimator.estimator.n_estimators
    copied.set_params(estimator__n_estimators=3)
    assert copied.estimator.n_estimators != estimator.estimator.n_estimators


def test_fit_and_crossing():
    """Fits ordered multi-level intervals and prevents crossed bounds."""
    X = np.arange(18, dtype=float).reshape(-1, 1)
    y = X[:, 0]
    estimator = CQR(GradientBoostingRegressor(n_estimators=2), confidence_level=0.8)

    assert estimator.fit(X[:8], y[:8]) is estimator
    assert estimator.conformalize(X[8:13], y[8:13]) is estimator
    prediction = estimator.predict(X[13:])

    assert prediction.shape == (5, 2)
    assert np.all(prediction[:, 0] <= prediction[:, 1])
    assert estimator.contains(X[13:], y[13:]).shape == (5,)


def test_one_sided():
    """Omits boundary quantiles for lower and upper one-sided intervals."""
    X = np.arange(18, dtype=float).reshape(-1, 1)
    y = X[:, 0]
    for gamma, infinite_bound in ((0, 0), (1, 1)):
        estimator = CQR(
            GradientBoostingRegressor(n_estimators=2),
            confidence_level=0.8,
            gamma=gamma,
        )
        estimator.fit(X[:8], y[:8]).conformalize(X[8:13], y[8:13])
        assert np.isinf(estimator.predict(X[13:])[:, infinite_bound]).all()
