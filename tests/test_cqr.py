"""Tests for conformalized quantile regression."""

import numpy as np
import pytest
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from pitcp import CQR
from pitcp.utils._quantile import QuantileEstimator


@pytest.mark.parametrize(
    "estimator",
    [
        HistGradientBoostingRegressor(max_iter=2),
        LGBMRegressor(n_estimators=2, verbosity=-1),
        XGBRegressor(n_estimators=2, max_depth=1, verbosity=0),
        CatBoostRegressor(iterations=2, depth=1, verbose=False),
    ],
)
def test_backend(estimator):
    """Configures each supported quantile backend."""
    models = QuantileEstimator(estimator, 0.8, 0.5)._configure_estimators()

    assert models
    assert all(model is not estimator for model in models)


def test_clone_and_nested_params():
    """Preserves nested estimator parameters while cloning."""
    estimator = CQR(LGBMRegressor(n_estimators=2), confidence_level=0.8)
    copied = clone(estimator).set_params(estimator__n_estimators=3)

    assert copied.estimator.n_estimators != estimator.estimator.n_estimators


def test_fit_and_crossing():
    """Fits ordered intervals and prevents crossed bounds."""
    X = np.arange(18, dtype=float).reshape(-1, 1)
    y = X[:, 0]
    estimator = CQR(
        CatBoostRegressor(iterations=2, depth=1, verbose=False), confidence_level=0.8
    )

    assert estimator.fit(X[:8], y[:8]) is estimator
    assert estimator.conformalize(X[8:13], y[8:13]) is estimator
    prediction = estimator.predict(X[13:])

    assert prediction.shape == (5, 2)
    assert np.all(prediction[:, 0] <= prediction[:, 1])
    assert estimator.contains(X[13:], y[13:]).shape == (5,)


@pytest.mark.parametrize(
    "estimator",
    [
        LGBMRegressor(n_estimators=2, verbosity=-1),
        XGBRegressor(n_estimators=2, max_depth=1, verbosity=0),
        CatBoostRegressor(iterations=2, depth=1, verbose=False),
    ],
)
@pytest.mark.parametrize(("gamma", "infinite_bound"), [(0, 0), (1, 1)])
def test_one_sided(estimator, gamma, infinite_bound):
    """Omits boundary quantiles for one-sided intervals."""
    X = np.arange(18, dtype=float).reshape(-1, 1)
    y = X[:, 0]
    model = CQR(estimator, confidence_level=0.8, gamma=gamma)
    model.fit(X[:8], y[:8]).conformalize(X[8:13], y[8:13])

    assert np.isinf(model.predict(X[13:])[:, infinite_bound]).all()


@pytest.mark.parametrize(
    "estimator",
    [
        LGBMRegressor(n_estimators=2, verbosity=-1),
        CatBoostRegressor(iterations=2, depth=1, verbose=False),
    ],
)
def test_multioutput(estimator):
    """Fits multiple target outputs."""
    X = np.arange(18, dtype=float).reshape(-1, 1)
    y = np.column_stack((X[:, 0], -X[:, 0]))
    model = CQR(estimator, confidence_level=0.8)
    model.fit(X[:8], y[:8]).conformalize(X[8:13], y[8:13])

    assert model.predict(X[13:]).shape == (5, 2, 2)


def test_unsupported():
    """Rejects unsupported estimators."""
    with pytest.raises(TypeError, match="HistGradientBoosting"):
        CQR(LinearRegression()).fit([[0], [1]], [0, 1])
