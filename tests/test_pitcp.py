import numpy as np
import pytest
import torch
import zuko
from sklearn.exceptions import NotFittedError

from pitcp import PITCP


def _datasets():
    """Creates deterministic training and calibration datasets.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Training features,
        training scores, calibration features, and calibration scores.
    """
    torch.manual_seed(42)
    x_train = np.linspace(-1.0, 1.0, 24)[:, None]
    s_train = np.abs(x_train[:, 0]) + 0.1
    x_cal = np.linspace(-0.9, 0.9, 12)[:, None]
    s_cal = np.abs(x_cal[:, 0]) + 0.1

    return x_train, s_train, x_cal, s_cal


def _exercise(estimator: torch.nn.Module):
    """Runs fitting, calibration, and prediction for one estimator family.

    Args:
        estimator (torch.nn.Module): Conditional density estimator to exercise.
    """
    x_train, s_train, x_cal, s_cal = _datasets()
    optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)
    predictor = PITCP(estimator, optimizer, n_epochs=1, batch_size=8, verbose=False)

    with pytest.raises(NotFittedError):
        predictor.predict(x_cal)

    assert predictor.fit(x_train, s_train) is predictor
    assert predictor.conformalize(x_cal, s_cal) is predictor

    quantiles = [0.5, 0.8]
    limits = predictor.predict(x_cal, quantile=quantiles)
    covered = predictor.predict_coverage(x_cal, s_cal, quantile=quantiles)

    assert limits.shape == (len(x_cal), len(quantiles))
    assert covered.shape == (len(x_cal), len(quantiles))
    assert covered.dtype == np.dtype(bool)


def test_flow():
    """Runs the public workflow with a conditional normalizing flow."""
    estimator = zuko.flows.SOSPF(features=1, context=1, hidden_features=(4, 4))
    _exercise(estimator)


def test_mixture():
    """Runs the public workflow with a conditional Gaussian mixture."""
    estimator = zuko.mixtures.GMM(
        features=1, context=1, components=2, hidden_features=(4, 4)
    )
    _exercise(estimator)
