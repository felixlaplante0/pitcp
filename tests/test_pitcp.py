"""Tests for the pitcp package."""

import numpy as np
import pytest
import torch
import zuko
from sklearn.exceptions import NotFittedError

from pitcp import PITCP, SCP


def _data():
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


def _exercise(estimator: torch.nn.Module, random_state: int | None = None):
    """Runs fitting, calibration, and prediction for one estimator family.

    Args:
        estimator (torch.nn.Module): Conditional density estimator to exercise.
    """
    x_train, s_train, x_cal, s_cal = _data()
    optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)
    predictor = PITCP(
        estimator,
        optimizer,
        n_epochs=1,
        batch_size=8,
        verbose=False,
        random_state=random_state,
    )

    assert predictor.get_params()["random_state"] == random_state
    with pytest.raises(NotFittedError):
        predictor.predict(x_cal)

    assert predictor.fit(x_train, s_train) is predictor
    assert predictor.conformalize(x_cal, s_cal) is predictor

    confidence_levels = [0.5, 0.8]
    limits = predictor.predict(x_cal, confidence_level=confidence_levels)
    covered = predictor.contains(x_cal, s_cal, confidence_level=confidence_levels)

    assert limits.shape == (len(x_cal), len(confidence_levels))
    assert covered.shape == (len(x_cal), len(confidence_levels))
    assert covered.dtype == np.dtype(bool)


@pytest.mark.parametrize(
    ("estimator_type", "options", "random_state"),
    [
        (zuko.flows.SOSPF, {}, None),
        (zuko.flows.SOSPF, {}, 42),
        (zuko.mixtures.GMM, {"components": 2}, None),
    ],
)
def test_pitcp(estimator_type, options, random_state):
    """Runs the public workflow with every conditional density family."""
    estimator = estimator_type(features=1, context=1, hidden_features=(4, 4), **options)
    _exercise(estimator, random_state)


def test_scp():
    """Runs scalar score calibration, prediction, and containment."""
    model = SCP().fit([0.1, 0.2, 0.3, 0.4])

    assert model.predict([[0], [1]], confidence_level=[0.5, 0.8]).shape == (2, 2)
    assert model.contains([0.1, 0.5], confidence_level=[0.5, 0.8]).shape == (2, 2)
