from collections.abc import Sequence

import numpy as np


def collapse(arr: np.ndarray):
    """Collapses the last dimension if it has size 1.

    Args:
        arr (np.ndarray): The array to convert.

    Returns:
        np.ndarray: The converted array.
    """
    if arr.shape[-1] > 1:
        return arr
    return arr.squeeze(-1)


def confidence_levels(value: float | Sequence[float]) -> np.ndarray:
    """Validates and normalizes confidence levels.

    Args:
        value (float | Sequence[float]): Scalar or ordered coverage levels with
            shape ``(n_levels,)``.

    Returns:
        np.ndarray: Ordered levels with shape ``(n_levels,)``.

    Raises:
        ValueError: If a level is outside the open unit interval.
    """
    levels = np.atleast_1d(np.asarray(value, dtype=float))
    if (
        levels.ndim != 1
        or not np.isfinite(levels).all()
        or np.any((levels <= 0) | (levels >= 1))
    ):
        raise ValueError("confidence levels must be finite values between 0 and 1")

    return levels
