import numpy as np
from sklearn.utils.validation import check_array, check_consistent_length, column_or_1d


def coverage_gap(
    labels: np.typing.ArrayLike, covered: np.typing.ArrayLike
) -> float | np.ndarray:
    """Computes the range of empirical coverage across labeled groups.

    Args:
        labels (np.typing.ArrayLike): Group labels with shape ``(n_samples,)``.
        covered (np.typing.ArrayLike): Boolean mask with shape ``(n_samples,)`` or
            ``(n_samples, n_levels)``.

    Returns:
        float | np.ndarray: Maximum minus minimum group coverage. A scalar is returned
            for one coverage mask and an array for multiple masks.

    Raises:
        ValueError: If inputs are empty, have invalid dimensions, or contain different
            sample counts.
    """
    groups = column_or_1d(labels)
    mask = np.asarray(check_array(covered, ensure_2d=False, dtype=bool))
    check_consistent_length(groups, mask)

    if mask.ndim == 1:
        mask = mask[:, None]

    rates = np.stack(
        [mask[groups == label].mean(axis=0) for label in np.unique(groups)]
    )
    gap = np.ptp(rates, axis=0)

    return float(gap[0]) if gap.size == 1 else gap
