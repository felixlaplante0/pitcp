from collections.abc import Sequence

import numpy as np
from scipy.stats import binom


def oracle_floor(
    labels: np.typing.ArrayLike, quantiles: Sequence[float]
) -> tuple[float, ...]:
    """Computes the expected CovGap of an exactly conditionally valid method.

    An exactly conditionally valid method covers each test point independently with
    probability equal to the target level, so the empirical coverage of a cluster of
    size ``m`` follows ``Binomial(m, level) / m``. The oracle floor is the expected
    range (maximum minus minimum) of the cluster coverages, computed exactly from the
    binomial cumulative distribution functions.

    Args:
        labels (np.typing.ArrayLike): Cluster labels with shape ``(n_samples,)``.
        quantiles (Sequence[float]): Target coverage levels in ``(0, 1)``.

    Returns:
        tuple[float, ...]: Oracle floor for each level in ``quantiles``.
    """
    sizes = np.unique(np.asarray(labels), return_counts=True)[1]

    # Coverages lie on the grids j / m, so all CDFs are constant between breakpoints
    grid = np.unique(np.concatenate([np.arange(m + 1) / m for m in sizes]))
    widths = np.diff(grid)
    counts = np.floor(grid[:-1, None] * sizes + 1e-9)

    floors = []
    for level in quantiles:
        cdfs = binom.cdf(counts, sizes, level)
        expected_max = np.sum(widths * (1 - np.prod(cdfs, axis=1)))
        expected_min = np.sum(widths * np.prod(1 - cdfs, axis=1))
        floors.append(float(expected_max - expected_min))

    return tuple(floors)
