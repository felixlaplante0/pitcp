from zuko.lazy import LazyDistribution  # type: ignore


def is_flow(estimator: LazyDistribution) -> bool:
    """Returns whether an object is a zuko normalizing flow.

    Args:
        estimator (LazyDistribution): The object instance to test.

    Returns:
        bool: Whether the object is a zuko normalizing flow.
    """
    return estimator.__class__.__module__.startswith("zuko.flows")


def is_mixture(estimator: LazyDistribution):
    """Returns whether an object is a zuko mixture.

    Args:
        estimator (LazyDistribution): The object instance to test.

    Returns:
        bool: Whether the object is a zuko mixture.
    """
    return estimator.__class__.__module__.startswith("zuko.mixtures")
