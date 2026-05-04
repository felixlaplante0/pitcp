from zuko.flows import Flow  # type: ignore
from zuko.mixtures import GMM  # type: ignore


def is_flow(estimator: object) -> bool:
    """Returns whether an object is a zuko normalizing flow.

    Args:
        estimator (object): The object instance to test.

    Returns:
        bool: Whether the object is a zuko normalizing flow.
    """
    return issubclass(type(estimator), Flow)


def is_mixture(estimator: object):
    """Returns whether an object is a zuko mixture.

    Args:
        estimator (object): The object instance to test.

    Returns:
        bool: Whether the object is a zuko mixture.
    """
    return issubclass(type(estimator), GMM)
