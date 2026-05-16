import torch
from torch.distributions import Normal
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


def correct_mixture(gmm: GMM, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Calculates the CDF of a Gaussian Mixture Model.

    Args:
        gmm (GMM): The Gaussian Mixture Model.
        x (torch.Tensor): The condition.
        s (torch.Tensor): The value to evaluate the CDF at.

    Returns:
        torch.Tensor: The CDF value.
    """
    dist = gmm(x)
    weights = dist.logits.softmax(dim=-1)

    if hasattr(dist.base, "base_dist"):
        means = dist.base.base_dist.loc.squeeze(-1)
        stds = dist.base.base_dist.scale.squeeze(-1).sqrt()
    else:
        means = dist.base.loc.squeeze(-1)
        stds = dist.base.covariance_matrix.squeeze((-2, -1)).sqrt()

    return (weights * Normal(means, stds).cdf(s)).sum(dim=-1, keepdim=True)


def invert_mixture(gmm: GMM, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Calculates the inverse CDF (quantile function) of a Gaussian Mixture Model.

    Args:
        gmm (GMM): The Gaussian Mixture Model.
        x (torch.Tensor): The condition.
        u (torch.Tensor): The quantile level.

    Returns:
        torch.Tensor: The quantile value.
    """
    dist = gmm(x)
    weights = dist.logits.softmax(dim=-1)

    if hasattr(dist.base, "base_dist"):
        means = dist.base.base_dist.loc.squeeze(-1)
        stds = dist.base.base_dist.scale.squeeze(-1).sqrt()
    else:
        means = dist.base.loc.squeeze(-1)
        stds = dist.base.covariance_matrix.squeeze((-2, -1)).sqrt()

    lo = (means - 10 * stds).min(dim=-1, keepdim=True).values
    hi = (means + 10 * stds).max(dim=-1, keepdim=True).values

    normal = Normal(means, stds)

    def _cdf(u: torch.Tensor):
        return (weights * normal.cdf(u)).sum(dim=-1, keepdim=True)

    while not torch.allclose(lo, hi, equal_nan=True):
        mid = 0.5 * (lo + hi)
        val = _cdf(mid)
        lo = torch.where(val < u, mid, lo)
        hi = torch.where(val >= u, mid, hi)

    return lo
