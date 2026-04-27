from numbers import Integral

import torch
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils._param_validation import Interval, validate_params  # type: ignore
from zuko.lazy import LazyDistribution  # type: ignore

from ._utils import is_flow, is_mixture


class PITCP(BaseEstimator):
    estimator: LazyDistribution
    optimizer: torch.optim.Optimizer
    n_epochs: int
    batch_size: int
    estimator_type_: str

    @validate_params(
        {
            "estimator": [LazyDistribution],
            "optimizer": [torch.optim.Optimizer],
            "n_epochs": [Interval(Integral, 1, None, closed="left")],
            "batch_size": [Interval(Integral, 1, None, closed="left")],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        estimator: LazyDistribution,
        optimizer: torch.optim.Optimizer,
        *,
        n_epochs: int = 10,
        batch_size: int = 128,
    ):
        self.estimator = estimator
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        if is_flow(self.estimator):
            self.estimator_type_ = "flow"
        elif is_mixture(self.estimator):
            self.estimator_type_ = "mixture"
        else:
            raise ValueError(
                "Estimator must be either a `zuko.flows` or `zuko.mixtures` submodule"
            )
