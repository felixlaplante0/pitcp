from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
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


class ScoreDataset(torch.utils.Dataset):
    """A PyTorch Dataset that computes base scores for given features and labels."""

    def __init__(
        self,
        data: Dataset,
        base_score: Callable[
            [np.typing.ArrayLike, np.typing.ArrayLike], np.typing.ArrayLike
        ],
    ):
        """Initializes the dataset with features, labels, and a base score function.

        Args:
            data (Dataset): The dataset containing input features and target labels.
            base_score (Callable[[np.typing.ArrayLike, np.typing.ArrayLike],
                np.typing.ArrayLike]): Function to compute nonconformity scores.
        """
        self.data = data
        self.base_score = base_score

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int | slice) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the features and base score for a given index.

        Args:
            idx (int | slice): The index or slice to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the features and base
                score.
        """
        x, y = self.data[idx]
        s = self.base_score(x, y)
        return torch.as_tensor(x), torch.as_tensor(s)
