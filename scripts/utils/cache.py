from collections import OrderedDict
from collections.abc import Callable
from typing import Any, TypeVar

import torch

CACHE_LIMIT = 32
T = TypeVar("T")


def cache(func: Callable[..., T]) -> Callable[..., T]:
    cache: OrderedDict[tuple[int, ...], T] = OrderedDict()

    def wrapper(*args: Any) -> T:
        key = tuple(
            hash(arr.detach().cpu().numpy().tobytes())
            for arr in args
            if isinstance(arr, torch.Tensor)
        )

        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        if len(cache) >= CACHE_LIMIT:
            cache.popitem(last=False)

        result = func(*args)
        cache[key] = result

        return result

    return wrapper
