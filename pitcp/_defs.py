from typing import NamedTuple


class CPResult(NamedTuple):
    is_covered: bool
    quantile: float
