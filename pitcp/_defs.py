from typing import NamedTuple


class CPResult(NamedTuple):
    """Conformal Prediction Result class.
    
    Attributes:
        is_covered (bool): Whether a test point is covered.
        quantile (float): The quantile of the PIT-corrected non-conformity scores.
    """

    is_covered: bool
    quantile: float
