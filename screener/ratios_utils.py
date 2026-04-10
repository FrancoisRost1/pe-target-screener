"""
ratios_utils.py — Shared low-level helpers for the ratio computation modules.

Lives separately from ratios.py / ratios_secondary.py so both can import
`_safe_divide` without creating an import cycle.
"""

import pandas as pd
import numpy as np


def _safe_divide(numerator, denominator, fill_zero_denom=True):
    """
    Safe element-wise division. Returns NaN on division by zero or missing input.
    Used by every ratio function in ratios.py and ratios_secondary.py.
    """
    if numerator is None or denominator is None:
        return pd.Series(dtype=float)
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (den == 0) | den.isna() | num.isna(), np.nan, num / den
        )
    return pd.Series(result, index=num.index if hasattr(num, "index") else None)
