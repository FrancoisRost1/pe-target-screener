"""
ratios.py — Financial ratio computation

Computes core PE/LBO screening metrics from raw financial data.
Each function handles division by zero and missing data defensively.

Secondary ratios (ROIC, growth, capex, FCF yield) are in ratios_secondary.py.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_all_ratios(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run all ratio computations in sequence."""
    from screener.ratios_secondary import (
        compute_roic, compute_ev_to_ebitda, compute_revenue_growth,
        compute_ebitda_growth, compute_capex_to_revenue, compute_fcf_yield_on_ev,
    )
    df = df.copy()
    tax_rate = cfg.get("assumptions", {}).get("tax_rate", 0.25)
    min_ic = cfg.get("assumptions", {}).get("min_invested_capital", 1)

    df = compute_ebitda_margin(df)
    df = compute_net_debt(df)
    df = compute_net_debt_to_ebitda(df)
    df = compute_interest_coverage(df)
    df = compute_fcf_conversion(df)
    df = compute_ocf_margin(df)
    df = compute_roic(df, tax_rate, min_ic)
    df = compute_ev_to_ebitda(df)
    df = compute_revenue_growth(df)
    df = compute_ebitda_growth(df)
    df = compute_capex_to_revenue(df)
    df = compute_fcf_yield_on_ev(df)

    logger.info("All ratios computed")
    return df


def compute_ebitda_margin(df: pd.DataFrame) -> pd.DataFrame:
    """EBITDA Margin = EBITDA / Revenue. PE funds prefer margin > 20%."""
    df["ebitda_margin"] = _safe_divide(df.get("ebitda"), df.get("revenue"))
    return df


def compute_net_debt(df: pd.DataFrame) -> pd.DataFrame:
    """Net Debt = Total Debt - Cash. Negative = net cash (attractive for LBO)."""
    debt = df.get("total_debt", pd.Series(dtype=float))
    cash = df.get("cash", pd.Series(dtype=float))
    df["net_debt"] = debt.fillna(0) - cash.fillna(0)
    return df


def compute_net_debt_to_ebitda(df: pd.DataFrame) -> pd.DataFrame:
    """Net Debt / EBITDA — current leverage multiple. Lower = better for LBO."""
    df["net_debt_to_ebitda"] = _safe_divide(df.get("net_debt"), df.get("ebitda"))
    return df


def compute_interest_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Interest Coverage = EBIT / |Interest Expense|. PE lenders require > 2x."""
    ebit = df.get("ebit", pd.Series(dtype=float))
    interest = df.get("interest_expense", pd.Series(dtype=float)).abs()
    df["interest_coverage"] = _safe_divide(ebit, interest)
    return df


def compute_fcf_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """FCF Conversion = FCF / EBITDA. PE loves high conversion (>70%)."""
    df["fcf_conversion"] = _safe_divide(df.get("free_cash_flow"), df.get("ebitda"))
    return df


def compute_ocf_margin(df: pd.DataFrame) -> pd.DataFrame:
    """OCF Margin = Operating Cash Flow / Revenue."""
    df["ocf_margin"] = _safe_divide(df.get("operating_cash_flow"), df.get("revenue"))
    return df


def _safe_divide(numerator, denominator, fill_zero_denom=True):
    """Safe element-wise division. Returns NaN on division by zero or missing input."""
    if numerator is None or denominator is None:
        return pd.Series(dtype=float)
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (den == 0) | den.isna() | num.isna(), np.nan, num / den
        )
    return pd.Series(result, index=num.index if hasattr(num, "index") else None)
