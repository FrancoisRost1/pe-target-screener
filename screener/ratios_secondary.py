"""
ratios_secondary.py — Secondary financial ratio computations

Contains ROIC, EV/EBITDA, growth, capex intensity, and FCF yield ratios.
All use `_safe_divide` from ratios_utils.py — importing from there (rather
than from ratios.py) keeps the dependency one-way and avoids any import
cycle with ratios.py.
"""

import pandas as pd
import logging
from screener.ratios_utils import _safe_divide

logger = logging.getLogger(__name__)


def compute_roic(df: pd.DataFrame, tax_rate: float = 0.25, min_ic: float = 1) -> pd.DataFrame:
    """
    ROIC = NOPAT / Invested Capital
    NOPAT = EBIT * (1 - tax rate)
    Strong ROIC (>15%) signals a defensible business with pricing power.
    """
    ebit = df.get("ebit", pd.Series(dtype=float))
    ic = df.get("invested_capital", pd.Series(dtype=float)).clip(lower=min_ic)
    nopat = ebit * (1 - tax_rate)
    df["nopat"] = nopat
    df["roic"] = _safe_divide(nopat, ic)
    return df


def compute_ev_to_ebitda(df: pd.DataFrame) -> pd.DataFrame:
    """
    EV / EBITDA — Primary LBO entry multiple.
    Lower = more attractive for buyout. Most PE deals happen 8–14x.
    """
    df["ev_to_ebitda"] = _safe_divide(df.get("enterprise_value"), df.get("ebitda"))
    return df


def compute_revenue_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Revenue Growth = (Revenue_t - Revenue_t-1) / Revenue_t-1
    Negative growth is a red flag for debt repayment capacity.
    """
    rev = df.get("revenue", pd.Series(dtype=float))
    prior = df.get("prior_revenue", pd.Series(dtype=float))
    df["revenue_growth"] = _safe_divide(rev - prior, prior.abs())
    return df


def compute_ebitda_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    EBITDA Growth = (EBITDA_t - EBITDA_t-1) / |EBITDA_t-1|
    Stable or growing EBITDA reduces refinancing risk.
    """
    ebitda = df.get("ebitda", pd.Series(dtype=float))
    prior = df.get("prior_ebitda", pd.Series(dtype=float))
    df["ebitda_growth"] = _safe_divide(ebitda - prior, prior.abs())
    return df


def compute_capex_to_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capex / Revenue — capital intensity proxy.
    PE prefers asset-light models where FCF flows to debt service.
    """
    df["capex_to_revenue"] = _safe_divide(df.get("capex"), df.get("revenue"))
    return df


def compute_fcf_yield_on_ev(df: pd.DataFrame) -> pd.DataFrame:
    """
    FCF Yield on EV = Free Cash Flow / Enterprise Value
    Above 6% is attractive for LBO; below 3% suggests an expensive deal.
    """
    df["fcf_yield_ev"] = _safe_divide(df.get("free_cash_flow"), df.get("enterprise_value"))
    return df
