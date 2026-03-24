"""
ratios.py — Financial ratio computation

Computes all 8 core PE/LBO screening metrics from raw financial data.
Each function is defensive: handles division by zero and missing data,
returns NaN rather than raising exceptions.

PE context: These ratios are the lens through which a buyout analyst
evaluates a business. Together they answer: Is this business good?
Does it generate cash? Can it handle debt? Is it cheap?
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_all_ratios(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run all ratio computations in sequence."""
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

    logger.info("All ratios computed")
    return df


def compute_ebitda_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    EBITDA Margin = EBITDA / Revenue
    Core profitability proxy. PE funds prefer margin > 20%.
    Higher margin = more resilient under leverage stress.
    """
    df["ebitda_margin"] = _safe_divide(df.get("ebitda"), df.get("revenue"))
    return df


def compute_net_debt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Net Debt = Total Debt - Cash
    Starting leverage position. PE adds debt on top of this.
    Negative net debt = net cash (very attractive for LBO).
    """
    debt = df.get("total_debt", pd.Series(dtype=float))
    cash = df.get("cash", pd.Series(dtype=float))
    df["net_debt"] = debt.fillna(0) - cash.fillna(0)
    return df


def compute_net_debt_to_ebitda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Net Debt / EBITDA
    Current leverage multiple. Signals available debt headroom.
    PE typically targets entry leverage of 4–6x EBITDA total.
    If company already at 3x+, headroom is limited.
    Lower = better for LBO candidacy.
    """
    df["net_debt_to_ebitda"] = _safe_divide(df.get("net_debt"), df.get("ebitda"))
    return df


def compute_interest_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interest Coverage = EBIT / Interest Expense
    Measures ability to service existing debt.
    PE lenders typically require > 2x minimum; > 5x = strong.
    Note: Use absolute value of interest expense (can be negative in yfinance).
    """
    ebit = df.get("ebit", pd.Series(dtype=float))
    interest = df.get("interest_expense", pd.Series(dtype=float)).abs()
    df["interest_coverage"] = _safe_divide(ebit, interest)
    return df


def compute_fcf_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    FCF Conversion = Free Cash Flow / EBITDA
    Quality of earnings — how much EBITDA actually becomes cash.
    PE loves high conversion (>70%): means fewer capex surprises,
    more cash available for debt service.
    """
    df["fcf_conversion"] = _safe_divide(df.get("free_cash_flow"), df.get("ebitda"))
    return df


def compute_ocf_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    OCF Margin = Operating Cash Flow / Revenue
    Cash profitability of the business.
    Complements EBITDA margin with a real cash view.
    """
    df["ocf_margin"] = _safe_divide(df.get("operating_cash_flow"), df.get("revenue"))
    return df


def compute_roic(df: pd.DataFrame, tax_rate: float = 0.25, min_ic: float = 1) -> pd.DataFrame:
    """
    ROIC = NOPAT / Invested Capital
    NOPAT = EBIT * (1 - tax rate)
    Measures how efficiently management deploys capital.
    Strong ROIC (>15%) signals a defensible business with pricing power.
    PE prefers high ROIC — it reflects real economic moat.
    """
    ebit = df.get("ebit", pd.Series(dtype=float))
    ic = df.get("invested_capital", pd.Series(dtype=float)).clip(lower=min_ic)
    nopat = ebit * (1 - tax_rate)
    df["nopat"] = nopat
    df["roic"] = _safe_divide(nopat, ic)
    return df


def compute_ev_to_ebitda(df: pd.DataFrame) -> pd.DataFrame:
    """
    EV / EBITDA
    Primary LBO entry multiple. Lower = more attractive for buyout.
    Most PE deals happen in the 8–14x range depending on sector/quality.
    Very high multiples (>18x) compress returns even for great businesses.
    """
    df["ev_to_ebitda"] = _safe_divide(df.get("enterprise_value"), df.get("ebitda"))
    return df


def compute_revenue_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Revenue Growth = (Revenue_t - Revenue_t-1) / Revenue_t-1
    Growth trajectory. PE doesn't require hypergrowth, but negative
    growth is a red flag for debt repayment capacity.
    """
    rev = df.get("revenue", pd.Series(dtype=float))
    prior = df.get("prior_revenue", pd.Series(dtype=float))
    df["revenue_growth"] = _safe_divide(rev - prior, prior.abs())
    return df


def compute_ebitda_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    EBITDA Growth = (EBITDA_t - EBITDA_t-1) / |EBITDA_t-1|
    Earnings momentum. Stable or growing EBITDA reduces refinancing risk.
    """
    ebitda = df.get("ebitda", pd.Series(dtype=float))
    prior = df.get("prior_ebitda", pd.Series(dtype=float))
    df["ebitda_growth"] = _safe_divide(ebitda - prior, prior.abs())
    return df


def compute_capex_to_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capex / Revenue
    Capital intensity proxy. High capex businesses are less attractive
    for LBO because capex competes with debt repayment.
    PE prefers asset-light models where FCF flows to debt service.
    """
    df["capex_to_revenue"] = _safe_divide(df.get("capex"), df.get("revenue"))
    return df


def _safe_divide(numerator, denominator, fill_zero_denom=True):
    """
    Safe element-wise division for pandas Series or scalars.
    Returns NaN on division by zero, NaN input, or None.
    """
    if numerator is None or denominator is None:
        return pd.Series(dtype=float)

    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (den == 0) | den.isna() | num.isna(),
            np.nan,
            num / den
        )
    return pd.Series(result, index=num.index if hasattr(num, "index") else None)
