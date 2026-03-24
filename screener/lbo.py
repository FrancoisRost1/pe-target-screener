"""
lbo.py — Simplified LBO return estimation

For each company, estimates:
- Max debt capacity (EBITDA × target leverage)
- Equity required at entry
- Approximate exit equity value
- Simplified IRR proxy

PE context: A screener without return estimation is incomplete.
Even a rough IRR tells you whether a deal can work at current prices.
The goal isn't precision — it's a quick filter: does the math work at all?
At 12x entry EV/EBITDA with 4x leverage, can a PE fund realistically
earn 20%+ IRR? This module answers that question in one pass.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_lbo_metrics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Main entry point. Runs all LBO estimation functions in sequence.
    Adds columns: max_debt, equity_required, irr_proxy, fcf_yield_equity.

    PE context: Called after winsorization so input metrics are already cleaned.
    """
    df = df.copy()
    df = compute_max_debt(df, cfg)
    df = compute_lbo_irr_proxy(df, cfg)
    df = compute_fcf_yield_on_equity(df)
    logger.info("LBO metrics computed")
    return df


def compute_max_debt(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Estimate max debt a PE fund could raise and equity check at entry.

    max_debt = EBITDA × target_leverage
        PE funds typically lever at 4–6x EBITDA. 4x is conservative for
        a screener default — avoids flagging overleveraged structures.

    equity_required = EV - max_debt
        This is what the sponsor must write as an equity cheque.
        If EV is not available, we skip (NaN).

    PE context: If equity_required is negative (max_debt > EV), the deal
    is theoretically all-debt. That's unrealistic — we cap at min 1 to
    avoid division errors downstream.
    """
    lbo = cfg.get("lbo", {})
    target_leverage = lbo.get("target_leverage", 4.0)

    if "ebitda" in df.columns:
        df["max_debt"] = df["ebitda"].clip(lower=0) * target_leverage
    else:
        df["max_debt"] = np.nan

    if "enterprise_value" in df.columns and "max_debt" in df.columns:
        df["equity_required"] = (df["enterprise_value"] - df["max_debt"]).clip(lower=1)
    else:
        df["equity_required"] = np.nan

    return df


def compute_lbo_irr_proxy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Simplified IRR proxy over the holding period.

    Methodology:
      exit_ebitda = ebitda × (1 + revenue_growth_clipped)^holding_period
        Revenue growth is clipped to [−5%, +15%] — extreme growth rates
        don't persist over a 5-year hold and would distort the IRR.

      exit_ev = exit_ebitda × exit_multiple
        Exit multiple defaults to 12x (conservative) to avoid fantasy returns.

      debt_repaid = FCF × holding_period × debt_repayment_rate
        We assume 40% of cumulative FCF goes to debt repayment (rest to fees,
        capex, and working capital). Simple but directionally correct.

      exit_equity = exit_ev − max(max_debt − debt_repaid, 0)
      irr_proxy = (exit_equity / equity_required)^(1/holding_period) − 1

    PE context: A fund targeting 20%+ IRR needs exit_equity / equity_required
    of roughly 2.5x over 5 years. This proxy surfaces those opportunities.
    IRR is capped at [−50%, +100%] to prevent outliers from dominating ranks.
    """
    lbo = cfg.get("lbo", {})
    holding_period = lbo.get("holding_period", 5)
    exit_multiple = lbo.get("exit_multiple", 12.0)
    debt_repayment_rate = lbo.get("debt_repayment_rate", 0.4)

    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        logger.warning("Missing columns for IRR proxy — skipping")
        df["irr_proxy"] = np.nan
        return df

    growth = df.get("revenue_growth", pd.Series(0.0, index=df.index)).fillna(0.0)
    growth_clipped = growth.clip(-0.05, 0.15)

    exit_ebitda = df["ebitda"].clip(lower=0) * ((1 + growth_clipped) ** holding_period)
    exit_ev = exit_ebitda * exit_multiple

    fcf = df.get("free_cash_flow", pd.Series(np.nan, index=df.index))
    debt_repaid = (fcf.clip(lower=0) * holding_period * debt_repayment_rate).fillna(0)
    debt_remaining = (df["max_debt"] - debt_repaid).clip(lower=0)

    exit_equity = exit_ev - debt_remaining

    # Avoid division by zero or negative equity_required
    eq_req = df["equity_required"].replace(0, np.nan)
    moic = exit_equity / eq_req
    moic = moic.clip(lower=0)  # MOIC can't be negative

    irr = moic ** (1.0 / holding_period) - 1
    df["irr_proxy"] = irr.clip(-0.5, 1.0)

    # Set to NaN where equity_required was invalid
    df.loc[df["equity_required"].isna() | (df["equity_required"] <= 1), "irr_proxy"] = np.nan

    return df


def compute_fcf_yield_on_equity(df: pd.DataFrame) -> pd.DataFrame:
    """
    FCF yield on sponsor equity.

    fcf_yield_equity = FCF / equity_required

    PE context: A fund earning 10%+ FCF yield on invested equity in year 1
    has a de-risked deal — it can service debt, return capital, and still
    compound. Anything below 5% suggests the entry price is too rich.
    """
    if "free_cash_flow" not in df.columns or "equity_required" not in df.columns:
        df["fcf_yield_equity"] = np.nan
        return df

    eq_req = df["equity_required"].replace(0, np.nan)
    df["fcf_yield_equity"] = (df["free_cash_flow"] / eq_req).clip(-0.5, 1.0)
    return df
