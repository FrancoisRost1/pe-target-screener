"""
lbo.py — LBO return estimation: debt capacity, orchestration, FCF yield.

The IRR engine and scenario functions live in lbo_scenarios.py to keep this
file under the 150-line limit and to enforce a one-way dependency
(lbo → lbo_scenarios, never the reverse).

All assumptions overridable via config.yaml lbo: section.
"""

import pandas as pd
import numpy as np
import logging

from screener.lbo_scenarios import (
    compute_lbo_irr_proxy, compute_scenario_irr, compute_irr_bridge,
)

logger = logging.getLogger(__name__)


def compute_lbo_metrics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Main entry point. Runs all LBO estimation functions in sequence.
    Adds: max_debt, equity_required, irr_proxy, irr_base,
          irr_upside, irr_downside, irr_driver_*, fcf_yield_equity.
    """
    df = df.copy()
    df = compute_max_debt(df, cfg)
    df = compute_lbo_irr_proxy(df, cfg)
    df = compute_scenario_irr(df, cfg)
    df = compute_irr_bridge(df, cfg)
    df = compute_fcf_yield_on_equity(df)
    logger.info("LBO metrics computed")
    return df


def compute_max_debt(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Estimate max debt and equity cheque at entry.
    max_debt = min(EBITDA × leverage, EV × 0.70); equity floored at 20% of EV.
    """
    lbo = cfg.get("lbo", {})
    target_leverage = lbo.get("target_leverage", 3.5)

    if "enterprise_value" not in df.columns or "ebitda" not in df.columns:
        df["max_debt"] = np.nan
        df["equity_required"] = np.nan
        return df

    valid_ev = df["enterprise_value"].notna() & (df["enterprise_value"] > 0)
    debt_from_ebitda = df["ebitda"].clip(lower=0) * target_leverage
    debt_cap_70pct = df["enterprise_value"] * 0.70
    max_debt_raw = debt_from_ebitda.where(debt_from_ebitda < debt_cap_70pct, debt_cap_70pct)

    equity_raw = df["enterprise_value"] - max_debt_raw
    equity_floor = df["enterprise_value"] * 0.20
    equity_required = equity_raw.where(equity_raw >= equity_floor, equity_floor)
    max_debt_final = df["enterprise_value"] - equity_required

    df["max_debt"] = max_debt_final.where(valid_ev, np.nan)
    df["equity_required"] = equity_required.where(valid_ev, np.nan)
    return df


def compute_fcf_yield_on_equity(df: pd.DataFrame) -> pd.DataFrame:
    """FCF yield on sponsor equity = FCF / equity_required. Capped at ±50%."""
    if "free_cash_flow" not in df.columns or "equity_required" not in df.columns:
        df["fcf_yield_equity"] = np.nan
        return df
    eq_req = df["equity_required"].replace(0, np.nan)
    df["fcf_yield_equity"] = (df["free_cash_flow"] / eq_req).clip(-0.5, 0.50)
    return df
