"""
lbo_scenarios.py — IRR scenario computation and bridge decomposition

Computes base/upside/downside IRR scenarios and decomposes the base-case
IRR into growth, deleveraging, and multiple drivers.
"""

import pandas as pd
import numpy as np
import logging
from screener.lbo import _compute_single_irr

logger = logging.getLogger(__name__)


def compute_lbo_irr_proxy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute base-case IRR proxy.

    exit_multiple = min(entry EV/EBITDA, config cap) — never exits above cap.

    PE context: Key discipline — the model NEVER exits above the configured cap.
    Most LBO models fail by assuming multiple expansion. This assumes contraction.
    """
    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        logger.warning("Missing columns for IRR proxy — skipping")
        df["irr_proxy"] = np.nan
        return df

    df["irr_proxy"] = _compute_single_irr(df, cfg, growth_delta=0.0, exit_multiple_delta=0.0)
    return df


def compute_scenario_irr(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute IRR under 3 scenarios for each company.

    Base:     growth as-is,     FCF at 100%,  exit multiple cap unchanged
    Upside:   growth + 2%,      FCF at 100%,  exit multiple cap + 0.5x
    Downside: growth - 3%,      FCF at 80%,   exit multiple cap - 1x
    """
    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        df["irr_base"] = np.nan
        df["irr_upside"] = np.nan
        df["irr_downside"] = np.nan
        return df

    df["irr_base"] = _compute_single_irr(
        df, cfg, growth_delta=0.00, exit_multiple_delta=0.0, fcf_conversion_mult=1.0
    )
    df["irr_upside"] = _compute_single_irr(
        df, cfg, growth_delta=+0.02, exit_multiple_delta=+0.5, fcf_conversion_mult=1.0
    )
    df["irr_downside"] = _compute_single_irr(
        df, cfg, growth_delta=-0.03, exit_multiple_delta=-1.0, fcf_conversion_mult=0.80
    )
    df["irr_proxy"] = df["irr_base"]

    spread = (df["irr_upside"] - df["irr_downside"]).median()
    logger.info(
        f"Scenario IRR computed — median spread: {spread:.1%} "
        f"(upside {df['irr_upside'].median():.1%} / base {df['irr_base'].median():.1%} "
        f"/ downside {df['irr_downside'].median():.1%})"
    )
    return df


def compute_irr_bridge(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Decompose the base-case IRR into its 3 main value drivers using isolation method.

    Method (isolation baseline):
      Baseline = zero growth + no debt repayment + exit at entry multiple
      Each driver = IRR with that factor enabled - baseline
    """
    required = {"ebitda", "equity_required", "max_debt", "irr_base"}
    if not required.issubset(df.columns):
        df["irr_driver_growth"] = np.nan
        df["irr_driver_deleveraging"] = np.nan
        df["irr_driver_multiple"] = np.nan
        return df

    df_zero_growth = df.copy()
    df_zero_growth["revenue_growth"] = 0.0
    cfg_no_delev = {**cfg, "lbo": {**cfg.get("lbo", {}), "debt_repayment_rate": 0.0}}

    irr_baseline = _compute_single_irr(
        df_zero_growth, cfg_no_delev, force_exit_at_entry_multiple=True
    )

    irr_with_growth = _compute_single_irr(
        df, cfg_no_delev, force_exit_at_entry_multiple=True
    )
    df["irr_driver_growth"] = (irr_with_growth - irr_baseline).round(4)

    irr_with_delev = _compute_single_irr(
        df_zero_growth, cfg, force_exit_at_entry_multiple=True
    )
    df["irr_driver_deleveraging"] = (irr_with_delev - irr_baseline).round(4)

    irr_with_cap = _compute_single_irr(df_zero_growth, cfg_no_delev)
    df["irr_driver_multiple"] = (irr_with_cap - irr_baseline).round(4)

    logger.info("IRR bridge (growth / deleveraging / multiple) computed via isolation method")
    return df
