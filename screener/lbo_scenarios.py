"""
lbo_scenarios.py, IRR engine, scenarios, and bridge decomposition.

Owns the per-company IRR engine (`_build_cashflows_and_compute_irr`), the
per-scenario wrapper (`_compute_single_irr`), and the public scenario / bridge
functions consumed by lbo.compute_lbo_metrics. Does NOT import from screener.lbo
— dependency is one-way (lbo → lbo_scenarios) so the import graph stays acyclic.
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
import logging

logger = logging.getLogger(__name__)


def _build_cashflows_and_compute_irr(
    equity_required: pd.Series, max_debt: pd.Series,
    annual_fcf: pd.Series, exit_ev: pd.Series,
    debt_repayment_rate: float, holding_period: int,
) -> pd.Series:
    """
    Build per-company cash flow schedules and compute IRR via npf.irr().
    Schedule: [-equity, 0, ..., exit_equity] where FCF sweeps reduce debt.
    Returns NaN for every row when holding_period < 1 (invalid timeline).
    """
    if holding_period is None or holding_period < 1:
        return pd.Series(np.nan, index=equity_required.index)

    irr_results = pd.Series(np.nan, index=equity_required.index)

    for idx in equity_required.index:
        eq = equity_required.get(idx)
        debt = max_debt.get(idx)
        fcf = annual_fcf.get(idx)
        ev_exit = exit_ev.get(idx)

        if any(pd.isna(v) for v in [eq, debt, fcf, ev_exit]):
            continue
        if eq <= 0:
            continue

        remaining_debt = debt
        for _ in range(holding_period):
            repayment = min(fcf * debt_repayment_rate, remaining_debt)
            remaining_debt = max(0, remaining_debt - repayment)

        exit_equity = ev_exit - remaining_debt
        if exit_equity <= 0:
            continue

        cashflows = [-eq] + [0.0] * (holding_period - 1) + [exit_equity]
        try:
            computed_irr = npf.irr(cashflows)
            if np.isnan(computed_irr) or np.isinf(computed_irr):
                continue
            irr_results.at[idx] = computed_irr
        except (ValueError, FloatingPointError):
            continue

    return irr_results.clip(-0.50, 0.40)


def _compute_single_irr(df: pd.DataFrame, cfg: dict,
                         growth_delta: float = 0.0,
                         exit_multiple_delta: float = 0.0,
                         fcf_conversion_mult: float = 1.0,
                         force_exit_at_entry_multiple: bool = False) -> pd.Series:
    """Compute IRR for a single scenario. Entry multiple floored at 6x. Negative EBITDA → NaN."""
    lbo = cfg.get("lbo", {})
    hp = lbo.get("holding_period", 5)
    cap = lbo.get("exit_multiple", 8.0)
    drr = lbo.get("debt_repayment_rate", 0.4)

    if not {"ebitda", "equity_required", "max_debt"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)

    valid_ebitda = df["ebitda"].notna() & (df["ebitda"] > 0)
    base_growth = df["revenue_growth"].fillna(0.0) if "revenue_growth" in df.columns \
        else pd.Series(0.0, index=df.index)
    growth = (base_growth + growth_delta).clip(-0.10, 0.20)

    if "enterprise_value" in df.columns and "ebitda" in df.columns:
        raw = df["enterprise_value"] / df["ebitda"].replace(0, np.nan)
        entry_mult = raw.clip(lower=6.0).fillna(cap)
    elif "ev_to_ebitda" in df.columns:
        entry_mult = df["ev_to_ebitda"].fillna(cap).clip(lower=6.0)
    else:
        entry_mult = pd.Series(cap, index=df.index)

    exit_mult = entry_mult if force_exit_at_entry_multiple else \
        (entry_mult.clip(upper=cap) + exit_multiple_delta).clip(lower=4.0)

    exit_ebitda = df["ebitda"].clip(lower=0) * ((1 + growth) ** hp)
    annual_fcf = df["free_cash_flow"].fillna(0.0).clip(lower=0) \
        if "free_cash_flow" in df.columns else pd.Series(0.0, index=df.index)

    irr = _build_cashflows_and_compute_irr(
        df["equity_required"], df["max_debt"], annual_fcf * fcf_conversion_mult,
        exit_ebitda * exit_mult, drr, hp)
    return irr.where(valid_ebitda, np.nan)


def compute_lbo_irr_proxy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute base-case IRR proxy. Exit multiple capped at config value."""
    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        logger.warning("Missing columns for IRR proxy, skipping")
        df["irr_proxy"] = np.nan
        return df

    df["irr_proxy"] = _compute_single_irr(df, cfg, growth_delta=0.0, exit_multiple_delta=0.0)
    return df


def compute_scenario_irr(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute base/upside/downside IRRs.
    Base = growth as-is, FCF 100%, cap unchanged.
    Upside = growth +2%, FCF 100%, cap +0.5x. Downside = growth -3%, FCF 80%, cap -1x.
    """
    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        df["irr_base"] = np.nan
        df["irr_upside"] = np.nan
        df["irr_downside"] = np.nan
        return df

    df["irr_base"] = _compute_single_irr(
        df, cfg, growth_delta=0.00, exit_multiple_delta=0.0, fcf_conversion_mult=1.0)
    df["irr_upside"] = _compute_single_irr(
        df, cfg, growth_delta=+0.02, exit_multiple_delta=+0.5, fcf_conversion_mult=1.0)
    df["irr_downside"] = _compute_single_irr(
        df, cfg, growth_delta=-0.03, exit_multiple_delta=-1.0, fcf_conversion_mult=0.80)
    df["irr_proxy"] = df["irr_base"]

    spread = (df["irr_upside"] - df["irr_downside"]).median()
    logger.info(
        f"Scenario IRR, spread {spread:.1%} (up {df['irr_upside'].median():.1%}"
        f" / base {df['irr_base'].median():.1%} / down {df['irr_downside'].median():.1%})"
    )
    return df


def compute_irr_bridge(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Decompose base-case IRR into 3 drivers: growth, deleveraging, multiple.
    Isolation method: each driver = IRR with that factor enabled vs baseline
    (zero growth + no debt repayment + exit at entry multiple).
    """
    required = {"ebitda", "equity_required", "max_debt", "irr_base"}
    if not required.issubset(df.columns):
        df["irr_driver_growth"] = np.nan
        df["irr_driver_deleveraging"] = np.nan
        df["irr_driver_multiple"] = np.nan
        return df

    df_zero = df.copy()
    df_zero["revenue_growth"] = 0.0
    cfg_no_delev = {**cfg, "lbo": {**cfg.get("lbo", {}), "debt_repayment_rate": 0.0}}

    base = _compute_single_irr(df_zero, cfg_no_delev, force_exit_at_entry_multiple=True)
    df["irr_driver_growth"] = (
        _compute_single_irr(df, cfg_no_delev, force_exit_at_entry_multiple=True) - base
    ).round(4)
    df["irr_driver_deleveraging"] = (
        _compute_single_irr(df_zero, cfg, force_exit_at_entry_multiple=True) - base
    ).round(4)
    df["irr_driver_multiple"] = (
        _compute_single_irr(df_zero, cfg_no_delev) - base
    ).round(4)

    logger.info("IRR bridge (growth / deleveraging / multiple) computed via isolation method")
    return df
