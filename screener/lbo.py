"""
lbo.py — LBO return estimation: debt capacity, IRR engine, FCF yield.

Delegates scenario IRR and bridge decomposition to lbo_scenarios.py.
All assumptions overridable via config.yaml lbo: section.
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
import logging

logger = logging.getLogger(__name__)

def compute_lbo_metrics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Main entry point. Runs all LBO estimation functions in sequence.
    Adds: max_debt, equity_required, irr_proxy, irr_base,
          irr_upside, irr_downside, irr_driver_*, fcf_yield_equity.
    """
    from screener.lbo_scenarios import (
        compute_lbo_irr_proxy, compute_scenario_irr, compute_irr_bridge,
    )
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


def _build_cashflows_and_compute_irr(
    equity_required: pd.Series, max_debt: pd.Series,
    annual_fcf: pd.Series, exit_ev: pd.Series,
    debt_repayment_rate: float, holding_period: int,
) -> pd.Series:
    """
    Build per-company cash flow schedules and compute IRR via npf.irr().
    Schedule: [-equity, 0, ..., exit_equity] where FCF sweeps reduce debt.
    """
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


def compute_fcf_yield_on_equity(df: pd.DataFrame) -> pd.DataFrame:
    """FCF yield on sponsor equity = FCF / equity_required. Capped at ±50%."""
    if "free_cash_flow" not in df.columns or "equity_required" not in df.columns:
        df["fcf_yield_equity"] = np.nan
        return df
    eq_req = df["equity_required"].replace(0, np.nan)
    df["fcf_yield_equity"] = (df["free_cash_flow"] / eq_req).clip(-0.5, 0.50)
    return df
