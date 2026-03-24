"""
lbo.py — Simplified LBO return estimation

For each company, estimates:
- Max debt capacity (capped at 70% of EV, minimum 20% equity)
- Equity required at entry
- Approximate exit equity value
- Simplified IRR proxy

Assumptions (all overridable via config.yaml lbo: section):
  target_leverage:    4.0x EBITDA (max debt, subject to 70% EV cap)
  holding_period:     5 years
  exit_multiple:      min(entry EV/EBITDA, config cap) — never exit above entry
  debt_repayment_rate: 40% of cumulative FCF reduces debt

PE context: A screener without return estimation is incomplete.
Even a rough IRR tells you whether a deal can work at current prices.
The goal isn't precision — it's a quick filter: does the math work at all?
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
    Estimate max debt a PE fund could raise and equity cheque at entry.

    max_debt = min(EBITDA × target_leverage, EV × 0.70)
        Capped at 70% of EV so the fund always writes at least 30% equity.
        In practice, lenders won't finance >65–70% of purchase price regardless
        of EBITDA leverage — this prevents the model from producing $0 equity cheques.

    equity_required = EV - max_debt
        Floored at 20% of EV — sponsor minimum equity is a market convention.
        Deals below 20% equity are considered too risky by most LPs and lenders.

    If EV is missing, zero, or negative: all LBO metrics set to NaN.

    PE context: The equity cheque is what the sponsor actually risks.
    Getting it right is more important than getting a precise IRR.
    """
    lbo = cfg.get("lbo", {})
    target_leverage = lbo.get("target_leverage", 4.0)

    ev = df.get("enterprise_value", pd.Series(np.nan, index=df.index))
    ebitda = df.get("ebitda", pd.Series(np.nan, index=df.index))

    if "enterprise_value" not in df.columns or "ebitda" not in df.columns:
        df["max_debt"] = np.nan
        df["equity_required"] = np.nan
        return df

    # Mask for valid EV (positive, non-NaN)
    valid_ev = df["enterprise_value"].notna() & (df["enterprise_value"] > 0)

    debt_from_ebitda = df["ebitda"].clip(lower=0) * target_leverage
    debt_cap_70pct = df["enterprise_value"] * 0.70

    max_debt_raw = debt_from_ebitda.where(
        debt_from_ebitda < debt_cap_70pct, debt_cap_70pct
    )

    equity_raw = df["enterprise_value"] - max_debt_raw
    equity_floor = df["enterprise_value"] * 0.20
    equity_required = equity_raw.where(equity_raw >= equity_floor, equity_floor)

    # Recalculate max_debt after applying equity floor
    max_debt_final = df["enterprise_value"] - equity_required

    df["max_debt"] = max_debt_final.where(valid_ev, np.nan)
    df["equity_required"] = equity_required.where(valid_ev, np.nan)

    return df


def compute_lbo_irr_proxy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Simplified IRR proxy over the holding period.

    Methodology:
      exit_ebitda = ebitda × (1 + revenue_growth.clip(-5%, +15%))^holding_period
        Revenue growth clipped — extreme rates don't persist over a 5yr hold.

      exit_multiple = min(entry EV/EBITDA, config exit_multiple cap)
        A key discipline: the model NEVER assumes exit above entry multiple.
        If you buy at 15x, you model 12x exit (config cap). This is the correct
        conservative assumption for any LBO model.

      exit_ev = exit_ebitda × exit_multiple
      debt_repaid = FCF.clip(0) × holding_period × debt_repayment_rate
      debt_remaining = max(max_debt - debt_repaid, 0)
      exit_equity = exit_ev - debt_remaining

      irr_proxy = (exit_equity / equity_required)^(1/holding_period) - 1

    Validity gates:
      - NaN if equity_required <= 0 (deal structure invalid)
      - NaN if exit_equity <= 0 (sponsor wiped out — not a real return)
      - Capped at [-50%, +100%] to suppress noise

    PE context: A fund targeting 20%+ IRR needs ~2.5x MOIC over 5 years.
    The exit multiple discipline is critical: most LBO models blow up because
    they assume multiple expansion. This model assumes contraction.
    """
    lbo = cfg.get("lbo", {})
    holding_period = lbo.get("holding_period", 5)
    exit_multiple_cap = lbo.get("exit_multiple", 12.0)
    debt_repayment_rate = lbo.get("debt_repayment_rate", 0.4)

    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        logger.warning("Missing columns for IRR proxy — skipping")
        df["irr_proxy"] = np.nan
        return df

    # Clipped growth for exit EBITDA projection
    growth = df["revenue_growth"].fillna(0.0) if "revenue_growth" in df.columns \
        else pd.Series(0.0, index=df.index)
    growth_clipped = growth.clip(-0.05, 0.15)
    exit_ebitda = df["ebitda"].clip(lower=0) * ((1 + growth_clipped) ** holding_period)

    # Exit multiple: never above entry, capped at config max
    entry_ev_ebitda = df["ev_to_ebitda"].fillna(exit_multiple_cap) \
        if "ev_to_ebitda" in df.columns else pd.Series(exit_multiple_cap, index=df.index)
    exit_multiple = entry_ev_ebitda.clip(upper=exit_multiple_cap)

    exit_ev = exit_ebitda * exit_multiple

    # Debt repayment from FCF
    fcf = df["free_cash_flow"].fillna(0.0) if "free_cash_flow" in df.columns \
        else pd.Series(0.0, index=df.index)
    debt_repaid = fcf.clip(lower=0) * holding_period * debt_repayment_rate
    debt_remaining = (df["max_debt"] - debt_repaid).clip(lower=0)

    exit_equity = exit_ev - debt_remaining

    # IRR calculation with validity gates
    eq_req = df["equity_required"]
    valid = (
        eq_req.notna() & (eq_req > 0) &
        exit_equity.notna() & (exit_equity > 0)
    )

    irr = pd.Series(np.nan, index=df.index)
    moic = exit_equity[valid] / eq_req[valid]
    irr[valid] = moic ** (1.0 / holding_period) - 1

    df["irr_proxy"] = irr.clip(-0.5, 1.0)
    df.loc[~valid, "irr_proxy"] = np.nan

    return df


def compute_fcf_yield_on_equity(df: pd.DataFrame) -> pd.DataFrame:
    """
    FCF yield on sponsor equity.

    fcf_yield_equity = FCF / equity_required

    PE context: A fund earning 10%+ FCF yield on invested equity in year 1
    has a de-risked deal — it can service debt, return capital, and compound.
    Anything below 5% suggests the entry price is too rich.
    """
    if "free_cash_flow" not in df.columns or "equity_required" not in df.columns:
        df["fcf_yield_equity"] = np.nan
        return df

    eq_req = df["equity_required"].replace(0, np.nan)
    df["fcf_yield_equity"] = (df["free_cash_flow"] / eq_req).clip(-0.5, 1.0)
    return df
