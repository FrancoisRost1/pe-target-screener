"""
lbo.py — Simplified LBO return estimation

For each company, estimates:
- Max debt capacity (capped at 70% of EV, minimum 20% equity)
- Equity required at entry
- Approximate exit equity value
- IRR proxy under Base / Upside / Downside scenarios

Assumptions (all overridable via config.yaml lbo: section):
  target_leverage:     4.0x EBITDA (max debt, subject to 70% EV cap)
  holding_period:      5 years
  exit_multiple:       min(entry EV/EBITDA, config cap) — never exit above entry
  debt_repayment_rate: 40% of annual FCF reduces debt each year

PE context: A screener without return estimation is incomplete.
Even a rough IRR tells you whether a deal can work at current prices.
The goal isn't precision — it's a quick filter: does the math work at all?
Showing Base / Upside / Downside forces the analyst to think about the range
of outcomes, not just the single-point estimate.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_lbo_metrics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Main entry point. Runs all LBO estimation functions in sequence.
    Adds columns: max_debt, equity_required, irr_proxy, irr_base,
                  irr_upside, irr_downside, fcf_yield_equity.

    PE context: Called after winsorization so input metrics are already cleaned.
    """
    df = df.copy()
    df = compute_max_debt(df, cfg)
    df = compute_lbo_irr_proxy(df, cfg)
    df = compute_scenario_irr(df, cfg)
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


def _amortize_debt(max_debt: pd.Series, annual_fcf: pd.Series,
                   debt_repayment_rate: float, holding_period: int) -> pd.Series:
    """
    Simulate realistic annual debt amortization over the holding period.

    Each year: repay min(FCF × repayment_rate, remaining_debt).
    Once debt hits zero, no further repayment (no negative debt).

    PE context: A bullet repayment assumption overstates paydown for low-FCF
    companies and understates it for strong cash generators. Annual amortization
    captures the compounding effect of debt reduction: as debt falls, interest
    expense drops, FCF improves — the virtuous LBO flywheel.

    Returns: debt_remaining at exit (scalar series, one value per company).
    """
    debt = max_debt.copy().fillna(0.0)
    fcf_clipped = annual_fcf.clip(lower=0)

    for _ in range(holding_period):
        annual_repayment = (fcf_clipped * debt_repayment_rate).clip(upper=debt)
        debt = (debt - annual_repayment).clip(lower=0)

    return debt


def _compute_irr_scenario(df: pd.DataFrame, holding_period: int,
                           exit_multiple_cap: float, debt_repayment_rate: float,
                           growth_delta: float = 0.0) -> pd.Series:
    """
    Compute IRR proxy for one scenario given growth and exit multiple assumptions.

    growth_delta: added to revenue_growth before clipping to [-5%, +15%]
    exit_multiple_cap: the maximum exit EV/EBITDA for this scenario

    Returns a Series of IRR values, NaN where deal structure is invalid.
    """
    growth = df["revenue_growth"].fillna(0.0) if "revenue_growth" in df.columns \
        else pd.Series(0.0, index=df.index)
    growth_clipped = (growth + growth_delta).clip(-0.05, 0.15)

    exit_ebitda = df["ebitda"].clip(lower=0) * ((1 + growth_clipped) ** holding_period)

    entry_ev_ebitda = df["ev_to_ebitda"].fillna(exit_multiple_cap) \
        if "ev_to_ebitda" in df.columns \
        else pd.Series(exit_multiple_cap, index=df.index)
    exit_mult = entry_ev_ebitda.clip(upper=exit_multiple_cap)

    exit_ev = exit_ebitda * exit_mult

    annual_fcf = df["free_cash_flow"].fillna(0.0) if "free_cash_flow" in df.columns \
        else pd.Series(0.0, index=df.index)

    debt_remaining = _amortize_debt(
        df["max_debt"], annual_fcf, debt_repayment_rate, holding_period
    )

    exit_equity = exit_ev - debt_remaining

    eq_req = df["equity_required"]
    valid = (
        eq_req.notna() & (eq_req > 0) &
        exit_equity.notna() & (exit_equity > 0)
    )

    irr = pd.Series(np.nan, index=df.index)
    moic = exit_equity[valid] / eq_req[valid]
    irr[valid] = moic ** (1.0 / holding_period) - 1

    return irr.clip(-0.5, 1.0)


def compute_lbo_irr_proxy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute base-case IRR proxy using realistic annual debt amortization.

    Methodology:
      exit_ebitda = ebitda × (1 + revenue_growth.clip(-5%, +15%))^holding_period

      exit_multiple = min(entry EV/EBITDA, config exit_multiple cap)
        Key discipline: the model NEVER exits above entry multiple.

      Debt amortization: each year, repay min(FCF × rate, remaining_debt)
        More realistic than a lump-sum repayment — prevents over-stating
        paydown for low-FCF companies.

      exit_equity = exit_ev - debt_remaining
      irr_proxy = (exit_equity / equity_required)^(1/holding_period) - 1

    Validity gates: NaN if equity_required ≤ 0 or exit_equity ≤ 0.
    """
    lbo = cfg.get("lbo", {})
    holding_period = lbo.get("holding_period", 5)
    exit_multiple_cap = lbo.get("exit_multiple", 10.0)
    debt_repayment_rate = lbo.get("debt_repayment_rate", 0.4)

    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        logger.warning("Missing columns for IRR proxy — skipping")
        df["irr_proxy"] = np.nan
        return df

    df["irr_proxy"] = _compute_irr_scenario(
        df, holding_period, exit_multiple_cap, debt_repayment_rate, growth_delta=0.0
    )
    return df


def compute_scenario_irr(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute IRR under 3 scenarios for each company.

    PE context: No IC approves a deal based on a single IRR number.
    The downside case is what kills deals. The upside case is what
    gets them approved. Showing the range = showing PE thinking.

    Base:     revenue_growth as-is,    exit_multiple = config default
    Upside:   revenue_growth + 2%,     exit_multiple cap + 2x
    Downside: revenue_growth - 2%,     exit_multiple cap - 2x

    irr_base mirrors irr_proxy (same assumptions) for consistency.
    """
    lbo = cfg.get("lbo", {})
    holding_period = lbo.get("holding_period", 5)
    exit_multiple_cap = lbo.get("exit_multiple", 10.0)
    debt_repayment_rate = lbo.get("debt_repayment_rate", 0.4)

    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        df["irr_base"] = np.nan
        df["irr_upside"] = np.nan
        df["irr_downside"] = np.nan
        return df

    df["irr_base"] = _compute_irr_scenario(
        df, holding_period, exit_multiple_cap, debt_repayment_rate, growth_delta=0.0
    )
    df["irr_upside"] = _compute_irr_scenario(
        df, holding_period, exit_multiple_cap + 2.0, debt_repayment_rate, growth_delta=0.02
    )
    df["irr_downside"] = _compute_irr_scenario(
        df, holding_period, max(exit_multiple_cap - 2.0, 4.0), debt_repayment_rate, growth_delta=-0.02
    )

    logger.info("Scenario IRR computed (base / upside / downside)")
    return df


def compute_fcf_yield_on_equity(df: pd.DataFrame) -> pd.DataFrame:
    """
    FCF yield on sponsor equity.

    fcf_yield_equity = FCF / equity_required

    PE context: A fund earning 10%+ FCF yield on invested equity in year 1
    has a de-risked deal — it can service debt, return capital, and compound.
    Anything below 5% suggests the entry price is too rich.

    Capped at 50% — values above this are data artifacts, not signal.
    """
    if "free_cash_flow" not in df.columns or "equity_required" not in df.columns:
        df["fcf_yield_equity"] = np.nan
        return df

    eq_req = df["equity_required"].replace(0, np.nan)
    df["fcf_yield_equity"] = (df["free_cash_flow"] / eq_req).clip(-0.5, 0.50)
    return df
