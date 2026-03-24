"""
lbo.py — Simplified LBO return estimation

For each company, estimates:
- Max debt capacity (capped at 70% of EV, minimum 20% equity)
- Equity required at entry
- IRR proxy under Base / Upside / Downside scenarios
- IRR bridge: decomposition into growth, deleveraging, and multiple drivers

Assumptions (all overridable via config.yaml lbo: section):
  target_leverage:     4.0x EBITDA (max debt, subject to 70% EV cap)
  holding_period:      5 years
  exit_multiple:       min(entry EV/EBITDA, cap) — never exit above cap without delta
  debt_repayment_rate: 40% of annual FCF reduces debt each year (realistic amortization)

PE context: A screener without return estimation is incomplete.
Even a rough IRR tells you whether a deal can work at current prices.
Showing Base / Upside / Downside forces the analyst to think about the range
of outcomes. The IRR bridge tells the IC where the return comes from.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_lbo_metrics(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Main entry point. Runs all LBO estimation functions in sequence.
    Adds columns: max_debt, equity_required, irr_proxy, irr_base,
                  irr_upside, irr_downside, irr_driver_*, fcf_yield_equity.

    PE context: Called after winsorization so input metrics are already cleaned.
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
    Estimate max debt a PE fund could raise and equity cheque at entry.

    max_debt = min(EBITDA × target_leverage, EV × 0.70)
        Capped at 70% of EV so the fund always writes at least 30% equity.

    equity_required = EV - max_debt
        Floored at 20% of EV — sponsor minimum equity is a market convention.

    If EV is missing, zero, or negative: all LBO metrics set to NaN.

    PE context: The equity cheque is what the sponsor actually risks.
    """
    lbo = cfg.get("lbo", {})
    target_leverage = lbo.get("target_leverage", 4.0)

    if "enterprise_value" not in df.columns or "ebitda" not in df.columns:
        df["max_debt"] = np.nan
        df["equity_required"] = np.nan
        return df

    valid_ev = df["enterprise_value"].notna() & (df["enterprise_value"] > 0)

    debt_from_ebitda = df["ebitda"].clip(lower=0) * target_leverage
    debt_cap_70pct = df["enterprise_value"] * 0.70

    max_debt_raw = debt_from_ebitda.where(
        debt_from_ebitda < debt_cap_70pct, debt_cap_70pct
    )

    equity_raw = df["enterprise_value"] - max_debt_raw
    equity_floor = df["enterprise_value"] * 0.20
    equity_required = equity_raw.where(equity_raw >= equity_floor, equity_floor)

    # Recalculate max_debt after equity floor
    max_debt_final = df["enterprise_value"] - equity_required

    df["max_debt"] = max_debt_final.where(valid_ev, np.nan)
    df["equity_required"] = equity_required.where(valid_ev, np.nan)

    return df


def _amortize_debt(max_debt: pd.Series, annual_fcf: pd.Series,
                   debt_repayment_rate: float, holding_period: int) -> pd.Series:
    """
    Simulate realistic annual debt amortization over the holding period.

    Each year: repay min(FCF × rate, remaining_debt).
    Once debt hits zero, no further repayment — no negative debt.

    PE context: Annual amortization captures the compounding effect of debt
    reduction: as debt falls, interest expense drops, FCF improves. This is
    more realistic than a lump-sum bullet repayment assumption.
    """
    debt = max_debt.copy().fillna(0.0)
    fcf_clipped = annual_fcf.clip(lower=0)

    for _ in range(holding_period):
        annual_repayment = (fcf_clipped * debt_repayment_rate).clip(upper=debt)
        debt = (debt - annual_repayment).clip(lower=0)

    return debt


def _compute_single_irr(df: pd.DataFrame, cfg: dict,
                         growth_delta: float = 0.0,
                         exit_multiple_delta: float = 0.0) -> pd.Series:
    """
    Compute IRR for a single scenario by adjusting growth and exit multiple.

    growth_delta: added to revenue_growth before clipping to realistic range.
      Wider clip range (-10%, +20%) vs base ensures real spread between scenarios.

    exit_multiple_delta: added to the base exit multiple (min(entry, cap)).
      Applied AFTER clipping at the cap — so upside can exceed entry for cheap names.
      Floored at 4x to prevent degenerate scenarios.

    PE context: Scenario analysis is only useful if the scenarios actually differ.
    The previous implementation clipped into the same value for high-growth companies.
    This version uses a wider range so upside and downside produce real spread.
    """
    lbo = cfg.get("lbo", {})
    holding_period = lbo.get("holding_period", 5)
    exit_multiple_cap = lbo.get("exit_multiple", 10.0)
    debt_repayment_rate = lbo.get("debt_repayment_rate", 0.4)

    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)

    # Growth: base + delta, clipped to realistic range (wider than base to enable spread)
    base_growth = df["revenue_growth"].fillna(0.0) if "revenue_growth" in df.columns \
        else pd.Series(0.0, index=df.index)
    growth = (base_growth + growth_delta).clip(-0.10, 0.20)

    # Exit multiple: min(entry, cap) + delta, floored at 4x
    entry_ev_ebitda = df["ev_to_ebitda"].fillna(exit_multiple_cap) \
        if "ev_to_ebitda" in df.columns \
        else pd.Series(exit_multiple_cap, index=df.index)
    exit_multiple = (
        entry_ev_ebitda.clip(upper=exit_multiple_cap) + exit_multiple_delta
    ).clip(lower=4.0)

    exit_ebitda = df["ebitda"].clip(lower=0) * ((1 + growth) ** holding_period)
    exit_ev = exit_ebitda * exit_multiple

    # Annual debt amortization
    annual_fcf = df["free_cash_flow"].fillna(0.0).clip(lower=0) \
        if "free_cash_flow" in df.columns \
        else pd.Series(0.0, index=df.index)
    debt_remaining = _amortize_debt(df["max_debt"], annual_fcf, debt_repayment_rate, holding_period)

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
    Compute base-case IRR proxy.

    exit_multiple = min(entry EV/EBITDA, config cap) — never exits above cap.
    Debt amortization uses annual loop per _amortize_debt().

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

    PE context: No IC approves a deal based on a single IRR number.
    The downside case is what kills deals. The upside case is what gets them approved.
    Showing the range = showing PE thinking.

    Base:     growth as-is,    exit multiple cap unchanged
    Upside:   growth + 2%,     exit multiple cap + 1x (allows modest expansion for cheap names)
    Downside: growth - 2%,     exit multiple cap - 1x (compression + growth miss)

    Key vs previous: growth clips to (-10%, +20%) so ±2% creates real spread
    even for high-growth companies. Exit delta is additive after base clip,
    ensuring scenarios differ even when entry << cap.
    """
    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        df["irr_base"] = np.nan
        df["irr_upside"] = np.nan
        df["irr_downside"] = np.nan
        return df

    df["irr_base"] = _compute_single_irr(df, cfg, growth_delta=0.00, exit_multiple_delta=0.0)
    df["irr_upside"] = _compute_single_irr(df, cfg, growth_delta=+0.02, exit_multiple_delta=+1.0)
    df["irr_downside"] = _compute_single_irr(df, cfg, growth_delta=-0.02, exit_multiple_delta=-1.0)

    # Keep irr_proxy as alias for backwards compatibility
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
    Decompose the base-case IRR into its 3 main value drivers.

    PE context: An IRR number alone is meaningless to an IC.
    They want to know WHERE the return comes from:
    - EBITDA growth contribution: what the company's organic growth adds vs flat
    - Deleveraging contribution: debt paydown → more equity value at exit
    - Multiple delta: effect of exit cap constraint vs exiting at entry multiple

    Method: compare base IRR against a counterfactual where each driver is removed.

    Growth driver:    irr_base - irr_with_zero_revenue_growth
    Deleverage driver: irr_base - irr_with_no_debt_repayment
    Multiple driver:  irr_base - irr_if_exit_at_entry_multiple (no cap constraint)
      → negative when entry > cap (compression hurts); 0 when entry <= cap (no compression)

    This is approximate attribution — directional correctness matters more than
    exact decomposition (which requires a Shapley value calculation).
    """
    required = {"ebitda", "equity_required", "max_debt", "irr_base"}
    if not required.issubset(df.columns):
        df["irr_driver_growth"] = np.nan
        df["irr_driver_deleveraging"] = np.nan
        df["irr_driver_multiple"] = np.nan
        return df

    irr_base = df["irr_base"]

    # --- Growth driver ---
    # Compare base (actual growth) vs zero growth
    df_zero_growth = df.copy()
    df_zero_growth["revenue_growth"] = 0.0
    irr_zero_growth = _compute_single_irr(df_zero_growth, cfg)
    df["irr_driver_growth"] = (irr_base - irr_zero_growth).round(4)

    # --- Deleverage driver ---
    # Compare base amortization vs no debt repayment (rate = 0)
    cfg_no_delev = {**cfg, "lbo": {**cfg.get("lbo", {}), "debt_repayment_rate": 0.0}}
    irr_no_delev = _compute_single_irr(df, cfg_no_delev)
    df["irr_driver_deleveraging"] = (irr_base - irr_no_delev).round(4)

    # --- Multiple driver ---
    # Compare base exit (capped) vs exiting at entry multiple (no cap constraint)
    # Positive exit_multiple cap of 999 effectively removes the cap
    cfg_no_cap = {**cfg, "lbo": {**cfg.get("lbo", {}), "exit_multiple": 999.0}}
    irr_no_cap = _compute_single_irr(df, cfg_no_cap)
    # irr_base uses min(entry, cap); irr_no_cap uses entry → irr_no_cap >= irr_base when entry > cap
    # driver = irr_base - irr_no_cap → negative = cap compresses multiple (hurts), 0 = no compression
    df["irr_driver_multiple"] = (irr_base - irr_no_cap).round(4)

    logger.info("IRR bridge (growth / deleveraging / multiple) computed")
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
