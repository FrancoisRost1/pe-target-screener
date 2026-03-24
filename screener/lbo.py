"""
lbo.py — Simplified LBO return estimation

For each company, estimates:
- Max debt capacity (capped at 70% of EV, minimum 20% equity)
- Equity required at entry
- IRR proxy under Base / Upside / Downside scenarios
- IRR bridge: decomposition into growth, deleveraging, and multiple drivers

Assumptions (all overridable via config.yaml lbo: section):
  target_leverage:     3.5x EBITDA (max debt, subject to 70% EV cap)
  holding_period:      5 years
  exit_multiple:       min(entry EV/EBITDA, cap) — never exit above cap without delta
  debt_repayment_rate: 40% of post-interest FCF reduces debt each year
  debt_interest_rate:  7% annual interest on outstanding debt (waterfall: interest first)

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
    target_leverage = lbo.get("target_leverage", 3.5)

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
                   debt_repayment_rate: float, holding_period: int,
                   interest_rate: float = 0.0) -> pd.Series:
    """
    Simulate realistic annual debt amortization over the holding period.

    Each year waterfall:
      1. Compute interest cost on outstanding debt (debt × interest_rate)
      2. FCF after interest = max(FCF - interest, 0)
      3. Repay min(FCF_after_interest × rate, remaining_debt)

    Once debt hits zero, no further repayment — no negative debt.

    PE context: Interest is the first call on FCF — it must be paid before any
    principal repayment. Without the interest waterfall, the model overstates
    debt paydown speed. At 7% on $500M of debt = $35M/yr less available for
    sweep. This materially affects the exit equity value, especially for
    low-FCF businesses where interest consumes most of free cash flow.
    """
    debt = max_debt.copy().fillna(0.0)
    fcf_clipped = annual_fcf.clip(lower=0)

    for _ in range(holding_period):
        interest_cost = debt * interest_rate
        fcf_after_interest = (fcf_clipped - interest_cost).clip(lower=0)
        annual_repayment = (fcf_after_interest * debt_repayment_rate).clip(upper=debt)
        debt = (debt - annual_repayment).clip(lower=0)

    return debt


def _compute_single_irr(df: pd.DataFrame, cfg: dict,
                         growth_delta: float = 0.0,
                         exit_multiple_delta: float = 0.0,
                         fcf_conversion_mult: float = 1.0,
                         force_exit_at_entry_multiple: bool = False) -> pd.Series:
    """
    Compute IRR for a single scenario.

    Parameters:
      growth_delta: added to revenue_growth before clipping to (-10%, +20%)
      exit_multiple_delta: additive shift on exit multiple (ignored if force_exit_at_entry_multiple)
      fcf_conversion_mult: FCF multiplier before amortization loop.
        1.0 = base/upside; 0.80 = downside (stress FCF by 20% — captures
        revenue miss + margin compression compounding effect on cash generation)
      force_exit_at_entry_multiple: if True, exit at raw entry multiple (no cap, no delta).
        Used in IRR bridge baseline to isolate the multiple driver.

    PE context: fcf_conversion_mult creates asymmetric downside — a revenue miss
    compresses margins, which compresses FCF more than linearly. Using 0.80× FCF
    in downside leaves more debt at exit, directly reducing equity proceeds.
    Entry multiple floored at 6x (FIX 4): values below 6x signal data errors —
    real LBO targets don't trade below 4-5x EBITDA.
    """
    lbo = cfg.get("lbo", {})
    holding_period = lbo.get("holding_period", 5)
    exit_multiple_cap = lbo.get("exit_multiple", 8.0)
    debt_repayment_rate = lbo.get("debt_repayment_rate", 0.4)
    interest_rate = lbo.get("debt_interest_rate", 0.07)

    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)

    # Growth: base + delta, clipped to realistic range (wider than base to enable spread)
    base_growth = df["revenue_growth"].fillna(0.0) if "revenue_growth" in df.columns \
        else pd.Series(0.0, index=df.index)
    growth = (base_growth + growth_delta).clip(-0.10, 0.20)

    # Entry multiple: compute from raw EV / EBITDA (bypasses NaN set by winsorization).
    # The winsorized ev_to_ebitda column sets < 6x to NaN for scoring, but for IRR
    # we must use the actual entry multiple floored at 6x — not fall back to the cap.
    # Falling back to the 8x cap for a 5x entry company models free multiple expansion.
    if "enterprise_value" in df.columns and "ebitda" in df.columns:
        raw_ev_ebitda = (
            df["enterprise_value"] / df["ebitda"].replace(0, np.nan)
        )
        entry_ev_ebitda = raw_ev_ebitda.clip(lower=6.0).fillna(exit_multiple_cap)
    elif "ev_to_ebitda" in df.columns:
        entry_ev_ebitda = df["ev_to_ebitda"].fillna(exit_multiple_cap).clip(lower=6.0)
    else:
        entry_ev_ebitda = pd.Series(exit_multiple_cap, index=df.index)

    if force_exit_at_entry_multiple:
        # Exit at raw entry multiple — used for IRR bridge baseline isolation
        exit_multiple = entry_ev_ebitda
    else:
        # Normal mode: cap at config, add delta, floor at 4x
        exit_multiple = (
            entry_ev_ebitda.clip(upper=exit_multiple_cap) + exit_multiple_delta
        ).clip(lower=4.0)

    exit_ebitda = df["ebitda"].clip(lower=0) * ((1 + growth) ** holding_period)
    exit_ev = exit_ebitda * exit_multiple

    # FCF stressed by fcf_conversion_mult (FIX 1: downside FCF compression)
    annual_fcf = df["free_cash_flow"].fillna(0.0).clip(lower=0) \
        if "free_cash_flow" in df.columns \
        else pd.Series(0.0, index=df.index)
    annual_fcf_stressed = annual_fcf * fcf_conversion_mult

    # Annual debt amortization with interest waterfall (FIX 3)
    debt_remaining = _amortize_debt(
        df["max_debt"], annual_fcf_stressed, debt_repayment_rate,
        holding_period, interest_rate
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
    # Hard cap at 40%: anything above is model noise from extreme entry multiples or FCF
    return irr.clip(-0.50, 0.40)


def compute_lbo_irr_proxy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute base-case IRR proxy.

    exit_multiple = min(entry EV/EBITDA, config cap) — never exits above cap.
    Debt amortization uses annual loop per _amortize_debt() with interest waterfall.

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

    Base:     growth as-is,     FCF at 100%,  exit multiple cap unchanged
    Upside:   growth + 2%,      FCF at 100%,  exit multiple cap + 0.5x
    Downside: growth - 3%,      FCF at 80%,   exit multiple cap - 1x
      → Downside uses fcf_conversion_mult=0.80 to stress FCF (revenue miss + margin compression)
      → Asymmetric: downside growth miss (-3%) is larger than upside beat (+2%) by design

    Key vs previous: growth clips to (-10%, +20%) so ±3%/2% creates real spread.
    Exit delta is additive after base clip, ensuring scenarios differ even when entry << cap.
    FCF stress in downside creates additional spread beyond just growth/multiple assumptions.
    """
    required_cols = {"ebitda", "equity_required", "max_debt"}
    if not required_cols.issubset(df.columns):
        df["irr_base"] = np.nan
        df["irr_upside"] = np.nan
        df["irr_downside"] = np.nan
        return df

    df["irr_base"] = _compute_single_irr(
        df, cfg, growth_delta=0.00, exit_multiple_delta=0.0, fcf_conversion_mult=1.0
    ).clip(-0.50, 0.40)

    df["irr_upside"] = _compute_single_irr(
        df, cfg, growth_delta=+0.02, exit_multiple_delta=+0.5, fcf_conversion_mult=1.0
    ).clip(-0.50, 0.40)

    df["irr_downside"] = _compute_single_irr(
        df, cfg, growth_delta=-0.03, exit_multiple_delta=-1.0, fcf_conversion_mult=0.80
    ).clip(-0.50, 0.40)

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
    Decompose the base-case IRR into its 3 main value drivers using isolation method.

    PE context: An IRR number alone is meaningless to an IC.
    They want to know WHERE the return comes from:
    - EBITDA growth contribution: what the company's organic growth adds vs zero growth
    - Deleveraging contribution: debt paydown → more equity value at exit
    - Multiple delta: effect of exit cap vs exiting at entry multiple
      → negative when entry > cap (compression hurts); ~0 when entry <= cap

    Method (isolation baseline):
      Baseline = zero growth + no debt repayment + exit at entry multiple
        (the "do nothing" scenario — measures the floor return from entry alone)

      Growth driver:     IRR(real growth, no delev, exit at entry) - baseline
      Deleverage driver: IRR(zero growth, real delev, exit at entry) - baseline
      Multiple driver:   IRR(zero growth, no delev, exit at cap)    - baseline
        → negative when cap < entry (multiple compression); ~0 when entry <= cap

    The three drivers approximately sum to irr_base — exact decomposition would
    require Shapley values; this isolation method is directionally correct and
    matches IC intuition about return attribution.
    """
    required = {"ebitda", "equity_required", "max_debt", "irr_base"}
    if not required.issubset(df.columns):
        df["irr_driver_growth"] = np.nan
        df["irr_driver_deleveraging"] = np.nan
        df["irr_driver_multiple"] = np.nan
        return df

    # --- Baseline: zero growth + no deleverage + exit at entry multiple ---
    df_zero_growth = df.copy()
    df_zero_growth["revenue_growth"] = 0.0
    cfg_no_delev = {**cfg, "lbo": {**cfg.get("lbo", {}), "debt_repayment_rate": 0.0}}

    irr_baseline = _compute_single_irr(
        df_zero_growth, cfg_no_delev, force_exit_at_entry_multiple=True
    )

    # --- Growth driver: add real growth vs baseline ---
    # (actual growth, no deleverage, exit at entry multiple)
    irr_with_growth = _compute_single_irr(
        df, cfg_no_delev, force_exit_at_entry_multiple=True
    )
    df["irr_driver_growth"] = (irr_with_growth - irr_baseline).round(4)

    # --- Deleverage driver: add real debt paydown vs baseline ---
    # (zero growth, real deleverage, exit at entry multiple)
    irr_with_delev = _compute_single_irr(
        df_zero_growth, cfg, force_exit_at_entry_multiple=True
    )
    df["irr_driver_deleveraging"] = (irr_with_delev - irr_baseline).round(4)

    # --- Multiple driver: exit at cap vs exit at entry multiple ---
    # (zero growth, no deleverage, normal exit — cap applies)
    # Positive when entry > cap (cap helps = entry too rich → not applicable for cheap names)
    # Negative when cap < entry (cap hurts = compression); ~0 when entry <= cap
    irr_with_cap = _compute_single_irr(
        df_zero_growth, cfg_no_delev
    )
    df["irr_driver_multiple"] = (irr_with_cap - irr_baseline).round(4)

    logger.info("IRR bridge (growth / deleveraging / multiple) computed via isolation method")
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
