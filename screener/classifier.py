"""
classifier.py — Debt capacity classification and red flag detection

This is the "business judgment" layer of the screener.
Pure ratios tell you numbers. This module adds interpretation.

PE context: A PE fund doesn't just want a high score — it needs to know
if the business can actually absorb acquisition debt (debt capacity)
and whether there are structural warning signs (red flags).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def classify_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Run debt capacity classification, red flag detection, and score penalties.

    Order matters: penalties are applied after flags so both share the same
    threshold definitions from config.
    """
    df = df.copy()
    dc_cfg = cfg.get("debt_capacity", {})
    rf_cfg = cfg.get("red_flags", {})

    df["debt_capacity"] = df.apply(lambda row: _classify_debt_capacity(row, dc_cfg), axis=1)
    df["red_flags"] = df.apply(lambda row: _detect_red_flags(row, rf_cfg), axis=1)
    df = compute_red_flag_penalty(df, cfg)
    df = compute_valuation_penalty(df, cfg)

    logger.info(f"Debt capacity: {df['debt_capacity'].value_counts().to_dict()}")
    logger.info(f"Red flags found in {(df['red_flags'] != '').sum()} companies")
    return df


def compute_red_flag_penalty(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Assign a numeric score penalty per red flag triggered.

    PE context: A high composite score should be discounted when the company
    carries structural red flags. This penalty is subtracted from pe_score
    in the adjustment step, ensuring flagged companies don't top the shortlist.

    Penalties:
      -10: Negative EBITDA (structural loss-maker)
      -10: Interest coverage < 2.0x (can barely service existing debt)
       -8: Net Debt/EBITDA > 5.0x (dangerously leveraged already)
       -8: FCF conversion < 20% (poor cash generation)
       -5: Revenue growth < -5% (declining business)
       -5: EV/EBITDA > 20x (too expensive for LBO)
       -5: CapEx/Revenue > 15% (capital intensive — hurts debt service)
    """
    df = df.copy()
    rf = cfg.get("red_flags", {})
    penalty = pd.Series(0.0, index=df.index)

    if "ebitda" in df.columns:
        penalty += df["ebitda"].fillna(0).lt(0) * -10

    if "interest_coverage" in df.columns:
        penalty += df["interest_coverage"].fillna(np.nan).lt(2.0).fillna(False) * -10

    if "net_debt_to_ebitda" in df.columns:
        penalty += df["net_debt_to_ebitda"].fillna(np.nan).gt(5.0).fillna(False) * -8

    if "fcf_conversion" in df.columns:
        penalty += df["fcf_conversion"].fillna(np.nan).lt(0.20).fillna(False) * -8

    if "revenue_growth" in df.columns:
        penalty += df["revenue_growth"].fillna(np.nan).lt(-0.05).fillna(False) * -5

    if "ev_to_ebitda" in df.columns:
        penalty += df["ev_to_ebitda"].fillna(np.nan).gt(20.0).fillna(False) * -5

    if "capex_to_revenue" in df.columns:
        penalty += df["capex_to_revenue"].fillna(np.nan).gt(0.15).fillna(False) * -5

    df["red_flag_penalty"] = penalty
    return df


def compute_valuation_penalty(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply a tiered penalty for expensive valuations.

    PE context: Valuation is the single biggest driver of IRR. Overpaying at
    entry is the most common cause of PE deal failure. This penalty directly
    discounts the score of richly-priced companies regardless of quality.

    Tiers (EV/EBITDA):
      > 18x  → -15 (deal almost certainly can't generate 20%+ IRR)
      > 14x  → -8  (stretched — needs significant multiple expansion or growth)
      > 12x  → -3  (slightly above LBO sweet spot)
      ≤ 12x  → 0   (within LBO-friendly range)
    """
    df = df.copy()
    penalty = pd.Series(0.0, index=df.index)

    if "ev_to_ebitda" in df.columns:
        ev = df["ev_to_ebitda"].fillna(np.nan)
        penalty = np.where(ev > 18, -15,
                  np.where(ev > 14, -8,
                  np.where(ev > 12, -3, 0)))
        penalty = pd.Series(penalty, index=df.index, dtype=float)
        # NaN EV/EBITDA → no penalty (benefit of the doubt)
        penalty[df["ev_to_ebitda"].isna()] = 0.0

    df["valuation_penalty"] = penalty
    return df


def apply_deal_killer_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multiplicative score penalties for deals with broken LBO math.

    PE context: An additive penalty (-20pts) is insufficient when a company
    has a 90+ raw score — it can still rank near the top despite a negative IRR.
    Multiplying the score (or zeroing it) ensures broken deals fall below all
    viable candidates, regardless of their quality metrics.

    Logic (applied to pe_score_adjusted, which must already exist):
      irr_proxy < 0 but >= -15%  → halve pe_score_adjusted (×0.5)
      irr_proxy < -15%           → set pe_score_adjusted = 0 (deal is dead)
      equity_required > EV × 90% → additional -10pts (almost no leverage)

    Also appends "Negative IRR" to red_flags for drill-down transparency.
    Records effective penalty in deal_killer_penalty column for display.
    """
    df = df.copy()

    if "pe_score_adjusted" not in df.columns:
        df["deal_killer_penalty"] = 0.0
        return df

    original_adj = df["pe_score_adjusted"].copy()

    if "red_flags" not in df.columns:
        df["red_flags"] = ""

    if "irr_proxy" in df.columns:
        irr = df["irr_proxy"]
        neg_irr_severe = irr.notna() & (irr < -0.15)
        neg_irr_mild = irr.notna() & (irr < 0) & ~neg_irr_severe

        # Halve score for mildly negative IRR
        df.loc[neg_irr_mild, "pe_score_adjusted"] = (
            df.loc[neg_irr_mild, "pe_score_adjusted"] * 0.5
        ).round(2)

        # Zero out for severely negative IRR
        df.loc[neg_irr_severe, "pe_score_adjusted"] = 0.0

        # Append flag (avoid duplicating if already present)
        flagged = neg_irr_mild | neg_irr_severe
        df.loc[flagged, "red_flags"] = df.loc[flagged].apply(
            lambda r: _append_flag(r.get("red_flags", ""), "Negative IRR"), axis=1
        )

    # Bloated equity: almost no leverage — LBO loses its return engine
    if "equity_required" in df.columns and "enterprise_value" in df.columns:
        bloated = (
            df["equity_required"].fillna(np.nan) > df["enterprise_value"].fillna(np.nan) * 0.90
        ).fillna(False)
        df.loc[bloated, "pe_score_adjusted"] = (
            df.loc[bloated, "pe_score_adjusted"] - 10
        ).clip(lower=0).round(2)
        df.loc[bloated, "red_flags"] = df.loc[bloated].apply(
            lambda r: _append_flag(r.get("red_flags", ""), "Deal math challenged"), axis=1
        )

    # Record effective impact for display in drill-down
    df["deal_killer_penalty"] = (df["pe_score_adjusted"] - original_adj).round(2)

    penalized = (df["deal_killer_penalty"] != 0).sum()
    logger.info(f"Deal killer penalty: {penalized} companies penalized")
    return df


def _append_flag(existing: str, new_flag: str) -> str:
    """Append a flag label to the pipe-separated red_flags string without duplicating."""
    if not existing or (isinstance(existing, float) and pd.isna(existing)):
        return new_flag
    if new_flag in str(existing):
        return str(existing)
    return str(existing) + " | " + new_flag


def _classify_debt_capacity(row: pd.Series, cfg: dict) -> str:
    """
    Classify a company's ability to take on additional LBO debt.

    High: Strong margins, high cash conversion, low existing leverage,
          solid interest coverage. Prime LBO candidate from a leverage standpoint.

    Medium: Acceptable profile but with one or more limiting factors.

    Low: Structural limitations — too little cash generation or already
         too leveraged. Difficult to finance a buyout.
    """
    high = cfg.get("high", {})
    low = cfg.get("low", {})

    ebitda_margin = row.get("ebitda_margin")
    fcf_conversion = row.get("fcf_conversion")
    net_debt_to_ebitda = row.get("net_debt_to_ebitda")
    interest_coverage = row.get("interest_coverage")

    # Check LOW first (disqualifying conditions)
    low_triggers = 0
    if _val_check(ebitda_margin, "<", low.get("max_ebitda_margin")):
        low_triggers += 1
    if _val_check(net_debt_to_ebitda, ">", low.get("max_net_debt_to_ebitda")):
        low_triggers += 1
    if _val_check(interest_coverage, "<", low.get("max_interest_coverage")):
        low_triggers += 1
    if low_triggers >= 2:
        return "Low"

    # Check HIGH (all conditions must be met)
    high_checks = [
        _val_check(ebitda_margin, ">=", high.get("min_ebitda_margin")),
        _val_check(fcf_conversion, ">=", high.get("min_fcf_conversion")),
        _val_check(net_debt_to_ebitda, "<=", high.get("max_net_debt_to_ebitda")),
        _val_check(interest_coverage, ">=", high.get("min_interest_coverage")),
    ]
    # All 4 must be met (and not None) for High
    if all(c is True for c in high_checks):
        return "High"

    return "Medium"


def _detect_red_flags(row: pd.Series, cfg: dict) -> str:
    """
    Detect warning signals that would give a PE analyst pause.
    Returns pipe-separated string of flag labels, empty if clean.

    These don't disqualify a company but are surfaced for review.
    """
    flags = []

    checks = [
        ("interest_coverage", "<", cfg.get("max_interest_coverage"), "Low interest coverage"),
        ("net_debt_to_ebitda", ">", cfg.get("max_net_debt_to_ebitda"), "High leverage"),
        ("fcf_conversion", "<", cfg.get("min_fcf_conversion"), "Weak cash conversion"),
        ("revenue_growth", "<", cfg.get("min_revenue_growth"), "Negative revenue growth"),
        ("ev_to_ebitda", ">", cfg.get("max_ev_to_ebitda"), "Expensive valuation"),
        ("capex_to_revenue", ">", cfg.get("max_capex_to_revenue"), "High capex intensity"),
    ]

    # Always flag negative EBITDA
    ebitda = row.get("ebitda")
    if ebitda is not None and not pd.isna(ebitda) and ebitda <= 0:
        flags.append("Negative EBITDA")

    for col, op, threshold, label in checks:
        val = row.get(col)
        if _val_check(val, op, threshold):
            flags.append(label)

    return " | ".join(flags)


def _val_check(value, operator: str, threshold):
    """
    Safe comparison that returns None if either value is missing.
    Returns True if condition is met, False otherwise.
    """
    if value is None or threshold is None:
        return None
    if pd.isna(value) or pd.isna(threshold):
        return None
    if operator == "<":
        return value < threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    if operator == ">=":
        return value >= threshold
    return None
