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
    """Run debt capacity classification and red flag detection."""
    df = df.copy()
    dc_cfg = cfg.get("debt_capacity", {})
    rf_cfg = cfg.get("red_flags", {})

    df["debt_capacity"] = df.apply(lambda row: _classify_debt_capacity(row, dc_cfg), axis=1)
    df["red_flags"] = df.apply(lambda row: _detect_red_flags(row, rf_cfg), axis=1)

    logger.info(f"Debt capacity: {df['debt_capacity'].value_counts().to_dict()}")
    logger.info(f"Red flags found in {(df['red_flags'] != '').sum()} companies")
    return df


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
