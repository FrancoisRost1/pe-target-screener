"""
classifier_rules.py, Deal killer penalties and debt capacity classification

Contains the multiplicative penalty logic for broken LBO math and the
rule-based debt capacity classification (High / Medium / Low).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_deal_killer_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multiplicative score penalties for deals with broken LBO math.

    Logic (applied to pe_score_adjusted):
      irr_proxy < 0 but >= -15%  → halve pe_score_adjusted (×0.5)
      irr_proxy < -15%           → set pe_score_adjusted = 0
      equity_required > EV × 90% → additional -10pts
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

        df.loc[neg_irr_mild, "pe_score_adjusted"] = (
            df.loc[neg_irr_mild, "pe_score_adjusted"] * 0.5
        ).round(2)
        df.loc[neg_irr_severe, "pe_score_adjusted"] = 0.0

        flagged = neg_irr_mild | neg_irr_severe
        df.loc[flagged, "red_flags"] = df.loc[flagged].apply(
            lambda r: _append_flag(r.get("red_flags", ""), "Negative IRR"), axis=1
        )

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

    df["deal_killer_penalty"] = (df["pe_score_adjusted"] - original_adj).round(2)
    penalized = (df["deal_killer_penalty"] != 0).sum()
    logger.info(f"Deal killer penalty: {penalized} companies penalized")
    return df


def classify_debt_capacity(row: pd.Series, cfg: dict) -> str:
    """
    Classify a company's ability to take on additional LBO debt.

    High: margin >20%, FCF conv >70%, leverage <2.5x, coverage >5x.
    Low: 2+ of margin <10%, leverage >4x, coverage <2.5x.
    Medium: everything else.
    """
    high = cfg.get("high", {})
    low = cfg.get("low", {})

    ebitda_margin = row.get("ebitda_margin")
    fcf_conversion = row.get("fcf_conversion")
    net_debt_to_ebitda = row.get("net_debt_to_ebitda")
    interest_coverage = row.get("interest_coverage")

    low_triggers = 0
    if _val_check(ebitda_margin, "<", low.get("max_ebitda_margin")):
        low_triggers += 1
    if _val_check(net_debt_to_ebitda, ">", low.get("max_net_debt_to_ebitda")):
        low_triggers += 1
    if _val_check(interest_coverage, "<", low.get("max_interest_coverage")):
        low_triggers += 1
    if low_triggers >= 2:
        return "Low"

    high_checks = [
        _val_check(ebitda_margin, ">=", high.get("min_ebitda_margin")),
        _val_check(fcf_conversion, ">=", high.get("min_fcf_conversion")),
        _val_check(net_debt_to_ebitda, "<=", high.get("max_net_debt_to_ebitda")),
        _val_check(interest_coverage, ">=", high.get("min_interest_coverage")),
    ]
    if all(c is True for c in high_checks):
        return "High"

    return "Medium"


def _val_check(value, operator: str, threshold):
    """Safe comparison returning None if either value is missing."""
    if value is None or threshold is None:
        return None
    if pd.isna(value) or pd.isna(threshold):
        return None
    ops = {"<": lambda a, b: a < b, ">": lambda a, b: a > b,
           "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b}
    return ops.get(operator, lambda a, b: None)(value, threshold)


def _append_flag(existing: str, new_flag: str) -> str:
    """Append a flag label to the pipe-separated red_flags string without duplicating."""
    if not existing or (isinstance(existing, float) and pd.isna(existing)):
        return new_flag
    if new_flag in str(existing):
        return str(existing)
    return str(existing) + " | " + new_flag
