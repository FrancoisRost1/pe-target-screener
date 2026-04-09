"""
classifier.py — Debt capacity classification, red flag detection, and score penalties

All thresholds and penalty magnitudes are loaded from config.yaml —
nothing is hardcoded. The same thresholds drive both flag detection
and score penalties (single source of truth).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def classify_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Run debt capacity classification, red flag detection, and score penalties.
    """
    from screener.classifier_rules import classify_debt_capacity
    df = df.copy()
    dc_cfg = cfg.get("debt_capacity", {})

    df["debt_capacity"] = df.apply(lambda row: classify_debt_capacity(row, dc_cfg), axis=1)
    df = detect_flags_and_penalties(df, cfg)
    df = compute_valuation_penalty(df, cfg)

    logger.info(f"Debt capacity: {df['debt_capacity'].value_counts().to_dict()}")
    logger.info(f"Red flags found in {(df['red_flags'] != '').sum()} companies")
    return df


def detect_flags_and_penalties(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Unified red flag detection and penalty computation.
    Both text flags and numeric penalties share the same config thresholds.
    """
    df = df.copy()
    rf = cfg.get("red_flags", {})
    rfp = cfg.get("red_flag_penalties", {})

    penalty = pd.Series(0.0, index=df.index)
    flags = pd.Series([""] * len(df), index=df.index)

    checks = [
        ("interest_coverage", "<", rf.get("max_interest_coverage"), "Low interest coverage", rfp.get("low_interest_coverage", -10)),
        ("net_debt_to_ebitda", ">", rf.get("max_net_debt_to_ebitda"), "High leverage", rfp.get("high_leverage", -8)),
        ("fcf_conversion", "<", rf.get("min_fcf_conversion"), "Weak cash conversion", rfp.get("weak_fcf_conversion", -8)),
        ("revenue_growth", "<", rf.get("min_revenue_growth"), "Negative revenue growth", rfp.get("negative_growth", -5)),
        ("ev_to_ebitda", ">", rf.get("max_ev_to_ebitda"), "Expensive valuation", rfp.get("expensive_valuation", -5)),
        ("capex_to_revenue", ">", rf.get("max_capex_to_revenue"), "High capex intensity", rfp.get("high_capex", -5)),
    ]

    if "ebitda" in df.columns:
        neg_ebitda = df["ebitda"].fillna(0) < 0
        penalty += neg_ebitda * rfp.get("negative_ebitda", -10)
        flags = _append_flags_series(flags, neg_ebitda, "Negative EBITDA")

    for col, op, threshold, label, pen_value in checks:
        if col not in df.columns or threshold is None:
            continue
        series = df[col].fillna(np.nan)
        triggered = series.lt(threshold).fillna(False) if op == "<" else series.gt(threshold).fillna(False)
        penalty += triggered * pen_value
        flags = _append_flags_series(flags, triggered, label)

    df["red_flags"] = flags
    df["red_flag_penalty"] = penalty
    return df


def _append_flags_series(flags: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    """Append a flag label to pipe-separated flag strings where mask is True."""
    result = flags.copy()
    for idx in mask.index[mask]:
        existing = result.at[idx]
        if not existing:
            result.at[idx] = label
        elif label not in existing:
            result.at[idx] = existing + " | " + label
    return result


def compute_valuation_penalty(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply a tiered penalty for expensive valuations.
    Tiers loaded from config.yaml valuation_penalties section.
    """
    df = df.copy()
    vp_cfg = cfg.get("valuation_penalties", {})
    tiers = sorted(
        vp_cfg.get("tiers", [{"threshold": 18.0, "penalty": -15},
                              {"threshold": 14.0, "penalty": -8},
                              {"threshold": 12.0, "penalty": -3}]),
        key=lambda t: t["threshold"],  # ascending: highest threshold overwrites last
    )

    penalty = pd.Series(0.0, index=df.index)
    if "ev_to_ebitda" in df.columns:
        ev = df["ev_to_ebitda"]
        result = np.zeros(len(df))
        for tier in tiers:
            result = np.where(ev > tier["threshold"], tier["penalty"], result)
        penalty = pd.Series(result, index=df.index, dtype=float)
        penalty[df["ev_to_ebitda"].isna()] = 0.0

    df["valuation_penalty"] = penalty
    return df
