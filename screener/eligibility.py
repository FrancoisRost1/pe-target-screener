"""
eligibility.py — Pre-screening eligibility filters

Applies size filters and hard PE quality gates before scoring.
All thresholds are loaded from config.yaml eligibility section.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def apply_eligibility_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply pre-screening eligibility filters.
    Two tiers:
      1. Size filters (from config): minimum revenue and EBITDA in absolute terms.
      2. Hard PE filters: thresholds from config.yaml eligibility section.

    PE context: Funds have minimum size requirements and won't consider businesses
    with structural deficiencies — negative EBITDA, insufficient interest coverage,
    or margins too thin to absorb acquisition debt.
    """
    e = cfg.get("eligibility", {})
    before = len(df)

    # ── Tier 1: size filters (config-driven) ──────────────────────────────────
    mask = pd.Series([True] * len(df), index=df.index)

    if "revenue" in df.columns and "min_revenue" in e:
        fails = df["revenue"].fillna(0) < e["min_revenue"]
        if fails.sum():
            logger.info(f"  Removed {fails.sum()} companies below min revenue")
        mask &= ~fails

    if "ebitda" in df.columns and "min_ebitda" in e:
        fails = df["ebitda"].fillna(0) < e["min_ebitda"]
        if fails.sum():
            logger.info(f"  Removed {fails.sum()} companies below min EBITDA")
        mask &= ~fails

    if "sector" in df.columns and "exclude_sectors" in e:
        fails = df["sector"].isin(e["exclude_sectors"])
        if fails.sum():
            logger.info(f"  Removed {fails.sum()} companies in excluded sectors")
        mask &= ~fails

    df = df[mask].reset_index(drop=True)

    # ── Tier 2: hard PE filters (thresholds from config) ─────────────────────
    before_hard = len(df)
    min_margin = e.get("min_ebitda_margin", 0.05)
    min_coverage = e.get("min_interest_coverage", 1.5)

    if "ebitda" in df.columns:
        fails = df["ebitda"].fillna(0) <= 0
        logger.info(f"  Hard filter: removed {fails.sum()} companies with EBITDA ≤ 0")
        df = df[~fails].reset_index(drop=True)

    if "ebitda_margin" in df.columns:
        fails = df["ebitda_margin"].fillna(0) < min_margin
        logger.info(f"  Hard filter: removed {fails.sum()} companies with EBITDA margin < {min_margin:.0%}")
        df = df[~fails].reset_index(drop=True)

    if "interest_coverage" in df.columns:
        fails = df["interest_coverage"].fillna(0) < min_coverage
        logger.info(f"  Hard filter: removed {fails.sum()} companies with interest coverage < {min_coverage:.1f}x")
        df = df[~fails].reset_index(drop=True)

    logger.info(f"Eligibility filter: {before} → {len(df)} companies "
                f"({before - before_hard} size, {before_hard - len(df)} hard PE filters)")
    return df
