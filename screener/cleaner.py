"""
cleaner.py — Data cleaning and normalization

Handles missing values, type casting, outlier detection, and
data quality flags before ratio computation begins.

PE context: Bad data leads to bad decisions. A screener is only as
credible as the data underneath it.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

NUMERIC_COLS = [
    "revenue", "prior_revenue", "ebitda", "prior_ebitda", "ebit",
    "net_income", "interest_expense", "da", "total_debt", "cash",
    "total_assets", "total_equity", "invested_capital", "capex",
    "operating_cash_flow", "free_cash_flow", "enterprise_value", "market_cap",
]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Normalize column names
    2. Cast numeric columns
    3. Remove duplicates
    4. Flag data quality issues
    """
    df = df.copy()
    df = _normalize_columns(df)
    df = _cast_numerics(df)
    df = _remove_duplicates(df)
    df = _flag_data_quality(df)
    logger.info(f"Cleaned dataset: {len(df)} companies")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip all column names."""
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df


def _cast_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Force all financial columns to numeric. Coerce errors to NaN."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate tickers, keep first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    removed = before - len(df)
    if removed:
        logger.warning(f"Removed {removed} duplicate tickers")
    return df


def _flag_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a data_quality column flagging structural issues.
    Companies with flags are kept but may be filtered later.
    """
    flags = pd.Series([""] * len(df), index=df.index)

    if "revenue" in df.columns:
        flags = flags.where(df["revenue"].notna(), flags + "missing_revenue;")
        flags = flags.where(~(df["revenue"] <= 0), flags + "negative_revenue;")

    if "ebitda" in df.columns:
        flags = flags.where(df["ebitda"].notna(), flags + "missing_ebitda;")
        flags = flags.where(~(df["ebitda"] <= 0), flags + "negative_ebitda;")

    if "enterprise_value" in df.columns:
        flags = flags.where(df["enterprise_value"].notna(), flags + "missing_ev;")

    if "interest_expense" in df.columns:
        flags = flags.where(df["interest_expense"].notna(), flags + "missing_interest;")

    df["data_quality_flags"] = flags.str.strip(";")
    return df


def winsorize_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme ratio values before scoring to prevent outliers from distorting
    percentile ranks. Winsorization is standard practice in quantitative screens.

    PE context: A 2620x interest coverage is a data artifact, not alpha signal.
    Capping at 50x preserves rank information while eliminating noise.
    Negative EV/EBITDA is set to NaN — it is meaningless in an LBO context.
    """
    df = df.copy()

    if "interest_coverage" in df.columns:
        df["interest_coverage"] = df["interest_coverage"].clip(upper=50.0)

    if "ev_to_ebitda" in df.columns:
        # Negative or below 6x EV/EBITDA = data error in an LBO context.
        # Real buyout targets don't trade below 4-5x; values below 6x are
        # almost always stale data or sector misclassification artifacts.
        df["ev_to_ebitda"] = df["ev_to_ebitda"].where(
            df["ev_to_ebitda"].isna() | (df["ev_to_ebitda"] >= 6.0), np.nan
        )

    if "net_debt_to_ebitda" in df.columns:
        df["net_debt_to_ebitda"] = df["net_debt_to_ebitda"].clip(-5.0, 15.0)

    if "revenue_growth" in df.columns:
        df["revenue_growth"] = df["revenue_growth"].clip(-0.5, 0.5)

    if "ebitda_growth" in df.columns:
        df["ebitda_growth"] = df["ebitda_growth"].clip(-0.5, 0.5)

    if "fcf_conversion" in df.columns:
        df["fcf_conversion"] = df["fcf_conversion"].clip(-0.5, 2.0)

    if "roic" in df.columns:
        df["roic"] = df["roic"].clip(-0.5, 1.0)

    if "fcf_yield_ev" in df.columns:
        df["fcf_yield_ev"] = df["fcf_yield_ev"].clip(-0.2, 0.5)

    logger.info("Winsorization applied to ratio columns")
    return df


def apply_eligibility_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply pre-screening eligibility filters.
    Two tiers:
      1. Size filters (from config): minimum revenue and EBITDA in absolute terms.
      2. Hard PE filters: non-negotiable quality gates applied after winsorization.

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

    # ── Tier 2: hard PE filters (run after winsorization) ────────────────────
    before_hard = len(df)

    if "ebitda" in df.columns:
        fails = df["ebitda"].fillna(0) <= 0
        logger.info(f"  Hard filter: removed {fails.sum()} companies with EBITDA ≤ 0")
        df = df[~fails].reset_index(drop=True)

    if "ebitda_margin" in df.columns:
        fails = df["ebitda_margin"].fillna(0) < 0.08
        logger.info(f"  Hard filter: removed {fails.sum()} companies with EBITDA margin < 8%")
        df = df[~fails].reset_index(drop=True)

    if "interest_coverage" in df.columns:
        fails = df["interest_coverage"].fillna(0) < 2.5
        logger.info(f"  Hard filter: removed {fails.sum()} companies with interest coverage < 2.5x")
        df = df[~fails].reset_index(drop=True)

    logger.info(f"Eligibility filter: {before} → {len(df)} companies "
                f"({before - before_hard} size, {before_hard - len(df)} hard PE filters)")
    return df
