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


def apply_eligibility_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply pre-screening eligibility filters from config.
    Removes companies that fail minimum criteria before scoring.

    PE context: Funds typically have minimum size requirements and
    won't look at businesses with structural deficiencies.
    """
    e = cfg.get("eligibility", {})
    before = len(df)
    mask = pd.Series([True] * len(df), index=df.index)

    if "revenue" in df.columns and "min_revenue" in e:
        mask &= df["revenue"].fillna(0) >= e["min_revenue"]

    if "ebitda" in df.columns and "min_ebitda" in e:
        mask &= df["ebitda"].fillna(0) >= e["min_ebitda"]

    if "sector" in df.columns and "exclude_sectors" in e:
        mask &= ~df["sector"].isin(e["exclude_sectors"])

    df = df[mask].reset_index(drop=True)
    logger.info(f"Eligibility filter: {before} → {len(df)} companies")
    return df
