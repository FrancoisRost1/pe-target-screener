"""
ranking.py — Final ranking and top-N selection

Sorts companies by pe_score, assigns rank, and extracts the shortlist.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

DISPLAY_COLS = [
    "rank", "ticker", "company", "sector",
    "ebitda_margin", "roic", "fcf_conversion",
    "net_debt_to_ebitda", "interest_coverage",
    "ev_to_ebitda", "revenue_growth",
    "debt_capacity", "red_flags",
    "quality_score", "cash_score", "leverage_score", "valuation_score",
    "pe_score",
]


def rank_companies(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by pe_score descending and assign rank."""
    df = df.copy()
    df = df.sort_values("pe_score", ascending=False, na_position="last")
    df["rank"] = range(1, len(df) + 1)
    logger.info(f"Ranked {len(df)} companies")
    return df.reset_index(drop=True)


def get_top_n(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top N candidates with selected display columns."""
    available = [c for c in DISPLAY_COLS if c in df.columns]
    top = df.head(n)[available].copy()
    return top
