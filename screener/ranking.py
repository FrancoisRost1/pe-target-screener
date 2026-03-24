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
    "irr_proxy", "irr_base", "irr_upside", "irr_downside",
    "irr_driver_growth", "irr_driver_deleveraging", "irr_driver_multiple",
    "max_debt", "equity_required", "fcf_yield_equity",
    "debt_capacity", "red_flags",
    "quality_score", "cash_score", "leverage_score", "valuation_score",
    "pe_score", "pe_score_raw", "pe_score_adjusted", "irr_score", "pe_score_final",
    "red_flag_penalty", "valuation_penalty", "deal_killer_penalty",
]


def rank_companies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by pe_score_adjusted (penalty-adjusted score) descending.
    Falls back to pe_score if pe_score_adjusted is not yet computed.

    PE context: Ranking by adjusted score ensures that companies with
    strong raw scores but serious red flags or expensive valuations
    don't appear at the top of the shortlist.
    """
    df = df.copy()
    sort_col = (
        "pe_score_final" if "pe_score_final" in df.columns else
        "pe_score_adjusted" if "pe_score_adjusted" in df.columns else
        "pe_score"
    )
    df = df.sort_values(sort_col, ascending=False, na_position="last")
    df["rank"] = range(1, len(df) + 1)
    logger.info(f"Ranked {len(df)} companies by {sort_col}")
    return df.reset_index(drop=True)


def get_top_n(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top N candidates with selected display columns."""
    available = [c for c in DISPLAY_COLS if c in df.columns]
    top = df.head(n)[available].copy()
    return top
