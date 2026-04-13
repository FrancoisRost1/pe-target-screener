"""
scoring.py, Weighted percentile scoring engine

Converts raw financial ratios into a normalized 0–100 PE attractiveness score.
Blend weights loaded from config.yaml scoring section.

Delegates penalty adjustments to scoring_adjustments.py.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Re-export so callers can import from screener.scoring
from screener.scoring_adjustments import (  # noqa: E402, F401
    apply_score_adjustments, compute_irr_blended_score, apply_irr_hurdle_penalty,
)


def score_universe_sector_adjusted(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Sector-adjusted scoring: blends universe-wide and within-sector percentile ranks.

    Per-metric: universe_rank + sector_rank → invert if needed → fill NaN → blend.
    Blend weights from config.yaml scoring section (default 0.60/0.40).
    """
    df = df.copy()
    weights = cfg.get("weights", {})
    invert = cfg.get("invert_metrics", [])
    scoring_cfg = cfg.get("scoring", {})
    sector_w = scoring_cfg.get("sector_weight", 0.60)
    universe_w = scoring_cfg.get("universe_weight", 0.40)
    has_sector = "sector" in df.columns and df["sector"].notna().any()

    blended_cols = []
    raw_u_ranks = []

    for metric, weight in weights.items():
        if weight == 0 or metric not in df.columns:
            continue

        score_col = f"score_{metric}"
        u_rank = df[metric].rank(method="average", na_option="keep", pct=True) * 100

        if has_sector:
            s_rank = df.groupby("sector")[metric].rank(
                method="average", na_option="keep", pct=True) * 100
            sector_size = df.groupby("sector")["sector"].transform("count")
            s_rank = s_rank.where(sector_size >= 3, u_rank)
        else:
            s_rank = u_rank.copy()

        if metric in invert:
            u_rank = 100 - u_rank
            s_rank = 100 - s_rank

        u_rank_filled = u_rank.fillna(50)
        s_rank_filled = s_rank.fillna(50)
        blended = sector_w * s_rank_filled + universe_w * u_rank_filled
        df[score_col] = blended

        blended_cols.append((score_col, weight))
        raw_u_ranks.append((u_rank_filled, weight))

    if not blended_cols:
        logger.error("No metrics could be scored")
        df["pe_score_raw"] = np.nan
        df["pe_score"] = np.nan
        return df

    total_weight = sum(w for _, w in blended_cols)
    df["pe_score_raw"] = sum(u * (w / total_weight) for u, w in raw_u_ranks).round(2)
    df["pe_score"] = sum(df[col] * (w / total_weight) for col, w in blended_cols).round(2)

    logger.info(f"Sector-adjusted scoring: {df['pe_score'].notna().sum()} companies")
    return df


def compute_sub_scores(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute 4 sub-scores for dashboard drill-down:
    quality, cash, leverage, valuation.
    """
    df = df.copy()
    sub_score_map = {
        "quality_score": ["ebitda_margin", "roic"],
        "cash_score": ["fcf_conversion", "ocf_margin"],
        "leverage_score": ["net_debt_to_ebitda", "interest_coverage"],
        "valuation_score": ["ev_to_ebitda"],
    }
    for sub_name, metrics in sub_score_map.items():
        available = [m for m in metrics if f"score_{m}" in df.columns]
        if available:
            df[sub_name] = df[[f"score_{m}" for m in available]].mean(axis=1).round(2)
        else:
            df[sub_name] = np.nan
    return df
