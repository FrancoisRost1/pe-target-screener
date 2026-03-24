"""
scoring.py — Weighted percentile scoring engine

Converts raw financial ratios into a normalized 0–100 PE attractiveness score.

Method: Percentile ranking per metric (robust to outliers), then weighted sum.
For metrics where lower = better (EV/EBITDA, Net Debt/EBITDA), rank is inverted.

PE context: We're not judging absolute levels — we're ranking companies
relative to each other within this universe. That's exactly how a screening
committee works: "who's best in class across these dimensions?"
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

RATIO_COLS = [
    "ebitda_margin", "roic", "fcf_conversion", "ocf_margin",
    "net_debt_to_ebitda", "interest_coverage",
    "revenue_growth", "ebitda_growth", "ev_to_ebitda",
]


def score_universe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Main scoring entry point.
    For each metric: compute percentile rank → apply direction → weight → sum.
    Returns df with individual metric scores + final pe_score.
    """
    df = df.copy()
    weights = cfg.get("weights", {})
    invert = cfg.get("invert_metrics", [])

    score_cols = []
    for metric, weight in weights.items():
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found in dataframe — skipping")
            continue

        score_col = f"score_{metric}"
        series = df[metric]

        # Percentile rank: 0 = worst, 100 = best in universe
        ranked = series.rank(method="average", na_option="keep", pct=True) * 100

        # Invert if lower value = better (e.g. EV/EBITDA: cheaper = higher score)
        if metric in invert:
            ranked = 100 - ranked

        df[score_col] = ranked
        score_cols.append((score_col, weight))

    # Weighted sum across available scored metrics
    if not score_cols:
        logger.error("No metrics could be scored")
        df["pe_score"] = np.nan
        return df

    # Normalize weights to sum to 1 (handles missing metrics gracefully)
    total_weight = sum(w for _, w in score_cols)
    weighted_scores = sum(
        df[col].fillna(50) * (w / total_weight)  # Fill NaN with 50 = neutral
        for col, w in score_cols
    )
    df["pe_score"] = weighted_scores.round(2)

    logger.info(f"Scored {df['pe_score'].notna().sum()} companies")
    return df


def compute_sub_scores(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute 4 sub-scores for dashboard drill-down:
    - quality_score: profitability + ROIC
    - cash_score: FCF conversion + OCF margin
    - leverage_score: debt/EBITDA + interest coverage
    - valuation_score: EV/EBITDA
    """
    df = df.copy()
    invert = cfg.get("invert_metrics", [])

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
