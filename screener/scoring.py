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


def score_universe_sector_adjusted(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Sector-adjusted scoring: blends universe-wide and within-sector percentile ranks.

    Formula per metric:
      universe_rank = percentile rank across all companies (0–100)
      sector_rank   = percentile rank within the company's sector (0–100)
      final_rank    = 0.6 × sector_rank + 0.4 × universe_rank

    Rationale: A 20% EBITDA margin in Consumer Staples is exceptional.
    The same margin in Technology is below-average. Sector-relative ranking
    rewards companies that stand out within their peer group — which is how
    PE deal teams actually evaluate opportunities.

    Outputs:
      pe_score_raw      — universe-only weighted score (preserved for reference)
      pe_score          — sector-adjusted blended score (used for ranking)
    """
    df = df.copy()
    weights = cfg.get("weights", {})
    invert = cfg.get("invert_metrics", [])

    has_sector = "sector" in df.columns and df["sector"].notna().any()
    score_cols = []

    for metric, weight in weights.items():
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found — skipping")
            continue

        series = df[metric]
        score_col = f"score_{metric}"

        # Universe-wide percentile rank
        u_rank = series.rank(method="average", na_option="keep", pct=True) * 100

        if has_sector:
            # Within-sector percentile rank; fall back to universe rank for
            # sectors with fewer than 3 companies (unstable percentiles)
            s_rank = df.groupby("sector")[metric].rank(
                method="average", na_option="keep", pct=True
            ) * 100
            sector_size = df.groupby("sector")["sector"].transform("count")
            s_rank = s_rank.where(sector_size >= 3, u_rank)
            blended = 0.6 * s_rank + 0.4 * u_rank
        else:
            blended = u_rank

        if metric in invert:
            u_rank = 100 - u_rank
            blended = 100 - blended

        df[score_col] = blended
        score_cols.append((score_col, u_rank if metric in invert else
                           series.rank(method="average", na_option="keep", pct=True) * 100,
                           weight))

    if not score_cols:
        logger.error("No metrics could be scored")
        df["pe_score_raw"] = np.nan
        df["pe_score"] = np.nan
        return df

    total_weight = sum(w for _, _, w in score_cols)

    # pe_score_raw: universe-only (for reference / comparison)
    raw_scores = []
    for score_col, u_rank_series, w in score_cols:
        metric = score_col.replace("score_", "")
        u_series = df[metric].rank(method="average", na_option="keep", pct=True) * 100
        if metric in invert:
            u_series = 100 - u_series
        raw_scores.append((u_series, w))

    df["pe_score_raw"] = sum(
        s.fillna(50) * (w / total_weight) for s, w in raw_scores
    ).round(2)

    # pe_score: sector-adjusted blended
    df["pe_score"] = sum(
        df[col].fillna(50) * (w / total_weight) for col, _, w in score_cols
    ).round(2)

    logger.info(f"Sector-adjusted scoring: {df['pe_score'].notna().sum()} companies")
    return df


def apply_score_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply red flag and valuation penalties to produce pe_score_adjusted.

    pe_score_adjusted = pe_score + red_flag_penalty + valuation_penalty, clipped [0, 100]

    PE context: A high raw score is necessary but not sufficient. A company
    with strong margins but dangerous leverage or an astronomic valuation
    deserves to be penalised before reaching a shortlist.
    """
    df = df.copy()
    penalty_cols = []
    if "red_flag_penalty" in df.columns:
        penalty_cols.append(df["red_flag_penalty"].fillna(0))
    if "valuation_penalty" in df.columns:
        penalty_cols.append(df["valuation_penalty"].fillna(0))

    total_penalty = sum(penalty_cols) if penalty_cols else 0
    df["pe_score_adjusted"] = (df["pe_score"] + total_penalty).clip(0, 100).round(2)
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
