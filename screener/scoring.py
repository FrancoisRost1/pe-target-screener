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
from screener.classifier import apply_deal_killer_penalty

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

    Per-metric formula (applied correctly at the metric level, not the score level):
      1. universe_rank = percentile rank across all companies (0–100)
      2. sector_rank   = percentile rank within company's sector (0–100)
         → fall back to universe_rank for sectors with < 3 companies
      3. Invert BOTH ranks independently if metric is in invert_metrics
         (lower value = better, e.g. EV/EBITDA)
      4. Fill NaN with 50 (neutral — no penalty for missing data)
      5. blended = 0.60 × sector_rank + 0.40 × universe_rank

    Weighted sum of blended scores → pe_score (sector-adjusted)
    Weighted sum of universe_rank  → pe_score_raw (universe-only, for comparison)

    Rationale: A 20% EBITDA margin in Consumer Staples is exceptional.
    The same margin in Technology is below-average. Sector-relative ranking
    rewards companies that stand out within their peer group.

    Key fix vs previous version: inversion is applied to each rank independently
    before blending, ensuring the direction correction is symmetric across both
    the universe and sector dimensions.
    """
    df = df.copy()
    weights = cfg.get("weights", {})
    invert = cfg.get("invert_metrics", [])
    has_sector = "sector" in df.columns and df["sector"].notna().any()

    blended_cols = []   # (score_col, weight) for pe_score
    raw_u_ranks = []    # (u_rank_series, weight) for pe_score_raw

    for metric, weight in weights.items():
        if weight == 0:
            continue  # Skip zero-weight metrics entirely
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not found — skipping")
            continue

        score_col = f"score_{metric}"

        # Step 1: Universe percentile rank
        u_rank = df[metric].rank(method="average", na_option="keep", pct=True) * 100

        # Step 2: Sector percentile rank (per-metric, not on blended scores)
        if has_sector:
            s_rank = df.groupby("sector")[metric].rank(
                method="average", na_option="keep", pct=True
            ) * 100
            # Fall back to universe rank for small sectors (< 3 companies)
            sector_size = df.groupby("sector")["sector"].transform("count")
            s_rank = s_rank.where(sector_size >= 3, u_rank)
        else:
            s_rank = u_rank.copy()

        # Step 3: Invert both ranks independently before blending
        if metric in invert:
            u_rank = 100 - u_rank
            s_rank = 100 - s_rank

        # Step 4: Fill NaN with 50 (neutral — no penalty for missing data)
        u_rank_filled = u_rank.fillna(50)
        s_rank_filled = s_rank.fillna(50)

        # Step 5: Blend — sector rank dominates (60%) but universe anchors (40%)
        blended = 0.60 * s_rank_filled + 0.40 * u_rank_filled
        df[score_col] = blended

        blended_cols.append((score_col, weight))
        raw_u_ranks.append((u_rank_filled, weight))

    if not blended_cols:
        logger.error("No metrics could be scored")
        df["pe_score_raw"] = np.nan
        df["pe_score"] = np.nan
        return df

    total_weight = sum(w for _, w in blended_cols)

    # pe_score_raw: universe-only weighted sum (visible in drill-down for comparison)
    df["pe_score_raw"] = sum(
        u * (w / total_weight) for u, w in raw_u_ranks
    ).round(2)

    # pe_score: sector-adjusted blended weighted sum
    df["pe_score"] = sum(
        df[col] * (w / total_weight) for col, w in blended_cols
    ).round(2)

    logger.info(f"Sector-adjusted scoring: {df['pe_score'].notna().sum()} companies")
    return df


def apply_score_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all score penalties to produce pe_score_adjusted.

    Step 1 — Additive penalties (red flags, expensive valuation):
      pe_score_adjusted = pe_score + red_flag_penalty + valuation_penalty
      Clipped to [0, 100].

    Step 2 — Deal killer (multiplicative, applied to the result of Step 1):
      irr < 0 but >= -15%  → pe_score_adjusted × 0.5
      irr < -15%           → pe_score_adjusted = 0
      equity bloated (>90% EV) → -10pts

    PE context: Additive penalties are insufficient for fundamentally broken
    deals — a company with a 90+ raw score could still rank near the top after
    a -20pt additive penalty. Multiplicative / zero penalties ensure broken
    LBO math collapses the score regardless of quality metrics. Think of Step 1
    as adjusting for imperfection; Step 2 as flagging deal-breakers.
    """
    df = df.copy()

    # Step 1: additive penalties — red flags and valuation
    penalty_cols = []
    for col in ("red_flag_penalty", "valuation_penalty"):
        if col in df.columns:
            penalty_cols.append(df[col].fillna(0))

    total_penalty = sum(penalty_cols) if penalty_cols else 0
    df["pe_score_adjusted"] = (df["pe_score"] + total_penalty).clip(0, 100).round(2)

    # Step 2: deal killer — modifies pe_score_adjusted in place, adds deal_killer_penalty
    df = apply_deal_killer_penalty(df)

    # Step 3: blend quality score with IRR signal for return-first ranking
    df = compute_irr_blended_score(df)

    return df


def compute_irr_blended_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blend the penalty-adjusted quality score with IRR signal for a return-first ranking.

    PE context: Return potential should drive ranking, not just business quality.
    A mediocre business at a cheap price (high IRR) beats a great business at a full
    price (low IRR) for LBO purposes. The blended score weights IRR at 40%, ensuring
    companies that don't pencil financially cannot rank above those that do.

    Formula:
      irr_score   = percentile rank of irr_base (0–100), neutral (50) for NaN
      pe_score_final = 0.60 × pe_score_adjusted + 0.40 × irr_score

    pe_score_adjusted already incorporates deal killer penalties (halve/zero for
    negative IRR), so pe_score_final double-penalizes broken deals — intentionally.
    """
    if "irr_base" not in df.columns:
        df["irr_score"] = np.nan
        df["pe_score_final"] = df["pe_score_adjusted"]
        return df

    irr_ranked = df["irr_base"].rank(method="average", na_option="keep", pct=True) * 100
    df["irr_score"] = irr_ranked.fillna(50)

    df["pe_score_final"] = (
        0.60 * df["pe_score_adjusted"].fillna(50) +
        0.40 * df["irr_score"]
    ).clip(0, 100).round(2)

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
