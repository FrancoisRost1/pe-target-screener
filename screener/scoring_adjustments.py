"""
scoring_adjustments.py, Score penalty adjustments, IRR blending, and hurdle logic

Produces pe_score_adjusted and pe_score_final from the raw pe_score
by applying additive penalties, deal-killer penalties, IRR blending,
and hurdle-rate enforcement.
"""

import pandas as pd
import numpy as np
import logging
from screener.classifier_rules import apply_deal_killer_penalty

logger = logging.getLogger(__name__)


def apply_score_adjustments(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Apply all score penalties to produce pe_score_adjusted, then pe_score_final.

    Step 1, Additive penalties (red flags, expensive valuation)
    Step 2, Deal killer (multiplicative for broken LBO math)
    Step 3, IRR blending (quality score + IRR signal)
    Step 4, IRR hurdle penalty
    """
    if cfg is None:
        cfg = {}

    df = df.copy()

    penalty_cols = []
    for col in ("red_flag_penalty", "valuation_penalty"):
        if col in df.columns:
            penalty_cols.append(df[col].fillna(0))

    total_penalty = sum(penalty_cols) if penalty_cols else 0
    df["pe_score_adjusted"] = (df["pe_score"] + total_penalty).clip(0, 100).round(2)

    df = apply_deal_killer_penalty(df)
    df = compute_irr_blended_score(df, cfg)
    df = apply_irr_hurdle_penalty(df, cfg)

    return df


def compute_irr_blended_score(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Blend quality score with IRR signal for return-first ranking.

    Formula: pe_score_final = quality_w × pe_score_adjusted + irr_w × irr_score
    Weights loaded from config.yaml scoring section.
    """
    if cfg is None:
        cfg = {}
    scoring_cfg = cfg.get("scoring", {})
    quality_w = scoring_cfg.get("irr_blend_quality", 0.60)
    irr_w = scoring_cfg.get("irr_blend_irr", 0.40)

    if "irr_base" not in df.columns:
        df["irr_score"] = np.nan
        df["pe_score_final"] = df["pe_score_adjusted"]
        return df

    irr_ranked = df["irr_base"].rank(method="average", na_option="keep", pct=True) * 100
    df["irr_score"] = irr_ranked.fillna(50)

    df["pe_score_final"] = (
        quality_w * df["pe_score_adjusted"].fillna(50) +
        irr_w * df["irr_score"]
    ).clip(0, 100).round(2)

    return df


def apply_irr_hurdle_penalty(df: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Apply IRR hurdle penalty to pe_score_final.

    Logic:
      irr_base < 0%           → pe_score_final = 0
      irr_base < hurdle_rate  → pe_score_final × hurdle_penalty_mult
      irr_base >= hurdle_rate → no change
    """
    if "pe_score_final" not in df.columns or "irr_base" not in df.columns:
        return df

    if cfg is None:
        cfg = {}
    scoring_cfg = cfg.get("scoring", {})
    hurdle_rate = scoring_cfg.get("hurdle_rate", 0.10)
    hurdle_mult = scoring_cfg.get("hurdle_penalty_mult", 0.40)

    df = df.copy()
    irr = df["irr_base"]

    if "red_flags" not in df.columns:
        df["red_flags"] = ""

    zero_mask = irr.notna() & (irr < 0)
    df.loc[zero_mask, "pe_score_final"] = 0.0

    hurdle_mask = irr.notna() & (irr >= 0) & (irr < hurdle_rate)
    df.loc[hurdle_mask, "pe_score_final"] = (
        df.loc[hurdle_mask, "pe_score_final"] * hurdle_mult
    ).round(2)

    below_hurdle = irr.notna() & (irr < hurdle_rate)
    for idx in df.index[below_hurdle]:
        existing = df.at[idx, "red_flags"]
        if not existing or (isinstance(existing, float) and pd.isna(existing)):
            df.at[idx, "red_flags"] = "IRR below hurdle"
        elif "IRR below hurdle" not in str(existing):
            df.at[idx, "red_flags"] = str(existing) + " | IRR below hurdle"

    penalized = below_hurdle.sum()
    logger.info(f"IRR hurdle penalty: {penalized} companies below {hurdle_rate:.0%} hurdle")
    return df
