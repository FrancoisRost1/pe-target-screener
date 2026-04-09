"""
changelog.py — Daily score comparison and change tracking

Compares today's scored output to the most recent previous run.
Display logic is in changelog_display.py.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Re-export display function
from screener.changelog_display import print_changelog_summary  # noqa: E402, F401


def generate_changelog(df_today: pd.DataFrame, cfg: dict) -> Optional[pd.DataFrame]:
    """
    Compare today's scored DataFrame to the most recent historical snapshot.
    Returns a changelog DataFrame, or None if no prior data exists.
    """
    history_dir = Path(cfg.get("output", {}).get("output_dir", "outputs")) / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_path = history_dir / f"scored_{today_str}.csv"

    cols_to_save = [
        "ticker", "company", "sector", "rank", "pe_score_final",
        "pe_score_adjusted", "irr_base", "debt_capacity", "red_flags"
    ]
    available = [c for c in cols_to_save if c in df_today.columns]
    df_today[available].to_csv(today_path, index=False)
    logger.info(f"Saved daily snapshot: {today_path}")

    prior = _find_prior_snapshot(history_dir, today_str)
    if prior is None:
        logger.info("No prior snapshot found — skipping changelog (first run)")
        return None

    df_prior = pd.read_csv(prior)
    logger.info(f"Comparing against prior snapshot: {prior.name}")

    changelog = _build_changelog(df_today, df_prior)
    changelog_path = history_dir / f"changelog_{today_str}.csv"
    changelog.to_csv(changelog_path, index=False)
    logger.info(f"Changelog saved: {changelog_path} ({len(changelog)} entries)")
    return changelog


def _find_prior_snapshot(history_dir: Path, today_str: str) -> Optional[Path]:
    """Find the most recent snapshot file before today."""
    snapshots = sorted(history_dir.glob("scored_*.csv"))
    prior_files = [f for f in snapshots if f.stem != f"scored_{today_str}"]
    return prior_files[-1] if prior_files else None


def _build_changelog(df_today: pd.DataFrame, df_prior: pd.DataFrame) -> pd.DataFrame:
    """Compare two scored DataFrames and produce a changelog."""
    score_col = "pe_score_final" if "pe_score_final" in df_today.columns else "pe_score_adjusted"

    today = df_today[["ticker", "company", "rank", score_col]].copy()
    today.columns = ["ticker", "company", "rank_today", "score_today"]

    prior_score_col = score_col if score_col in df_prior.columns else "pe_score_adjusted"
    prior = df_prior[["ticker", "rank", prior_score_col]].copy()
    prior.columns = ["ticker", "rank_prior", "score_prior"]

    merged = today.merge(prior, on="ticker", how="outer")

    rows = []
    for _, r in merged.iterrows():
        ticker = r["ticker"]
        company = r.get("company", "")

        if pd.isna(r.get("rank_prior")):
            rows.append({"ticker": ticker, "company": company, "change_type": "NEW",
                         "rank_today": int(r["rank_today"]) if pd.notna(r.get("rank_today")) else None,
                         "rank_prior": None, "rank_change": None,
                         "score_today": round(r["score_today"], 1) if pd.notna(r.get("score_today")) else None,
                         "score_prior": None, "score_change": None})
        elif pd.isna(r.get("rank_today")):
            rows.append({"ticker": ticker, "company": company, "change_type": "DROPPED",
                         "rank_today": None, "rank_prior": int(r["rank_prior"]),
                         "rank_change": None, "score_today": None,
                         "score_prior": round(r["score_prior"], 1) if pd.notna(r.get("score_prior")) else None,
                         "score_change": None})
        else:
            rank_change = int(r["rank_prior"] - r["rank_today"])
            score_change = round(r["score_today"] - r["score_prior"], 1)
            ctype = "UP" if rank_change > 0 else "DOWN" if rank_change < 0 else "UNCHANGED"
            rows.append({"ticker": ticker, "company": company, "change_type": ctype,
                         "rank_today": int(r["rank_today"]), "rank_prior": int(r["rank_prior"]),
                         "rank_change": rank_change, "score_today": round(r["score_today"], 1),
                         "score_prior": round(r["score_prior"], 1), "score_change": score_change})

    changelog = pd.DataFrame(rows)
    type_order = {"NEW": 0, "DROPPED": 1, "UP": 2, "DOWN": 3, "UNCHANGED": 4}
    changelog["_sort"] = changelog["change_type"].map(type_order)
    changelog = changelog.sort_values(["_sort", "rank_today"], ascending=[True, True])
    return changelog.drop(columns=["_sort"]).reset_index(drop=True)
