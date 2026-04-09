"""
main.py — Pipeline orchestrator

Runs the full PE screening pipeline:
load → clean → ratios → winsorize → lbo → eligibility → score → classify → rank → export

Usage:
    python main.py
    python main.py --no-fetch   (use existing raw data)
    python main.py --top 30     (change top-N)
"""

import argparse
import sys
import yaml
import logging
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table

from screener.loader import load_universe, fetch_universe
from screener.cleaner import clean, winsorize_ratios, apply_eligibility_filters
from screener.ratios import compute_all_ratios
from screener.lbo import compute_lbo_metrics
from screener.scoring import score_universe_sector_adjusted, compute_sub_scores, apply_score_adjustments
from screener.classifier import classify_all
from screener.ranking import rank_companies, get_top_n
from screener.exporter import export_results
from screener.summary import add_memos
from screener.changelog import generate_changelog, print_changelog_summary
from screener.exclusion_report import generate_exclusion_report, print_exclusion_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
console = Console()

_SCRIPT_DIR = Path(__file__).parent


def load_config(path: str = None) -> dict:
    if path is None:
        path = _SCRIPT_DIR / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(cfg: dict, fetch: bool = True, top_n: int = None):
    if top_n is None:
        top_n = cfg.get("output", {}).get("top_n", 20)

    # 1. Load or fetch data
    raw_path = cfg["data"]["raw_output"]
    if fetch:
        console.print("[bold cyan]Fetching data from yfinance...[/bold cyan]")
        universe = load_universe(cfg["data"]["universe_file"])
        df = fetch_universe(universe, save_path=raw_path)
    elif Path(raw_path).exists():
        console.print(f"[cyan]Loading cached data from {raw_path}[/cyan]")
        df = pd.read_csv(raw_path)
    else:
        console.print(f"[bold red]Error: --no-fetch specified but cache not found at {raw_path}[/bold red]")
        sys.exit(1)

    console.print(f"[green]Loaded: {len(df)} companies[/green]")

    # 2. Clean (type casting, dedup, quality flags)
    df = clean(df)

    # 3. Compute financial ratios
    df = compute_all_ratios(df, cfg)

    # 4. Winsorize (cap outliers BEFORE scoring and filters)
    df = winsorize_ratios(df)

    # 5. LBO return estimation (needs winsorized ratios)
    df = compute_lbo_metrics(df, cfg)

    # --- Save pre-filter snapshot for exclusion report ---
    df_pre_filter = df.copy()

    # 6. Eligibility filters (size + hard PE quality gates, run after ratios)
    df = apply_eligibility_filters(df, cfg)

    if df.empty:
        console.print("[bold red]No companies passed eligibility filters. Exiting.[/bold red]")
        return df, pd.DataFrame()

    # 7. Score — sector-adjusted percentile ranking
    df = score_universe_sector_adjusted(df, cfg)
    df = compute_sub_scores(df, cfg)

    # 8. Classify — debt capacity, red flags, score penalties
    df = classify_all(df, cfg)

    # 9. Apply penalty adjustments to produce pe_score_adjusted, then pe_score_final
    df = apply_score_adjustments(df, cfg)

    # 10. Rank by pe_score_final
    df = rank_companies(df)
    df = add_memos(df, top_n=top_n)
    top = get_top_n(df, n=top_n)

    # 11. Export
    export_results(df, top, cfg)

    # 12. Display
    _print_summary(top, top_n)

    # 13. Exclusion report — why companies were filtered out
    exclusion_report = generate_exclusion_report(df_pre_filter, df, cfg)
    print_exclusion_summary(exclusion_report)

    # 14. Changelog — compare to previous run
    changelog = generate_changelog(df, cfg)
    print_changelog_summary(changelog)

    console.print(f"\n[bold green]Done. Results saved to {cfg['output']['output_dir']}[/bold green]")
    return df, top


def _fmt_irr(v):
    return f"{v:.1%}" if not pd.isna(v) else "N/A"


def _print_summary(top, n):
    console.print(f"\n[bold]Top {n} PE Targets[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    cols_to_show = ["rank", "company", "sector", "pe_score_final", "irr_base", "irr_downside", "irr_upside", "debt_capacity"]
    present_cols = [col for col in cols_to_show if col in top.columns]
    for col in present_cols:
        table.add_column(col.replace("_", " ").title())
    for _, row in top.head(10).iterrows():
        final = row.get("pe_score_final", row.get("pe_score_adjusted", row.get("pe_score", 0)))
        values = {
            "rank": str(int(row.get("rank", 0))),
            "company": str(row.get("company", "")),
            "sector": str(row.get("sector", "")),
            "pe_score_final": f"{final:.1f}",
            "irr_base": _fmt_irr(row.get("irr_base")),
            "irr_downside": _fmt_irr(row.get("irr_downside")),
            "irr_upside": _fmt_irr(row.get("irr_upside")),
            "debt_capacity": str(row.get("debt_capacity", "")),
        }
        table.add_row(*[values[col] for col in present_cols])
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PE Target Screener")
    parser.add_argument("--no-fetch", action="store_true", help="Use cached raw data")
    parser.add_argument("--top", type=int, default=None, help="Number of top targets (must be positive)")
    parser.add_argument("--config", default=None, help="Config file path (default: config.yaml next to main.py)")
    args = parser.parse_args()

    if args.top is not None and args.top <= 0:
        parser.error("--top must be a positive integer")

    cfg = load_config(args.config)
    run(cfg, fetch=not args.no_fetch, top_n=args.top)
