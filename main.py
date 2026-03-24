"""
main.py — Pipeline orchestrator

Runs the full PE screening pipeline:
load → clean → ratios → score → classify → rank → export

Usage:
    python main.py
    python main.py --no-fetch   (use existing raw data)
    python main.py --top 30     (change top-N)
"""

import argparse
import yaml
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table

from screener.loader import load_universe, fetch_universe
from screener.cleaner import clean, apply_eligibility_filters
from screener.ratios import compute_all_ratios
from screener.scoring import score_universe, compute_sub_scores
from screener.classifier import classify_all
from screener.ranking import rank_companies, get_top_n
from screener.exporter import export_results
from screener.summary import add_memos

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run(cfg: dict, fetch: bool = True, top_n: int = None):
    top_n = top_n or cfg.get("output", {}).get("top_n", 20)

    # 1. Load or fetch data
    raw_path = cfg["data"]["raw_output"]
    if fetch or not Path(raw_path).exists():
        console.print("[bold cyan]Fetching data from yfinance...[/bold cyan]")
        universe = load_universe(cfg["data"]["universe_file"])
        df = fetch_universe(universe, save_path=raw_path)
    else:
        import pandas as pd
        console.print(f"[cyan]Loading cached data from {raw_path}[/cyan]")
        df = pd.read_csv(raw_path)

    console.print(f"[green]Loaded: {len(df)} companies[/green]")

    # 2. Clean
    df = clean(df)
    df = apply_eligibility_filters(df, cfg)

    # 3. Ratios
    df = compute_all_ratios(df, cfg)

    # 4. Score
    df = score_universe(df, cfg)
    df = compute_sub_scores(df, cfg)

    # 5. Classify
    df = classify_all(df, cfg)

    # 6. Rank
    df = rank_companies(df)
    df = add_memos(df, top_n=top_n)
    top = get_top_n(df, n=top_n)

    # 7. Export
    export_results(df, top, cfg)

    # 8. Display
    _print_summary(top, top_n)

    console.print(f"\n[bold green]Done. Results saved to {cfg['output']['output_dir']}[/bold green]")
    return df, top


def _print_summary(top, n):
    console.print(f"\n[bold]Top {n} PE Targets[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    for col in ["rank", "company", "sector", "pe_score", "debt_capacity"]:
        if col in top.columns:
            table.add_column(col.replace("_", " ").title())
    for _, row in top.head(10).iterrows():
        table.add_row(
            str(int(row.get("rank", 0))),
            str(row.get("company", "")),
            str(row.get("sector", "")),
            f"{row.get('pe_score', 0):.1f}",
            str(row.get("debt_capacity", "")),
        )
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PE Target Screener")
    parser.add_argument("--no-fetch", action="store_true", help="Use cached raw data")
    parser.add_argument("--top", type=int, default=None, help="Number of top targets")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, fetch=not args.no_fetch, top_n=args.top)
