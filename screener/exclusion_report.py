"""
exclusion_report.py, Document why companies were excluded from scoring

Captures every company that was filtered out and the specific reason,
so analysts can review whether exclusions are justified.

PE context: Knowing why a company was excluded is as important as
knowing why one was included. A company filtered for low margins today
might be worth watching if margins are trending up.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_exclusion_report(
    df_raw: pd.DataFrame,
    df_scored: pd.DataFrame,
    cfg: dict
) -> pd.DataFrame:
    """
    Compare the raw (pre-filter) DataFrame to the scored (post-filter) DataFrame.
    For every company in raw but not in scored, determine the exclusion reason.

    Returns a DataFrame with columns: ticker, company, sector, reason, details
    """
    # Get tickers that survived
    scored_tickers = set(df_scored["ticker"].unique())
    excluded = df_raw[~df_raw["ticker"].isin(scored_tickers)].copy()

    if excluded.empty:
        logger.info("No companies excluded, all passed filters")
        return pd.DataFrame(columns=["ticker", "company", "sector", "reason", "details"])

    rows = []
    e = cfg.get("eligibility", {})

    for _, row in excluded.iterrows():
        ticker = row.get("ticker", "")
        company = row.get("company", ticker)
        sector = row.get("sector", "Unknown")
        reasons = []

        # Check: missing from fetch (no income statement)
        if pd.isna(row.get("revenue")) and pd.isna(row.get("ebitda")):
            reasons.append(("No financial data", "yfinance returned no income statement"))

        # Check: size, revenue too low
        min_rev = e.get("min_revenue", 0)
        rev = row.get("revenue", 0)
        if pd.notna(rev) and rev < min_rev and min_rev > 0:
            reasons.append(("Below min revenue", f"Revenue: ${rev/1e6:,.0f}M (min: ${min_rev/1e6:,.0f}M)"))

        # Check: size, EBITDA too low
        min_ebitda = e.get("min_ebitda", 0)
        ebitda = row.get("ebitda", 0)
        if pd.notna(ebitda) and ebitda < min_ebitda and min_ebitda > 0:
            reasons.append(("Below min EBITDA", f"EBITDA: ${ebitda/1e6:,.0f}M (min: ${min_ebitda/1e6:,.0f}M)"))

        # Check: negative EBITDA
        if pd.notna(ebitda) and ebitda <= 0:
            reasons.append(("Negative EBITDA", f"EBITDA: ${ebitda/1e6:,.0f}M"))

        # Check: low EBITDA margin
        ebitda_margin = row.get("ebitda_margin")
        if pd.notna(ebitda_margin) and ebitda_margin < 0.08:
            reasons.append(("EBITDA margin < 8%", f"Margin: {ebitda_margin:.1%}"))

        # Check: low interest coverage
        int_cov = row.get("interest_coverage")
        if pd.notna(int_cov) and int_cov < 2.5:
            reasons.append(("Interest coverage < 2.5x", f"Coverage: {int_cov:.1f}x"))

        # Check: excluded sector
        if "exclude_sectors" in e and sector in e.get("exclude_sectors", []):
            reasons.append(("Excluded sector", f"Sector: {sector}"))

        # If no specific reason found, mark as unknown
        if not reasons:
            reasons.append(("Unknown", "Excluded but reason not matched, check pipeline logic"))

        for reason, details in reasons:
            rows.append({
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "reason": reason,
                "details": details,
            })

    report = pd.DataFrame(rows)
    report = report.sort_values(["reason", "ticker"]).reset_index(drop=True)

    # Save report
    output_dir = Path(cfg.get("output", {}).get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "exclusion_report.csv"
    report.to_csv(report_path, index=False)
    logger.info(f"Exclusion report saved: {report_path} ({len(report)} entries, {excluded.shape[0]} companies)")

    return report


def print_exclusion_summary(report: pd.DataFrame):
    """Print a human-readable summary of exclusions grouped by reason."""
    if report is None or report.empty:
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold]Exclusion Report[/bold]")

    # Summary by reason
    reason_counts = report["reason"].value_counts()
    for reason, count in reason_counts.items():
        console.print(f"  [yellow]• {reason}:[/yellow] {count} companies")

    # Detailed table
    table = Table(show_header=True, header_style="bold yellow", title="Excluded Companies")
    table.add_column("Ticker")
    table.add_column("Company")
    table.add_column("Sector")
    table.add_column("Reason")
    table.add_column("Details")

    for _, r in report.iterrows():
        table.add_row(
            str(r["ticker"]),
            str(r["company"]),
            str(r["sector"]),
            str(r["reason"]),
            str(r["details"]),
        )
    console.print(table)
