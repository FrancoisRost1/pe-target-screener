"""
changelog_display.py — Rich console display for the daily changelog.
"""

import pandas as pd


def print_changelog_summary(changelog: pd.DataFrame):
    """Print a human-readable summary of the changelog."""
    if changelog is None or changelog.empty:
        return

    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold]Daily Changelog[/bold]")

    new = changelog[changelog["change_type"] == "NEW"]
    dropped = changelog[changelog["change_type"] == "DROPPED"]
    movers = changelog[changelog["change_type"].isin(["UP", "DOWN"])]

    if len(new):
        console.print(f"  [green]+ {len(new)} new companies entered[/green]")
    if len(dropped):
        console.print(f"  [red]- {len(dropped)} companies dropped out[/red]")

    if len(movers):
        table = Table(show_header=True, header_style="bold cyan", title="Biggest Movers")
        table.add_column("Ticker")
        table.add_column("Company")
        table.add_column("Rank Change")
        table.add_column("Score Change")
        table.add_column("Current Rank")

        up = movers[movers["rank_change"] > 0].nlargest(5, "rank_change")
        down = movers[movers["rank_change"] < 0].nsmallest(5, "rank_change")
        show = pd.concat([up, down])

        for _, r in show.iterrows():
            rc = r["rank_change"]
            arrow = f"[green]▲ {rc}[/green]" if rc > 0 else f"[red]▼ {abs(rc)}[/red]"
            sc = r["score_change"]
            sc_str = f"+{sc}" if sc > 0 else str(sc)
            table.add_row(
                str(r["ticker"]), str(r["company"]),
                arrow, sc_str, str(r.get("rank_today", ""))
            )
        console.print(table)
