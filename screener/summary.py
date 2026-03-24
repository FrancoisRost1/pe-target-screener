"""
summary.py — Auto-generated investment memo snippets

Generates a plain-English investment summary for each top candidate.
No AI needed — pure rule-based logic that reads like an analyst wrote it.

PE context: A screening memo doesn't just list numbers.
It tells the story of why a company is interesting or not.
"""

import pandas as pd
import numpy as np


def generate_memo(row: pd.Series) -> str:
    """
    Generate a concise investment summary for a single company.
    Highlights strengths, flags watchpoints, and states debt capacity.
    """
    company = row.get("company", row.get("ticker", "This company"))
    strengths = []
    watchpoints = []

    # Profitability
    em = row.get("ebitda_margin")
    if em and not pd.isna(em):
        if em > 0.25:
            strengths.append(f"strong EBITDA margin ({em:.1%})")
        elif em > 0.15:
            strengths.append(f"solid EBITDA margin ({em:.1%})")
        else:
            watchpoints.append(f"thin EBITDA margin ({em:.1%})")

    # Cash conversion
    fcf = row.get("fcf_conversion")
    if fcf and not pd.isna(fcf):
        if fcf > 0.75:
            strengths.append(f"high FCF conversion ({fcf:.0%})")
        elif fcf < 0.40:
            watchpoints.append(f"weak FCF conversion ({fcf:.0%})")

    # Leverage
    nd_ebitda = row.get("net_debt_to_ebitda")
    if nd_ebitda is not None and not pd.isna(nd_ebitda):
        if nd_ebitda < 1.5:
            strengths.append(f"low existing leverage ({nd_ebitda:.1f}x Net Debt/EBITDA)")
        elif nd_ebitda > 3.5:
            watchpoints.append(f"elevated leverage ({nd_ebitda:.1f}x Net Debt/EBITDA)")

    # Interest coverage
    ic = row.get("interest_coverage")
    if ic and not pd.isna(ic):
        if ic > 6:
            strengths.append(f"strong interest coverage ({ic:.1f}x)")
        elif ic < 3:
            watchpoints.append(f"limited interest coverage ({ic:.1f}x)")

    # ROIC
    roic = row.get("roic")
    if roic and not pd.isna(roic):
        if roic > 0.18:
            strengths.append(f"high ROIC ({roic:.1%})")
        elif roic < 0.08:
            watchpoints.append(f"low ROIC ({roic:.1%})")

    # Valuation
    ev_ebitda = row.get("ev_to_ebitda")
    if ev_ebitda and not pd.isna(ev_ebitda):
        if ev_ebitda < 10:
            strengths.append(f"attractive entry multiple ({ev_ebitda:.1f}x EV/EBITDA)")
        elif ev_ebitda > 18:
            watchpoints.append(f"premium valuation ({ev_ebitda:.1f}x EV/EBITDA)")

    # Growth
    rev_growth = row.get("revenue_growth")
    if rev_growth is not None and not pd.isna(rev_growth):
        if rev_growth < -0.03:
            watchpoints.append(f"declining revenue ({rev_growth:.1%} YoY)")
        elif rev_growth > 0.10:
            strengths.append(f"strong revenue growth ({rev_growth:.1%} YoY)")

    # Compose memo
    dc = row.get("debt_capacity", "Medium")
    score = row.get("pe_score", "N/A")
    rank = row.get("rank", "N/A")

    parts = [f"#{rank} — {company} (PE Score: {score:.1f}/100, Debt Capacity: {dc})."]

    if strengths:
        parts.append("Key strengths: " + ", ".join(strengths) + ".")
    if watchpoints:
        parts.append("Main watchpoints: " + ", ".join(watchpoints) + ".")

    flags = row.get("red_flags", "")
    if flags:
        parts.append(f"Red flags: {flags}.")

    return " ".join(parts)


def add_memos(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Add investment memo column to top N companies."""
    df = df.copy()
    df["investment_memo"] = ""
    top_mask = df["rank"] <= top_n
    df.loc[top_mask, "investment_memo"] = df[top_mask].apply(generate_memo, axis=1)
    return df
