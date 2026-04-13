"""
summary.py, Auto-generated investment memo snippets

Generates a structured Bull / Bear / Risk memo for each top candidate.
No AI needed, pure rule-based logic that reads like an analyst wrote it.

Format mirrors a real IC one-pager: BULL / BEAR / KEY RISK.
"""

import pandas as pd
import numpy as np


def generate_memo(row: pd.Series) -> str:
    """Generate a structured Bull / Bear / Key Risk memo per company."""
    company = row.get("company", row.get("ticker", "This company"))
    dc = row.get("debt_capacity", "Medium")
    score = row.get("pe_score_adjusted", row.get("pe_score", float("nan")))
    rank = row.get("rank", "N/A")
    irr = row.get("irr_proxy")

    bull, bear = [], []

    em = row.get("ebitda_margin")
    if em is not None and not pd.isna(em):
        if em > 0.30: bull.append(f"exceptional EBITDA margin ({em:.1%})")
        elif em > 0.20: bull.append(f"strong EBITDA margin ({em:.1%})")
        elif em > 0.12: bull.append(f"solid EBITDA margin ({em:.1%})")
        else: bear.append(f"thin EBITDA margin ({em:.1%})")

    fcf = row.get("fcf_conversion")
    if fcf is not None and not pd.isna(fcf):
        if fcf > 0.75: bull.append(f"high FCF conversion ({fcf:.0%})")
        elif fcf > 0.55: bull.append(f"good FCF conversion ({fcf:.0%})")
        elif fcf < 0.35: bear.append(f"weak FCF conversion ({fcf:.0%})")

    nd = row.get("net_debt_to_ebitda")
    if nd is not None and not pd.isna(nd):
        if nd < 1.0: bull.append(f"minimal existing leverage ({nd:.1f}x)")
        elif nd < 2.0: bull.append(f"low leverage ({nd:.1f}x)")
        elif nd > 3.5: bear.append(f"elevated leverage ({nd:.1f}x)")

    ic = row.get("interest_coverage")
    if ic is not None and not pd.isna(ic):
        if ic > 8: bull.append(f"strong interest coverage ({ic:.1f}x)")
        elif ic < 3.5: bear.append(f"thin interest coverage ({ic:.1f}x)")

    roic = row.get("roic")
    if roic is not None and not pd.isna(roic):
        if roic > 0.20: bull.append(f"high ROIC ({roic:.1%})")
        elif roic < 0.07: bear.append(f"low ROIC ({roic:.1%})")

    ev_ebitda = row.get("ev_to_ebitda")
    if ev_ebitda is not None and not pd.isna(ev_ebitda):
        if ev_ebitda < 9: bull.append(f"attractive entry multiple ({ev_ebitda:.1f}x)")
        elif ev_ebitda < 12: bull.append(f"LBO-friendly valuation ({ev_ebitda:.1f}x)")
        elif ev_ebitda > 16: bear.append(f"premium valuation ({ev_ebitda:.1f}x)")

    rev_g = row.get("revenue_growth")
    if rev_g is not None and not pd.isna(rev_g):
        if rev_g > 0.10: bull.append(f"strong revenue growth ({rev_g:.1%} YoY)")
        elif rev_g > 0.04: bull.append(f"steady revenue growth ({rev_g:.1%} YoY)")
        elif rev_g < -0.02: bear.append(f"declining revenue ({rev_g:.1%} YoY)")

    if irr is not None and not pd.isna(irr):
        if irr > 0.20: bull.append(f"IRR proxy ~{irr:.0%}, deal math works")
        elif irr < 0.10: bear.append(f"IRR proxy ~{irr:.0%}, returns look stretched")

    if not bull: bull.append("defensive business model with stable cash generation")
    if not bear:
        if ev_ebitda is not None and not pd.isna(ev_ebitda) and ev_ebitda > 10:
            bear.append(f"valuation at {ev_ebitda:.1f}x limits return upside")
        elif rev_g is not None and not pd.isna(rev_g) and rev_g < 0.05:
            bear.append("limited organic growth constrains deleveraging pace")
        else: bear.append("limited public float may complicate exit process")

    flags_str = row.get("red_flags", "")
    flags_list = [f.strip() for f in flags_str.split(" | ") if f.strip()] if flags_str else []

    if flags_list: key_risk = flags_list[0]
    elif ev_ebitda is not None and not pd.isna(ev_ebitda) and ev_ebitda > 14:
        key_risk = "Premium valuation compresses returns"
    elif rev_g is not None and not pd.isna(rev_g) and rev_g < 0.03:
        key_risk = "Limited organic growth constrains deleveraging"
    else: key_risk = "Execution risk on operational improvement plan"

    score_str = f"{score:.0f}" if isinstance(score, float) and not pd.isna(score) else "N/A"
    irr_str = f"~{irr:.0%}" if irr is not None and not pd.isna(irr) else "N/A"

    header = f"#{rank}, {company} (Score: {score_str}/100 | Debt Capacity: {dc} | IRR: {irr_str})"
    bull_text = "; ".join(bull[:3]) + "."
    bear_text = "; ".join(bear[:2]) + "."

    irr_b = row.get("irr_base")
    growth_d, delev_d, mult_d = row.get("irr_driver_growth"), row.get("irr_driver_deleveraging"), row.get("irr_driver_multiple")
    drivers_line = ""
    if all(v is not None and not pd.isna(v) for v in [irr_b, growth_d, delev_d, mult_d]):
        drivers_line = (f"RETURN DRIVERS: Base IRR ~{irr_b:.0%} driven by "
                        f"EBITDA growth ({growth_d:+.0%}), debt paydown ({delev_d:+.0%}), multiple ({mult_d:+.0%}).")

    lines = [header, "", f"BULL CASE: {bull_text}", f"BEAR CASE: {bear_text}", f"KEY RISK:  {key_risk}."]
    if drivers_line: lines.append(drivers_line)
    return "\n".join(lines)


def add_memos(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Add structured investment memo column to top N companies."""
    df = df.copy()
    df["investment_memo"] = ""
    top_mask = df["rank"] <= top_n
    df.loc[top_mask, "investment_memo"] = df[top_mask].apply(generate_memo, axis=1)
    return df
