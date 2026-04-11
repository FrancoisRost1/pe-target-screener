"""
tab_detail.py — Company drill-down detail view for the Streamlit dashboard.

Renders metric cards, score decomposition, investment memo, radar chart,
and penalty breakdown. LBO expander is in tab_lbo_detail.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app.helpers import (
    fmt_pct, fmt_irr, fmt_mult, fmt_score,
    fmt_millions, fmt_fcf_yield_equity, debt_capacity_color,
)
from app.tab_lbo_detail import render_lbo_expander


def render_company_detail(row: pd.Series, cfg: dict = None):
    """Render the full company drill-down view."""
    col1, col2 = st.columns([2, 1])

    with col1:
        _render_metrics(row, cfg)
        _render_memo_and_flags(row)

    with col2:
        _render_radar(row)
        _render_penalties(row)


def _render_metrics(row: pd.Series, cfg: dict = None):
    """Render metric cards and LBO breakdown expander."""
    st.markdown(f"### {row.get('company', row.get('ticker'))}")
    st.caption(f"**Sector:** {row.get('sector', 'n/a')}  |  **Rank:** #{int(row.get('rank', 0))}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Score", fmt_score(row.get("pe_score_final", row.get("pe_score_adjusted", row.get("pe_score")))))
    m2.metric("EBITDA Margin", fmt_pct(row.get("ebitda_margin")))
    m3.metric("FCF Conversion", fmt_pct(row.get("fcf_conversion")))
    m4.metric("EV/EBITDA", fmt_mult(row.get("ev_to_ebitda")))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Base IRR", fmt_irr(row.get("irr_base", row.get("irr_proxy"))))
    m6.metric("Max Debt", fmt_millions(row.get("max_debt")))
    m7.metric("Equity Required", fmt_millions(row.get("equity_required")))
    m8.metric("FCF Yield / Equity", fmt_fcf_yield_equity(row.get("fcf_yield_equity")))

    raw = row.get("pe_score_raw", row.get("pe_score"))
    adj = row.get("pe_score_adjusted")
    final = row.get("pe_score_final")
    if raw is not None and adj is not None and final is not None \
            and not pd.isna(raw) and not pd.isna(adj) and not pd.isna(final):
        st.caption(
            f"Raw: {raw:.1f} | Adjusted: {adj:.1f} | Final (IRR-blended): {final:.1f} "
            f"(delta {final - raw:+.1f} vs raw)"
        )

    render_lbo_expander(row, cfg)


def _render_memo_and_flags(row: pd.Series):
    """Render investment memo and red flags."""
    memo = row.get("investment_memo", "")
    if memo:
        st.markdown("**Investment Memo**")
        st.markdown(memo.replace("\n", "  \n"))
    flags = row.get("red_flags", "")
    if flags and not (isinstance(flags, float) and np.isnan(flags)):
        st.warning(f"**Red Flags:** {flags}")


def _render_radar(row: pd.Series):
    """Render radar chart of sub-scores."""
    sub_scores = {"Quality": row.get("quality_score"), "Cash Flow": row.get("cash_score"),
                  "Leverage": row.get("leverage_score"), "Valuation": row.get("valuation_score")}
    valid = {k: v for k, v in sub_scores.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    if not valid:
        return
    categories = list(valid.keys())
    values = list(valid.values())
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(76, 120, 168, 0.3)", line_color="#4C78A8"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      showlegend=False, height=280, margin=dict(t=30, b=20), title="Sub-Score Radar")
    st.plotly_chart(fig, use_container_width=True)


def _render_penalties(row: pd.Series):
    """Render debt capacity badge and penalty breakdown."""
    dc = row.get("debt_capacity", "n/a")
    st.metric("Debt Capacity", f"{debt_capacity_color(dc)} {dc}")
    rf_pen = row.get("red_flag_penalty", 0) or 0
    val_pen = row.get("valuation_penalty", 0) or 0
    dk_pen = row.get("deal_killer_penalty", 0) or 0
    parts = []
    if rf_pen: parts.append(f"Red flags {rf_pen:+.0f}")
    if val_pen: parts.append(f"Valuation {val_pen:+.0f}")
    if dk_pen: parts.append(f"Deal killer {dk_pen:+.0f}")
    if parts:
        st.caption("Score penalties: " + " | ".join(parts))
