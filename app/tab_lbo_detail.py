"""
tab_lbo_detail.py | LBO deal breakdown expander with scenario IRR and bridge charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app.helpers import fmt_irr, fmt_irr_delta, fmt_mult, fmt_millions
from style_inject import apply_plotly_theme, styled_kpi, styled_divider, styled_section_label, TOKENS


def render_lbo_expander(row: pd.Series, cfg: dict = None):
    """Render the LBO deal breakdown expander with scenario IRR and bridge charts."""
    if cfg is None:
        cfg = {}
    lbo_cfg = cfg.get("lbo", {})
    holding_period = lbo_cfg.get("holding_period", 5)
    exit_multiple_cap = lbo_cfg.get("exit_multiple", 12.0)
    debt_repayment_rate = lbo_cfg.get("debt_repayment_rate", 0.4)

    ev_val = row.get("enterprise_value")
    ebitda_val = row.get("ebitda")
    max_debt_val = row.get("max_debt")
    eq_req_val = row.get("equity_required")

    if not all(v is not None and not pd.isna(v) for v in [ev_val, ebitda_val, max_debt_val, eq_req_val]):
        return
    if ev_val <= 0:
        return

    growth = row.get("revenue_growth") or 0.0
    if pd.isna(growth):
        growth = 0.0
    growth_clipped = max(-0.05, min(0.15, growth))
    exit_ebitda = max(ebitda_val, 0) * ((1 + growth_clipped) ** holding_period)

    entry_ev_ebitda = row.get("ev_to_ebitda") or exit_multiple_cap
    if pd.isna(entry_ev_ebitda):
        entry_ev_ebitda = exit_multiple_cap
    exit_mult = min(entry_ev_ebitda, exit_multiple_cap)

    exit_ev = exit_ebitda * exit_mult
    fcf_val = row.get("free_cash_flow") or 0.0
    if pd.isna(fcf_val):
        fcf_val = 0.0
    debt_repaid = max(fcf_val, 0) * holding_period * debt_repayment_rate
    debt_remaining = max(max_debt_val - debt_repaid, 0)
    exit_equity = exit_ev - debt_remaining
    moic = exit_equity / eq_req_val if eq_req_val > 0 else float("nan")

    with st.expander("LBO Deal Breakdown"):
        lev_exp = lbo_cfg.get("target_leverage", 3.5)
        st.markdown(f"""
**Base Case Assumptions**
| Parameter | Value |
|---|---|
| Holding period | {int(holding_period)} years |
| Entry multiple | {fmt_mult(row.get("ev_to_ebitda"))} EV/EBITDA |
| Exit multiple (cap) | {exit_multiple_cap:.0f}x EV/EBITDA |
| Target leverage | {lev_exp:.1f}x EBITDA |
| FCF to debt repayment | {debt_repayment_rate:.0%} annually |
""")
        st.divider()
        b1, b2, b3 = st.columns(3)
        with b1:
            styled_kpi("ENTRY EV", fmt_millions(ev_val))
            styled_kpi("MAX DEBT", fmt_millions(max_debt_val))
            styled_kpi("EQUITY CHEQUE", fmt_millions(eq_req_val))
        with b2:
            styled_kpi("EXIT EBITDA", fmt_millions(exit_ebitda))
            styled_kpi("EXIT MULTIPLE", f"{exit_mult:.1f}x")
            styled_kpi("EXIT EV", fmt_millions(exit_ev))
        with b3:
            styled_kpi("DEBT REPAID", fmt_millions(debt_repaid))
            styled_kpi("DEBT REMAINING", fmt_millions(debt_remaining))
            styled_kpi("EXIT EQUITY", fmt_millions(exit_equity))
        styled_divider()

        moic_col, irr_col = st.columns(2)
        with moic_col:
            styled_kpi("MOIC", f"{moic:.2f}x" if not pd.isna(moic) else "n/a")
        with irr_col:
            styled_kpi("BASE IRR", fmt_irr(row.get("irr_base", row.get("irr_proxy"))))

        _render_scenario_irr(row)
        _render_irr_bridge(row)


def _render_scenario_irr(row: pd.Series):
    """Render the 3-scenario IRR view with bar chart."""
    styled_section_label("IRR SCENARIOS")
    irr_base_val = row.get("irr_base", row.get("irr_proxy"))
    irr_up_val = row.get("irr_upside")
    irr_dn_val = row.get("irr_downside")

    s1, s2, s3 = st.columns(3)
    with s1:
        delta = fmt_irr_delta(irr_dn_val, irr_base_val)
        styled_kpi("DOWNSIDE", fmt_irr(irr_dn_val), delta=delta or "", delta_color=TOKENS["accent_danger"])
    with s2:
        styled_kpi("BASE", fmt_irr(irr_base_val))
    with s3:
        delta = fmt_irr_delta(irr_up_val, irr_base_val)
        styled_kpi("UPSIDE", fmt_irr(irr_up_val), delta=delta or "", delta_color=TOKENS["accent_success"])

    scenario_vals = {"Downside": irr_dn_val, "Base": irr_base_val, "Upside": irr_up_val}
    valid = {k: v for k, v in scenario_vals.items() if v is not None and not pd.isna(v)}
    if not valid:
        return

    bar_colors = {
        "Downside": TOKENS["accent_danger"],
        "Base": TOKENS["accent_primary"],
        "Upside": TOKENS["accent_success"],
    }
    fig = go.Figure()
    for label, irr_val in valid.items():
        fig.add_trace(go.Bar(x=[irr_val * 100], y=[label], orientation="h",
                             marker_color=bar_colors.get(label, TOKENS["accent_secondary"]), name=label,
                             text=[f"{irr_val:.1%}"], textposition="outside", showlegend=False))
    fig.add_vline(x=20, line_dash="dash", line_color=TOKENS["text_muted"],
                  annotation_text="20% hurdle", annotation_position="top right",
                  annotation_font_size=10)
    x_min = min(min(v * 100 for v in valid.values()) - 5, -5)
    x_max = max(max(v * 100 for v in valid.values()) + 10, 30)
    fig.update_layout(
        height=240,
        xaxis=dict(title="IRR (%)", range=[x_min, x_max]),
        yaxis=dict(title=""),
        bargap=0.3,
        title="IRR by Scenario",
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, width="stretch")


def _render_irr_bridge(row: pd.Series):
    """Render the IRR decomposition bar chart."""
    growth_drv = row.get("irr_driver_growth")
    delev_drv = row.get("irr_driver_deleveraging")
    mult_drv = row.get("irr_driver_multiple")

    if not all(v is not None and not pd.isna(v) for v in [growth_drv, delev_drv, mult_drv]):
        return
    if not any(abs(v) > 0.001 for v in [growth_drv, delev_drv, mult_drv]):
        return

    styled_section_label("IRR DECOMPOSITION")
    labels = ["EBITDA Growth", "Debt Paydown", "Multiple Delta"]
    vals = [growth_drv * 100, delev_drv * 100, mult_drv * 100]
    colors = [TOKENS["accent_success"] if v >= 0 else TOKENS["accent_danger"] for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=vals, marker_color=colors,
                         text=[f"{v:+.1f}%" for v in vals], textposition="outside", showlegend=False))
    fig.add_hline(y=0, line_dash="solid", line_color=TOKENS["text_muted"], line_width=1)
    y_abs_max = max(abs(v) for v in vals) * 1.4 or 5
    irr_base_val = row.get("irr_base", row.get("irr_proxy"))
    fig.update_layout(
        height=260,
        yaxis=dict(title="IRR contribution (%)", range=[-y_abs_max, y_abs_max]),
        xaxis=dict(title=""),
        title=f"Base IRR: {fmt_irr(irr_base_val)} by driver",
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, width="stretch")
