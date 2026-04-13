"""
tab_table.py | KPI cards and top targets table rendering.
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.helpers import fmt_pct, fmt_irr, fmt_mult, fmt_score, debt_capacity_color
from style_inject import styled_kpi, styled_section_label, TOKENS


def render_kpis(df, df_filtered, score_col, run_cfg):
    """Render 5 KPI cards."""
    styled_section_label("KPIS")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        styled_kpi("SCREENED", f"{len(df)}")
    with c2:
        styled_kpi("FILTERED", f"{len(df_filtered)}")
    with c3:
        avg_score = f"{df_filtered[score_col].mean():.1f}" if len(df_filtered) else "n/a"
        styled_kpi("AVG SCORE", avg_score)
    with c4:
        styled_kpi("HIGH DEBT CAP", f"{int((df_filtered['debt_capacity'] == 'High').sum())}")

    irr_col = "irr_base" if "irr_base" in df_filtered.columns else "irr_proxy"
    irr_median = df_filtered[irr_col].median() if irr_col in df_filtered.columns else None
    with c5:
        irr_str = fmt_irr(irr_median) if irr_median is not None and not np.isnan(irr_median) else "n/a"
        styled_kpi("MEDIAN IRR", irr_str)

    lbo_cfg = run_cfg.get("lbo", {})
    if irr_median is not None and not np.isnan(irr_median) and irr_median < 0.12:
        st.info(
            f"Market context. Under current assumptions "
            f"({int(lbo_cfg.get('holding_period', 5))}yr hold, "
            f"{lbo_cfg.get('exit_multiple', 10.0):.0f}x exit, "
            f"{lbo_cfg.get('target_leverage', 4.0):.1f}x leverage), "
            f"most public comps do not meet typical PE return thresholds (20%+ IRR). "
            f"Median base IRR: {fmt_irr(irr_median)}. "
            f"Adjust LBO assumptions in the sidebar to stress-test scenarios."
        )


def render_top_table(df_top, top_n, score_col, run_cfg):
    """Render the top targets table with formatted columns."""
    styled_section_label(f"TOP {top_n}")

    TABLE_COLS = [
        "rank", "ticker", "company", "sector",
        "ebitda_margin", "fcf_conversion", "net_debt_to_ebitda",
        "ev_to_ebitda", "irr_base", "irr_downside", "debt_capacity", "pe_score_final",
    ]
    display_df = df_top[[c for c in TABLE_COLS if c in df_top.columns]].copy()

    for col in ["ebitda_margin", "fcf_conversion"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_pct)
    for col in ["net_debt_to_ebitda", "ev_to_ebitda"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_mult)
    for col in ["irr_base", "irr_downside"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_irr)
    if "pe_score_final" in display_df.columns:
        display_df["pe_score_final"] = display_df["pe_score_final"].apply(fmt_score)
    if "debt_capacity" in display_df.columns:
        display_df["debt_capacity"] = display_df["debt_capacity"].apply(
            lambda v: f"{debt_capacity_color(v)} {v}")

    col_rename = {c: c.replace("_", " ").title() for c in display_df.columns}
    col_rename["irr_base"] = "IRR (Base)"
    col_rename["irr_downside"] = "IRR (Down)"
    col_rename["pe_score_final"] = "Final Score"
    display_df.rename(columns=col_rename, inplace=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    lbo_cfg = run_cfg.get("lbo", {})
    st.caption(
        f"IRR (Base): {int(lbo_cfg.get('holding_period', 5))}yr hold, "
        f"{lbo_cfg.get('exit_multiple', 10.0):.0f}x exit cap. "
        f"IRR (Down): exit cap minus 1x, growth minus 3%. "
        "Annual amortization at 40% FCF. Simplified model, indicative only."
    )

    csv = df_top.to_csv(index=False).encode("utf-8")
    st.download_button("Download Top Targets CSV", csv, "top_targets.csv", "text/csv")
