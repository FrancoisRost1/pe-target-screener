"""
tab_charts.py — Chart sections for the PE Target Screener dashboard.

Renders: score distribution, debt capacity pie, deal quadrant,
and top opportunities sections.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app.helpers import fmt_irr, debt_capacity_color


def render_charts(df_filtered: pd.DataFrame, df_top: pd.DataFrame,
                  score_col: str):
    """Render the charts section of the dashboard."""
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Score Distribution")
        fig = px.histogram(
            df_filtered, x=score_col, nbins=20,
            color_discrete_sequence=["#4C78A8"],
            labels={score_col: "Adjusted Score"},
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.subheader("Debt Capacity Breakdown")
        dc_counts = df_filtered["debt_capacity"].value_counts().reset_index()
        dc_counts.columns = ["Category", "Count"]
        fig2 = px.pie(
            dc_counts, names="Category", values="Count",
            color="Category",
            color_discrete_map={"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"},
        )
        fig2.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)


def render_deal_quadrant(df_top: pd.DataFrame, score_col: str):
    """Render the deal quadrant chart (Quality vs Valuation)."""
    st.subheader("Deal Quadrant — Quality vs Valuation")
    quad_df = df_top.dropna(subset=["ev_to_ebitda", "quality_score"]).copy()

    if quad_df.empty:
        return

    quad_df["_bubble"] = quad_df["irr_proxy"].abs().fillna(0.15).clip(lower=0.05)

    color_map = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}

    fig3 = px.scatter(
        quad_df,
        x="ev_to_ebitda", y="quality_score",
        size="_bubble", color="debt_capacity",
        hover_name="company" if "company" in quad_df.columns else "ticker",
        hover_data={
            "ev_to_ebitda": ":.1f", "quality_score": ":.1f",
            "irr_proxy": ":.1%", "_bubble": False, "debt_capacity": False,
        },
        color_discrete_map=color_map,
        labels={
            "ev_to_ebitda": "EV/EBITDA — Entry Valuation (lower = better)",
            "quality_score": "Quality Score (higher = better)",
        },
        size_max=45,
    )

    x_mid = quad_df["ev_to_ebitda"].median()
    y_mid = quad_df["quality_score"].median()
    x_min, x_max = quad_df["ev_to_ebitda"].min(), quad_df["ev_to_ebitda"].max()
    y_min, y_max = quad_df["quality_score"].min(), quad_df["quality_score"].max()
    x_pad = (x_max - x_min) * 0.10
    y_pad = (y_max - y_min) * 0.10

    zone_shapes = [
        dict(type="rect", xref="x", yref="y", x0=x_min - x_pad, x1=x_mid,
             y0=y_mid, y1=y_max + y_pad, fillcolor="rgba(46, 204, 113, 0.08)",
             line_width=0, layer="below"),
        dict(type="rect", xref="x", yref="y", x0=x_mid, x1=x_max + x_pad,
             y0=y_mid, y1=y_max + y_pad, fillcolor="rgba(243, 156, 18, 0.08)",
             line_width=0, layer="below"),
        dict(type="rect", xref="x", yref="y", x0=x_min - x_pad, x1=x_mid,
             y0=y_min - y_pad, y1=y_mid, fillcolor="rgba(243, 156, 18, 0.08)",
             line_width=0, layer="below"),
        dict(type="rect", xref="x", yref="y", x0=x_mid, x1=x_max + x_pad,
             y0=y_min - y_pad, y1=y_mid, fillcolor="rgba(231, 76, 60, 0.08)",
             line_width=0, layer="below"),
    ]
    fig3.update_layout(shapes=zone_shapes)

    fig3.add_hline(y=y_mid, line_dash="dash", line_color="rgba(150,150,150,0.4)", line_width=1)
    fig3.add_vline(x=x_mid, line_dash="dash", line_color="rgba(150,150,150,0.4)", line_width=1)

    quadrant_labels = [
        (x_min, y_max, "🎯 Sweet Spot", "left", "top"),
        (x_max, y_max, "💰 Quality Premium", "right", "top"),
        (x_min, y_min, "⚠️ Value Trap?", "left", "bottom"),
        (x_max, y_min, "❌ Avoid", "right", "bottom"),
    ]
    for qx, qy, text, xanchor, yanchor in quadrant_labels:
        fig3.add_annotation(
            x=qx, y=qy, text=text, showarrow=False,
            font=dict(size=11, color="rgba(180,180,180,0.8)"),
            xanchor=xanchor, yanchor=yanchor,
        )

    label_col = "ticker" if "ticker" in quad_df.columns else "company"
    top5_tickers = (
        quad_df.nlargest(5, score_col) if score_col in quad_df.columns
        else quad_df.head(5)
    )
    for _, r in top5_tickers.iterrows():
        fig3.add_annotation(
            x=r["ev_to_ebitda"], y=r["quality_score"],
            text=str(r.get(label_col, "")),
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor="rgba(200,200,200,0.6)", font=dict(size=10),
            ax=15, ay=-20,
        )

    fig3.update_layout(height=500, margin=dict(t=30, b=30))
    st.plotly_chart(fig3, use_container_width=True)


def render_top_opportunities(df_filtered: pd.DataFrame, score_col: str):
    """Render the top opportunities and watch list sections."""
    st.subheader("🎯 Top Opportunities")
    opp_left, opp_right = st.columns(2)

    with opp_left:
        st.markdown("**🟢 Best LBO Candidates**")
        score_75th = df_filtered[score_col].quantile(0.75) if len(df_filtered) else 0
        best_mask = (
            (df_filtered["debt_capacity"].isin(["High"]) |
             (df_filtered.get("quality_score", pd.Series(0, index=df_filtered.index)) > 60)) &
            (df_filtered[score_col] >= score_75th)
        )
        if "irr_proxy" in df_filtered.columns:
            best_mask &= df_filtered["irr_proxy"].fillna(0) > 0.15
        best = df_filtered[best_mask].head(3)
        if best.empty:
            st.markdown("_No companies match this filter._")
        else:
            for _, r in best.iterrows():
                dc = r.get("debt_capacity", "—")
                irr = r.get("irr_proxy")
                irr_str = fmt_irr(irr) if irr is not None and not (isinstance(irr, float) and np.isnan(irr)) else "N/A"
                name = r.get("company", r.get("ticker", "—"))
                score_val = r.get(score_col, float("nan"))
                st.markdown(
                    f"**#{int(r.get('rank', 0))} — {name}** | "
                    f"Score: {score_val:.0f} | IRR: {irr_str} | {debt_capacity_color(dc)}"
                )

    with opp_right:
        st.markdown("**🔴 Watch List (red flags)**")
        watch = df_filtered[df_filtered["red_flags"].fillna("") != ""].copy()
        if not watch.empty:
            watch["_flag_count"] = watch["red_flags"].str.count(r"\|") + 1
            watch = watch.sort_values("_flag_count", ascending=False).head(3)
        if watch.empty:
            st.markdown("_No companies with red flags in current filter._")
        else:
            for _, r in watch.iterrows():
                name = r.get("company", r.get("ticker", "—"))
                flags = r.get("red_flags", "")
                st.markdown(f"**{name}** | ⚠️ {flags}")
