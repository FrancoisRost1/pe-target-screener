"""
streamlit_app.py — Interactive PE Target Screener Dashboard (V2)

Run with: streamlit run app/streamlit_app.py

V2 changes:
- pe_score_adjusted as primary ranking metric (raw score visible in drill-down)
- IRR proxy column in top targets table
- 5 KPI cards including Median IRR Proxy
- Deal Quadrant chart (Quality vs Valuation) replaces bubble chart
- LBO metrics row in company drill-down
- Bull/Bear/Key Risk memo format with proper line breaks
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from screener.cleaner import clean, winsorize_ratios, apply_eligibility_filters
from screener.ratios import compute_all_ratios
from screener.lbo import compute_lbo_metrics
from screener.scoring import score_universe_sector_adjusted, compute_sub_scores, apply_score_adjustments
from screener.classifier import classify_all
from screener.ranking import rank_companies, get_top_n
from screener.summary import add_memos

st.set_page_config(
    page_title="PE Target Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """V2 pipeline: winsorize → LBO → eligibility → sector-adjusted score → penalties."""
    df = clean(df_raw)
    df = compute_all_ratios(df, cfg)
    df = winsorize_ratios(df)
    df = compute_lbo_metrics(df, cfg)
    df = apply_eligibility_filters(df, cfg)
    df = score_universe_sector_adjusted(df, cfg)
    df = compute_sub_scores(df, cfg)
    df = classify_all(df, cfg)
    df = apply_score_adjustments(df)
    df = rank_companies(df)
    df = add_memos(df, top_n=len(df))
    return df


def fmt_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.1%}"


def fmt_irr(val):
    """Format IRR proxy as ~XX% or — if NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"~{val:.0%}"


def fmt_mult(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.1f}x"


def fmt_score(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.1f}"


def fmt_millions(val):
    """Format a raw dollar value as $XXXm."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"${val / 1_000_000:.0f}m"


def fmt_fcf_yield_equity(val):
    """Format FCF yield on equity — cap display at >50% to flag outliers."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if val >= 0.50:
        return ">50%"
    return f"{val:.1%}"


def debt_capacity_color(val):
    colors = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    return colors.get(val, "⚪")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar(cfg: dict) -> Tuple[Optional[pd.DataFrame], dict, dict, dict]:
    st.sidebar.title("⚙️ Screener Controls")

    # Data source
    st.sidebar.header("1. Data Source")
    data_option = st.sidebar.radio(
        "Choose data",
        ["Use processed results", "Upload CSV"],
        index=0,
    )

    df_raw = None
    if data_option == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload companies CSV", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.sidebar.success(f"Loaded {len(df_raw)} rows")
    else:
        processed_path = Path(__file__).parent.parent / "outputs" / "companies_scored.csv"
        raw_path = Path(__file__).parent.parent / "data" / "raw" / "companies_raw.csv"
        for p in [processed_path, raw_path]:
            if p.exists():
                df_raw = pd.read_csv(p)
                st.sidebar.info(f"Loaded {len(df_raw)} companies from {p.name}")
                break
        if df_raw is None:
            st.sidebar.warning("No data found. Run `python main.py` first or upload a CSV.")

    # Scoring weights
    st.sidebar.header("2. Scoring Weights")
    st.sidebar.caption("Adjust weights to reflect your investment thesis")

    default_w = cfg.get("weights", {})
    custom_weights = {}
    for metric, default_val in default_w.items():
        label = metric.replace("_", " ").title()
        custom_weights[metric] = st.sidebar.slider(
            label, min_value=0.0, max_value=0.30, value=float(default_val), step=0.01
        )

    # Normalize to sum to 1
    total = sum(custom_weights.values())
    if total > 0:
        custom_weights = {k: v / total for k, v in custom_weights.items()}

    # LBO assumptions
    st.sidebar.header("3. LBO Assumptions")
    st.sidebar.caption("Override model parameters — affects IRR estimates")
    lbo_defaults = cfg.get("lbo", {})
    lbo_overrides = {
        "exit_multiple": st.sidebar.slider(
            "Entry → Exit Multiple (x)", 6.0, 16.0,
            float(lbo_defaults.get("exit_multiple", 12.0)), step=0.5,
        ),
        "holding_period": st.sidebar.slider(
            "Holding Period (years)", 3, 7,
            int(lbo_defaults.get("holding_period", 5)),
        ),
        "target_leverage": st.sidebar.slider(
            "Target Leverage (x EBITDA)", 2.0, 6.0,
            float(lbo_defaults.get("target_leverage", 4.0)), step=0.25,
        ),
    }

    # Filters
    st.sidebar.header("4. Filters")
    filters = {
        "min_score": st.sidebar.slider("Min Adjusted Score", 0, 100, 0),
        "debt_capacity": st.sidebar.multiselect(
            "Debt Capacity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
        ),
        "no_red_flags": st.sidebar.checkbox("Exclude companies with red flags"),
        "top_n": st.sidebar.slider("Top N to display", 5, 50, 20),
    }

    return df_raw, custom_weights, lbo_overrides, filters


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    st.title("📊 Private Equity Target Screener")
    st.caption("Identify LBO candidates — sector-adjusted scoring, IRR proxy, valuation penalty.")

    df_raw, custom_weights, lbo_overrides, filters = sidebar(cfg)

    if df_raw is None:
        st.info("👈 Configure your data source in the sidebar to get started.")
        _show_methodology()
        return

    run_cfg = {**cfg, "weights": custom_weights, "lbo": {**cfg.get("lbo", {}), **lbo_overrides}}

    with st.spinner("Running screening pipeline..."):
        try:
            df = run_pipeline(df_raw, run_cfg)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            return

    # Apply filters against pe_score_adjusted
    score_col = "pe_score_adjusted" if "pe_score_adjusted" in df.columns else "pe_score"
    mask = pd.Series([True] * len(df), index=df.index)
    if filters["min_score"] > 0:
        mask &= df[score_col].fillna(0) >= filters["min_score"]
    if filters["debt_capacity"]:
        mask &= df["debt_capacity"].isin(filters["debt_capacity"])
    if filters["no_red_flags"]:
        mask &= df["red_flags"].fillna("") == ""
    df_filtered = df[mask].reset_index(drop=True)
    df_filtered["rank"] = range(1, len(df_filtered) + 1)

    top_n = filters["top_n"]
    df_top = df_filtered.head(top_n)

    # ── KPI Cards (5) ────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Companies Screened", len(df))
    c2.metric("After Filters", len(df_filtered))
    c3.metric(
        "Avg Adjusted Score",
        f"{df_filtered[score_col].mean():.1f}" if len(df_filtered) else "—",
    )
    c4.metric("High Debt Capacity", int((df_filtered["debt_capacity"] == "High").sum()))
    irr_median = df_filtered["irr_proxy"].median() if "irr_proxy" in df_filtered.columns else None
    c5.metric(
        "Median IRR Proxy",
        fmt_irr(irr_median) if irr_median is not None and not np.isnan(irr_median) else "—",
    )

    # ── Top Targets Table ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"🏆 Top {top_n} Candidates")

    TABLE_COLS = [
        "rank", "ticker", "company", "sector",
        "ebitda_margin", "fcf_conversion", "net_debt_to_ebitda",
        "ev_to_ebitda", "irr_proxy", "debt_capacity", "pe_score_adjusted",
    ]
    display_df = df_top[[c for c in TABLE_COLS if c in df_top.columns]].copy()

    for col in ["ebitda_margin", "fcf_conversion"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_pct)
    for col in ["net_debt_to_ebitda", "ev_to_ebitda"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_mult)
    if "irr_proxy" in display_df.columns:
        display_df["irr_proxy"] = display_df["irr_proxy"].apply(fmt_irr)
    if "pe_score_adjusted" in display_df.columns:
        display_df["pe_score_adjusted"] = display_df["pe_score_adjusted"].apply(fmt_score)
    if "debt_capacity" in display_df.columns:
        display_df["debt_capacity"] = display_df["debt_capacity"].apply(
            lambda v: f"{debt_capacity_color(v)} {v}"
        )

    col_rename = {c: c.replace("_", " ").title() for c in display_df.columns}
    col_rename["irr_proxy"] = "IRR Est."
    display_df.rename(columns=col_rename, inplace=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    lbo_cfg = run_cfg.get("lbo", {})
    st.caption(
        f"IRR estimates assume {int(lbo_cfg.get('holding_period', 5))}yr hold, "
        f"{lbo_cfg.get('exit_multiple', 12.0):.0f}x exit, "
        f"{lbo_cfg.get('target_leverage', 4.0):.1f}x entry leverage. "
        "Simplified model — for indicative purposes only."
    )

    csv = df_top.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Top Targets CSV", csv, "top_targets.csv", "text/csv")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("---")
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

    # ── Deal Quadrant ─────────────────────────────────────────────────────────
    st.subheader("Deal Quadrant — Quality vs Valuation")
    quad_df = df_top.dropna(subset=["ev_to_ebitda", "quality_score"]).copy()

    if not quad_df.empty:
        # Bubble size: abs(irr_proxy), default 0.15 for NaN
        quad_df["_bubble"] = quad_df["irr_proxy"].abs().fillna(0.15)
        # Clamp to a minimum so every company gets a visible bubble
        quad_df["_bubble"] = quad_df["_bubble"].clip(lower=0.05)

        color_map = {"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"}

        fig3 = px.scatter(
            quad_df,
            x="ev_to_ebitda",
            y="quality_score",
            size="_bubble",
            color="debt_capacity",
            hover_name="company" if "company" in quad_df.columns else "ticker",
            hover_data={
                "ev_to_ebitda": ":.1f",
                "quality_score": ":.1f",
                "irr_proxy": ":.1%",
                "_bubble": False,
                "debt_capacity": False,
            },
            color_discrete_map=color_map,
            labels={
                "ev_to_ebitda": "EV/EBITDA — Entry Valuation (lower = better)",
                "quality_score": "Quality Score (higher = better)",
            },
            size_max=45,
        )

        # Quadrant reference lines (medians)
        x_mid = quad_df["ev_to_ebitda"].median()
        y_mid = quad_df["quality_score"].median()

        x_min = quad_df["ev_to_ebitda"].min()
        x_max = quad_df["ev_to_ebitda"].max()
        y_min = quad_df["quality_score"].min()
        y_max = quad_df["quality_score"].max()

        # Add 10% padding so zones extend beyond data points
        x_pad = (x_max - x_min) * 0.10
        y_pad = (y_max - y_min) * 0.10

        # Colored background zones: Sweet Spot (green), Value Trap + Quality Premium (yellow), Avoid (red)
        zone_shapes = [
            # 🎯 Sweet Spot — low valuation, high quality (bottom-left x, top-right y)
            dict(
                type="rect", xref="x", yref="y",
                x0=x_min - x_pad, x1=x_mid,
                y0=y_mid, y1=y_max + y_pad,
                fillcolor="rgba(46, 204, 113, 0.08)",
                line_width=0, layer="below",
            ),
            # 💰 Quality Premium — high valuation, high quality
            dict(
                type="rect", xref="x", yref="y",
                x0=x_mid, x1=x_max + x_pad,
                y0=y_mid, y1=y_max + y_pad,
                fillcolor="rgba(243, 156, 18, 0.08)",
                line_width=0, layer="below",
            ),
            # ⚠️ Value Trap — low valuation, low quality
            dict(
                type="rect", xref="x", yref="y",
                x0=x_min - x_pad, x1=x_mid,
                y0=y_min - y_pad, y1=y_mid,
                fillcolor="rgba(243, 156, 18, 0.08)",
                line_width=0, layer="below",
            ),
            # ❌ Avoid — high valuation, low quality
            dict(
                type="rect", xref="x", yref="y",
                x0=x_mid, x1=x_max + x_pad,
                y0=y_min - y_pad, y1=y_mid,
                fillcolor="rgba(231, 76, 60, 0.08)",
                line_width=0, layer="below",
            ),
        ]
        fig3.update_layout(shapes=zone_shapes)

        fig3.add_hline(y=y_mid, line_dash="dash", line_color="rgba(150,150,150,0.4)", line_width=1)
        fig3.add_vline(x=x_mid, line_dash="dash", line_color="rgba(150,150,150,0.4)", line_width=1)

        # Quadrant labels
        quadrant_labels = [
            (x_min, y_max, "🎯 Sweet Spot", "left", "top"),
            (x_max, y_max, "💰 Quality Premium", "right", "top"),
            (x_min, y_min, "⚠️ Value Trap?", "left", "bottom"),
            (x_max, y_min, "❌ Avoid", "right", "bottom"),
        ]
        for qx, qy, text, xanchor, yanchor in quadrant_labels:
            fig3.add_annotation(
                x=qx, y=qy, text=text,
                showarrow=False,
                font=dict(size=11, color="rgba(180,180,180,0.8)"),
                xanchor=xanchor, yanchor=yanchor,
            )

        # Ticker labels for top 5 by pe_score_adjusted
        label_col = "ticker" if "ticker" in quad_df.columns else "company"
        top5_tickers = (
            quad_df.nlargest(5, score_col) if score_col in quad_df.columns
            else quad_df.head(5)
        )
        for _, r in top5_tickers.iterrows():
            fig3.add_annotation(
                x=r["ev_to_ebitda"],
                y=r["quality_score"],
                text=str(r.get(label_col, "")),
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowcolor="rgba(200,200,200,0.6)",
                font=dict(size=10),
                ax=15, ay=-20,
            )

        fig3.update_layout(height=500, margin=dict(t=30, b=30))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Top Opportunities ─────────────────────────────────────────────────────
    st.markdown("---")
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

    # ── Company Drill-Down ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Company Detail")

    col_key = "company" if "company" in df_top.columns else "ticker"
    company_list = df_top[col_key].tolist()
    selected = st.selectbox("Select a company", company_list)

    if selected:
        row = df_top[df_top[col_key] == selected].iloc[0]
        _show_company_detail(row, run_cfg)


def _show_company_detail(row: pd.Series, cfg: dict):
    """Render V2 detailed view: two metric rows + LBO breakdown expander + memo."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {row.get('company', row.get('ticker'))}")
        st.caption(f"**Sector:** {row.get('sector', '—')}  |  **Rank:** #{int(row.get('rank', 0))}")

        # Row 1: adjusted score + core metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Adj. Score", fmt_score(row.get("pe_score_adjusted", row.get("pe_score"))))
        m2.metric("EBITDA Margin", fmt_pct(row.get("ebitda_margin")))
        m3.metric("FCF Conversion", fmt_pct(row.get("fcf_conversion")))
        m4.metric("EV/EBITDA", fmt_mult(row.get("ev_to_ebitda")))

        # Row 2: LBO metrics
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("IRR Proxy", fmt_irr(row.get("irr_proxy")))
        m6.metric("Max Debt", fmt_millions(row.get("max_debt")))
        m7.metric("Equity Required", fmt_millions(row.get("equity_required")))
        m8.metric("FCF Yield / Equity", fmt_fcf_yield_equity(row.get("fcf_yield_equity")))

        # Raw score for comparison
        raw = row.get("pe_score_raw", row.get("pe_score"))
        adj = row.get("pe_score_adjusted")
        if raw is not None and adj is not None and not pd.isna(raw) and not pd.isna(adj):
            st.caption(f"Raw Score (universe-only): {raw:.1f} → Adjusted: {adj:.1f} "
                       f"(Δ {adj - raw:+.1f})")

        # LBO Deal Breakdown expander
        lbo_cfg = cfg.get("lbo", {})
        holding_period = lbo_cfg.get("holding_period", 5)
        exit_multiple_cap = lbo_cfg.get("exit_multiple", 12.0)
        debt_repayment_rate = lbo_cfg.get("debt_repayment_rate", 0.4)

        ev_val = row.get("enterprise_value")
        ebitda_val = row.get("ebitda")
        max_debt_val = row.get("max_debt")
        eq_req_val = row.get("equity_required")

        if (ev_val is not None and not pd.isna(ev_val) and ev_val > 0
                and ebitda_val is not None and not pd.isna(ebitda_val)
                and max_debt_val is not None and not pd.isna(max_debt_val)
                and eq_req_val is not None and not pd.isna(eq_req_val)):

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

            with st.expander("🧮 LBO Deal Breakdown"):
                st.caption(
                    f"Assumptions: {holding_period}yr hold · "
                    f"{exit_mult:.1f}x exit multiple · "
                    f"{debt_repayment_rate:.0%} FCF → debt repayment"
                )
                b1, b2, b3 = st.columns(3)
                b1.metric("Entry EV", fmt_millions(ev_val))
                b1.metric("Max Debt", fmt_millions(max_debt_val))
                b1.metric("Equity Cheque", fmt_millions(eq_req_val))
                b2.metric("Exit EBITDA", fmt_millions(exit_ebitda))
                b2.metric("Exit Multiple", f"{exit_mult:.1f}x")
                b2.metric("Exit EV", fmt_millions(exit_ev))
                b3.metric("Debt Repaid", fmt_millions(debt_repaid))
                b3.metric("Debt Remaining", fmt_millions(debt_remaining))
                b3.metric("Exit Equity", fmt_millions(exit_equity))
                st.markdown("---")
                moic_col, irr_col = st.columns(2)
                moic_col.metric("MOIC", f"{moic:.2f}x" if not pd.isna(moic) else "—")
                irr_col.metric("IRR Proxy", fmt_irr(row.get("irr_proxy")))

        # Investment memo with proper line breaks
        memo = row.get("investment_memo", "")
        if memo:
            st.markdown("**📋 Investment Memo**")
            st.markdown(memo.replace("\n", "  \n"))

        # Red flags
        flags = row.get("red_flags", "")
        if flags and not (isinstance(flags, float) and np.isnan(flags)):
            st.warning(f"⚠️ **Red Flags:** {flags}")

    with col2:
        # Radar chart of sub-scores
        sub_scores = {
            "Quality": row.get("quality_score"),
            "Cash Flow": row.get("cash_score"),
            "Leverage": row.get("leverage_score"),
            "Valuation": row.get("valuation_score"),
        }
        valid = {k: v for k, v in sub_scores.items()
                 if v is not None and not (isinstance(v, float) and np.isnan(v))}

        if valid:
            categories = list(valid.keys())
            values = list(valid.values())
            values_closed = values + [values[0]]
            categories_closed = categories + [categories[0]]

            fig = go.Figure(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor="rgba(76, 120, 168, 0.3)",
                line_color="#4C78A8",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=280,
                margin=dict(t=30, b=20),
                title="Sub-Score Radar",
            )
            st.plotly_chart(fig, use_container_width=True)

        dc = row.get("debt_capacity", "—")
        st.metric("Debt Capacity", f"{debt_capacity_color(dc)} {dc}")

        # Penalty breakdown
        rf_pen = row.get("red_flag_penalty", 0) or 0
        val_pen = row.get("valuation_penalty", 0) or 0
        dk_pen = row.get("deal_killer_penalty", 0) or 0
        penalty_parts = []
        if rf_pen:
            penalty_parts.append(f"Red flags {rf_pen:+.0f}")
        if val_pen:
            penalty_parts.append(f"Valuation {val_pen:+.0f}")
        if dk_pen:
            penalty_parts.append(f"Deal killer {dk_pen:+.0f}")
        if penalty_parts:
            st.caption("Score penalties: " + " | ".join(penalty_parts))


def _show_methodology():
    """Show methodology section when no data is loaded."""
    with st.expander("📖 Methodology — V2", expanded=True):
        st.markdown("""
        ### How the screener works (V2)

        This tool evaluates companies across **5 dimensions** a PE fund cares about:

        | Dimension | Metrics | Weight |
        |---|---|---|
        | Profitability | EBITDA Margin, ROIC | 25% |
        | Cash Generation | FCF Conversion, OCF Margin | 25% |
        | Leverage Capacity | Net Debt/EBITDA, Interest Coverage | 20% |
        | Growth Quality | Revenue Growth, EBITDA Growth | 15% |
        | Valuation | EV/EBITDA | 15% |

        **V2 upgrades:**
        - **Winsorization** — extreme ratios capped before scoring (e.g. 2600x coverage → 50x)
        - **Sector normalization** — 60% sector rank + 40% universe rank
        - **LBO layer** — IRR proxy, max debt, equity required per company
        - **Valuation penalty** — EV/EBITDA > 12x penalises score (up to −15pts)
        - **Red flag penalty** — structural issues reduce score (up to −38pts)
        - **Hard PE filters** — EBITDA margin < 8%, coverage < 2.5x, negative EBITDA removed

        **Debt Capacity** is classified separately using rule-based thresholds:
        - 🟢 **High** = margin >20%, FCF conversion >70%, leverage <2.5x, coverage >5x
        - 🟡 **Medium** = intermediate profile
        - 🔴 **Low** = margin <10%, leverage >4x, or coverage <2.5x
        """)


if __name__ == "__main__":
    main()
