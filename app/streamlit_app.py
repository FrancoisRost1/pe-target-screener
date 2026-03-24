"""
streamlit_app.py — Interactive PE Target Screener Dashboard

Run with: streamlit run app/streamlit_app.py

Features:
- Upload your own company dataset or use default universe
- Adjust scoring weights via sliders
- Filter by sector, debt capacity, minimum score
- Top-N ranked table with color coding
- Score distribution charts
- Company drill-down with sub-scores and investment memo
- Download results as CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from screener.cleaner import clean, apply_eligibility_filters
from screener.ratios import compute_all_ratios
from screener.scoring import score_universe, compute_sub_scores
from screener.classifier import classify_all
from screener.ranking import rank_companies, get_top_n
from screener.summary import add_memos

st.set_page_config(
    page_title="PE Target Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = clean(df_raw)
    df = apply_eligibility_filters(df, cfg)
    df = compute_all_ratios(df, cfg)
    df = score_universe(df, cfg)
    df = compute_sub_scores(df, cfg)
    df = classify_all(df, cfg)
    df = rank_companies(df)
    df = add_memos(df, top_n=len(df))
    return df


def fmt_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.1%}"


def fmt_mult(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.1f}x"


def fmt_score(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.1f}"


def debt_capacity_color(val):
    colors = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    return colors.get(val, "⚪")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar(cfg: dict) -> tuple[pd.DataFrame | None, dict, dict]:
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
        processed_path = Path(__file__).parent.parent / "data" / "processed" / "companies_scored.csv"
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

    # Filters
    st.sidebar.header("3. Filters")
    filters = {
        "min_score": st.sidebar.slider("Min PE Score", 0, 100, 0),
        "debt_capacity": st.sidebar.multiselect(
            "Debt Capacity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
        ),
        "no_red_flags": st.sidebar.checkbox("Exclude companies with red flags"),
        "top_n": st.sidebar.slider("Top N to display", 5, 50, 20),
    }

    return df_raw, custom_weights, filters


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()

    st.title("📊 Private Equity Target Screener")
    st.caption("Identify LBO candidates based on PE-style financial metrics and weighted scoring.")

    df_raw, custom_weights, filters = sidebar(cfg)

    if df_raw is None:
        st.info("👈 Configure your data source in the sidebar to get started.")
        _show_methodology()
        return

    # Run pipeline with custom weights
    run_cfg = {**cfg, "weights": custom_weights}

    with st.spinner("Running screening pipeline..."):
        try:
            df = run_pipeline(df_raw, run_cfg)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return

    # Apply filters
    mask = pd.Series([True] * len(df))
    if filters["min_score"] > 0:
        mask &= df["pe_score"].fillna(0) >= filters["min_score"]
    if filters["debt_capacity"]:
        mask &= df["debt_capacity"].isin(filters["debt_capacity"])
    if filters["no_red_flags"]:
        mask &= df["red_flags"].fillna("") == ""
    df_filtered = df[mask].reset_index(drop=True)

    # Re-rank after filter
    df_filtered["rank"] = range(1, len(df_filtered) + 1)

    top_n = filters["top_n"]
    df_top = df_filtered.head(top_n)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies Screened", len(df))
    col2.metric("After Filters", len(df_filtered))
    col3.metric("Avg PE Score", f"{df_filtered['pe_score'].mean():.1f}")
    col4.metric("High Debt Capacity", int((df_filtered["debt_capacity"] == "High").sum()))

    # ── Top Targets Table ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"🏆 Top {top_n} Candidates")

    display_df = df_top[[c for c in [
        "rank", "ticker", "company", "sector",
        "ebitda_margin", "fcf_conversion", "net_debt_to_ebitda",
        "interest_coverage", "ev_to_ebitda", "revenue_growth",
        "debt_capacity", "pe_score"
    ] if c in df_top.columns]].copy()

    for col in ["ebitda_margin", "fcf_conversion", "revenue_growth"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_pct)
    for col in ["net_debt_to_ebitda", "interest_coverage", "ev_to_ebitda"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt_mult)

    display_df["debt_capacity"] = display_df["debt_capacity"].apply(
        lambda v: f"{debt_capacity_color(v)} {v}"
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download button
    csv = df_top.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Top Targets CSV", csv, "top_targets.csv", "text/csv")

    # ── Charts ───────────────────────────────────────────────────────────────
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Score Distribution")
        fig = px.histogram(
            df_filtered, x="pe_score", nbins=20,
            color_discrete_sequence=["#4C78A8"],
            labels={"pe_score": "PE Score"},
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

    # Bubble chart: EBITDA Margin vs EV/EBITDA
    st.subheader("Margin vs Valuation (bubble = PE Score)")
    bubble_df = df_top.dropna(subset=["ebitda_margin", "ev_to_ebitda", "pe_score"])
    if not bubble_df.empty:
        fig3 = px.scatter(
            bubble_df,
            x="ev_to_ebitda", y="ebitda_margin",
            size="pe_score", color="debt_capacity",
            hover_name="company",
            color_discrete_map={"High": "#2ecc71", "Medium": "#f39c12", "Low": "#e74c3c"},
            labels={"ev_to_ebitda": "EV/EBITDA (lower = cheaper)", "ebitda_margin": "EBITDA Margin"},
            size_max=40,
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Company Drill-Down ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Company Detail")

    company_list = df_top["company"].tolist() if "company" in df_top.columns else df_top["ticker"].tolist()
    selected = st.selectbox("Select a company", company_list)

    if selected:
        col_key = "company" if "company" in df_top.columns else "ticker"
        row = df_top[df_top[col_key] == selected].iloc[0]
        _show_company_detail(row)


def _show_company_detail(row: pd.Series):
    """Render detailed view for a single company."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {row.get('company', row.get('ticker'))}")
        st.caption(f"**Sector:** {row.get('sector', '—')}  |  **Rank:** #{int(row.get('rank', 0))}")

        # Key metrics grid
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PE Score", fmt_score(row.get("pe_score")))
        m2.metric("EBITDA Margin", fmt_pct(row.get("ebitda_margin")))
        m3.metric("FCF Conversion", fmt_pct(row.get("fcf_conversion")))
        m4.metric("EV/EBITDA", fmt_mult(row.get("ev_to_ebitda")))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Net Debt/EBITDA", fmt_mult(row.get("net_debt_to_ebitda")))
        m6.metric("Interest Coverage", fmt_mult(row.get("interest_coverage")))
        m7.metric("ROIC", fmt_pct(row.get("roic")))
        m8.metric("Revenue Growth", fmt_pct(row.get("revenue_growth")))

        # Investment memo
        memo = row.get("investment_memo", "")
        if memo:
            st.info(f"📝 **Investment Memo**\n\n{memo}")

        # Red flags
        flags = row.get("red_flags", "")
        if flags:
            st.warning(f"⚠️ **Red Flags:** {flags}")

    with col2:
        # Radar chart of sub-scores
        sub_scores = {
            "Quality": row.get("quality_score"),
            "Cash Flow": row.get("cash_score"),
            "Leverage": row.get("leverage_score"),
            "Valuation": row.get("valuation_score"),
        }
        valid = {k: v for k, v in sub_scores.items() if v is not None and not np.isnan(v)}

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
                margin=dict(t=20, b=20),
                title="Sub-Score Radar",
            )
            st.plotly_chart(fig, use_container_width=True)

        dc = row.get("debt_capacity", "—")
        st.metric("Debt Capacity", f"{debt_capacity_color(dc)} {dc}")


def _show_methodology():
    """Show methodology section when no data is loaded."""
    with st.expander("📖 Methodology", expanded=True):
        st.markdown("""
        ### How the screener works

        This tool evaluates companies across **5 dimensions** a PE fund cares about:

        | Dimension | Metrics | Weight |
        |---|---|---|
        | Profitability | EBITDA Margin, ROIC | 25% |
        | Cash Generation | FCF Conversion, OCF Margin | 25% |
        | Leverage Capacity | Net Debt/EBITDA, Interest Coverage | 20% |
        | Growth Quality | Revenue Growth, EBITDA Growth | 15% |
        | Valuation | EV/EBITDA | 15% |

        **Scoring method:** Percentile ranking within the universe.
        Companies are ranked relative to each other — not against absolute thresholds.

        **Debt Capacity** is classified separately using rule-based thresholds:
        - 🟢 **High** = margin >20%, FCF conversion >70%, leverage <2.5x, coverage >5x
        - 🟡 **Medium** = intermediate profile
        - 🔴 **Low** = margin <10%, leverage >4x, or coverage <2.5x
        """)


if __name__ == "__main__":
    main()
