"""
streamlit_app.py — Interactive PE Target Screener Dashboard

Run with: streamlit run app/streamlit_app.py

Orchestrates the pipeline and delegates rendering to tab modules.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import yaml

from screener.cleaner import clean, winsorize_ratios, apply_eligibility_filters
from screener.ratios import compute_all_ratios
from screener.lbo import compute_lbo_metrics
from screener.scoring import score_universe_sector_adjusted, compute_sub_scores, apply_score_adjustments
from screener.classifier import classify_all
from screener.ranking import rank_companies
from screener.summary import add_memos
from app.sidebar import sidebar
from app.tab_table import render_kpis, render_top_table
from app.tab_charts import render_charts, render_deal_quadrant, render_top_opportunities
from app.tab_detail import render_company_detail

st.set_page_config(page_title="PE Target Screener",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_config():
    """Load config.yaml from project root."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def run_pipeline(df_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run the full screening pipeline. Cached so repeat slider values are instant."""
    df = clean(df_raw)
    df = compute_all_ratios(df, cfg)
    df = winsorize_ratios(df)
    df = compute_lbo_metrics(df, cfg)
    df = apply_eligibility_filters(df, cfg)
    df = score_universe_sector_adjusted(df, cfg)
    df = compute_sub_scores(df, cfg)
    df = classify_all(df, cfg)
    df = apply_score_adjustments(df, cfg)
    df = rank_companies(df)
    df = add_memos(df, top_n=len(df))
    return df


def main():
    """Main dashboard entry point."""
    cfg = load_config()
    st.title("Private Equity Target Screener")
    st.caption("Identify LBO candidates using sector-adjusted scoring, IRR proxy, and valuation penalty.")

    df_raw, custom_weights, lbo_overrides, filters = sidebar(cfg)
    if df_raw is None:
        st.info("Configure your data source in the sidebar to get started.")
        return

    run_cfg = {**cfg, "weights": custom_weights, "lbo": {**cfg.get("lbo", {}), **lbo_overrides}}

    with st.spinner("Running screening pipeline..."):
        try:
            df = run_pipeline(df_raw, run_cfg)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            return

    score_col = ("pe_score_final" if "pe_score_final" in df.columns else
                 "pe_score_adjusted" if "pe_score_adjusted" in df.columns else "pe_score")
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

    st.markdown("---")
    render_kpis(df, df_filtered, score_col, run_cfg)
    st.markdown("---")
    render_top_table(df_top, top_n, score_col, run_cfg)
    st.markdown("---")
    render_charts(df_filtered, df_top, score_col)
    render_deal_quadrant(df_top, score_col)
    st.markdown("---")
    render_top_opportunities(df_filtered, score_col)
    st.markdown("---")

    st.subheader("Company Detail")
    col_key = "company" if "company" in df_top.columns else "ticker"
    selected = st.selectbox("Select a company", df_top[col_key].tolist())
    if selected:
        row = df_top[df_top[col_key] == selected].iloc[0]
        render_company_detail(row, cfg=run_cfg)


if __name__ == "__main__":
    main()
