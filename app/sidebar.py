"""
sidebar.py — Streamlit sidebar controls for the PE Target Screener.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str) -> pd.DataFrame:
    """Cached CSV read so the file is only parsed once per session."""
    return pd.read_csv(path)


def sidebar(cfg: dict) -> Tuple[Optional[pd.DataFrame], dict, dict, dict]:
    """Render sidebar controls and return user selections."""
    st.sidebar.title("Screener Controls")

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
        # Lookup order:
        #   1. data/snapshot/  → committed shipped sample (always present, even on Streamlit Cloud)
        #   2. data/raw/       → local yfinance cache (gitignored, present after `python3 main.py`)
        #   3. outputs/        → local pipeline output (gitignored)
        root = Path(__file__).parent.parent
        candidates = [
            root / "data" / "snapshot" / "companies_raw.csv",
            root / "data" / "raw" / "companies_raw.csv",
            root / "outputs" / "companies_scored.csv",
        ]
        for p in candidates:
            if p.exists():
                df_raw = _read_csv_cached(str(p))
                st.sidebar.info(f"Loaded {len(df_raw)} companies from `{p.parent.name}/{p.name}`")
                break

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
    st.sidebar.caption("Override model parameters to affect IRR estimates")
    lbo_defaults = cfg.get("lbo", {})
    lbo_overrides = {
        "exit_multiple": st.sidebar.slider(
            "Exit Multiple (x)", 6.0, 14.0,
            float(lbo_defaults.get("exit_multiple", 8.0)), step=0.5,
        ),
        "holding_period": st.sidebar.slider(
            "Holding Period (years)", 3, 7,
            int(lbo_defaults.get("holding_period", 5)),
        ),
        "target_leverage": st.sidebar.slider(
            "Target Leverage (x EBITDA)", 2.0, 6.0,
            float(lbo_defaults.get("target_leverage", 3.5)), step=0.25,
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
