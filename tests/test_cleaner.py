"""Tests for data cleaning."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from screener.cleaner import clean, apply_eligibility_filters

def test_clean_removes_duplicates():
    df = pd.DataFrame({"ticker": ["AAPL", "AAPL", "MSFT"], "revenue": [100, 100, 200]})
    result = clean(df)
    assert len(result) == 2

def test_clean_flags_negative_revenue():
    df = pd.DataFrame({"ticker": ["BAD"], "revenue": [-100], "ebitda": [10]})
    result = clean(df)
    assert "negative_revenue" in result.loc[0, "data_quality_flags"]

def test_eligibility_filter():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "revenue": [200_000_000, 50_000_000, 500_000_000],
        "ebitda": [20_000_000, 5_000_000, 50_000_000],
        "sector": ["Industrials", "Industrials", "Financials"],
        "data_quality_flags": ["", "", ""],
    })
    cfg = {"eligibility": {
        "min_revenue": 100_000_000,
        "min_ebitda": 10_000_000,
        "exclude_sectors": ["Financials"],
    }}
    result = apply_eligibility_filters(df, cfg)
    assert len(result) == 1
    assert result.iloc[0]["ticker"] == "A"
