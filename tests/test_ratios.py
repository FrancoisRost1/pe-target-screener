"""Tests for ratio calculations."""
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from screener.ratios import (
    compute_ebitda_margin, compute_net_debt, compute_net_debt_to_ebitda,
    compute_interest_coverage, compute_fcf_conversion,
)
from screener.ratios_secondary import (
    compute_roic, compute_ev_to_ebitda, compute_revenue_growth,
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "company": ["Alpha", "Beta", "Gamma"],
        "revenue": [1_000_000, 500_000, 200_000],
        "ebitda": [250_000, 50_000, -10_000],
        "ebit": [200_000, 30_000, -20_000],
        "total_debt": [300_000, 400_000, 100_000],
        "cash": [100_000, 50_000, 20_000],
        "interest_expense": [20_000, 40_000, 5_000],
        "free_cash_flow": [180_000, 20_000, -5_000],
        "operating_cash_flow": [200_000, 40_000, 0],
        "invested_capital": [800_000, 600_000, 150_000],
        "enterprise_value": [2_000_000, 800_000, 100_000],
        "prior_revenue": [900_000, 480_000, 210_000],
        "prior_ebitda": [220_000, 45_000, 5_000],
    })

def test_ebitda_margin(sample_df):
    df = compute_ebitda_margin(sample_df)
    assert abs(df.loc[0, "ebitda_margin"] - 0.25) < 0.001
    assert abs(df.loc[1, "ebitda_margin"] - 0.10) < 0.001
    assert df.loc[2, "ebitda_margin"] < 0  # Negative EBITDA

def test_net_debt(sample_df):
    df = compute_net_debt(sample_df)
    assert df.loc[0, "net_debt"] == 200_000
    assert df.loc[1, "net_debt"] == 350_000

def test_net_debt_to_ebitda(sample_df):
    df = compute_net_debt(sample_df)
    df = compute_net_debt_to_ebitda(df)
    assert abs(df.loc[0, "net_debt_to_ebitda"] - 0.8) < 0.01

def test_interest_coverage(sample_df):
    df = compute_interest_coverage(sample_df)
    assert abs(df.loc[0, "interest_coverage"] - 10.0) < 0.01

def test_fcf_conversion(sample_df):
    df = compute_fcf_conversion(sample_df)
    assert abs(df.loc[0, "fcf_conversion"] - 0.72) < 0.01

def test_zero_division_handled(sample_df):
    """Ensure no exceptions on zero denominators."""
    df = sample_df.copy()
    df.loc[0, "revenue"] = 0
    df = compute_ebitda_margin(df)
    assert pd.isna(df.loc[0, "ebitda_margin"])

def test_roic(sample_df):
    cfg = {"assumptions": {"tax_rate": 0.25, "min_invested_capital": 1}}
    df = compute_roic(sample_df, tax_rate=0.25, min_ic=1)
    # NOPAT = 200k * 0.75 = 150k; IC = 800k → ROIC = 0.1875
    assert abs(df.loc[0, "roic"] - 0.1875) < 0.001

def test_revenue_growth(sample_df):
    df = compute_revenue_growth(sample_df)
    # (1M - 900k) / 900k = 11.1%
    assert abs(df.loc[0, "revenue_growth"] - (100_000/900_000)) < 0.001
