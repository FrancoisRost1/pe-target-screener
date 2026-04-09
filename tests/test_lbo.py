"""Tests for LBO estimation functions."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from screener.lbo import compute_lbo_metrics, _build_cashflows_and_compute_irr
from screener.lbo_scenarios import compute_scenario_irr
from screener.scoring_adjustments import apply_irr_hurdle_penalty


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_company(ev=1_000, ebitda=100, fcf=60, revenue_growth=0.05,
                  total_debt=200, cash=50) -> pd.DataFrame:
    """
    Build a minimal single-company DataFrame for LBO testing.
    Values are in $M. Defaults represent a healthy mid-market LBO candidate.
    """
    return pd.DataFrame({
        "ticker": ["TEST"],
        "company": ["TestCo"],
        "enterprise_value": [float(ev)],
        "ebitda": [float(ebitda)],
        "free_cash_flow": [float(fcf)],
        "revenue_growth": [float(revenue_growth)],
        "total_debt": [float(total_debt)],
        "cash": [float(cash)],
    })


def _base_cfg() -> dict:
    return {
        "lbo": {
            "target_leverage": 3.5,
            "holding_period": 5,
            "exit_multiple": 8.0,
            "debt_repayment_rate": 0.4,
            "debt_interest_rate": 0.07,  # kept for config compat; not used in model
        }
    }


# ---------------------------------------------------------------------------
# _build_cashflows_and_compute_irr tests
# ---------------------------------------------------------------------------

def test_irr_positive_for_profitable_deal():
    """A profitable deal (exit equity > entry equity) should have positive IRR."""
    eq = pd.Series([300.0])
    debt = pd.Series([700.0])
    fcf = pd.Series([50.0])
    exit_ev = pd.Series([1500.0])
    irr = _build_cashflows_and_compute_irr(eq, debt, fcf, exit_ev, 0.4, 5)
    assert irr.iloc[0] > 0, "Profitable deal should have positive IRR"


def test_irr_nan_for_zero_equity():
    """Zero equity should produce NaN IRR (invalid deal)."""
    eq = pd.Series([0.0])
    debt = pd.Series([700.0])
    fcf = pd.Series([50.0])
    exit_ev = pd.Series([1500.0])
    irr = _build_cashflows_and_compute_irr(eq, debt, fcf, exit_ev, 0.4, 5)
    assert pd.isna(irr.iloc[0])


def test_irr_nan_when_exit_equity_negative():
    """If exit EV < remaining debt, exit equity is negative → NaN IRR."""
    eq = pd.Series([300.0])
    debt = pd.Series([700.0])
    fcf = pd.Series([0.0])  # no FCF → no debt paydown
    exit_ev = pd.Series([500.0])  # exit EV < debt → negative exit equity
    irr = _build_cashflows_and_compute_irr(eq, debt, fcf, exit_ev, 0.4, 5)
    assert pd.isna(irr.iloc[0])


def test_irr_capped_at_40pct():
    """IRR should be capped at 40% regardless of deal economics."""
    eq = pd.Series([100.0])
    debt = pd.Series([0.0])
    fcf = pd.Series([0.0])
    exit_ev = pd.Series([10_000.0])  # massive exit → would be >100% IRR uncapped
    irr = _build_cashflows_and_compute_irr(eq, debt, fcf, exit_ev, 0.4, 5)
    assert irr.iloc[0] <= 0.40


# ---------------------------------------------------------------------------
# compute_scenario_irr tests
# ---------------------------------------------------------------------------

def _build_lbo_df(ev=800, ebitda=100, fcf=60, growth=0.06) -> pd.DataFrame:
    """Build df that has already gone through compute_max_debt."""
    from screener.lbo import compute_max_debt
    df = _make_company(ev=ev, ebitda=ebitda, fcf=fcf, revenue_growth=growth)
    df = compute_max_debt(df, _base_cfg())
    return df


def test_scenario_irr_ordering():
    """
    For a typical profitable company: upside IRR > base IRR > downside IRR.

    PE context: This is the fundamental sanity check on the scenario model.
    If it doesn't hold, the growth/exit/FCF deltas are not producing spread.
    """
    df = _build_lbo_df()
    df = compute_scenario_irr(df, _base_cfg())

    # All three must be non-NaN for a healthy company
    assert df["irr_base"].notna().all(), "Base IRR should not be NaN for healthy company"
    assert df["irr_upside"].notna().all(), "Upside IRR should not be NaN"
    assert df["irr_downside"].notna().all(), "Downside IRR should not be NaN"

    # Ordering: upside >= base >= downside
    assert df["irr_upside"].iloc[0] >= df["irr_base"].iloc[0], \
        "Upside IRR must be >= base IRR"
    assert df["irr_base"].iloc[0] >= df["irr_downside"].iloc[0], \
        "Base IRR must be >= downside IRR"


def test_scenario_irr_spread_exists():
    """
    Upside and downside should differ by at least a few percentage points.
    The scenario deltas (+2%/-3% growth, ±0.5x/1x exit, 80% FCF stress) must
    produce meaningful spread — otherwise the scenario model adds no IC value.
    """
    df = _build_lbo_df(ev=600, ebitda=80, fcf=50, growth=0.05)
    df = compute_scenario_irr(df, _base_cfg())

    spread = df["irr_upside"].iloc[0] - df["irr_downside"].iloc[0]
    assert spread >= 0.04, f"Scenario spread too narrow: {spread:.1%} (expected ≥ 4%)"


def test_scenario_irr_capped_at_40pct():
    """IRR should never exceed 40% (hard cap — model noise above this level)."""
    # Very cheap entry + strong FCF should hit the cap
    df = _build_lbo_df(ev=300, ebitda=100, fcf=80, growth=0.10)
    df = compute_scenario_irr(df, _base_cfg())

    assert (df["irr_base"] <= 0.40).all(), "Base IRR must be capped at 40%"
    assert (df["irr_upside"] <= 0.40).all(), "Upside IRR must be capped at 40%"


def test_scenario_irr_negative_ev_gives_nan():
    """Companies with missing or zero EV should produce NaN LBO metrics."""
    df = _make_company(ev=0, ebitda=100, fcf=60)
    df = compute_scenario_irr(df, _base_cfg())
    assert df["irr_base"].isna().all(), "Zero EV should result in NaN IRR"


# ---------------------------------------------------------------------------
# apply_irr_hurdle_penalty tests
# ---------------------------------------------------------------------------

def _make_scored_df(irr_values: list, base_score: float = 80.0) -> pd.DataFrame:
    """Build a scored df for testing IRR hurdle penalty."""
    n = len(irr_values)
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "irr_base": irr_values,
        "pe_score_final": [base_score] * n,
        "red_flags": [""] * n,
    })


def test_hurdle_penalty_negative_irr_zeroes_score():
    """
    irr_base < 0% → pe_score_final must be set to 0.
    PE context: A deal with negative IRR cannot return capital — it should
    never appear in the ranked shortlist regardless of business quality.
    """
    df = _make_scored_df([-0.05])
    df = apply_irr_hurdle_penalty(df)
    assert df["pe_score_final"].iloc[0] == 0.0


def test_hurdle_penalty_sub_10pct_reduces_score():
    """
    0% <= irr_base < 10% → pe_score_final × 0.40 (60% reduction).
    PE context: Below fund hurdle (8-10%), the deal is stretched — it can
    technically complete but won't generate carry. Severe score haircut.
    """
    df = _make_scored_df([0.06], base_score=80.0)
    df = apply_irr_hurdle_penalty(df)
    assert abs(df["pe_score_final"].iloc[0] - 32.0) < 0.01  # 80 × 0.40


def test_hurdle_penalty_above_hurdle_unchanged():
    """irr_base >= 10% → score must be untouched."""
    df = _make_scored_df([0.20], base_score=75.0)
    df = apply_irr_hurdle_penalty(df)
    assert df["pe_score_final"].iloc[0] == 75.0


def test_hurdle_penalty_adds_red_flag():
    """Companies below 10% IRR should receive 'IRR below hurdle' in red_flags."""
    df = _make_scored_df([0.05, 0.20])
    df = apply_irr_hurdle_penalty(df)
    assert "IRR below hurdle" in df["red_flags"].iloc[0]
    assert "IRR below hurdle" not in df["red_flags"].iloc[1]


def test_hurdle_penalty_no_duplicate_flag():
    """Re-applying the penalty should not duplicate the red flag string."""
    df = _make_scored_df([0.05])
    df = apply_irr_hurdle_penalty(df)
    df = apply_irr_hurdle_penalty(df)  # second application
    flag_count = df["red_flags"].iloc[0].count("IRR below hurdle")
    assert flag_count == 1, "Flag should appear exactly once even after re-application"


def test_hurdle_penalty_mixed_portfolio():
    """Correct penalties applied to a mixed portfolio in one call."""
    df = _make_scored_df([-0.10, 0.05, 0.15, 0.30], base_score=60.0)
    df = apply_irr_hurdle_penalty(df)
    assert df["pe_score_final"].iloc[0] == 0.0        # negative → zero
    assert abs(df["pe_score_final"].iloc[1] - 24.0) < 0.01   # 0-10% → ×0.4
    assert df["pe_score_final"].iloc[2] == 60.0       # 15% → unchanged
    assert df["pe_score_final"].iloc[3] == 60.0       # 30% → unchanged


# ---------------------------------------------------------------------------
# Needed import for approx comparison
# ---------------------------------------------------------------------------
try:
    from pytest import approx as pytest_approx
except ImportError:
    def pytest_approx(val, rel=1e-6, abs=1e-6):
        return val
