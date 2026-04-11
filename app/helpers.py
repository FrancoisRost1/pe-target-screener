"""
helpers.py — Shared formatting and utility functions for the Streamlit dashboard.
"""

import numpy as np
import pandas as pd


def fmt_pct(val):
    """Format a decimal as a percentage string, or — if NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{val:.1%}"


def fmt_irr(val):
    """Format IRR proxy as ~XX% or — if NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"~{val:.0%}"


def fmt_irr_delta(val, base):
    """Format IRR delta vs base case as +X% / -X% or None if either is NaN."""
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [val, base]):
        return None
    diff = val - base
    return f"{diff:+.0%}"


def fmt_mult(val):
    """Format a multiple as X.Xx or — if NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{val:.1f}x"


def fmt_score(val):
    """Format a score as X.X or — if NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{val:.1f}"


def fmt_millions(val):
    """Format a raw dollar value as $XXXm."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"${val / 1_000_000:.0f}m"


def fmt_fcf_yield_equity(val):
    """Format FCF yield on equity — cap display at >50% to flag outliers."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    if val >= 0.50:
        return ">50%"
    return f"{val:.1%}"


def debt_capacity_color(val):
    """Return a short text indicator for debt capacity classification."""
    labels = {"High": "HIGH", "Medium": "MED", "Low": "LOW"}
    return labels.get(val, "N/A")
