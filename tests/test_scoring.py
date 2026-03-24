"""Tests for scoring engine."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from screener.scoring import score_universe

def test_score_range():
    """All scores should be between 0 and 100."""
    df = pd.DataFrame({
        "company": [f"Co{i}" for i in range(10)],
        "ebitda_margin": np.random.uniform(0.05, 0.40, 10),
        "roic": np.random.uniform(0.05, 0.30, 10),
        "fcf_conversion": np.random.uniform(0.3, 0.9, 10),
        "ocf_margin": np.random.uniform(0.05, 0.30, 10),
        "net_debt_to_ebitda": np.random.uniform(0, 5, 10),
        "interest_coverage": np.random.uniform(1, 15, 10),
        "revenue_growth": np.random.uniform(-0.05, 0.20, 10),
        "ebitda_growth": np.random.uniform(-0.05, 0.20, 10),
        "ev_to_ebitda": np.random.uniform(6, 20, 10),
    })
    cfg = {"weights": {
        "ebitda_margin": 0.15, "roic": 0.10, "fcf_conversion": 0.15,
        "ocf_margin": 0.10, "net_debt_to_ebitda": 0.10, "interest_coverage": 0.10,
        "revenue_growth": 0.10, "ebitda_growth": 0.05, "ev_to_ebitda": 0.15
    }, "invert_metrics": ["net_debt_to_ebitda", "ev_to_ebitda"]}
    df = score_universe(df, cfg)
    assert df["pe_score"].between(0, 100).all()

def test_weights_sum_doesnt_matter():
    """Even if weights don't sum to 1, result should be normalized."""
    df = pd.DataFrame({
        "company": ["A", "B", "C"],
        "ebitda_margin": [0.30, 0.15, 0.10],
        "ev_to_ebitda": [8.0, 12.0, 16.0],
    })
    cfg = {"weights": {"ebitda_margin": 5, "ev_to_ebitda": 5},
           "invert_metrics": ["ev_to_ebitda"]}
    df = score_universe(df, cfg)
    # A should rank #1 (high margin, low EV/EBITDA)
    assert df.loc[0, "pe_score"] > df.loc[2, "pe_score"]
