# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (fetches live data from yfinance)
python main.py

# Run without refetching (use cached data/raw/companies_raw.csv)
python main.py --no-fetch

# Change number of top targets shown
python main.py --top 30

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_lbo.py -v

# Launch Streamlit dashboard
streamlit run app/streamlit_app.py
```

## Architecture

The pipeline runs in a strict linear sequence defined in `main.py::run()`:

```
fetch → clean → ratios → winsorize → lbo → [snapshot] → eligibility → score → classify → adjust → rank → export
```

The pre-filter snapshot (`df_pre_filter`) is saved before eligibility filters so the exclusion report can explain which companies were dropped and why.

### Scoring pipeline (most complex part)

`scoring.py` has three distinct scoring stages, each building on the previous:

1. **`score_universe_sector_adjusted()`** → produces `pe_score`
   - Each metric is percentile-ranked universe-wide (`u_rank`) and within sector (`s_rank`)
   - Inversion applied independently to each rank before blending (critical for correctness)
   - Blended: `0.60 × s_rank + 0.40 × u_rank`, fall back to `u_rank` for sectors with < 3 companies
   - NaN filled with 50 (neutral — no penalty for missing data)

2. **`apply_score_adjustments()`** → produces `pe_score_adjusted`, then `pe_score_final`
   - Step 1: additive penalties (red flags, expensive valuation) → `pe_score_adjusted`
   - Step 2: deal killer — multiplicative (`×0.5`) or zero for negative IRR → modifies `pe_score_adjusted`
   - Step 3: `compute_irr_blended_score()` → `pe_score_final = 0.60 × pe_score_adjusted + 0.40 × irr_score`
   - Step 4: `apply_irr_hurdle_penalty()` → collapses `pe_score_final` for IRR < 10%

3. **Final ranking** is by `pe_score_final`, not `pe_score_adjusted`.

### LBO model (`screener/lbo.py`)

Key design decisions:
- **No separate interest subtraction**: `debt_repayment_rate=0.40` means 40% of reported FCF sweeps principal. The 60% retained implicitly covers LBO interest + incremental costs. yfinance FCF is already post-interest, so subtracting again would double-count.
- **Entry multiple floor**: raw `ev_to_ebitda` clipped to 6x minimum in IRR calculations. `ev_to_ebitda < 6x` is set to NaN in `cleaner.py` for scoring (no signal), but the floor prevents "free multiple expansion" artifacts in IRR math.
- **IRR capped at 40%**: `clip(-0.50, 0.40)` on all scenarios. Unrealistic IRRs above this indicate data issues, not real returns.
- **Scenario asymmetry**: downside = −3% growth, −1x exit, 80% FCF conversion; upside = +2% growth, +0.5x exit.

### Config-driven design

All thresholds, weights, and LBO assumptions live in `config.yaml`. Nothing is hardcoded. Key sections:
- `weights:` — scoring metric weights (must sum to 1.0; `ebitda_growth` is intentionally 0)
- `invert_metrics:` — metrics where lower = better (EV/EBITDA, Net Debt/EBITDA)
- `lbo:` — exit multiple cap (8x), leverage (3.5x), holding period (5yr), repayment rate (40%)
- `eligibility:` — hard filters run before scoring (min revenue $100M, min EBITDA $10M, excludes Financials/RE/Utilities)

### Module responsibilities (one file = one job)

| Module | Responsibility |
|--------|---------------|
| `loader.py` | yfinance fetch + column validation |
| `cleaner.py` | Type casting, dedup, winsorization, eligibility filters |
| `ratios.py` | 11 PE-relevant metrics (all via `_safe_divide()`) |
| `lbo.py` | IRR estimation: base/up/down scenarios + IRR bridge attribution |
| `scoring.py` | Sector-adjusted percentile scoring → `pe_score_final` |
| `classifier.py` | Debt capacity (High/Med/Low), red flags, deal killer penalty |
| `ranking.py` | Sort by `pe_score_final`, assign rank, top-N slice |
| `changelog.py` | Diff vs previous run to detect rank changes |
| `exclusion_report.py` | Explain which companies were filtered and why |
| `summary.py` | Auto-generate investment memo per company |

### Data flow columns

Key columns produced at each stage:
- After `ratios.py`: `ebitda_margin`, `roic`, `fcf_conversion`, `ev_to_ebitda`, `net_debt_to_ebitda`, `interest_coverage`, `revenue_growth`, `ocf_margin`, `fcf_yield_ev`
- After `lbo.py`: `max_debt`, `equity_required`, `irr_base`, `irr_upside`, `irr_downside`, `irr_driver_growth`, `irr_driver_deleveraging`, `irr_driver_multiple`
- After `scoring.py`: `score_<metric>` for each metric, `pe_score`, `pe_score_raw`, `quality_score`, `cash_score`, `leverage_score`, `valuation_score`
- After `classifier.py`: `debt_capacity`, `red_flags`, `red_flag_penalty`, `valuation_penalty`, `deal_killer_penalty`
- After `apply_score_adjustments()`: `pe_score_adjusted`, `irr_score`, `pe_score_final`

## Session journal

Full session history (Sessions 1–11, design decisions, top-5 results per run) is in `claude.md` (lowercase).
