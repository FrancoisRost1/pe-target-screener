# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run full pipeline (fetches live data from yfinance)
python3 main.py

# Run without refetching (use cached data/raw/companies_raw.csv)
python3 main.py --no-fetch

# Change number of top targets shown
python3 main.py --top 30

# Use a non-default config file
python3 main.py --config path/to/other.yaml

# Run all tests
pytest tests/ -v

# Run a single test file (or single test by name)
pytest tests/test_lbo.py -v
pytest tests/test_lbo.py::test_compute_max_debt -v

# Launch Streamlit dashboard
streamlit run app/streamlit_app.py
```

`python` is not installed on this machine — always use `python3`.

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

### LBO model (`screener/lbo.py` + `screener/lbo_scenarios.py`)

`lbo.py` is the entry point and owns `compute_max_debt`, `compute_lbo_metrics`, and `compute_fcf_yield_on_equity`. The IRR engine itself (`_build_cashflows_and_compute_irr`, `_compute_single_irr`) and all scenario / bridge functions live in `lbo_scenarios.py`. Dependency is strictly one-way (`lbo → lbo_scenarios`) so the import graph stays acyclic — no lazy/inline imports needed.

`_build_cashflows_and_compute_irr` returns NaN for every row when `holding_period < 1` to guard against degenerate timelines.

Key design decisions:
- **No separate interest subtraction**: `debt_repayment_rate=0.40` means 40% of reported FCF sweeps principal. The 60% retained implicitly covers LBO interest + incremental costs. yfinance FCF is already post-interest, so subtracting again would double-count.
- **Max debt is the lesser of two caps**: `max_debt = min(EBITDA × target_leverage, EV × 0.70)`. The 70% LTV cap binds for low-margin businesses where 3.5x EBITDA would imply implausibly thin equity.
- **Equity floored at 20% of EV**: if the leverage math implies a smaller cheque, equity is bumped up to 20% of EV and `max_debt` is recomputed as `EV − equity`. No sub-20% sponsor cheques.
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

Several modules were split to honour the ~150 line rule. The split pairs use a strict one-way dependency (parent → child) so the import graph is a DAG with no cycles — verified by an AST scan. **Edit the file where the function actually lives.**

| Module | Responsibility |
|--------|---------------|
| `loader.py` | yfinance fetch + column validation |
| `cleaner.py` | Type casting, dedup, winsorization. Re-exports `apply_eligibility_filters` from `eligibility.py`. |
| `eligibility.py` | Hard pre-scoring filters (size, margin, sector exclusions) |
| `ratios_utils.py` | Shared low-level helper `_safe_divide()` — separate file so `ratios.py` and `ratios_secondary.py` can both import it without a cycle. |
| `ratios.py` | Core PE metrics (margin, leverage, coverage, conversion). Imports `_safe_divide` from `ratios_utils.py` and re-exports the secondary ratio functions from `ratios_secondary.py`. |
| `ratios_secondary.py` | Secondary ratios: ROIC, EV/EBITDA, growth, capex intensity, FCF yield. Imports `_safe_divide` from `ratios_utils.py` (never from `ratios.py`). |
| `lbo.py` | Top-level orchestrator: `compute_lbo_metrics`, `compute_max_debt`, `compute_fcf_yield_on_equity`. Imports scenario functions from `lbo_scenarios.py` at module level. |
| `lbo_scenarios.py` | IRR engine (`_build_cashflows_and_compute_irr`, `_compute_single_irr`) + scenario IRRs + 3-driver bridge attribution. Does NOT import from `lbo.py`. |
| `scoring.py` | Sector-adjusted percentile scoring → `pe_score`, sub-scores. Re-exports `apply_score_adjustments`, `compute_irr_blended_score`, `apply_irr_hurdle_penalty` from `scoring_adjustments.py`. |
| `scoring_adjustments.py` | Penalty/IRR-blend pipeline that turns `pe_score` into `pe_score_final`. Imports `apply_deal_killer_penalty` from `classifier_rules.py` (not `classifier.py`). |
| `classifier.py` | Top-level orchestrator: debt capacity, red flag detection, valuation penalty |
| `classifier_rules.py` | Pure rule functions: `classify_debt_capacity`, `apply_deal_killer_penalty` |
| `ranking.py` | Sort by `pe_score_final`, assign rank, top-N slice |
| `changelog.py` | Diff vs previous run to detect rank changes |
| `changelog_display.py` | Console rendering of the changelog |
| `exclusion_report.py` | Explain which companies were filtered and why |
| `exporter.py` | CSV + Excel writers (top-N and full universe) |
| `summary.py` | Auto-generate investment memo per company |

### Streamlit app structure (`app/`)

`streamlit_app.py` is an orchestrator only — it builds the sidebar and dispatches to tab modules. Edit the tab files for content changes:

| File | Tab |
|------|-----|
| `sidebar.py` | Filters and global controls |
| `tab_table.py` | Top-N table view |
| `tab_charts.py` | Sector and metric charts |
| `tab_detail.py` | Single-company drill-down |
| `tab_lbo_detail.py` | LBO scenario + IRR bridge view |
| `helpers.py` | Shared formatters and small utilities |

### Data flow columns

Key columns produced at each stage:
- After `ratios.py`: `ebitda_margin`, `roic`, `fcf_conversion`, `ev_to_ebitda`, `net_debt_to_ebitda`, `interest_coverage`, `revenue_growth`, `ocf_margin`, `fcf_yield_ev`
- After `lbo.py`: `max_debt`, `equity_required`, `irr_base`, `irr_upside`, `irr_downside`, `irr_driver_growth`, `irr_driver_deleveraging`, `irr_driver_multiple`
- After `scoring.py`: `score_<metric>` for each metric, `pe_score`, `pe_score_raw`, `quality_score`, `cash_score`, `leverage_score`, `valuation_score`
- After `classifier.py`: `debt_capacity`, `red_flags`, `red_flag_penalty`, `valuation_penalty`, `deal_killer_penalty`
- After `apply_score_adjustments()`: `pe_score_adjusted`, `irr_score`, `pe_score_final`

## Session journal

Full session history (Sessions 1–11, design decisions, top-5 results per run) is in `claude.md` (lowercase).
