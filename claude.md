# claude.md — Private Equity Target Screener

> Dual purpose: instructions for Claude Code + living journal of the project.
> Update this file after every significant session or decision.

---

## 1. What this project is

A Python-based screening engine that identifies potential LBO/buyout candidates from a universe of public companies.  
It fetches real financial data (via `yfinance`), computes PE-relevant metrics, scores each company using a weighted model, classifies debt capacity, flags red flags, and surfaces a ranked shortlist.  
Final deliverable: a modular Python pipeline + interactive Streamlit dashboard.

**GitHub repo name:** `pe-target-screener`

---

## 2. Instructions for Claude Code

### Philosophy
- Always write modular code. One file = one responsibility.
- Never put business logic in `main.py`. It only orchestrates.
- All weights, thresholds, and assumptions live in `config.yaml`. Never hardcode them.
- Handle edge cases: division by zero, missing values, negative EBITDA, negative EV.
- Every function must have a docstring explaining the PE rationale, not just what the code does.

### Module responsibilities
| File | Role |
|---|---|
| `screener/loader.py` | Fetch data from yfinance, validate columns |
| `screener/cleaner.py` | Type casting, NaN handling, outlier detection |
| `screener/ratios.py` | Compute all financial ratios |
| `screener/scoring.py` | Normalize metrics, apply weights → pe_score |
| `screener/classifier.py` | Debt capacity (High/Medium/Low) + red flags |
| `screener/ranking.py` | Sort, assign rank, build top-N list |
| `screener/exporter.py` | Output to CSV and Excel |
| `screener/summary.py` | Auto-generate investment memo snippet per company |
| `app/streamlit_app.py` | Full interactive dashboard |

### Coding standards
- Use `pandas` for all data manipulation
- Use `numpy` only where needed (percentile, clip)
- Config loaded once via `utils/config_loader.py`, passed as dict
- All ratio functions take a `pd.DataFrame`, return a `pd.DataFrame`
- Scoring uses **percentile ranking** (not min-max) — more robust to outliers
- For metrics where lower = better (Net Debt/EBITDA, EV/EBITDA), invert the rank
- `red_flags` column = pipe-separated string of flag labels, empty string if none

### Data source
- Primary: `yfinance` — real public company financials
- Universe: ~100 mid/large cap companies across 6–8 sectors (Industrials, Healthcare, Technology, Consumer Staples, Business Services, Packaging, Infrastructure, Energy)
- Ticker list stored in `data/sample/universe_tickers.csv`
- Raw fetched data saved to `data/raw/companies_raw.csv`
- Processed + scored data saved to `data/processed/companies_scored.csv`

### Error handling rules
- Division by zero → return `np.nan`, never raise
- If EBITDA ≤ 0, company is flagged but not dropped (unless filter_ineligible=True)
- If a required column is missing after fetch, log warning and skip that metric
- Never break the pipeline on a single company's bad data

---

## 3. Project Roadmap

### Phase 1 — Foundation ✅
- [x] Repo structure
- [x] claude.md
- [x] README.md
- [x] config.yaml
- [x] requirements.txt
- [x] .gitignore

### Phase 2 — Data ✅
- [x] Build universe ticker list (90 real tickers, 8 sectors)
- [x] loader.py — fetch from yfinance with graceful error handling
- [x] cleaner.py — type casting, dedup, data quality flags, eligibility filters

### Phase 3 — Analytics ✅
- [x] ratios.py — 11 metrics with PE rationale docstrings
- [x] scoring.py — percentile ranking + weighted scoring + 4 sub-scores
- [x] classifier.py — debt capacity (High/Medium/Low) + red flag detection

### Phase 4 — Output ✅
- [x] ranking.py — sort, assign rank, top-N
- [x] exporter.py — CSV + Excel
- [x] summary.py — auto investment memo generator

### Phase 5 — Dashboard ✅
- [x] streamlit_app.py — full interactive UI (filters, charts, radar, drill-down, download)

### Phase 6 — Quality ✅
- [x] tests/test_ratios.py — 8 ratio tests
- [x] tests/test_scoring.py — 2 scoring tests
- [x] tests/test_cleaner.py — 3 cleaner + eligibility tests
- [x] tests/test_classifier.py — debt capacity test
- [x] All 14 tests passing ✅
- [x] notebooks/exploratory_analysis.ipynb

### Phase 7 — GitHub Push ✅
- [x] Push to GitHub
- [x] Run `python main.py` with real yfinance data
- [x] Validate outputs with real data
- [ ] Screenshot dashboard for README

---

## 4. Metrics used and PE rationale

| Metric | Formula | Why PE cares | Higher = Better |
|---|---|---|---|
| EBITDA Margin | EBITDA / Revenue | Core profitability proxy | ✅ |
| ROIC | NOPAT / Invested Capital | Capital efficiency | ✅ |
| FCF Conversion | FCF / EBITDA | Cash quality | ✅ |
| Net Debt / EBITDA | (Debt - Cash) / EBITDA | Leverage headroom | ❌ |
| Interest Coverage | EBIT / Interest Expense | Debt service capacity | ✅ |
| EV / EBITDA | EV / EBITDA | Entry valuation | ❌ |
| Revenue Growth | YoY % | Growth trajectory | ✅ |
| OCF Margin | Operating CF / Revenue | Cash profitability | ✅ |

---

## 5. Scoring model

Weights defined in `config.yaml`. Default:

| Metric | Weight |
|---|---|
| EBITDA Margin | 15% |
| ROIC | 10% |
| FCF Conversion | 15% |
| OCF Margin | 10% |
| Net Debt / EBITDA | 10% |
| Interest Coverage | 10% |
| Revenue Growth | 10% |
| EBITDA Growth | 5% |
| EV / EBITDA | 15% |

Method: percentile rank per metric (0→100), invert where lower = better, then weighted sum.

---

## 6. Debt Capacity Classification

| Label | Criteria |
|---|---|
| High | EBITDA margin > 20% AND FCF conversion > 70% AND Net Debt/EBITDA < 2.5x AND Interest Coverage > 5x |
| Medium | Intermediate profile |
| Low | EBITDA margin < 10% OR Net Debt/EBITDA > 4x OR Interest Coverage < 2.5x |

---

## 7. Session Journal

### Session 1 — Project initialization
**Date:** 2026-03-24  
**What was done:**
- Defined full project scope and architecture
- Decided on yfinance as data source (real public data, free, GitHub-friendly)
- Created full repo directory structure
- Created claude.md (this file)
- Created README.md skeleton

**Key decisions:**
- Option 3 (Streamlit dashboard) as final target
- Percentile ranking for scoring (more robust than min-max)
- Real data via yfinance over fictive dataset
- All config externalised in YAML

**Next session:**
- Write config.yaml
- Write requirements.txt
- Build universe ticker list
- Write loader.py

---

### Session 2 — Full pipeline build + test validation
**Date:** 2026-03-24  
**What was done:**
- Wrote all 8 screener modules (loader, cleaner, ratios, scoring, classifier, ranking, exporter, summary)
- Wrote main.py orchestrator with CLI args (`--no-fetch`, `--top`, `--config`)
- Wrote full Streamlit dashboard (streamlit_app.py) with:
  - KPI cards, top-N table, histogram, pie chart, bubble chart
  - Company drill-down with radar sub-score chart
  - Adjustable weight sliders, sector/debt capacity filters
  - CSV download button
- Wrote 14 unit tests across ratios, scoring, cleaner, classifier
- All 14 tests passing ✅ (validated without network access)
- Created exploratory_analysis.ipynb with 6 analysis sections
- Added .gitignore
- Network disabled in this environment — yfinance fetch will work locally

**Key decisions:**
- `_safe_divide()` utility in ratios.py handles all zero/NaN cases centrally
- Scoring fills NaN metric scores with 50 (neutral) — avoids penalizing companies with missing data
- Debt capacity uses "2 of 3 low triggers" logic — single bad metric doesn't disqualify
- Streamlit re-runs full pipeline on weight/filter change (fast enough at 100 companies)
- `.gitignore` excludes `data/raw/`, `data/processed/`, `outputs/` — users run the pipeline themselves

**Issues encountered & fixed:**
- yfinance `interest_expense` can return negative values → added `.abs()` in `compute_interest_coverage`
- yfinance `capex` is negative by convention → added `abs()` in loader and ratios
- `_safe_get` must handle both missing index keys and `pd.isna` separately

**Next session:**
- Push to GitHub
- Run `python main.py` locally with real yfinance data
- Validate that real data flows cleanly through pipeline
- Add screenshot of Streamlit dashboard to README

*Last updated: 2026-03-24 — Session 2*

---

### Session 3 — Pipeline validation + first commit
**Date:** 2026-03-24
**What was done:**
- Created missing runtime directories: `data/raw/`, `data/processed/`, `outputs/`
- Installed all requirements via `pip3 install -r requirements.txt`
- Ran `pytest tests/ -v` → 13/13 tests passing ✅
- Fixed Python 3.9 incompatibility in `classifier.py`: `bool | None` return type annotation (union syntax requires 3.10+) → removed annotation, logic unchanged
- Ran `python main.py` with live yfinance data:
  - 80/90 tickers fetched successfully (10 skipped — no income statement data)
  - 78 companies scored after eligibility filter
  - Debt capacity breakdown: High: 8 | Medium: 64 | Low: 6
  - Red flags detected in 33 companies
  - Outputs written: `top_targets.csv`, `companies_scored.csv`, `screening_summary.xlsx`
- Launched Streamlit dashboard → confirmed live at http://localhost:8501
- Initialized git repo and made first commit

**Top 5 PE Targets (real data, 2026-03-24):**
| Rank | Company | Sector | PE Score | Debt Capacity |
|---|---|---|---|---|
| 1 | Cal-Maine Foods | Consumer Staples | 88.7 | Medium |
| 2 | Stride Inc | Specialty | 77.2 | High |
| 3 | Graco Inc | Industrials | 73.1 | High |
| 4 | Copart Inc | Business Services | 71.9 | Medium |
| 5 | Microsoft Corp | Technology | 71.2 | Medium |

**Issues encountered & fixed:**
- `bool | None` union type annotation in `classifier.py:109` not supported on Python 3.9 → removed return type annotation

**Next session:**
- Take Streamlit dashboard screenshot for README
- Push to GitHub (`git push origin main`)

*Last updated: 2026-03-24 — Session 3*

---

### Session 5 — IRR model fixes, FCF yield on EV, LBO UI, memo quality
**Date:** 2026-03-24
**What was done:**

**FIX 1 — IRR proxy math corrected:**
- Exit multiple now `min(entry EV/EBITDA, config cap)` — never exits above entry multiple
- IRR set to NaN if `equity_required <= 0` OR `exit_equity <= 0` (both conditions)
- This fixed Cal-Maine Foods IRR showing N/A (entry EV/EBITDA was below exit cap → now correctly uses entry multiple)

**FIX 2 — Equity Required floor enforced:**
- `max_debt = min(EBITDA × leverage, EV × 70%)` — debt never exceeds 70% of EV
- `equity_required = max(EV - max_debt, EV × 20%)` — minimum 20% equity floor
- EV missing/zero/negative → all LBO metrics set to NaN
- Eliminated phantom $0m equity cheque results

**FIX 3 — Memo quality: no more empty bull/bear:**
- If `bear` list empty: fallback to valuation comment (>10x), growth comment (<5%), or float comment
- If `bull` list empty: fallback to "defensive business model with stable cash generation"

**FIX 4 — New ratio: FCF Yield on EV:**
- `fcf_yield_ev = FCF / EV` — real cash return vs. price paid
- Added to `ratios.py` + `compute_all_ratios()`
- Winsorized to [-20%, +50%]
- Weight in config: 5% (from `ebitda_growth` which moved to 0%)

**FIX 5 — Dashboard: LBO assumptions sidebar (Section 3):**
- New sliders: Exit Multiple (6x–16x), Holding Period (3–7yr), Target Leverage (2x–6x)
- Overrides flow into `run_cfg["lbo"]` → pipeline recalculates IRR live
- Filters section renumbered to Section 4

**FIX 6 — Dashboard: Top Opportunities panel:**
- 🟢 Best LBO Candidates: debt_capacity=High OR quality_score>60, irr_proxy>15%, top 25% score
- 🔴 Watch List: companies with red flags sorted by flag count descending

**FIX 7 — Dashboard: IRR column renamed to "IRR Est." + assumptions caption**
- Caption shows hold period, exit multiple, leverage used — sets expectations clearly

**Top 5 PE Targets (post-fix, 2026-03-24):**
| Rank | Company | Score | IRR Est. | Equity Required | Debt Capacity |
|---|---|---|---|---|---|
| 1 | Cal-Maine Foods | 91.5 | ~46% | $762m | Medium |
| 2 | Stride Inc | 81.3 | ~30% | $1,717m | High |
| 3 | ExlService Holdings | 70.8 | ~20% | $3,240m | Medium |
| 4 | Graco Inc | 68.6 | ~-1% | $10,507m | High |
| 5 | Cognizant Technology | 66.4 | ~18% | $12,619m | Medium |

**Validation:** All top 10 companies have valid equity_required (>$1m) and non-NaN IRR proxy ✅

**Files changed:** `screener/lbo.py`, `screener/ratios.py`, `screener/cleaner.py`, `screener/summary.py`, `config.yaml`, `app/streamlit_app.py`

*Last updated: 2026-03-24 — Session 5*

---

### Session 6 — Deal Killer Logic, sector norm fix, LBO breakdown, FCF yield cap, quadrant zones
**Date:** 2026-03-24
**What was done:**

**FIX 1 — 🧮 LBO Deal Breakdown expander in drill-down:**
- Added `cfg` parameter to `_show_company_detail(row, cfg)` in `streamlit_app.py`
- Updated call site in `main()` to pass `run_cfg`
- Expander computes full LBO waterfall inline from row: EV → max debt → equity cheque → exit EBITDA → exit EV → debt repaid → debt remaining → exit equity → MOIC + IRR
- Displayed in 3-column grid with assumption caption (hold period, exit multiple, debt repayment rate)

**FIX 2 — FCF Yield on Equity cap:**
- `screener/lbo.py`: clipped `fcf_yield_equity` from `clip(-0.5, 1.0)` → `clip(-0.5, 0.50)`
- Added `fmt_fcf_yield_equity()` formatter to `streamlit_app.py`: returns ">50%" if val ≥ 0.50
- Used in drill-down Row 2 replacing `fmt_pct`

**FIX 3 — Deal Killer Logic:**
- Added `apply_deal_killer_penalty(df)` to `screener/classifier.py`
- Penalties: irr_proxy < 0 → -20pts; irr_proxy < -0.20 → -35pts; equity_required > EV × 90% → -10pts
- Also appends "Negative IRR" or "Deal math challenged" to `red_flags`
- Updated `apply_score_adjustments()` in `screener/scoring.py` to import and call this function
- `pe_score_adjusted` now includes `deal_killer_penalty` in addition to red_flag and valuation penalties
- 21 companies penalized in current universe

**FIX 4 — Sector normalization rewrite:**
- `score_universe_sector_adjusted()` in `screener/scoring.py` rewrote to:
  - Apply inversion independently to u_rank AND s_rank before blending (not after)
  - Fall back to u_rank for sectors with < 3 companies
  - Skip zero-weight metrics entirely
  - Track `raw_u_ranks` separately for `pe_score_raw`
- Visible impact: Graco (negative IRR) dropped from top 5; new entrants Civitas Resources, Jack Henry, AptarGroup, Sonoco

**FIX 5 — Deal Quadrant colored background zones:**
- Added 4 `rect` shapes behind scatter bubbles via `fig3.update_layout(shapes=zone_shapes)`:
  - 🟢 Sweet Spot (low EV, high quality): rgba(46,204,113, 0.08)
  - 🟡 Quality Premium + Value Trap: rgba(243,156,18, 0.08)
  - 🔴 Avoid (high EV, low quality): rgba(231,76,60, 0.08)
- Zones extend 10% beyond data range for visual polish

**Top 5 PE Targets (post Session 6 fixes, 2026-03-24):**
| Rank | Company | Sector | Adj. Score | IRR Est. | Debt Capacity |
|---|---|---|---|---|---|
| 1 | Cal-Maine Foods | Consumer Staples | 91.5 | ~46% | Medium |
| 2 | Stride Inc | Specialty | 81.3 | ~30% | High |
| 3 | ExlService Holdings | Business Services | 70.8 | ~20% | Medium |
| 4 | Cognizant Technology | Technology | 66.4 | ~18% | Medium |
| 5 | Civitas Resources | Energy | 65.0 | ~39% | Medium |

**Validation:**
- All top 10 have positive IRR ✅
- Graco (~-1% IRR) correctly penalized and dropped from top 10 ✅
- 21 companies penalized by deal killer logic ✅
- 54 companies scored after eligibility filters ✅

**Files changed:** `screener/classifier.py`, `screener/scoring.py`, `screener/lbo.py`, `app/streamlit_app.py`

*Last updated: 2026-03-24 — Session 6*

---

### Session 7 — Realistic amortization, scenario IRR, deal killer rewrite, table upgrade
**Date:** 2026-03-24
**What was done:**

**FIX 1 — Realistic annual debt amortization in LBO model:**
- Replaced lump-sum `debt_repaid = FCF × years × rate` with year-by-year loop
- Each year: `repay = min(FCF × rate, remaining_debt)` — no "negative debt" possible
- Extracted `_amortize_debt(max_debt, annual_fcf, rate, years)` helper in `lbo.py`
- Extracted `_compute_irr_scenario(df, ...)` helper to avoid code duplication across scenarios
- Impact: ExlService IRR dropped 20.4% → 16.0% (more realistic — carries $1.1B debt at exit)

**FIX 2 — Exit multiple default + scenario awareness:**
- `config.yaml`: `exit_multiple` changed from 12.0 → 10.0 (more conservative)
- Added `exit_multiple_min: 8.0` and `exit_multiple_max: 14.0` for documentation
- Streamlit sidebar default updated from 12x → 10x

**FIX 3 — Deal killer: multiplicative penalty replacing additive:**
- `apply_deal_killer_penalty()` in `classifier.py` now modifies `pe_score_adjusted` directly:
  - `irr < 0` but `>= -15%`: multiply pe_score_adjusted × 0.5
  - `irr < -15%`: set pe_score_adjusted = 0 (deal is dead)
  - equity > 90% of EV: subtract 10pts (still bloated)
- Added `_append_flag()` helper to avoid duplicate red_flag entries
- `apply_score_adjustments()` in `scoring.py` restructured: additive penalties first → pe_score_adjusted set → then deal killer applied to that result
- 29 companies penalized (was 21 with additive approach)

**FIX 4 — Scenario Analysis: Base / Upside / Downside IRR:**
- Added `compute_scenario_irr(df, cfg)` to `lbo.py`
- Base: growth as-is, exit cap = config (10x)
- Upside: growth +2%, exit cap + 2x (12x)
- Downside: growth -2%, exit cap - 2x (8x)
- Each scenario uses full amortization loop via `_compute_irr_scenario()`
- Stores `irr_base`, `irr_upside`, `irr_downside` — `irr_proxy` unchanged for backwards compat
- Note: companies with growth already at ±15% clip show identical scenarios (by design)

**FIX 5 — Scenario IRR display in dashboard drill-down:**
- Added `fmt_irr_delta()` helper: formats ±X% difference vs base
- Row 2 of drill-down: "IRR Proxy" → "Base IRR" (uses `irr_base`)
- LBO Deal Breakdown expander enhanced:
  - 3-scenario metric row with delta arrows (Downside / Base / Upside)
  - Horizontal bar chart (red/blue/green) with 20% hurdle rate dashed line at 20%
  - Height: 200px

**FIX 6 — Top Targets table updated:**
- `irr_proxy` replaced with `irr_base` + `irr_downside` columns
- Column headers: "IRR (Base)" and "IRR (Down)"
- Table caption updated to explain both columns

**Top 5 PE Targets (post Session 7, 2026-03-24):**
| Rank | Company | Sector | Adj. Score | IRR Base | IRR Down | IRR Up | Debt Capacity |
|---|---|---|---|---|---|---|---|
| 1 | Cal-Maine Foods | Consumer Staples | 91.5 | ~46% | ~46% | ~46% | Medium |
| 2 | Stride Inc | Specialty | 81.3 | ~30% | ~30% | ~30% | High |
| 3 | ExlService Holdings | Business Services | 70.8 | ~16% | ~8% | ~22% | Medium |
| 4 | Cognizant Technology | Technology | 66.4 | ~18% | ~15% | ~21% | Medium |
| 5 | Civitas Resources | Energy | 65.0 | ~39% | ~39% | ~39% | Medium |

**Validation:**
- All top 10 have positive IRR base ✅
- Amortization: debt_remaining NOT always 0 (Stride: $1.3B remaining, Cognizant: $11B remaining) ✅
- 29 companies penalized by deal killer (vs 21 in Session 6 — stricter logic) ✅
- Cal-Maine/Stride/Civitas scenario IRR identical: revenue growth already at 15% clip ceiling ✅ (expected)

**Files changed:** `screener/lbo.py`, `screener/classifier.py`, `screener/scoring.py`, `screener/ranking.py`, `config.yaml`, `app/streamlit_app.py`

*Last updated: 2026-03-24 — Session 7*

---

### Session 8 — Scenario spread fix, IRR bridge waterfall, IRR-blended score, market context
**Date:** 2026-03-24
**What was done:**

**FIX 1 — Scenario IRR: real spread between Base / Upside / Downside:**
- Rewrote `_compute_single_irr(df, cfg, growth_delta, exit_multiple_delta)` in `lbo.py`
- Key changes:
  1. Growth clips to `(-0.10, +0.20)` — wider range ensures ±2% delta produces real spread
  2. Exit multiple: `(entry.clip(upper=cap) + exit_multiple_delta).clip(lower=4.0)` — additive after clipping, so upside allows modest expansion even for cheap names
  3. Scenarios: upside = +2% growth + 1x exit; downside = -2% growth - 1x exit
- Result: 96.2% of companies have upside-downside spread ≥ 5% (target: 80%) ✅
- Median spread: 9.7% across universe
- Note: ~4% of companies still show identical scenarios because entry EV/EBITDA < 3x causes all scenarios to hit the 4x floor AND growth > 20% hits the ceiling

**FIX 2 — IRR Bridge: decompose IRR into 3 value drivers:**
- Added `compute_irr_bridge(df, cfg)` to `lbo.py`
- Three drivers (counterfactual attribution):
  1. `irr_driver_growth`: irr_base - irr_if_revenue_growth=0 (growth contribution)
  2. `irr_driver_deleveraging`: irr_base - irr_if_no_debt_repayment (paydown contribution)
  3. `irr_driver_multiple`: irr_base - irr_if_no_exit_cap (0 when entry < cap, negative when cap compresses)
- Implemented via cfg/df overrides to `_compute_single_irr` (no extra function signatures needed)
- Called in `compute_lbo_metrics()` after `compute_scenario_irr()`
- IRR bridge for top 3 (2026-03-24):
  - Cal-Maine: Growth +31%, Deleverage +4%, Multiple 0% (entry < cap, no compression)
  - Stride: Growth +26%, Deleverage +3%, Multiple 0%
  - Civitas: Growth +33%, Deleverage +2%, Multiple 0%

**FIX 3 — IRR bridge waterfall chart in dashboard drill-down:**
- Added vertical bar chart in LBO Deal Breakdown expander (after scenario bar chart)
- Bars: EBITDA Growth / Deleveraging / Multiple Δ — green if +, red if −
- Title: "IRR = ~XX% | Value Driver Attribution"
- Height: 220px, Y-axis auto-scaled to ±max driver value + 40% headroom

**FIX 4 — Market context alert when median IRR < 12%:**
- Added `st.info()` block after KPI cards in `main()` of streamlit_app.py
- Shown only when `irr_median < 0.12` — signals to user that public market valuations are rich
- Displays holding period, exit multiple, leverage from current sidebar settings

**FIX 5 — IRR-blended final score (`pe_score_final`):**
- Added `compute_irr_blended_score(df)` to `scoring.py`, called from `apply_score_adjustments()`
- Formula: `pe_score_final = 0.60 × pe_score_adjusted + 0.40 × irr_score`
  where `irr_score = percentile rank of irr_base (0–100)`
- Double-penalizes broken deals (deal killer already collapsed pe_score_adjusted)
- `ranking.py`: now sorts by `pe_score_final`
- `main.py` summary table: shows pe_score_final instead of pe_score_adjusted
- Dashboard: `score_col` uses pe_score_final, table header = "Final Score"
- Drill-down: "Raw → Adj → Final (IRR-blended)" caption

**Top 5 PE Targets (post Session 8, 2026-03-24):**
| Rank | Company | Sector | Final Score | IRR Base | IRR Down | IRR Up | Debt Capacity |
|---|---|---|---|---|---|---|---|
| 1 | Cal-Maine Foods | Consumer Staples | 94.9 | ~86% | ~86% | ~86% | Medium |
| 2 | Stride Inc | Specialty | 86.6 | ~33% | ~26% | ~40% | High |
| 3 | Civitas Resources | Energy | 78.3 | ~70% | ~70% | ~70% | Medium |
| 4 | ExlService Holdings | Business Services | 76.5 | ~16% | ~11% | ~21% | Medium |
| 5 | Cognizant Technology | Technology | 75.4 | ~18% | ~10% | ~26% | Medium |

**Validation:**
- 96.2% of companies have upside-downside spread ≥ 5% ✅
- All top 10 have positive irr_base ✅
- IRR bridge growth driver dominates for high-growth companies ✅
- Multiple driver = 0% for top companies (they trade below exit cap — cheap) ✅
- Ranked by pe_score_final ✅

**Files changed:** `screener/lbo.py`, `screener/scoring.py`, `screener/ranking.py`, `main.py`, `app/streamlit_app.py`

*Last updated: 2026-03-24 — Session 8*

---

### Session 9 — IRR cap 40%, realistic LBO defaults, return attribution labels, assumptions table
**Date:** 2026-03-24
**What was done:**

**FIX 1 — Cap IRR to realistic PE ranges:**
- `config.yaml`: `target_leverage` 4.0 → 3.5x; `exit_multiple` 10.0 → 8.0x; min/max updated to 7.0/12.0
- `lbo.py` `_compute_single_irr()`: hard cap changed from `clip(-0.5, 1.0)` → `clip(-0.50, 0.40)`
- `lbo.py` `compute_scenario_irr()`: explicit `.clip(-0.50, 0.40)` on each scenario column
- Sidebar sliders: Exit Multiple default 10.0→8.0, range 6→14; Target Leverage default 4.0→3.5
- Results: Cal-Maine capped at 40% (was 86%), Civitas at 40% (was 70%), Stride at 31% ✅
- Viable deals (positive IRR) median: 9.5% — in 8-18% target range ✅
- No company above 40% ✅
- Note: full 54-company universe median = -6.2% (expected — most public cos don't pencil as LBOs at 8x exit/3.5x leverage). Market context alert fires correctly.

**FIX 2 — IRR Bridge label improvements:**
- Section header: "IRR Bridge — Value Drivers" → "IRR Decomposition (PE Return Attribution)"
- Driver labels: "EBITDA Growth" → "EBITDA Growth contribution"; "Deleveraging" → "Debt paydown (deleveraging)"; "Multiple Δ" → "Multiple contraction / expansion vs entry"
- Chart title: "IRR = X | Drivers" → "Base IRR: X — attribution by value driver"

**FIX 3 — LBO assumptions summary table at top of expander:**
- Added markdown table with 5 parameters at top of 🧮 LBO Deal Breakdown expander
- Shows: Holding period, Entry multiple (actual EV/EBITDA), Exit multiple cap, Target leverage, FCF→debt repayment rate
- Replaced the old `st.caption()` line with the table
- Added `st.divider()` after the table for visual separation
- Updated `_show_company_detail(row, cfg: dict = None)` signature (default=None for safety)
- Updated call site to `_show_company_detail(row, cfg=run_cfg)` (keyword arg)

**Top 5 PE Targets (post Session 9, 2026-03-24):**
| Rank | Company | Sector | Final Score | IRR Base | IRR Down | IRR Up | Debt Capacity |
|---|---|---|---|---|---|---|---|
| 1 | Cal-Maine Foods | Consumer Staples | 94.5 | ~40% | ~40% | ~40% | Medium |
| 2 | Stride Inc | Specialty | 86.6 | ~31% | ~24% | ~37% | High |
| 3 | Civitas Resources | Energy | 78.7 | ~40% | ~40% | ~40% | Medium |
| 4 | ExlService Holdings | Business Services | 76.5 | ~10% | ~4% | ~15% | Medium |
| 5 | Cognizant Technology | Technology | 76.2 | ~17% | ~9% | ~23% | Medium |

**Validation:**
- No company above 40% IRR ✅
- Viable deals median: 9.5% (in 8-18% range) ✅
- Market context alert fires (full universe median < 12%) ✅
- Deal killer now penalizes 36 companies (up from 29 — more deals don't pencil at 8x/3.5x) ✅

**Files changed:** `config.yaml`, `screener/lbo.py`, `app/streamlit_app.py`

*Last updated: 2026-03-24 — Session 9*
