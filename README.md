# Private Equity Target Screener

A Python-based screening engine that identifies potential LBO/buyout acquisition candidates from a universe of public companies.  
Built to reflect the analytical logic of a private equity analyst — not just financial ratios, but structured judgment about leverage capacity, cash quality, and entry valuation.

---

## Why this project exists

A PE fund doesn't look for "good companies." It looks for companies that can:
- support significant leverage without breaking
- generate stable, predictable free cash flow
- be acquired at a reasonable entry multiple
- offer operational improvement potential

This screener quantifies those criteria, scores each company, and surfaces the most attractive LBO candidates from a broad universe.

---

## Features

- **Real financial data** fetched via `yfinance` (public companies, no API key needed)
- **8 core PE metrics** computed from raw financials
- **Weighted scoring engine** — configurable via `config.yaml`, no code change needed
- **Debt capacity classification** — High / Medium / Low based on PE-style rules
- **Red flag detection** — automated warning signals per company
- **Investment memo snippet** — auto-generated summary for top candidates
- **CSV + Excel export** — ready for further analysis
- **Interactive Streamlit dashboard** — filters, sliders, drill-down, downloadable output

---

## Metrics

| Metric | Formula | PE Rationale |
|---|---|---|
| EBITDA Margin | EBITDA / Revenue | Core profitability proxy |
| ROIC | NOPAT / Invested Capital | Capital efficiency |
| FCF Conversion | FCF / EBITDA | Cash quality |
| OCF Margin | Operating CF / Revenue | Cash profitability |
| Net Debt / EBITDA | (Debt − Cash) / EBITDA | Leverage headroom available |
| Interest Coverage | EBIT / Interest Expense | Debt service capacity |
| EV / EBITDA | EV / EBITDA | Entry valuation attractiveness |
| Revenue Growth | YoY % | Growth trajectory |

---

## Repository structure

```
pe-target-screener/
├── README.md
├── claude.md               # Project journal + Claude Code instructions
├── requirements.txt
├── config.yaml             # All weights, thresholds, assumptions
├── main.py                 # Orchestrator — runs the full pipeline
├── data/
│   ├── raw/                # Raw data fetched from yfinance
│   ├── processed/          # Scored and enriched dataset
│   └── sample/             # Universe ticker list
├── screener/
│   ├── loader.py           # Data fetching and validation
│   ├── cleaner.py          # Type casting, NaN handling, outliers
│   ├── ratios.py           # All financial ratio calculations
│   ├── scoring.py          # Percentile-based weighted scoring
│   ├── classifier.py       # Debt capacity + red flag logic
│   ├── ranking.py          # Final ranking and top-N selection
│   ├── exporter.py         # CSV and Excel output
│   └── summary.py          # Auto-generated investment memo snippets
├── app/
│   └── streamlit_app.py    # Interactive dashboard
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
│   ├── test_ratios.py
│   ├── test_scoring.py
│   └── test_cleaner.py
└── outputs/
    ├── top_targets.csv
    └── screening_summary.xlsx
```

---

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the screening pipeline
python main.py

# Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## Example output

| Rank | Company | Sector | EBITDA Margin | FCF Conv. | Net Debt/EBITDA | Interest Cov. | PE Score | Debt Capacity |
|---|---|---|---|---|---|---|---|---|
| 1 | Firm A | Industrials | 28.1% | 81% | 1.2x | 9.5x | 91.4 | High |
| 2 | Firm B | Healthcare | 24.7% | 76% | 0.8x | 12.0x | 88.9 | High |
| 3 | Firm C | Business Svcs | 21.3% | 68% | 2.1x | 6.2x | 84.1 | Medium |

---

## Configuration

All scoring weights and thresholds are defined in `config.yaml`:

```yaml
weights:
  ebitda_margin: 0.15
  roic: 0.10
  fcf_conversion: 0.15
  ...

thresholds:
  debt_capacity_high_coverage: 5.0
  debt_capacity_high_leverage: 2.5
  ...
```

No code change needed to adjust the model.

---

## Roadmap

- [x] Modular Python pipeline
- [x] Real data via yfinance
- [x] Weighted scoring engine
- [x] Debt capacity classifier
- [x] Red flag detection
- [x] CSV/Excel export
- [ ] Streamlit dashboard (in progress)
- [ ] Sector-specific scoring profiles
- [ ] Monte Carlo scenario screening
- [ ] DCF-based floor valuation overlay

---

## Screenshots

![PE Target Screener Dashboard](assets/dashboard_screenshot.png)

## Stack

- Python 3.10+
- pandas, numpy
- yfinance
- pyyaml
- openpyxl
- streamlit
- matplotlib / plotly
- pytest
