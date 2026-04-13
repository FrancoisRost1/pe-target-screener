"""
exporter.py, Output generation (CSV and Excel)

Exports the full scored dataset and top-N shortlist.
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

PCT_COLS = ["ebitda_margin", "roic", "fcf_conversion", "ocf_margin",
            "revenue_growth", "ebitda_growth", "capex_to_revenue"]
MULT_COLS = ["net_debt_to_ebitda", "ev_to_ebitda", "interest_coverage"]


def export_results(df_full: pd.DataFrame, df_top: pd.DataFrame, cfg: dict):
    """Export full dataset and top-N list to CSV and optionally Excel."""
    out_cfg = cfg.get("output", {})
    out_dir = Path(out_cfg.get("output_dir", "outputs/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_cfg.get("export_csv", True):
        full_path = out_dir / "companies_scored.csv"
        df_full.to_csv(full_path, index=False)
        logger.info(f"Full results saved: {full_path}")

        top_path = out_dir / "top_targets.csv"
        df_top.to_csv(top_path, index=False)
        logger.info(f"Top targets saved: {top_path}")

    if out_cfg.get("export_excel", True):
        excel_path = out_dir / "screening_summary.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_top.to_excel(writer, sheet_name="Top Targets", index=False)
            df_full.to_excel(writer, sheet_name="Full Universe", index=False)
        logger.info(f"Excel saved: {excel_path}")
