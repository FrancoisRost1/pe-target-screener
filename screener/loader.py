"""
loader.py, Data fetching and validation

Fetches financial data for a universe of public companies via yfinance.
Saves raw output to data/raw/companies_raw.csv.

PE context: We need income statement, balance sheet, and cash flow data
to compute the 8 core LBO screening metrics.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_universe(filepath: str) -> list:
    """
    Load the list of tickers from the universe CSV file.
    Skips comment lines starting with '#'.
    """
    df = pd.read_csv(filepath, comment="#")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].str.strip()
    logger.info(f"Loaded universe: {len(df)} tickers from {filepath}")
    return df.to_dict(orient="records")


def _safe_get(series, keys):
    """Try multiple key names, return first match or None."""
    for key in keys:
        if key in series.index and pd.notna(series[key]):
            return float(series[key])
    return None


def fetch_company_data(ticker_info: dict):
    """
    Fetch annual financial data for a single company from yfinance.
    PE context: annual data reflects the medium-term view a PE fund takes.
    """
    ticker = ticker_info["ticker"]
    try:
        stock = yf.Ticker(ticker)

        income_all = stock.financials
        if income_all is None or income_all.empty:
            logger.warning(f"{ticker}: No income statement data")
            return None
        income = income_all.iloc[:, 0]

        balance = stock.balance_sheet
        if balance is None or balance.empty:
            logger.warning(f"{ticker}: No balance sheet data")
            return None
        balance = balance.iloc[:, 0]

        cashflow = stock.cashflow
        if cashflow is None or cashflow.empty:
            logger.warning(f"{ticker}: No cash flow data")
            return None
        cashflow = cashflow.iloc[:, 0]

        # Prior year for growth calculations
        prior_revenue = None
        prior_ebitda = None
        if income_all.shape[1] >= 2:
            prior = income_all.iloc[:, 1]
            prior_revenue = _safe_get(prior, ["Total Revenue"])
            p_ebitda = _safe_get(prior, ["EBITDA"])
            p_ebit = _safe_get(prior, ["EBIT", "Operating Income"])
            p_da = _safe_get(prior, ["Reconciled Depreciation", "Depreciation And Amortization"])
            prior_ebitda = p_ebitda if p_ebitda else (p_ebit + p_da if p_ebit and p_da else None)

        info = stock.info or {}

        revenue = _safe_get(income, ["Total Revenue"])
        ebit = _safe_get(income, ["EBIT", "Operating Income"])
        da = _safe_get(income, ["Reconciled Depreciation", "Depreciation And Amortization", "Depreciation"])
        ebitda = _safe_get(income, ["EBITDA"]) or (ebit + da if ebit and da else None)
        interest_expense = _safe_get(income, ["Interest Expense"])
        net_income = _safe_get(income, ["Net Income"])

        total_debt = _safe_get(balance, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"])
        cash = _safe_get(balance, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"])
        total_assets = _safe_get(balance, ["Total Assets"])
        total_equity = _safe_get(balance, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
        invested_capital = _safe_get(balance, ["Invested Capital"])

        capex_raw = _safe_get(cashflow, ["Capital Expenditure", "Purchase Of Property Plant And Equipment"])
        capex = abs(capex_raw) if capex_raw is not None else None
        operating_cf = _safe_get(cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        fcf = _safe_get(cashflow, ["Free Cash Flow"]) or (operating_cf - capex if operating_cf and capex else None)

        return {
            "ticker": ticker,
            "company": ticker_info.get("company", info.get("longName", ticker)),
            "sector": ticker_info.get("sector", info.get("sector", "Unknown")),
            "country": ticker_info.get("country", info.get("country", "Unknown")),
            "revenue": revenue,
            "prior_revenue": prior_revenue,
            "ebitda": ebitda,
            "prior_ebitda": prior_ebitda,
            "ebit": ebit,
            "net_income": net_income,
            "interest_expense": interest_expense,
            "da": da,
            "total_debt": total_debt,
            "cash": cash,
            "total_assets": total_assets,
            "total_equity": total_equity,
            "invested_capital": invested_capital,
            "capex": capex,
            "operating_cash_flow": operating_cf,
            "free_cash_flow": fcf,
            "enterprise_value": info.get("enterpriseValue"),
            "market_cap": info.get("marketCap"),
        }

    except Exception as e:
        logger.error(f"{ticker}: Failed, {e}")
        return None


def fetch_universe(universe: list, save_path: str = None) -> pd.DataFrame:
    """Fetch data for all tickers. Skips failures gracefully."""
    results = []
    total = len(universe)
    for i, ticker_info in enumerate(universe, 1):
        logger.info(f"[{i}/{total}] Fetching {ticker_info['ticker']}...")
        data = fetch_company_data(ticker_info)
        if data:
            results.append(data)
    df = pd.DataFrame(results)
    logger.info(f"Fetched: {len(df)}/{total} companies")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved to {save_path}")
    return df
