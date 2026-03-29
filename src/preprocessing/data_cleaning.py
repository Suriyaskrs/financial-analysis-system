"""
src/preprocessing/data_cleaning.py
------------------------------------
Cleans raw financial data and engineers derived KPI features.
All transformations are pure functions (input df → output df) for testability.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_financial_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for financial_records DataFrame.
    Steps:
      1. Parse and sort dates
      2. Remove duplicates
      3. Handle missing/negative values
      4. Normalize currency (all values in thousands £/$ for consistency)
    """
    df = df.copy()

    # 1. Date parsing and sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["company_id", "date"]).reset_index(drop=True)

    # 2. Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["company_id", "date"])
    if len(df) < before:
        logger.warning(f"Removed {before - len(df)} duplicate rows.")

    # 3. Handle missing values — forward-fill within company group, then 0
    numeric_cols = ["revenue", "expenses", "profit", "operating_cost", "assets", "liabilities"]
    df[numeric_cols] = (
        df.groupby("company_id")[numeric_cols]
        .transform(lambda x: x.ffill().bfill())
        .fillna(0)
    )

    # 4. Clamp negative assets/liabilities (data quality safeguard)
    df["assets"]      = df["assets"].clip(lower=0)
    df["liabilities"] = df["liabilities"].clip(lower=0)

    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived financial metrics to the DataFrame.
    All ratios are per-row calculations; growth rates use prior quarter within company.
    
    New columns added:
        profit_margin      — net profit as % of revenue
        operating_ratio    — operating cost / revenue (efficiency metric)
        expense_ratio      — total expenses / revenue
        revenue_growth     — QoQ revenue growth rate
        profit_growth      — QoQ profit growth rate
        debt_to_assets     — liabilities / assets (leverage)
        return_on_assets   — profit / assets (ROA)
        equity             — assets - liabilities
    """
    df = df.copy()
    df = df.sort_values(["company_id", "date"])

    # Guard against division by zero
    rev = df["revenue"].replace(0, np.nan)
    ast = df["assets"].replace(0, np.nan)

    df["profit_margin"]   = (df["profit"] / rev).round(4)
    df["operating_ratio"] = (df["operating_cost"] / rev).round(4)
    df["expense_ratio"]   = (df["expenses"] / rev).round(4)
    df["debt_to_assets"]  = (df["liabilities"] / ast).round(4)
    df["return_on_assets"]= (df["profit"] / ast).round(4)
    df["equity"]          = (df["assets"] - df["liabilities"]).round(2)

    # QoQ growth rates per company
    df["revenue_growth"] = (
        df.groupby("company_id")["revenue"]
        .pct_change()
        .round(4)
    )
    df["profit_growth"] = (
        df.groupby("company_id")["profit"]
        .pct_change()
        .round(4)
    )

    # Replace inf/-inf from divide-by-zero edge cases
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def add_rolling_metrics(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Add rolling averages (trailing 4-quarter / 1-year window) per company.
    Useful for smoothing seasonal volatility.
    """
    df = df.copy().sort_values(["company_id", "date"])

    for metric in ["revenue", "profit", "profit_margin"]:
        col_name = f"{metric}_rolling_{window}q"
        df[col_name] = (
            df.groupby("company_id")[metric]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
            .round(2)
        )

    return df


def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Single entry-point: clean → engineer features → rolling metrics."""
    df = clean_financial_records(df)
    df = engineer_features(df)
    df = add_rolling_metrics(df)
    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df
