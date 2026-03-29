"""
src/analytics/kpi_calculator.py
---------------------------------
Computes all KPIs, trend statistics, and cross-company comparisons.
Results are stored back to the database kpi_results table.

Key KPIs:
  - Revenue Growth Rate (QoQ and YoY)
  - Profit Margin
  - Operating Margin
  - Expense Ratio
  - Return on Assets (ROA)
  - Debt-to-Assets Ratio
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.data_loader import get_connection, get_financial_records, get_all_companies

logger = logging.getLogger(__name__)


# ── Core KPI Computations ─────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all KPI columns for a financial_records DataFrame.
    Returns a summary DataFrame with one row per (company_id, date).
    """
    df = df.copy().sort_values(["company_id", "date"])
    rev = df["revenue"].replace(0, np.nan)
    ast = df["assets"].replace(0, np.nan)

    df["profit_margin"]    = (df["profit"] / rev).round(4)
    df["operating_ratio"]  = (df["operating_cost"] / rev).round(4)
    df["expense_ratio"]    = (df["expenses"] / rev).round(4)
    df["return_on_assets"] = (df["profit"] / ast).round(4)
    df["revenue_growth"]   = df.groupby("company_id")["revenue"].pct_change().round(4)

    return df


def save_kpis_to_db(df_kpi: pd.DataFrame) -> None:
    """Persist KPI results to the kpi_results table."""
    cols = ["company_id", "date", "profit_margin", "operating_ratio",
            "revenue_growth", "expense_ratio", "return_on_assets"]
    save_df = df_kpi[cols].copy()
    save_df["date"] = save_df["date"].astype(str)

    conn = get_connection()
    save_df.to_sql("kpi_results", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    logger.info(f"Saved {len(save_df)} KPI rows to database.")


# ── Summary Statistics ────────────────────────────────────────────────────────

def company_summary(df: pd.DataFrame, companies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate KPIs per company into a single summary row.
    Returns a DataFrame sorted by average profit margin (descending).
    """
    df = compute_kpis(df)

    summary = df.groupby("company_id").agg(
        total_revenue      = ("revenue",        "sum"),
        avg_profit_margin  = ("profit_margin",  "mean"),
        avg_expense_ratio  = ("expense_ratio",  "mean"),
        avg_roa            = ("return_on_assets","mean"),
        avg_revenue_growth = ("revenue_growth", "mean"),
        latest_profit      = ("profit",         "last"),
        quarters           = ("date",           "count"),
    ).reset_index()

    summary = summary.merge(companies_df[["company_id","company_name","sector","country"]],
                            on="company_id", how="left")

    for col in ["avg_profit_margin","avg_expense_ratio","avg_roa","avg_revenue_growth"]:
        summary[col] = (summary[col] * 100).round(2)
    summary["total_revenue"]   = summary["total_revenue"].round(0)
    summary["latest_profit"]   = summary["latest_profit"].round(0)

    return summary.sort_values("avg_profit_margin", ascending=False).reset_index(drop=True)


def yoy_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Year-over-Year revenue growth per company.
    Compares same quarter across years.
    """
    df = df.copy()
    df["year"]    = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    yoy = df.merge(
        df[["company_id","year","quarter","revenue"]].rename(columns={"revenue":"prev_revenue","year":"prev_year"}),
        left_on=["company_id","year","quarter"],
        right_on=["company_id","prev_year","quarter"],
        how="left"
    )
    # prev_year should be current year - 1
    mask = yoy["year"] == yoy["prev_year"] + 1
    yoy.loc[~mask, "prev_revenue"] = np.nan
    yoy["yoy_growth"] = ((yoy["revenue"] - yoy["prev_revenue"]) / yoy["prev_revenue"]).round(4)

    return yoy[["company_id","date","year","quarter","revenue","prev_revenue","yoy_growth"]]


def sector_comparison(df: pd.DataFrame, companies_df: pd.DataFrame) -> pd.DataFrame:
    """Compare average profit margin and revenue growth by sector."""
    df = compute_kpis(df)
    df = df.merge(companies_df[["company_id","sector"]], on="company_id", how="left")

    sector = df.groupby("sector").agg(
        avg_profit_margin  = ("profit_margin",  "mean"),
        avg_expense_ratio  = ("expense_ratio",  "mean"),
        avg_roa            = ("return_on_assets","mean"),
        total_revenue      = ("revenue",        "sum"),
        companies          = ("company_id",     "nunique"),
    ).reset_index()

    for col in ["avg_profit_margin","avg_expense_ratio","avg_roa"]:
        sector[col] = (sector[col] * 100).round(2)

    return sector.sort_values("avg_profit_margin", ascending=False)


def compare_profit(df: pd.DataFrame, company_a_id: int, company_b_id: int) -> pd.DataFrame:
    """Side-by-side quarterly profit comparison between two companies."""
    a = df[df["company_id"] == company_a_id][["date","profit"]].rename(columns={"profit": f"profit_co{company_a_id}"})
    b = df[df["company_id"] == company_b_id][["date","profit"]].rename(columns={"profit": f"profit_co{company_b_id}"})
    return a.merge(b, on="date", how="outer").sort_values("date")


def latest_kpi_snapshot(df: pd.DataFrame, companies_df: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent quarter KPIs for every company — used in dashboard header."""
    df = compute_kpis(df)
    latest = df.sort_values("date").groupby("company_id").last().reset_index()
    latest = latest.merge(companies_df[["company_id","company_name","sector"]], on="company_id")

    cols_pct = ["profit_margin","operating_ratio","expense_ratio","return_on_assets","revenue_growth"]
    for c in cols_pct:
        if c in latest.columns:
            latest[c] = (latest[c] * 100).round(2)

    return latest


# ── SQL Analytics Queries (for interview demonstration) ───────────────────────

SQL_QUERIES = {
    "top_revenue_companies": """
        SELECT c.company_name, c.sector,
               ROUND(SUM(f.revenue), 0)       AS total_revenue,
               ROUND(AVG(f.profit / NULLIF(f.revenue,0)) * 100, 2) AS avg_profit_margin_pct
        FROM financial_records f
        JOIN companies c ON f.company_id = c.company_id
        GROUP BY c.company_id
        ORDER BY total_revenue DESC
    """,
    "quarterly_revenue_trend": """
        SELECT c.company_name, f.date,
               ROUND(f.revenue, 0) AS revenue,
               ROUND(f.profit, 0)  AS profit
        FROM financial_records f
        JOIN companies c ON f.company_id = c.company_id
        ORDER BY c.company_id, f.date
    """,
    "expense_ratio_by_quarter": """
        SELECT c.company_name,
               strftime('%Y', f.date) AS year,
               ROUND(AVG(f.expenses / NULLIF(f.revenue, 0)) * 100, 2) AS avg_expense_ratio_pct
        FROM financial_records f
        JOIN companies c ON f.company_id = c.company_id
        GROUP BY c.company_id, year
        ORDER BY c.company_id, year
    """,
    "profit_ranking_latest": """
        SELECT c.company_name, c.sector,
               f.date, ROUND(f.profit, 0) AS latest_profit,
               ROUND(f.profit / NULLIF(f.revenue,0) * 100, 2) AS profit_margin_pct
        FROM financial_records f
        JOIN companies c ON f.company_id = c.company_id
        WHERE f.date = (SELECT MAX(date) FROM financial_records WHERE company_id = f.company_id)
        ORDER BY latest_profit DESC
    """,
    "yoy_growth_window": """
        SELECT c.company_name,
               strftime('%Y', f.date) AS year,
               ROUND(SUM(f.revenue), 0) AS annual_revenue
        FROM financial_records f
        JOIN companies c ON f.company_id = c.company_id
        GROUP BY c.company_id, year
        ORDER BY c.company_id, year
    """,
}


def run_sql_query(query_name: str) -> pd.DataFrame:
    """Execute a named SQL query and return results as DataFrame."""
    if query_name not in SQL_QUERIES:
        raise ValueError(f"Unknown query '{query_name}'. Available: {list(SQL_QUERIES.keys())}")
    conn = get_connection()
    df = pd.read_sql(SQL_QUERIES[query_name], conn)
    conn.close()
    return df


def run_full_analytics() -> dict:
    """Run the full analytics pipeline. Returns dict of result DataFrames."""
    df_records   = get_financial_records()
    df_companies = get_all_companies()

    df_kpi = compute_kpis(df_records)
    save_kpis_to_db(df_kpi)

    return {
        "kpi_data":         df_kpi,
        "company_summary":  company_summary(df_records, df_companies),
        "sector_comparison":sector_comparison(df_records, df_companies),
        "yoy_growth":       yoy_growth(df_records),
        "latest_snapshot":  latest_kpi_snapshot(df_records, df_companies),
    }
