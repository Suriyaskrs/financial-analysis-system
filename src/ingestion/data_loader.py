"""
src/ingestion/data_loader.py
-----------------------------
Handles all data ingestion:
  - Load CSV files into the SQLite database
  - Validate incoming data (types, nulls, ranges)
  - Extensible: plug in API fetchers (yfinance, Alpha Vantage) when online

Used by: app.py, pipeline.py
"""

import os
import sqlite3
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "database", "finance.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "database", "schema.sql")
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_data")


# ── Database helpers ──────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection, creating the DB and schema if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist yet."""
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_financial_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean a financial_records DataFrame.
    Returns cleaned df; raises ValueError if critical columns are missing.
    """
    required = ["company_id", "date", "revenue", "expenses", "profit",
                "operating_cost", "assets", "liabilities"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.drop_duplicates(subset=["company_id", "date"])
    df = df.dropna(subset=["company_id", "date", "revenue"])
    df["date"] = pd.to_datetime(df["date"])

    # Numeric coercion
    num_cols = ["revenue", "expenses", "profit", "operating_cost", "assets", "liabilities"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    after = len(df)
    if before != after:
        logger.warning(f"Dropped {before - after} invalid rows during validation.")

    return df


def validate_companies(df: pd.DataFrame) -> pd.DataFrame:
    required = ["company_id", "company_name", "sector", "country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.drop_duplicates(subset=["company_id"])
    return df


# ── CSV Loaders ───────────────────────────────────────────────────────────────

def load_csv_data(companies_path: str = None, records_path: str = None) -> dict:
    """
    Load companies and financial_records CSVs into the SQLite database.
    Uses default paths if none provided.
    Returns dict with row counts.
    """
    companies_path = companies_path or os.path.join(RAW_DATA_DIR, "companies.csv")
    records_path   = records_path   or os.path.join(RAW_DATA_DIR, "financial_records.csv")

    df_companies = validate_companies(pd.read_csv(companies_path))
    df_records   = validate_financial_records(pd.read_csv(records_path))

    conn = get_connection()
    df_companies.to_sql("companies",          conn, if_exists="replace", index=False)
    df_records.to_sql(  "financial_records",  conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    logger.info(f"Loaded {len(df_companies)} companies, {len(df_records)} records.")
    return {"companies": len(df_companies), "records": len(df_records)}


# ── Read helpers (used by analytics/dashboard) ────────────────────────────────

def get_all_companies() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM companies ORDER BY company_id", conn)
    conn.close()
    return df


def get_financial_records(company_id: int = None) -> pd.DataFrame:
    conn = get_connection()
    if company_id:
        df = pd.read_sql(
            "SELECT * FROM financial_records WHERE company_id = ? ORDER BY date",
            conn, params=(company_id,)
        )
    else:
        df = pd.read_sql("SELECT * FROM financial_records ORDER BY company_id, date", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_kpi_results(company_id: int = None) -> pd.DataFrame:
    conn = get_connection()
    if company_id:
        df = pd.read_sql(
            "SELECT * FROM kpi_results WHERE company_id = ? ORDER BY date",
            conn, params=(company_id,)
        )
    else:
        df = pd.read_sql("SELECT * FROM kpi_results ORDER BY company_id, date", conn)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_forecast_results(company_id: int = None) -> pd.DataFrame:
    conn = get_connection()
    if company_id:
        df = pd.read_sql(
            "SELECT * FROM forecast_results WHERE company_id = ? ORDER BY forecast_date",
            conn, params=(company_id,)
        )
    else:
        df = pd.read_sql("SELECT * FROM forecast_results ORDER BY company_id, forecast_date", conn)
    conn.close()
    if not df.empty:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


def get_anomaly_flags(company_id: int = None) -> pd.DataFrame:
    conn = get_connection()
    if company_id:
        df = pd.read_sql(
            "SELECT * FROM anomaly_flags WHERE company_id = ? ORDER BY date",
            conn, params=(company_id,)
        )
    else:
        df = pd.read_sql("SELECT * FROM anomaly_flags ORDER BY company_id, date", conn)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ── API Fetcher stub (expandable when network is available) ───────────────────

def fetch_api_data(symbol: str, source: str = "yfinance") -> pd.DataFrame:
    """
    Fetch financial data from an external API.
    Currently returns a helpful message when offline.
    
    HOW TO EXPAND:
    ──────────────
    When you have internet access, install:
        pip install yfinance alpha-vantage
    
    Then replace the body of this function with:
    
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.financials.T.reset_index()
        df.columns = ["date"] + list(df.columns[1:])
        return df
    
    Or for Alpha Vantage:
        from alpha_vantage.fundamentaldata import FundamentalData
        fd = FundamentalData(key=os.getenv("ALPHA_VANTAGE_KEY"))
        data, _ = fd.get_income_statement_quarterly(symbol)
        return pd.DataFrame(data)
    """
    raise ConnectionError(
        f"API fetch unavailable offline. Symbol '{symbol}' requested from '{source}'.\n"
        "To enable: pip install yfinance alpha-vantage and set ALPHA_VANTAGE_KEY env var.\n"
        "See src/ingestion/data_loader.py → fetch_api_data() for expansion instructions."
    )


def store_in_database(df: pd.DataFrame, table: str) -> int:
    """Generic function to store a DataFrame into any DB table."""
    conn = get_connection()
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.commit()
    count = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn).iloc[0]["n"]
    conn.close()
    return count
