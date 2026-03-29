-- ============================================================
-- Financial Performance Analytics Platform — Database Schema
-- Compatible with SQLite (default), MySQL, and PostgreSQL
-- ============================================================

CREATE TABLE IF NOT EXISTS companies (
    company_id   INTEGER PRIMARY KEY,
    company_name TEXT    NOT NULL,
    sector       TEXT    NOT NULL,
    country      TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS financial_records (
    record_id      INTEGER PRIMARY KEY,
    company_id     INTEGER NOT NULL,
    date           DATE    NOT NULL,
    revenue        REAL    NOT NULL,
    expenses       REAL    NOT NULL,
    profit         REAL    NOT NULL,
    operating_cost REAL    NOT NULL,
    assets         REAL    NOT NULL,
    liabilities    REAL    NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

CREATE TABLE IF NOT EXISTS kpi_results (
    kpi_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id     INTEGER NOT NULL,
    date           DATE    NOT NULL,
    profit_margin  REAL,
    operating_ratio REAL,
    revenue_growth REAL,
    expense_ratio  REAL,
    return_on_assets REAL,
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

CREATE TABLE IF NOT EXISTS forecast_results (
    forecast_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id         INTEGER NOT NULL,
    forecast_date      DATE    NOT NULL,
    predicted_revenue  REAL,
    predicted_profit   REAL,
    lower_bound        REAL,
    upper_bound        REAL,
    model_used         TEXT,
    mae                REAL,
    mape               REAL,
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

CREATE TABLE IF NOT EXISTS anomaly_flags (
    anomaly_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id   INTEGER NOT NULL,
    date         DATE    NOT NULL,
    metric       TEXT    NOT NULL,
    value        REAL    NOT NULL,
    z_score      REAL    NOT NULL,
    severity     TEXT    NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_records_company ON financial_records(company_id);
CREATE INDEX IF NOT EXISTS idx_records_date    ON financial_records(date);
CREATE INDEX IF NOT EXISTS idx_kpi_company     ON kpi_results(company_id);
CREATE INDEX IF NOT EXISTS idx_forecast_company ON forecast_results(company_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_company  ON anomaly_flags(company_id);
