# Financial Performance Analytics & Forecasting Platform
<!--
> **Built for HSBC FinOps / Financial Analyst Internship Applications**
-->
> Python · Pandas · Scikit-learn · SQLite · Matplotlib · Streamlit

---

## What This Project Does

A full end-to-end financial analytics system that:

- Ingests quarterly financial data (CSV or live APIs when online)
- Cleans and validates data through an automated pipeline
- Calculates 5 core KPIs per company per quarter
- Forecasts the next 4 quarters of revenue and profit using 3 models
- Detects financial anomalies using Z-score, IQR, and Isolation Forest
- Visualises everything in an interactive 8-page Streamlit dashboard
- Exports polished Excel reports (7 sheets) and CSV data

**Dataset:** 5 companies · 5 sectors · 24 quarters (2019–2024) · 120 records

---

## Project Structure

```
financial_analytics_system/
│
├── app.py                          ← Streamlit dashboard (run this)
├── pipeline.py                     ← Master pipeline runner
├── requirements.txt
├── README.md
│
├── data/
│   └── raw_data/
│       ├── generate_data.py        ← Synthetic data generator
│       ├── companies.csv           ← 5 companies across 5 sectors
│       └── financial_records.csv  ← 120 quarterly records
│
├── database/
│   ├── schema.sql                  ← SQLite schema (5 tables)
│   └── finance.db                  ← Auto-created on first run
│
├── src/
│   ├── ingestion/
│   │   └── data_loader.py          ← CSV loader, DB helpers, API stub
│   │
│   ├── preprocessing/
│   │   └── data_cleaning.py        ← Cleaning + feature engineering
│   │
│   ├── analytics/
│   │   └── kpi_calculator.py       ← KPIs, SQL queries, sector analysis
│   │
│   ├── forecasting/
│   │   └── forecasting_engine.py   ← 3 forecast models + auto-selector
│   │
│   ├── anomaly_detection/
│   │   └── anomaly_detector.py     ← Z-score, IQR, Isolation Forest
│   │
│   ├── visualization/
│   │   └── charts.py               ← 7 chart types (matplotlib)
│   │
│   └── reporting/
│       └── report_generator.py     ← Excel (7 sheets) + CSV export
│
└── reports/                        ← Auto-generated charts + reports
```

---

## What YOU Need to Do (Setup Steps)

### Step 1 — Install Python

Make sure you have **Python 3.9 or higher**.

Check your version:
```bash
python --version
```

Download from https://python.org if needed.

---

### Step 2 — Clone or Copy the Project

If you received this as a zip, unzip it to a folder of your choice.

If you have git:
```bash
git clone <your-repo-url>
cd financial_analytics_system
```

---

### Step 3 — Create a Virtual Environment (Recommended)

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

---

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, streamlit, openpyxl.

**Note:** All core features work fully offline. Optional packages (Prophet, ARIMA, yfinance) can be added later — see the Expansion section below.

---

### Step 5 — Run the Full Pipeline (First Time)

This builds the database, runs all analytics, forecasting, anomaly detection, and saves reports:

```bash
python pipeline.py
```

Expected output:
```
08:19:15  INFO  ── STEP 1: Data Ingestion
08:19:16  INFO  Loaded: 5 companies, 120 records
08:19:16  INFO  ── STEP 2: KPI Analytics
                TechVentures Ltd   51.9% profit margin
                NovaPharma Inc     50.1% profit margin
                ...
08:19:19  INFO  ── STEP 3: Forecasting  (20 rows, 4 quarters × 5 companies)
08:19:20  INFO  ── STEP 4: Anomaly Detection  (52 flags, 1 SEVERE)
08:19:21  INFO  ── STEP 5: Report Generation
08:19:26  INFO  ✓ Full pipeline complete in ~11s
```

---

### Step 6 — Launch the Dashboard

```bash
streamlit run app.py
```

Then open your browser at: **http://localhost:8501**

---

## Running Individual Steps

You can run any single pipeline step:

```bash
python pipeline.py --step ingest      # reload data from CSV
python pipeline.py --step analytics   # recalculate KPIs
python pipeline.py --step forecast    # regenerate forecasts
python pipeline.py --step anomaly     # rerun anomaly detection
python pipeline.py --step report      # regenerate Excel/CSV reports
python pipeline.py --step charts      # save all chart PNGs
```

---

## Dashboard Pages

| Page | What It Shows |
|------|--------------|
| Overview | KPI summary cards, revenue trend, company table, heatmap |
| Revenue & Trends | Per-company deep-dive, expense pie chart, full KPI table |
| KPI Analysis | Profit margin trends, cross-company comparison |
| Forecasting | 4-quarter forecast chart with confidence bands, model metrics |
| Anomaly Detection | Timeline with flagged points, anomaly table by company |
| Sector Comparison | Grouped bar chart, sector KPI table |
| SQL Explorer | 5 pre-built SQL queries + custom SQL editor |
| Reports & Export | Download Excel report, CSV, or any raw dataset |

---

## KPIs Calculated

| KPI | Formula | What It Tells You |
|-----|---------|-------------------|
| Profit Margin | profit / revenue | How much of each £ of revenue becomes profit |
| Operating Ratio | operating_cost / revenue | Cost efficiency — lower is better |
| Expense Ratio | expenses / revenue | Total cost burden relative to revenue |
| Return on Assets | profit / assets | How well assets are generating profit |
| Revenue Growth | (current - prev) / prev | Quarter-on-quarter growth rate |

---

## Forecasting Models

Three models run automatically. The one with the lowest MAPE is used:

| Model | How It Works | Best For |
|-------|-------------|----------|
| Seasonal Trend Decomposition | Separates trend + quarterly seasonal factors | Data with clear Q1/Q4 patterns |
| Polynomial Regression | Degree-2 curve fit on time index | Accelerating or decelerating trends |
| Holt's Double Exponential Smoothing | Level + trend smoothing (alpha=0.3, beta=0.1) | Stable trends with gradual changes |

**Evaluation metrics:** MAE (Mean Absolute Error), RMSE, MAPE (Mean Absolute Percentage Error)

Typical MAPE on this dataset: **2.5% – 7%** (good for quarterly financial forecasting)

---

## Anomaly Detection

Three methods run in parallel and results are deduplicated:

| Method | Technique | Flags |
|--------|----------|-------|
| Z-Score | \|z\| > 2.0 = WARNING, \|z\| > 3.0 = SEVERE | Per-metric per-company outliers |
| IQR Fence | Outside Q1 - 1.5×IQR or Q3 + 1.5×IQR | Distribution-based outliers |
| Isolation Forest | sklearn, contamination=8% | Multivariate anomalies |
| Debt Ratio | liabilities/assets > 0.70 | Leverage risk flags |

**Known anomaly in dataset:** GlobalRetail Q2 2022 — 45% expense spike injected deliberately. The system correctly flags it as SEVERE (z-score: -3.99).

---

## Database Tables

```sql
companies          -- company_id, name, sector, country
financial_records  -- revenue, expenses, profit, operating_cost, assets, liabilities
kpi_results        -- profit_margin, operating_ratio, expense_ratio, revenue_growth, ROA
forecast_results   -- predicted_revenue, predicted_profit, confidence intervals, model, MAE, MAPE
anomaly_flags      -- metric, value, z_score, severity (WARNING / SEVERE)
```

All tables are queryable from the SQL Explorer page in the dashboard.

---

## Expanding the Project (When You Have Internet)

### Add Prophet (Better Forecasting)

```bash
pip install prophet
```

Then open `src/forecasting/forecasting_engine.py` and scroll to the **EXPANSION GUIDE** at the bottom of the file. Copy the `_prophet_forecast()` function into the file and add `"prophet"` to the `best_forecast()` models dict. That's it — the auto-selector will pick Prophet if it has the lowest MAPE.

---

### Add ARIMA

```bash
pip install statsmodels
```

Same file, same EXPANSION GUIDE — copy `_arima_forecast()` and add `"arima"` to the models dict.

---

### Connect to Yahoo Finance (Live Stock Data)

```bash
pip install yfinance
```

Open `src/ingestion/data_loader.py` → `fetch_api_data()`. The expansion instructions are written inside the function docstring. Replace the body with:

```python
import yfinance as yf
ticker = yf.Ticker(symbol)
df = ticker.quarterly_financials.T.reset_index()
return df
```

---

### Connect to Alpha Vantage (Fundamental Data)

```bash
pip install alpha-vantage
```

Set your API key:
```bash
# On Mac/Linux:
export ALPHA_VANTAGE_KEY="your_key_here"

# On Windows:
set ALPHA_VANTAGE_KEY=your_key_here
```

Get a free key at: https://www.alphavantage.co/support/#api-key

Same function in `data_loader.py` — the ARIMA/Alpha Vantage example is already written in the docstring.

---

### Use a Real Database (MySQL / PostgreSQL)

The project uses SQLite by default (zero setup). To switch:

1. Install the driver: `pip install mysqlclient` or `pip install psycopg2`
2. Open `src/ingestion/data_loader.py`
3. Change `DB_PATH` to a SQLAlchemy connection string:
   ```python
   DB_PATH = "mysql+mysqlclient://user:password@localhost/finance_db"
   ```
4. Replace `sqlite3.connect(DB_PATH)` with `sqlalchemy.create_engine(DB_PATH).connect()`

---

### Add PDF Reports

```bash
pip install fpdf2
```

Add a `generate_pdf_report()` function in `src/reporting/report_generator.py` using the fpdf2 library. A skeleton is easy to build — see https://py-fpdf2.readthedocs.io

---

## Using Your Own Data

To load your own financial CSV instead of the synthetic data:

1. Your CSV must have these columns:
   ```
   company_id, date, revenue, expenses, profit, operating_cost, assets, liabilities
   ```
2. Create a matching `companies.csv`:
   ```
   company_id, company_name, sector, country
   ```
3. Run:
   ```bash
   python pipeline.py --step ingest
   ```
   Or from Python:
   ```python
   from src.ingestion.data_loader import load_csv_data
   load_csv_data("path/to/your_companies.csv", "path/to/your_records.csv")
   ```

---

## Tech Stack Summary (for CV / Resume)

```
Language:     Python 3.9+
Data:         Pandas, NumPy
ML:           Scikit-learn (Isolation Forest, Polynomial Regression)
Stats:        SciPy (seasonal decomposition, exponential smoothing)
Database:     SQLite (schema.sql, 5 tables, indexed)
Visualisation:Matplotlib, Seaborn
Dashboard:    Streamlit (8 pages, sidebar controls)
Reports:      OpenPyXL (7-sheet Excel workbook)
```

---

## Troubleshooting

**`ModuleNotFoundError`** — make sure your virtual environment is activated and you ran `pip install -r requirements.txt`.

**`No such table: financial_records`** — run `python pipeline.py` first to initialise the database.

**`streamlit: command not found`** — run `pip install streamlit` then try again.

**Charts show blank** — run `python pipeline.py --step charts` to regenerate them.

**Dashboard says "Run pipeline first"** — the database hasn't been initialised. Run `python pipeline.py`.

---

## License

MIT — free to use, modify, and include in your portfolio.
