"""
pipeline.py
-----------
Master pipeline: runs every module in order.
Use this to initialise the system from scratch OR refresh all data.

Usage:
    python pipeline.py                   # full run
    python pipeline.py --step ingest     # only ingestion
    python pipeline.py --step analytics  # only KPIs
    python pipeline.py --step forecast   # only forecasting
    python pipeline.py --step anomaly    # only anomaly detection
    python pipeline.py --step report     # only reports
"""

import logging
import argparse
import sys
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))


def step_ingest():
    logger.info("── STEP 1: Data Ingestion ──────────────────────────")
    from src.ingestion.data_loader import load_csv_data
    result = load_csv_data()
    logger.info(f"Loaded: {result['companies']} companies, {result['records']} records")


def step_analytics():
    logger.info("── STEP 2: KPI Analytics ───────────────────────────")
    from src.analytics.kpi_calculator import run_full_analytics
    results = run_full_analytics()
    summary = results["company_summary"]
    logger.info(f"KPIs computed for {len(summary)} companies")
    print("\nCompany Summary (sorted by Profit Margin):")
    print(summary[["company_name","sector","avg_profit_margin","avg_revenue_growth"]].to_string(index=False))


def step_forecast():
    logger.info("── STEP 3: Forecasting ─────────────────────────────")
    from src.forecasting.forecasting_engine import run_all_forecasts
    fc = run_all_forecasts(horizon=4)
    logger.info(f"Generated {len(fc)} forecast rows (4 quarters × {len(fc)//4 if len(fc) else 0} companies)")
    if not fc.empty:
        print("\nForecast Preview (next 4 quarters):")
        from src.ingestion.data_loader import get_all_companies
        comp = get_all_companies()
        fc_display = fc.merge(comp[["company_id","company_name"]], on="company_id")
        print(fc_display[["company_name","forecast_date","predicted_revenue","mape"]].to_string(index=False))


def step_anomaly():
    logger.info("── STEP 4: Anomaly Detection ───────────────────────")
    from src.anomaly_detection.anomaly_detector import run_anomaly_detection, anomaly_summary
    from src.ingestion.data_loader import get_all_companies
    anomalies = run_anomaly_detection()
    companies = get_all_companies()
    if not anomalies.empty:
        smry = anomaly_summary(anomalies, companies)
        print("\nAnomaly Summary:")
        print(smry.to_string(index=False))
        severe = anomalies[anomalies["severity"] == "SEVERE"]
        if not severe.empty:
            print(f"\n⚠  {len(severe)} SEVERE anomalies detected!")
            print(severe[["company_id","date","metric","value","z_score"]].to_string(index=False))
    else:
        logger.info("No anomalies detected.")


def step_report():
    logger.info("── STEP 5: Report Generation ───────────────────────")
    from src.reporting.report_generator import generate_all_reports
    paths = generate_all_reports()
    logger.info(f"Excel report: {paths['excel']}")
    logger.info(f"CSV report:   {paths['csv']}")


def step_charts():
    logger.info("── STEP 6: Saving Charts ───────────────────────────")
    from src.visualization.charts import save_all_charts
    save_all_charts()


STEPS = {
    "ingest":    step_ingest,
    "analytics": step_analytics,
    "forecast":  step_forecast,
    "anomaly":   step_anomaly,
    "report":    step_report,
    "charts":    step_charts,
}


def run_full_pipeline():
    start = time.time()
    logger.info("═══════════════════════════════════════════════════")
    logger.info("  Financial Analytics Platform — Full Pipeline Run")
    logger.info("═══════════════════════════════════════════════════")

    for name, fn in STEPS.items():
        try:
            fn()
        except Exception as e:
            logger.error(f"Step '{name}' failed: {e}")
            raise

    elapsed = time.time() - start
    logger.info(f"\n✓ Full pipeline complete in {elapsed:.1f}s")
    logger.info("Run 'streamlit run app.py' to launch the dashboard.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Analytics Pipeline")
    parser.add_argument("--step", choices=list(STEPS.keys()),
                        help="Run a single pipeline step")
    args = parser.parse_args()

    if args.step:
        STEPS[args.step]()
    else:
        run_full_pipeline()
