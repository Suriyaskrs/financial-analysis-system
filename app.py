"""
app.py — Financial Performance Analytics Dashboard
----------------------------------------------------
Streamlit dashboard providing interactive access to all analytics modules.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinAnalytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (lazy — only after sys.path set) ──────────────────────────────────
from src.ingestion.data_loader import (
    get_all_companies, get_financial_records,
    get_forecast_results, get_anomaly_flags, load_csv_data
)
from src.analytics.kpi_calculator import (
    compute_kpis, company_summary, sector_comparison,
    latest_kpi_snapshot, run_sql_query, SQL_QUERIES
)
from src.forecasting.forecasting_engine import run_all_forecasts, get_fitted_and_forecast
from src.anomaly_detection.anomaly_detector import run_anomaly_detection, anomaly_summary
from src.reporting.report_generator import generate_excel_report, generate_csv_report
from src.visualization.charts import (
    plot_revenue_trend, plot_profit_margin, plot_forecast,
    plot_expense_breakdown, plot_sector_comparison,
    plot_anomaly_timeline, plot_kpi_heatmap
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetric"] {
    background: #F0F7FF;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 4px solid #2563EB;
}
[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
.block-container { padding-top: 1.5rem; }
h1 { color: #1E3A5F; font-size: 1.9rem !important; }
h2 { color: #1E3A5F; font-size: 1.3rem !important; }
div[data-testid="stSidebar"] { background: #F8FAFC; }
.status-ok    { color: #059669; font-weight: 700; }
.status-warn  { color: #D97706; font-weight: 700; }
.status-bad   { color: #DC2626; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Cache data loaders ────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def cached_companies():       return get_all_companies()

@st.cache_data(ttl=300)
def cached_records():
    df = get_financial_records()
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=300)
def cached_forecasts():       return get_forecast_results()

@st.cache_data(ttl=300)
def cached_anomalies():       return get_anomaly_flags()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 FinAnalytics")
    st.markdown("*HSBC FinOps Project*")
    st.divider()

    page = st.radio("Navigation", [
        "🏠  Overview",
        "📈  Revenue & Trends",
        "🎯  KPI Analysis",
        "🔮  Forecasting",
        "🚨  Anomaly Detection",
        "🏢  Sector Comparison",
        "🔍  SQL Explorer",
        "📥  Reports & Export",
    ])

    st.divider()
    st.markdown("**Pipeline Controls**")

    if st.button("🔄 Reload Data from CSV"):
        with st.spinner("Reloading..."):
            load_csv_data()
            st.cache_data.clear()
        st.success("Data reloaded!")

    if st.button("🔮 Run Forecasts"):
        with st.spinner("Running forecasting models..."):
            run_all_forecasts(horizon=4)
            st.cache_data.clear()
        st.success("Forecasts updated!")

    if st.button("🚨 Detect Anomalies"):
        with st.spinner("Running anomaly detection..."):
            run_anomaly_detection()
            st.cache_data.clear()
        st.success("Anomaly detection complete!")

    st.divider()
    companies_df = cached_companies()
    company_options = dict(zip(companies_df["company_name"], companies_df["company_id"]))
    selected_company_name = st.selectbox("Filter company", list(company_options.keys()))
    selected_cid = company_options[selected_company_name]


# ── Load core data ────────────────────────────────────────────────────────────
try:
    companies_df = cached_companies()
    df_records   = cached_records()
    df_forecasts = cached_forecasts()
    df_anomalies = cached_anomalies()
    data_ok = True
except Exception as e:
    st.error(f"Database not initialised. Run `python pipeline.py` first.\n\nError: {e}")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("Financial Performance Analytics Platform")
    st.markdown("*Quarterly financial data · 5 companies · 2019–2024 · 24 periods*")

    # KPI summary cards
    snap = latest_kpi_snapshot(df_records, companies_df)

    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    for i, (_, row) in enumerate(snap.iterrows()):
        with cols[i]:
            pm = row.get("profit_margin", 0)
            rg = row.get("revenue_growth", 0)
            st.metric(
                label=row["company_name"].split()[0],
                value=f"{pm:.1f}%",
                delta=f"Rev Δ {rg:+.1f}%",
            )

    st.divider()

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Revenue Trend — All Companies")
        fig = plot_revenue_trend()
        st.pyplot(fig)

    with col_right:
        st.subheader("Company Summary")
        summary = company_summary(df_records, companies_df)
        display_cols = {
            "company_name":      "Company",
            "sector":            "Sector",
            "avg_profit_margin": "Profit Margin %",
            "avg_revenue_growth":"Rev Growth %",
            "total_revenue":     "Total Revenue (£k)",
        }
        st.dataframe(
            summary.rename(columns=display_cols)[list(display_cols.values())],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Profit Margin Trends")
        st.pyplot(plot_profit_margin())
    with col_b:
        st.subheader("KPI Heatmap — Latest Quarter")
        st.pyplot(plot_kpi_heatmap())


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Revenue & Trends
# ═══════════════════════════════════════════════════════════════════════════════
elif "Revenue" in page:
    st.title(f"Revenue & Trend Analysis — {selected_company_name}")

    df_co = df_records[df_records["company_id"] == selected_cid].sort_values("date")
    df_kpi = compute_kpis(df_co)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    latest = df_kpi.iloc[-1]
    prev   = df_kpi.iloc[-2] if len(df_kpi) > 1 else latest

    c1.metric("Latest Revenue",     f"£{latest['revenue']:,.0f}k",
              delta=f"{(latest['revenue']-prev['revenue'])/prev['revenue']*100:+.1f}%")
    c2.metric("Latest Profit",      f"£{latest['profit']:,.0f}k",
              delta=f"{(latest['profit']-prev['profit'])/max(abs(prev['profit']),1)*100:+.1f}%")
    c3.metric("Profit Margin",      f"{latest['profit_margin']*100:.1f}%")
    c4.metric("Expense Ratio",      f"{latest['expense_ratio']*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quarterly Revenue")
        st.pyplot(plot_revenue_trend(selected_cid))
    with col2:
        st.subheader("Revenue Distribution")
        st.pyplot(plot_expense_breakdown(selected_cid))

    st.subheader("Historical KPI Data")
    disp = df_kpi[["date","revenue","expenses","profit","profit_margin",
                   "operating_ratio","expense_ratio","return_on_assets","revenue_growth"]].copy()
    disp["date"]           = disp["date"].dt.strftime("%Y-Q%q" if hasattr(disp["date"].dt, 'quarter') else "%Y-%m")
    disp["profit_margin"]  = (disp["profit_margin"]  * 100).round(2)
    disp["operating_ratio"]= (disp["operating_ratio"] * 100).round(2)
    disp["expense_ratio"]  = (disp["expense_ratio"]   * 100).round(2)
    disp["return_on_assets"]=(disp["return_on_assets"]* 100).round(2)
    disp["revenue_growth"] = (disp["revenue_growth"]  * 100).round(2)
    st.dataframe(disp.rename(columns={
        "profit_margin":  "Profit Margin %",
        "operating_ratio":"Op Ratio %",
        "expense_ratio":  "Exp Ratio %",
        "return_on_assets":"ROA %",
        "revenue_growth": "Rev Growth %",
    }), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: KPI Analysis
# ═══════════════════════════════════════════════════════════════════════════════
elif "KPI" in page:
    st.title("KPI Analysis")

    tab1, tab2 = st.tabs(["Company Deep-Dive", "All Companies"])

    with tab1:
        st.subheader(f"KPI Deep-Dive: {selected_company_name}")
        df_co  = df_records[df_records["company_id"] == selected_cid]
        df_kpi = compute_kpis(df_co)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_profit_margin(selected_cid))
        with col2:
            st.pyplot(plot_expense_breakdown(selected_cid))

    with tab2:
        st.subheader("All Companies — Summary")
        summary = company_summary(df_records, companies_df)
        st.dataframe(summary.drop(columns=["company_id","quarters"], errors="ignore"),
                     use_container_width=True, hide_index=True)

        st.subheader("KPI Heatmap")
        st.pyplot(plot_kpi_heatmap())


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Forecasting
# ═══════════════════════════════════════════════════════════════════════════════
elif "Forecast" in page:
    st.title(f"Revenue Forecasting — {selected_company_name}")

    if df_forecasts.empty:
        st.warning("No forecasts in DB. Click 'Run Forecasts' in the sidebar.")
    else:
        fc_co = df_forecasts[df_forecasts["company_id"] == selected_cid]

        c1, c2, c3 = st.columns(3)
        if not fc_co.empty:
            next_q = fc_co.iloc[0]
            c1.metric("Next Quarter Revenue", f"£{next_q['predicted_revenue']:,.0f}k")
            c2.metric("Model MAE",            f"£{next_q['mae']:,.0f}k")
            c3.metric("Model MAPE",           f"{next_q['mape']:.1f}%")

        st.subheader("Forecast Chart")
        fc_data = get_fitted_and_forecast(selected_cid, horizon=4)
        st.pyplot(plot_forecast(selected_cid, fc_data))

        st.subheader("Forecast Table")
        disp = fc_co[["forecast_date","predicted_revenue","predicted_profit",
                       "lower_bound","upper_bound","model_used","mae","mape"]].copy()
        disp["forecast_date"] = pd.to_datetime(disp["forecast_date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(disp.rename(columns={
            "forecast_date":    "Date",
            "predicted_revenue":"Predicted Revenue (£k)",
            "predicted_profit": "Predicted Profit (£k)",
            "lower_bound":      "Lower 95% CI",
            "upper_bound":      "Upper 95% CI",
            "model_used":       "Model",
        }), use_container_width=True, hide_index=True)

        st.info("""
        **Models used (auto-selected by lowest MAPE):**
        - **Seasonal Trend Decomposition** — trend + quarterly seasonal factors
        - **Polynomial Regression** — degree-2 curve fit on time index
        - **Holt's Double Exponential Smoothing** — level + trend smoothing

        *To add Prophet or ARIMA: see `src/forecasting/forecasting_engine.py` → EXPANSION GUIDE*
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════
elif "Anomaly" in page:
    st.title("Anomaly Detection")

    if df_anomalies.empty:
        st.warning("No anomaly data. Click 'Detect Anomalies' in the sidebar.")
    else:
        severe_count = len(df_anomalies[df_anomalies["severity"] == "SEVERE"])
        warn_count   = len(df_anomalies[df_anomalies["severity"] == "WARNING"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Flags",    len(df_anomalies))
        c2.metric("⚠️ Severe",      severe_count)
        c3.metric("⚡ Warnings",     warn_count)

        st.subheader(f"Anomaly Timeline — {selected_company_name}")
        st.pyplot(plot_anomaly_timeline(selected_cid))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("All Anomaly Flags")
            disp = df_anomalies.merge(companies_df[["company_id","company_name"]], on="company_id")
            st.dataframe(
                disp[["company_name","date","metric","value","z_score","severity"]].rename(columns={
                    "company_name": "Company", "z_score": "Z-Score", "severity": "Severity"
                }),
                use_container_width=True, hide_index=True
            )
        with col2:
            st.subheader("Summary by Company")
            smry = anomaly_summary(df_anomalies, companies_df)
            st.dataframe(smry, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Sector Comparison
# ═══════════════════════════════════════════════════════════════════════════════
elif "Sector" in page:
    st.title("Sector Comparison")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Profit Margin vs Expense Ratio by Sector")
        st.pyplot(plot_sector_comparison())
    with col2:
        st.subheader("Sector KPI Table")
        sec = sector_comparison(df_records, companies_df)
        st.dataframe(sec, use_container_width=True, hide_index=True)

    st.subheader("All Company Revenue Overlay")
    st.pyplot(plot_revenue_trend())


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SQL Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif "SQL" in page:
    st.title("SQL Analytics Explorer")
    st.markdown("Run pre-built SQL queries against the live database — just as you would in an analyst interview.")

    query_labels = {
        "top_revenue_companies":  "Top companies by total revenue",
        "quarterly_revenue_trend":"Quarterly revenue trend all companies",
        "expense_ratio_by_quarter":"Annual expense ratio by company",
        "profit_ranking_latest":  "Profit ranking (latest quarter)",
        "yoy_growth_window":      "Year-over-year annual revenue",
    }
    selected_query = st.selectbox("Select query", list(query_labels.keys()),
                                  format_func=lambda k: query_labels[k])

    with st.expander("📄 View SQL", expanded=True):
        st.code(SQL_QUERIES[selected_query].strip(), language="sql")

    if st.button("▶  Run Query"):
        with st.spinner("Executing..."):
            result = run_sql_query(selected_query)
        st.success(f"Returned {len(result)} rows")
        st.dataframe(result, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Custom SQL")
    st.markdown("*For demo: only SELECT queries on `financial_records`, `companies`, `kpi_results`, `forecast_results`, `anomaly_flags`*")
    custom_sql = st.text_area("Write your SQL:", height=120,
                               placeholder="SELECT c.company_name, AVG(f.profit/f.revenue) AS avg_margin\nFROM financial_records f\nJOIN companies c ON f.company_id = c.company_id\nGROUP BY c.company_name")
    if st.button("▶  Run Custom Query") and custom_sql.strip():
        try:
            from src.ingestion.data_loader import get_connection
            import pandas as pd
            conn = get_connection()
            result = pd.read_sql(custom_sql, conn)
            conn.close()
            st.dataframe(result, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Query error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Reports & Export
# ═══════════════════════════════════════════════════════════════════════════════
elif "Report" in page:
    st.title("Reports & Export")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Excel Report")
        st.markdown("""
        Multi-sheet Excel workbook containing:
        - Cover page
        - Executive Summary
        - Company KPIs (all quarters)
        - Sector Analysis
        - Forecast Results
        - Anomaly Flags
        - Raw Financial Data
        """)
        if st.button("Generate Excel Report"):
            with st.spinner("Generating..."):
                path = generate_excel_report()
            with open(path, "rb") as f:
                st.download_button(
                    "⬇️ Download Excel Report",
                    data=f.read(),
                    file_name=os.path.basename(path),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    with col2:
        st.subheader("📄 CSV Report")
        st.markdown("""
        Flat CSV file containing all KPI metrics per company per quarter.
        Ready for import into Power BI, Excel, or further Python analysis.
        """)
        if st.button("Generate CSV Report"):
            with st.spinner("Generating..."):
                path = generate_csv_report()
            with open(path, "rb") as f:
                st.download_button(
                    "⬇️ Download CSV Report",
                    data=f.read(),
                    file_name=os.path.basename(path),
                    mime="text/csv"
                )

    st.divider()
    st.subheader("Quick Data Export")
    export_choice = st.selectbox("Export dataset", [
        "Financial Records", "Companies", "KPI Results", "Forecasts", "Anomaly Flags"
    ])
    export_map = {
        "Financial Records": df_records,
        "Companies":         companies_df,
        "KPI Results":       compute_kpis(df_records),
        "Forecasts":         df_forecasts,
        "Anomaly Flags":     df_anomalies,
    }
    df_export = export_map[export_choice]
    st.dataframe(df_export.head(20), use_container_width=True, hide_index=True)
    csv_bytes = df_export.to_csv(index=False).encode()
    st.download_button(
        f"⬇️ Download {export_choice} CSV",
        data=csv_bytes,
        file_name=f"{export_choice.lower().replace(' ','_')}.csv",
        mime="text/csv"
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("""
<small>
Financial Analytics Platform<br>
Built with Python · Pandas · Scikit-learn<br>
Matplotlib · Streamlit · SQLite<br><br>
<em>HSBC FinOps Internship Project</em>
</small>
""", unsafe_allow_html=True)
