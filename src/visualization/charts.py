"""
src/visualization/charts.py
-----------------------------
Generates all financial charts as matplotlib figures.
Returns Figure objects (for embedding in Streamlit) or saves to PNG.
All charts use a consistent professional colour palette.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.data_loader import get_financial_records, get_all_companies, get_forecast_results, get_anomaly_flags
from src.analytics.kpi_calculator import compute_kpis, sector_comparison

PALETTE = ["#2563EB","#059669","#D97706","#DC2626","#7C3AED"]
LIGHT_PALETTE = ["#DBEAFE","#D1FAE5","#FEF3C7","#FEE2E2","#EDE9FE"]

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        120,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
os.makedirs(OUT_DIR, exist_ok=True)


def _company_colors(companies_df: pd.DataFrame) -> dict:
    return {row["company_id"]: PALETTE[i % len(PALETTE)]
            for i, (_, row) in enumerate(companies_df.iterrows())}


def fmt_k(val, pos):
    """Format axis ticks as £12.3k or £1.2M."""
    if abs(val) >= 1e6:
        return f"£{val/1e6:.1f}M"
    if abs(val) >= 1e3:
        return f"£{val/1e3:.0f}k"
    return f"£{val:.0f}"


# ── 1. Revenue Trend ──────────────────────────────────────────────────────────

def plot_revenue_trend(company_id: int = None, save: bool = False) -> plt.Figure:
    """Line chart of quarterly revenue for one or all companies."""
    df       = get_financial_records(company_id)
    comp_df  = get_all_companies()
    colors   = _company_colors(comp_df)
    name_map = dict(zip(comp_df["company_id"], comp_df["company_name"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    for cid, group in df.groupby("company_id"):
        group = group.sort_values("date")
        ax.plot(group["date"], group["revenue"],
                label=name_map.get(cid, f"Co {cid}"),
                color=colors.get(cid, "#999"),
                linewidth=2, marker="o", markersize=3)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.set_title("Quarterly Revenue Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Revenue")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(OUT_DIR, "revenue_trend.png"), bbox_inches="tight")
    return fig


# ── 2. Profit Margin Trend ────────────────────────────────────────────────────

def plot_profit_margin(company_id: int = None, save: bool = False) -> plt.Figure:
    """Profit margin (%) over time per company."""
    df       = get_financial_records(company_id)
    df       = compute_kpis(df)
    comp_df  = get_all_companies()
    colors   = _company_colors(comp_df)
    name_map = dict(zip(comp_df["company_id"], comp_df["company_name"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    for cid, group in df.groupby("company_id"):
        group = group.sort_values("date")
        ax.plot(group["date"], group["profit_margin"] * 100,
                label=name_map.get(cid, f"Co {cid}"),
                color=colors.get(cid, "#999"),
                linewidth=2)

    ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.set_title("Profit Margin Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Profit Margin %")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(OUT_DIR, "profit_margin.png"), bbox_inches="tight")
    return fig


# ── 3. Forecast Chart ─────────────────────────────────────────────────────────

def plot_forecast(company_id: int, forecast_data: dict, save: bool = False) -> plt.Figure:
    """
    Revenue forecast chart: historical actuals + fitted line + future forecast with CI band.
    forecast_data comes from forecasting_engine.get_fitted_and_forecast()
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    hist_dates    = pd.to_datetime(forecast_data["historical_dates"])
    future_dates  = pd.to_datetime(forecast_data["future_dates"])
    actual_rev    = forecast_data["actual_revenue"]
    fitted_rev    = forecast_data["fitted_revenue"]
    fc_rev        = forecast_data["forecast_revenue"]
    fc_lower      = forecast_data["forecast_lower"]
    fc_upper      = forecast_data["forecast_upper"]

    # Historical actual
    ax.plot(hist_dates, actual_rev, color=PALETTE[0], linewidth=2,
            marker="o", markersize=3, label="Actual revenue", zorder=3)

    # Fitted (in-sample)
    ax.plot(hist_dates, fitted_rev, color=PALETTE[0], linewidth=1.5,
            linestyle="--", alpha=0.6, label="Model fit")

    # Forecast + CI
    ax.plot(future_dates, fc_rev, color=PALETTE[2], linewidth=2.5,
            marker="D", markersize=4, label="Forecast", zorder=3)
    ax.fill_between(future_dates, fc_lower, fc_upper,
                    color=PALETTE[2], alpha=0.15, label="95% CI")

    # Vertical divider
    ax.axvline(hist_dates[-1], color="#999", linewidth=1, linestyle=":", alpha=0.7)
    ax.text(hist_dates[-1], ax.get_ylim()[0] if ax.get_ylim()[0] else 0,
            "  Forecast →", fontsize=9, color="#666")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    mae  = forecast_data["revenue_metrics"]["mae"]
    mape = forecast_data["revenue_metrics"]["mape"]
    model = forecast_data["revenue_model"]

    comp_df  = get_all_companies()
    name_map = dict(zip(comp_df["company_id"], comp_df["company_name"]))
    title    = f"Revenue Forecast — {name_map.get(company_id, f'Company {company_id}')}"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Revenue")
    ax.legend(fontsize=9)
    ax.text(0.01, 0.02, f"Model: {model}  |  MAE: {fmt_k(mae,0)}  |  MAPE: {mape:.1f}%",
            transform=ax.transAxes, fontsize=8, color="#666")
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(OUT_DIR, f"forecast_co{company_id}.png"), bbox_inches="tight")
    return fig


# ── 4. Expense Breakdown Pie ──────────────────────────────────────────────────

def plot_expense_breakdown(company_id: int, save: bool = False) -> plt.Figure:
    """Pie chart: operating cost vs other expenses vs profit for latest quarter."""
    df = get_financial_records(company_id).sort_values("date")
    if df.empty:
        return plt.figure()

    latest = df.iloc[-1]
    op_cost   = max(latest["operating_cost"], 0)
    other_exp = max(latest["expenses"] - op_cost, 0)
    profit    = max(latest["profit"], 0)
    total     = op_cost + other_exp + profit

    if total == 0:
        return plt.figure()

    labels = ["Operating Cost", "Other Expenses", "Net Profit"]
    sizes  = [op_cost, other_exp, profit]
    colors = [PALETTE[3], PALETTE[2], PALETTE[1]]

    # Remove zero slices
    pairs  = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    labels = [p[0] for p in pairs]
    sizes  = [p[1] for p in pairs]
    colors = [p[2] for p in pairs]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontsize(10)

    comp_df  = get_all_companies()
    name_map = dict(zip(comp_df["company_id"], comp_df["company_name"]))
    # Format quarter manually since strftime does not support %q
    if hasattr(latest["date"], "strftime"):
        dt = latest["date"]
        year = dt.year
        month = dt.month
        quarter_num = (month - 1) // 3 + 1
        quarter = f"{year} Q{quarter_num}"
    else:
        quarter = str(latest["date"])
    ax.set_title(f"Revenue Distribution — {name_map.get(company_id, '')}\n{quarter}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(OUT_DIR, f"expense_pie_co{company_id}.png"), bbox_inches="tight")
    return fig


# ── 5. Sector Comparison Bar Chart ────────────────────────────────────────────

def plot_sector_comparison(save: bool = False) -> plt.Figure:
    """Grouped bar chart comparing avg profit margin and avg expense ratio by sector."""
    df   = get_financial_records()
    comp = get_all_companies()
    sec  = sector_comparison(df, comp)

    fig, ax = plt.subplots(figsize=(9, 5))
    x   = np.arange(len(sec))
    w   = 0.35

    bars1 = ax.bar(x - w/2, sec["avg_profit_margin"], w, label="Avg Profit Margin %",
                   color=PALETTE[1], alpha=0.85)
    bars2 = ax.bar(x + w/2, sec["avg_expense_ratio"], w, label="Avg Expense Ratio %",
                   color=PALETTE[3], alpha=0.85)

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(sec["sector"], fontsize=10)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Sector Comparison — Profit Margin vs Expense Ratio",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(OUT_DIR, "sector_comparison.png"), bbox_inches="tight")
    return fig


# ── 6. Anomaly Timeline ───────────────────────────────────────────────────────

def plot_anomaly_timeline(company_id: int, save: bool = False) -> plt.Figure:
    """Revenue line with anomaly points highlighted in red."""
    df_rec = get_financial_records(company_id).sort_values("date")
    df_an  = get_anomaly_flags(company_id)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_rec["date"], df_rec["revenue"], color=PALETTE[0],
            linewidth=2, label="Revenue")

    if not df_an.empty:
        rev_an = df_an[df_an["metric"] == "revenue"]
        if not rev_an.empty:
            an_dates = pd.to_datetime(rev_an["date"])
            an_vals  = df_rec.set_index("date").reindex(an_dates)["revenue"].values
            ax.scatter(an_dates, an_vals, color=PALETTE[3], zorder=5,
                       s=80, label="Anomaly flagged", marker="^")

        exp_an = df_an[df_an["metric"] == "expenses"]
        if not exp_an.empty:
            ax.plot(df_rec["date"], df_rec["expenses"], color=PALETTE[2],
                    linewidth=1.5, linestyle="--", alpha=0.7, label="Expenses")
            an_dates_e = pd.to_datetime(exp_an["date"])
            an_vals_e  = df_rec.set_index("date").reindex(an_dates_e)["expenses"].values
            ax.scatter(an_dates_e, an_vals_e, color=PALETTE[3], zorder=5,
                       s=80, marker="v")

    comp_df  = get_all_companies()
    name_map = dict(zip(comp_df["company_id"], comp_df["company_name"]))
    ax.set_title(f"Anomaly Detection — {name_map.get(company_id, '')}",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.set_xlabel("Quarter")
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(OUT_DIR, f"anomaly_co{company_id}.png"), bbox_inches="tight")
    return fig


# ── 7. KPI Summary Heatmap ────────────────────────────────────────────────────

def plot_kpi_heatmap(save: bool = False) -> plt.Figure:
    """Heatmap of latest-quarter KPIs across all companies."""
    from src.analytics.kpi_calculator import latest_kpi_snapshot
    comp_df  = get_all_companies()
    df_rec   = get_financial_records()
    snap     = latest_kpi_snapshot(df_rec, comp_df)

    kpi_cols = ["profit_margin","operating_ratio","expense_ratio","return_on_assets"]
    kpi_labels = ["Profit Margin %","Operating Ratio %","Expense Ratio %","ROA %"]
    data = snap[kpi_cols].values
    companies = snap["company_name"].tolist()

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(data.T, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(companies)))
    ax.set_xticklabels([c.split()[0] for c in companies], fontsize=10)
    ax.set_yticks(range(len(kpi_labels)))
    ax.set_yticklabels(kpi_labels, fontsize=10)

    for i in range(len(companies)):
        for j in range(len(kpi_cols)):
            ax.text(i, j, f"{data[i,j]:.1f}", ha="center", va="center",
                    fontsize=9, color="black", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Value (%)")
    ax.set_title("KPI Heatmap — Latest Quarter", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(OUT_DIR, "kpi_heatmap.png"), bbox_inches="tight")
    return fig


def save_all_charts():
    """Save all charts to the reports/ directory."""
    comp_df = get_all_companies()
    from src.forecasting.forecasting_engine import get_fitted_and_forecast

    plot_revenue_trend(save=True)
    plot_profit_margin(save=True)
    plot_sector_comparison(save=True)
    plot_kpi_heatmap(save=True)

    for _, row in comp_df.iterrows():
        cid = row["company_id"]
        plot_expense_breakdown(cid, save=True)
        plot_anomaly_timeline(cid, save=True)
        fc_data = get_fitted_and_forecast(cid)
        plot_forecast(cid, fc_data, save=True)

    print(f"All charts saved to {OUT_DIR}/")
