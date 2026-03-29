"""
src/anomaly_detection/anomaly_detector.py
-------------------------------------------
Detects unusual financial patterns using:
  1. Z-score analysis (primary — always available)
  2. IQR-based outlier detection
  3. Isolation Forest (sklearn — available in this project)

Flags:
  - Sudden expense spikes
  - Unusual revenue drops
  - Abnormal profit changes
  - High debt-to-assets ratio

Results stored in anomaly_flags table.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.data_loader import get_connection, get_financial_records, get_all_companies

logger = logging.getLogger(__name__)

# Thresholds
Z_SCORE_THRESHOLD    = 2.0   # flag if |z| > 2.0
Z_SCORE_SEVERE       = 3.0   # severe if |z| > 3.0
ISOLATION_CONTAMINATION = 0.08  # expect ~8% anomalies


# ── Z-Score Detection ─────────────────────────────────────────────────────────

def zscore_anomalies(df: pd.DataFrame, metrics: list = None) -> pd.DataFrame:
    """
    Compute Z-scores for each metric per company.
    Returns rows where |z_score| > Z_SCORE_THRESHOLD.
    """
    if metrics is None:
        metrics = ["revenue", "expenses", "profit", "operating_cost"]

    records = []
    for company_id, group in df.groupby("company_id"):
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].values.astype(float)
            if len(values) < 4:
                continue

            mean = np.mean(values)
            std  = np.std(values)
            if std == 0:
                continue

            z_scores = (values - mean) / std

            for i, (_, row) in enumerate(group.iterrows()):
                z = z_scores[i]
                if abs(z) > Z_SCORE_THRESHOLD:
                    severity = "SEVERE" if abs(z) > Z_SCORE_SEVERE else "WARNING"
                    records.append({
                        "company_id": company_id,
                        "date":       row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                        "metric":     metric,
                        "value":      round(float(row[metric]), 2),
                        "z_score":    round(z, 3),
                        "severity":   severity,
                        "method":     "Z-Score",
                    })

    return pd.DataFrame(records)


# ── IQR-Based Detection ───────────────────────────────────────────────────────

def iqr_anomalies(df: pd.DataFrame, metrics: list = None) -> pd.DataFrame:
    """
    IQR fence method: flag values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Complements Z-score (handles non-normal distributions better).
    """
    if metrics is None:
        metrics = ["revenue", "expenses", "profit"]

    records = []
    for company_id, group in df.groupby("company_id"):
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].values.astype(float)
            q1, q3 = np.percentile(values, [25, 75])
            iqr    = q3 - q1
            lower  = q1 - 1.5 * iqr
            upper  = q3 + 1.5 * iqr

            for _, row in group.iterrows():
                v = float(row[metric])
                if v < lower or v > upper:
                    records.append({
                        "company_id": company_id,
                        "date":       row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                        "metric":     metric,
                        "value":      round(v, 2),
                        "z_score":    round((v - np.mean(values)) / (np.std(values) or 1), 3),
                        "severity":   "WARNING",
                        "method":     "IQR",
                    })

    return pd.DataFrame(records)


# ── Isolation Forest ──────────────────────────────────────────────────────────

def isolation_forest_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multivariate anomaly detection using Isolation Forest (sklearn).
    Uses revenue, expenses, profit, operating_cost as features.
    """
    feature_cols = ["revenue", "expenses", "profit", "operating_cost"]
    available = [c for c in feature_cols if c in df.columns]

    records = []
    for company_id, group in df.groupby("company_id"):
        X = group[available].values.astype(float)
        if len(X) < 6:  # need minimum samples
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso = IsolationForest(
            contamination=ISOLATION_CONTAMINATION,
            random_state=42,
            n_estimators=100
        )
        preds  = iso.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
        scores = iso.score_samples(X_scaled) # more negative = more anomalous

        for i, (_, row) in enumerate(group.iterrows()):
            if preds[i] == -1:
                # Determine dominant anomalous metric
                z_vals = np.abs((X[i] - X.mean(axis=0)) / (X.std(axis=0) + 1e-9))
                dominant_metric = available[np.argmax(z_vals)]
                records.append({
                    "company_id": company_id,
                    "date":       row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
                    "metric":     dominant_metric,
                    "value":      round(float(row[dominant_metric]), 2),
                    "z_score":    round(float(-scores[i]), 3),  # invert for readability
                    "severity":   "WARNING",
                    "method":     "IsolationForest",
                })

    return pd.DataFrame(records)


# ── Debt Risk Flags ───────────────────────────────────────────────────────────

def debt_risk_flags(df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
    """Flag quarters where liabilities/assets > threshold (high leverage risk)."""
    df = df.copy()
    df["debt_ratio"] = df["liabilities"] / df["assets"].replace(0, np.nan)
    risky = df[df["debt_ratio"] > threshold].copy()

    if risky.empty:
        return pd.DataFrame()

    risky["metric"]   = "debt_to_assets"
    risky["value"]    = risky["debt_ratio"].round(4)
    risky["z_score"]  = ((risky["debt_ratio"] - df["debt_ratio"].mean()) / (df["debt_ratio"].std() or 1)).round(3)
    risky["severity"] = risky["debt_ratio"].apply(lambda x: "SEVERE" if x > 0.80 else "WARNING")
    risky["method"]   = "DebtRatioThreshold"
    risky["date"]     = risky["date"].astype(str)

    return risky[["company_id","date","metric","value","z_score","severity","method"]]


# ── Combined Detection + DB Storage ──────────────────────────────────────────

def run_anomaly_detection(save_to_db: bool = True) -> pd.DataFrame:
    """
    Run all detection methods, deduplicate, save to DB.
    Returns combined anomaly DataFrame.
    """
    df = get_financial_records()
    df["date"] = pd.to_datetime(df["date"])

    z_anom   = zscore_anomalies(df)
    iqr_anom = iqr_anomalies(df)
    iso_anom = isolation_forest_anomalies(df)
    debt_anom = debt_risk_flags(df)

    frames = [f for f in [z_anom, iqr_anom, iso_anom, debt_anom] if not f.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep most severe per (company, date, metric)
    severity_order = {"SEVERE": 0, "WARNING": 1}
    combined["_sev_rank"] = combined["severity"].map(severity_order)
    combined = (
        combined.sort_values("_sev_rank")
        .drop_duplicates(subset=["company_id", "date", "metric"])
        .drop(columns=["_sev_rank","method"])
        .reset_index(drop=True)
    )

    if save_to_db:
        conn = get_connection()
        combined.to_sql("anomaly_flags", conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(combined)} anomaly flags.")

    return combined


def anomaly_summary(anomalies: pd.DataFrame, companies_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise anomaly counts per company for the dashboard."""
    if anomalies.empty:
        return pd.DataFrame()

    summary = anomalies.groupby("company_id").agg(
        total_flags  = ("metric",   "count"),
        severe_flags = ("severity", lambda x: (x == "SEVERE").sum()),
        metrics_affected = ("metric", lambda x: ", ".join(x.unique())),
    ).reset_index()

    return summary.merge(companies_df[["company_id","company_name"]], on="company_id")
