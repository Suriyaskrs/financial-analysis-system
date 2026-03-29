"""
generate_data.py
----------------
Generates realistic synthetic quarterly financial data for 5 companies
across 3 sectors covering 2019–2024 (24 quarters).
Run this ONCE to create the base dataset used by the entire system.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

COMPANIES = [
    {"company_id": 1, "company_name": "AlphaBank Corp",      "sector": "Banking",    "country": "UK"},
    {"company_id": 2, "company_name": "TechVentures Ltd",    "sector": "Technology", "country": "US"},
    {"company_id": 3, "company_name": "GlobalRetail PLC",    "sector": "Retail",     "country": "UK"},
    {"company_id": 4, "company_name": "NovaPharma Inc",      "sector": "Healthcare", "country": "US"},
    {"company_id": 5, "company_name": "EnergyFlow Group",    "sector": "Energy",     "country": "UK"},
]

BASE = {
    1: {"revenue": 5000, "growth": 0.03, "margin": 0.22, "op_ratio": 0.55, "asset_base": 45000},
    2: {"revenue": 3200, "growth": 0.08, "margin": 0.28, "op_ratio": 0.48, "asset_base": 28000},
    3: {"revenue": 8500, "growth": 0.02, "margin": 0.09, "op_ratio": 0.72, "asset_base": 15000},
    4: {"revenue": 2100, "growth": 0.06, "margin": 0.31, "op_ratio": 0.50, "asset_base": 18000},
    5: {"revenue": 4200, "growth": 0.01, "margin": 0.14, "op_ratio": 0.65, "asset_base": 32000},
}

quarters = pd.date_range("2019-01-01", periods=24, freq="QS")

records = []
record_id = 1

for comp in COMPANIES:
    cid = comp["company_id"]
    b = BASE[cid]
    revenue = b["revenue"]

    for i, date in enumerate(quarters):
        # Seasonal factor: Q4 is stronger for retail/tech
        seasonal = 1.0 + 0.05 * np.sin(2 * np.pi * i / 4)

        # COVID shock Q1-Q2 2020 (quarters 4 and 5)
        shock = 1.0
        if i in [4, 5] and cid in [1, 3, 5]:
            shock = np.random.uniform(0.78, 0.88)
        elif i in [4, 5] and cid == 2:
            shock = np.random.uniform(1.02, 1.12)  # tech benefited

        # Inject a deliberate anomaly: Company 3 Q2 2022 expense spike (index 13)
        anomaly_expense = 1.0
        if cid == 3 and i == 13:
            anomaly_expense = 1.45  # 45% expense spike for anomaly detection demo

        revenue = revenue * (1 + b["growth"] / 4 + np.random.normal(0, 0.008)) * seasonal * shock
        expenses = revenue * b["op_ratio"] * anomaly_expense * np.random.uniform(0.97, 1.03)
        op_cost = expenses * np.random.uniform(0.55, 0.65)
        profit = revenue - expenses
        assets = b["asset_base"] * (1 + 0.02 * i / 4) * np.random.uniform(0.98, 1.02)
        liabilities = assets * np.random.uniform(0.42, 0.58)

        records.append({
            "record_id":      record_id,
            "company_id":     cid,
            "date":           date.strftime("%Y-%m-%d"),
            "revenue":        round(revenue, 2),
            "expenses":       round(expenses, 2),
            "profit":         round(profit, 2),
            "operating_cost": round(op_cost, 2),
            "assets":         round(assets, 2),
            "liabilities":    round(liabilities, 2),
        })
        record_id += 1

df_companies = pd.DataFrame(COMPANIES)
df_records   = pd.DataFrame(records)

out = os.path.join(os.path.dirname(__file__))
df_companies.to_csv(os.path.join(out, "companies.csv"),         index=False)
df_records.to_csv(  os.path.join(out, "financial_records.csv"), index=False)

print(f"Generated {len(df_records)} financial records for {len(df_companies)} companies.")
print("Files saved: companies.csv, financial_records.csv")
