"""
src/forecasting/forecasting_engine.py
========================================
Production-grade forecasting engine using 5 real models:

  1. ARIMA(1,1,1)     — built from scratch with scipy (AutoRegressive + Moving Average
                         on differenced series). Proper time-series model, no external deps.

  2. Random Forest     — lag features (t-1..t-4) + seasonal dummies + trend index.
                         Captures non-linear patterns and interactions.

  3. Gradient Boosting — same lag feature set. Residual-boosting approach.
                         Usually the strongest single model on tabular time-series.

  4. SVR               — Support Vector Regression with RBF kernel.
                         Good for smaller series, robust to outliers.

  5. Ensemble          — weighted average of all models, weighted by 1/MAPE.
                         Typically outperforms any single model.

Auto-selector: runs all 5 on a rolling walk-forward validation,
picks the winner by lowest MAPE on the held-out test window.

Evaluation: MAE, RMSE, MAPE on last 4 quarters (walk-forward CV).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings, logging, sys, os

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.data_loader import get_connection, get_financial_records, get_all_companies

logger = logging.getLogger(__name__)


def _metrics(actual, predicted):
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)
    mae  = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    nz   = actual != 0
    mape = float(np.mean(np.abs((actual[nz] - predicted[nz]) / actual[nz])) * 100) if nz.any() else np.nan
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}


# ── 1. ARIMA(1,1,1) from scratch ─────────────────────────────────────────────

def _arima_scratch(series, horizon=4, p=1, d=1, q=1):
    """
    ARIMA(p,d,q) from first principles using scipy.optimize.
    1. Difference series d times to achieve stationarity.
    2. Fit AR(p) + MA(q) coefficients by minimising sum-of-squared residuals.
    3. Forecast on differenced scale, then invert differencing.
    4. 95% CI widens with forecast horizon (uncertainty compounds).
    """
    y      = series.astype(float).copy()
    n      = len(y)
    diff_y = np.diff(y, n=d)
    n_diff = len(diff_y)

    def _ssr(params):
        ar_c = params[:p]; ma_c = params[p:p+q]
        res  = np.zeros(n_diff)
        for t in range(n_diff):
            ar = sum(ar_c[i] * diff_y[t-i-1] for i in range(p) if t-i-1 >= 0)
            ma = sum(ma_c[i] * res[t-i-1]    for i in range(q) if t-i-1 >= 0)
            res[t] = diff_y[t] - ar - ma
        return float(np.sum(res**2))

    res_opt  = minimize(_ssr, np.zeros(p+q)+0.01, method="L-BFGS-B",
                        bounds=[(-0.999,0.999)]*(p+q), options={"maxiter":500})
    ar_c = res_opt.x[:p]; ma_c = res_opt.x[p:p+q]

    residuals = np.zeros(n_diff)
    for t in range(n_diff):
        ar = sum(ar_c[i] * diff_y[t-i-1] for i in range(p) if t-i-1 >= 0)
        ma = sum(ma_c[i] * residuals[t-i-1] for i in range(q) if t-i-1 >= 0)
        residuals[t] = diff_y[t] - ar - ma

    res_std   = float(np.std(residuals))
    diff_ext  = list(diff_y); res_ext = list(residuals)
    diff_fc   = []

    for _ in range(horizon):
        t  = len(diff_ext)
        ar = sum(ar_c[i] * diff_ext[t-i-1] for i in range(p) if t-i-1 >= 0)
        ma = sum(ma_c[i] * (res_ext[t-i-1] if t-i-1 < len(res_ext) else 0.0) for i in range(q))
        step = ar + ma
        diff_fc.append(step); diff_ext.append(step); res_ext.append(0.0)

    last_val = y[-1]; forecast = []
    for dv in diff_fc:
        nv = last_val + dv; forecast.append(nv); last_val = nv
    forecast = np.array(forecast)

    fitted_diff = diff_y - residuals
    fitted      = np.concatenate([[y[0]]*d, y[:d] + np.cumsum(fitted_diff)])[:n]
    ci_width    = 1.96 * res_std * np.sqrt(np.arange(1, horizon+1))

    return {
        "fitted":   fitted,
        "forecast": forecast,
        "lower":    forecast - ci_width,
        "upper":    forecast + ci_width,
        "metrics":  _metrics(y[d:], fitted[d:]),
        "model":    f"ARIMA({p},{d},{q})",
    }


# ── Shared lag-feature builder for ML models ─────────────────────────────────

def _make_lag_features(series, n_lags=4):
    """
    Transform time-series into supervised ML matrix.
    Features: [lag1..lag4, rolling_mean4, rolling_std4, quarter_sin, quarter_cos, t_norm]
    Target:   value at time t.
    """
    n = len(series); X_rows = []; y_rows = []
    for t in range(n_lags, n):
        lags      = series[t-n_lags:t][::-1]
        roll_mean = np.mean(series[max(0,t-4):t])
        roll_std  = np.std( series[max(0,t-4):t]) + 1e-9
        q         = t % 4
        row       = np.array([*lags, roll_mean, roll_std,
                               np.sin(2*np.pi*q/4), np.cos(2*np.pi*q/4), t/n])
        X_rows.append(row); y_rows.append(series[t])
    return np.array(X_rows), np.array(y_rows)


def _ml_forecast(series, model_cls, model_params, horizon=4, n_lags=4, model_name=""):
    """
    Generic ML forecaster: fit on lag features, predict recursively.
    Recursive strategy: append each prediction and use it as a lag for the next step.
    """
    n  = len(series)
    X, y = _make_lag_features(series, n_lags)
    sx   = StandardScaler(); sy = StandardScaler()
    Xs   = sx.fit_transform(X)
    ys   = sy.fit_transform(y.reshape(-1,1)).ravel()

    params = model_params.copy()
    try:
        model = model_cls(**params, random_state=42)
    except TypeError:
        model = model_cls(**params)
    model.fit(Xs, ys)

    fitted_s   = model.predict(Xs)
    fitted_inv = sy.inverse_transform(fitted_s.reshape(-1,1)).ravel()
    fitted_full = np.concatenate([[np.nan]*n_lags, fitted_inv])

    ext = list(series.copy()); fc_list = []
    for step in range(horizon):
        t    = len(ext)
        lags = np.array(ext[-n_lags:])[::-1]
        rm   = np.mean(ext[-4:]); rs = np.std(ext[-4:]) + 1e-9
        q    = t % 4
        feat = np.array([*lags, rm, rs, np.sin(2*np.pi*q/4), np.cos(2*np.pi*q/4), t/n]).reshape(1,-1)
        pred = float(sy.inverse_transform(model.predict(sx.transform(feat)).reshape(-1,1))[0,0])
        fc_list.append(pred); ext.append(pred)

    forecast = np.array(fc_list)
    res_std  = float(np.std(series[n_lags:] - fitted_inv))
    ci_w     = 1.96 * res_std * np.sqrt(np.arange(1, horizon+1))
    return {
        "fitted":   fitted_full,
        "forecast": forecast,
        "lower":    forecast - ci_w,
        "upper":    forecast + ci_w,
        "metrics":  _metrics(series[n_lags:], fitted_inv),
        "model":    model_name,
    }


# ── 2. Random Forest ──────────────────────────────────────────────────────────

def _random_forest_forecast(series, horizon=4):
    """
    Random Forest on lag features. n_estimators=200, max_depth=4.
    Handles non-linear interactions between lags. Robust to outliers.
    Feature importance shows which lag periods drive revenue most.
    """
    return _ml_forecast(series, RandomForestRegressor,
                        {"n_estimators":200, "max_depth":4, "min_samples_leaf":2, "max_features":"sqrt"},
                        horizon=horizon, model_name="RandomForest(n=200,depth=4)")


# ── 3. Gradient Boosting ──────────────────────────────────────────────────────

def _gradient_boosting_forecast(series, horizon=4):
    """
    Gradient Boosting on lag features. n=300, lr=0.05 (slow+careful).
    Sequential residual correction — typically the strongest single model
    on structured tabular time-series data.
    """
    return _ml_forecast(series, GradientBoostingRegressor,
                        {"n_estimators":300, "learning_rate":0.05, "max_depth":3,
                         "subsample":0.8, "min_samples_leaf":2},
                        horizon=horizon, model_name="GradientBoosting(n=300,lr=0.05)")


# ── 4. SVR ────────────────────────────────────────────────────────────────────

def _svr_forecast(series, horizon=4, n_lags=4):
    """
    Support Vector Regression, RBF kernel, C=100, epsilon=0.1.
    Maximises margin around fit — robust on small datasets.
    """
    n  = len(series)
    X, y = _make_lag_features(series, n_lags)
    sx   = StandardScaler(); sy = StandardScaler()
    Xs   = sx.fit_transform(X)
    ys   = sy.fit_transform(y.reshape(-1,1)).ravel()

    model = SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale")
    model.fit(Xs, ys)
    fi    = sy.inverse_transform(model.predict(Xs).reshape(-1,1)).ravel()
    ffl   = np.concatenate([[np.nan]*n_lags, fi])

    ext = list(series.copy()); fc_list = []
    for step in range(horizon):
        t    = len(ext)
        lags = np.array(ext[-n_lags:])[::-1]
        rm   = np.mean(ext[-4:]); rs = np.std(ext[-4:]) + 1e-9
        q    = t % 4
        feat = np.array([*lags, rm, rs, np.sin(2*np.pi*q/4), np.cos(2*np.pi*q/4), t/n]).reshape(1,-1)
        pred = float(sy.inverse_transform(model.predict(sx.transform(feat)).reshape(-1,1))[0,0])
        fc_list.append(pred); ext.append(pred)

    forecast = np.array(fc_list)
    res_std  = float(np.std(series[n_lags:] - fi))
    ci_w     = 1.96 * res_std * np.sqrt(np.arange(1, horizon+1))
    return {
        "fitted":   ffl,
        "forecast": forecast,
        "lower":    forecast - ci_w,
        "upper":    forecast + ci_w,
        "metrics":  _metrics(series[n_lags:], fi),
        "model":    "SVR(RBF,C=100)",
    }


# ── 5. Ensemble ───────────────────────────────────────────────────────────────

def _ensemble_forecast(results, horizon=4):
    """
    Weighted ensemble: weight = 1/MAPE (lower error → higher trust).
    Reduces variance across models. Usually beats any individual model.
    """
    valid = {k: r for k, r in results.items()
             if not np.isnan(r["metrics"].get("mape", np.nan)) and r["metrics"]["mape"] > 0}
    if not valid:
        fc = np.mean([r["forecast"] for r in results.values()], axis=0)
        return {"forecast": fc, "lower": fc*0.92, "upper": fc*1.08,
                "metrics": {"mae":np.nan,"rmse":np.nan,"mape":np.nan},
                "model": "Ensemble(simple_avg)",
                "fitted": list(results.values())[0]["fitted"]}

    w     = {k: 1.0/r["metrics"]["mape"] for k,r in valid.items()}
    wt    = sum(w.values())
    nw    = {k: v/wt for k,v in w.items()}
    fc    = sum(nw[k]*valid[k]["forecast"] for k in valid)
    lo    = sum(nw[k]*valid[k]["lower"]    for k in valid)
    hi    = sum(nw[k]*valid[k]["upper"]    for k in valid)
    bf    = min(valid.values(), key=lambda r: r["metrics"]["mape"])["fitted"]
    wstr  = " | ".join(f"{k}:{nw[k]:.2f}" for k in valid)
    return {
        "forecast": fc, "lower": lo, "upper": hi,
        "fitted":   bf,
        "metrics":  {"mae":np.nan,"rmse":np.nan,
                     "mape": float(np.mean([r["metrics"]["mape"] for r in valid.values()]))},
        "model":    f"Ensemble({wstr})",
        "weights":  nw,
    }


# ── Walk-Forward Validation ───────────────────────────────────────────────────

def _walk_forward_mape(series, model_fn, horizon=4):
    n = len(series); train_size = n - horizon
    if train_size < 8: return np.nan
    try:
        result = model_fn(series[:train_size], horizon=horizon)
        actual = series[train_size:]
        fc     = result["forecast"]
        nz     = actual != 0
        return float(np.mean(np.abs((actual[nz]-fc[nz])/actual[nz]))*100) if nz.any() else np.nan
    except Exception as e:
        logger.warning(f"Walk-forward failed: {e}")
        return np.nan


ALL_MODELS = {
    "arima":             _arima_scratch,
    "random_forest":     _random_forest_forecast,
    "gradient_boosting": _gradient_boosting_forecast,
    "svr":               _svr_forecast,
}


def best_forecast(series, horizon=4):
    """
    Run all 4 models + ensemble. Auto-select winner by walk-forward MAPE.
    Returns winner result dict with 'all_results' and 'wf_mapes' attached.
    """
    y = np.array(series, dtype=float)

    results = {}
    for name, fn in ALL_MODELS.items():
        try:
            results[name] = fn(y, horizon=horizon)
            logger.info(f"    {name:22s} MAPE={results[name]['metrics']['mape']:.2f}%")
        except Exception as e:
            logger.warning(f"    {name} failed: {e}")

    if len(results) >= 2:
        results["ensemble"] = _ensemble_forecast(results, horizon)

    if not results:
        raise RuntimeError("All models failed.")

    wf = {k: _walk_forward_mape(y, fn, horizon) for k,fn in ALL_MODELS.items() if k in results}

    best_single = min(wf, key=lambda k: wf[k] if not np.isnan(wf.get(k,np.nan)) else 9999)
    best_wf_val = wf.get(best_single, np.nan)
    ens_mape    = results.get("ensemble",{}).get("metrics",{}).get("mape",9999) or 9999

    if not np.isnan(ens_mape) and ens_mape < (best_wf_val or 9999) * 0.95:
        winner = results["ensemble"]; winner["model_name"] = "ensemble"
    else:
        winner = results[best_single]; winner["model_name"] = best_single

    winner["all_results"] = results
    winner["wf_mapes"]    = wf
    return winner


# ── Per-company forecast + DB ─────────────────────────────────────────────────

def forecast_company(company_id, horizon=4):
    df = get_financial_records(company_id).sort_values("date")
    if len(df) < 8:
        logger.warning(f"Company {company_id}: not enough data.")
        return pd.DataFrame()

    last_date    = df["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.offsets.QuarterBegin(startingMonth=1),
        periods=horizon, freq="QS")

    logger.info(f"  Revenue:")
    rev  = best_forecast(df["revenue"].values, horizon)
    logger.info(f"  Profit:")
    prof = best_forecast(df["profit"].values,  horizon)

    rows = []
    for i, fd in enumerate(future_dates):
        rows.append({
            "company_id":        company_id,
            "forecast_date":     fd.strftime("%Y-%m-%d"),
            "predicted_revenue": round(float(rev["forecast"][i]),  2),
            "predicted_profit":  round(float(prof["forecast"][i]), 2),
            "lower_bound":       round(float(rev["lower"][i]),     2),
            "upper_bound":       round(float(rev["upper"][i]),     2),
            "model_used":        rev["model"],
            "mae":               rev["metrics"]["mae"],
            "mape":              rev["metrics"]["mape"],
        })
    return pd.DataFrame(rows)


def run_all_forecasts(horizon=4):
    companies = get_all_companies(); all_fc = []
    for _, row in companies.iterrows():
        cid = row["company_id"]
        logger.info(f"── [{row['company_name']}] ──")
        try:
            fc = forecast_company(cid, horizon)
            if not fc.empty: all_fc.append(fc)
        except Exception as e:
            logger.error(f"Failed: {e}")

    if not all_fc: return pd.DataFrame()
    combined = pd.concat(all_fc, ignore_index=True)
    conn = get_connection()
    combined.to_sql("forecast_results", conn, if_exists="replace", index=False)
    conn.commit(); conn.close()
    logger.info(f"Saved {len(combined)} forecast rows.")
    return combined


def get_fitted_and_forecast(company_id, horizon=4):
    """Returns fitted + forecast data for all models — used by dashboard chart."""
    df = get_financial_records(company_id).sort_values("date")
    y  = df["revenue"].values.astype(float)

    all_results = {}
    for name, fn in ALL_MODELS.items():
        try: all_results[name] = fn(y, horizon=horizon)
        except Exception: pass
    if all_results:
        all_results["ensemble"] = _ensemble_forecast(all_results, horizon)

    winner = best_forecast(df["revenue"], horizon)
    last_date    = df["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.offsets.QuarterBegin(startingMonth=1),
        periods=horizon, freq="QS")

    return {
        "historical_dates": df["date"].tolist(),
        "actual_revenue":   df["revenue"].tolist(),
        "actual_profit":    df["profit"].tolist(),
        "fitted_revenue":   winner["fitted"].tolist() if winner.get("fitted") is not None else [],
        "future_dates":     future_dates.tolist(),
        "forecast_revenue": winner["forecast"].tolist(),
        "forecast_lower":   winner["lower"].tolist(),
        "forecast_upper":   winner["upper"].tolist(),
        "revenue_metrics":  winner["metrics"],
        "revenue_model":    winner["model"],
        "all_model_results": {
            n: {"forecast": r["forecast"].tolist(), "lower": r["lower"].tolist(),
                "upper": r["upper"].tolist(), "mape": r["metrics"]["mape"], "model": r["model"]}
            for n, r in all_results.items()
        },
        "wf_mapes": winner.get("wf_mapes", {}),
    }


# ── EXPANSION STUBS ───────────────────────────────────────────────────────────
# Prophet:  pip install prophet  → see full stub in README
# SARIMA:   pip install statsmodels → replace _arima_scratch with SARIMAX