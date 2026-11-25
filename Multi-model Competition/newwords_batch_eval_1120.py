#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
newwords_batch_eval_v4.py
Enhanced batch evaluator:
- Fits Logistic-1, Gompertz, Logistic-2wave (sum of two logistics) on cumulative.
- Derives increments, compares RMSE on a hold-out test segment vs ARIMA and Naive baselines.
- Computes:
    * logisticness_score (continuous [0,1]) as before;
    * model tendency: best model label (argmin test RMSE among candidates);
    * per-model probabilities (soft preference from RMSE);
    * tendency confidence.
- Writes summary.csv (one row per file) and scores_long.csv (long-form metrics).
- Optionally saves per-word plots & params JSON.

Usage:
    python newwords_batch_eval_v4.py --input_dir ./csvs --output_dir ./out --plots
    python newwords_batch_eval_v4.py --input_dir ./csvs --output_dir ./out --numeric_is_cumulative --bootstrap 200

Dependencies:
    pip install numpy pandas matplotlib scipy statsmodels
"""
import argparse, os, math, json, warnings
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

scipy_ok = True
try:
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
except Exception:
    scipy_ok = False
    curve_fit = None
    find_peaks = None

arima_ok = True
try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception:
    arima_ok = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True); return path

def detect_time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        try:
            dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().mean() > 0.7:
                df[c] = dt
                return c
        except Exception:
            pass
    df["__t__"] = np.arange(len(df), dtype=float)
    return "__t__"

def detect_numeric_col(df: pd.DataFrame, exclude: List[str]) -> str:
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        for c in df.columns:
            if c in exclude: continue
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass
        numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric column found.")
    return max(numeric_cols, key=lambda c: df[c].var(skipna=True))

def make_train_test_split(n: int, train_frac: float) -> Tuple[int,int]:
    split = max(5, int(n * train_frac))
    split = min(split, max(1, n - 1))
    return split, n - split

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2); ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

def aic_bic_ll_gaussian(y_true, y_pred, k_params):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    resid = y_true - y_pred
    sigma2 = np.mean(resid ** 2) if n > 0 else np.nan
    if n == 0 or not np.isfinite(sigma2) or sigma2 <= 0: return np.nan, np.nan, np.nan
    ll = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
    aic = 2 * k_params - 2 * ll
    bic = k_params * math.log(n) - 2 * ll
    return float(aic), float(bic), float(ll)

def logistic_cum(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))

def logistic_inc_from_params(t, K, r, t0):
    Nt = logistic_cum(t, K, r, t0); Nt_1 = logistic_cum(t - 1.0, K, r, t0)
    return Nt - Nt_1

def gompertz_cum(t, K, b, m):
    return K * np.exp(-np.exp(-b * (t - m)))

def gompertz_inc_from_params(t, K, b, m):
    Nt = gompertz_cum(t, K, b, m); Nt_1 = gompertz_cum(t - 1.0, K, b, m)
    return Nt - Nt_1

def logistic2_cum(t, K1, r1, t01, K2, r2, t02):
    return logistic_cum(t, K1, r1, t01) + logistic_cum(t, K2, r2, t02)

def logistic2_inc_from_params(t, K1, r1, t01, K2, r2, t02):
    return logistic_inc_from_params(t, K1, r1, t01) + logistic_inc_from_params(t, K2, r2, t02)

def fit_curve(func, t, y, p0, bounds, maxfev=60000):
    if not scipy_ok: return None
    try:
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds, maxfev=maxfev)
        return [float(x) for x in popt]
    except Exception:
        return None

def naive_forecast_increments(train_y, horizon):
    last = float(train_y[-1]) if len(train_y) else 0.0
    return np.full(horizon, last, dtype=float)

def arima_forecast_increments(train_y, horizon):
    if not arima_ok or len(train_y) < 5:
        return naive_forecast_increments(train_y, horizon)
    try:
        model = ARIMA(train_y, order=(1, 0, 0))
        res = model.fit()
        fc = res.forecast(steps=horizon)
        return np.asarray(fc, dtype=float)
    except Exception:
        return naive_forecast_increments(train_y, horizon)

def compute_logisticness(metrics: Dict, pred_store: Dict, N_test, y_train, y_test, w_r=0.5, w_a=0.5, aic_scale=10.0):
    log_rmse = metrics.get(("logistic_increments","test","RMSE"), np.inf)
    nai_rmse = metrics.get(("naive_increments","test","RMSE"), np.inf)
    ari_rmse = metrics.get(("arima_increments","test","RMSE"), np.inf)
    baseline_rmse = float(min(nai_rmse, ari_rmse))
    if not np.isfinite(log_rmse) or baseline_rmse<=0 or not np.isfinite(baseline_rmse):
        rel_rmse_imp = 0.0
    else:
        rel_rmse_imp = (baseline_rmse - log_rmse) / baseline_rmse
        if rel_rmse_imp < 0: rel_rmse_imp = 0.0
    I_AIC = 0.0
    try:
        Nhat_log = pred_store.get("Nhat_logistic", (None,None))[1]
        Nhat_gom = pred_store.get("Nhat_gompertz", (None,None))[1]
        if Nhat_log is not None and Nhat_gom is not None and len(N_test)>0:
            aic_log = aic_bic_ll_gaussian(N_test, Nhat_log, k_params=3)[0]
            aic_gom = aic_bic_ll_gaussian(N_test, Nhat_gom, k_params=3)[0]
            if np.isfinite(aic_log) and np.isfinite(aic_gom):
                delta_aic = aic_gom - aic_log
                raw = 1.0/(1.0 + math.exp(-delta_aic / aic_scale))
                I_AIC = max(0.0, (raw - 0.5) / 0.5)
    except Exception:
        I_AIC = 0.0
    S = float(w_r * rel_rmse_imp + w_a * I_AIC)
    return S, rel_rmse_imp, I_AIC, baseline_rmse

def bootstrap_logisticness(pred_store, y_train, y_test, baseline_rmse, I_AIC, w_r=0.5, w_a=0.5, B=200):
    if "yhat_logistic" not in pred_store:
        return (np.nan, np.nan)
    ytr, yts = pred_store["yhat_logistic"]
    resid_train = (y_train - ytr) if len(ytr)>0 else np.array([])
    if len(resid_train)==0 or baseline_rmse<=0: return (np.nan, np.nan)
    scores=[]
    for _ in range(B):
        samp = np.random.choice(resid_train, size=len(y_test), replace=True)
        pseudo = yts + samp
        pseudo_rmse = float(np.sqrt(np.mean((pseudo - yts)**2)))
        rel = max(0.0, (baseline_rmse - pseudo_rmse)/baseline_rmse) if baseline_rmse>0 else 0.0
        scores.append(w_r*rel + w_a*I_AIC)
    lo = float(np.percentile(scores, 2.5)); hi = float(np.percentile(scores,97.5))
    return lo, hi

def soft_preferences(rmse_dict: Dict[str, float]) -> Dict[str, float]:
    items = [(k,v) for k,v in rmse_dict.items() if np.isfinite(v)]
    if not items: return {k:0.0 for k in rmse_dict.keys()}
    vals = np.array([v for _,v in items], dtype=float)
    tau = float(np.median(vals)) if np.median(vals)>0 else (float(np.mean(vals))+1e-8)
    if tau<=0: tau = 1.0
    weights = np.exp(-vals / tau)
    Z = float(np.sum(weights))
    probs = {k:0.0 for k in rmse_dict.keys()}
    for (k,_),w in zip(items, weights):
        probs[k] = float(w / Z) if Z>0 else 0.0
    return probs

def process_file(path: str, out_dir: str, train_frac: float=0.7, min_points: int=10, numeric_is_cumulative: bool=False, make_plots: bool=False, bootstrap_B: int=200):
    try:
        df_raw = pd.read_csv(path)
    except Exception as e:
        return {"file": path, "word": os.path.splitext(os.path.basename(path))[0], "status": f"read_error: {e}"}
    df = df_raw.copy()
    time_col = detect_time_col(df)
    df = df.sort_values(time_col).reset_index(drop=True)
    num_col = detect_numeric_col(df, exclude=[time_col])
    time = df[time_col].values
    y_raw = df[num_col].astype(float).values
    if numeric_is_cumulative:
        N = y_raw.copy(); y = np.diff(np.r_[0.0, N]); y[y<0]=0.0
    else:
        y = y_raw.copy(); y[y<0]=0.0; N = np.cumsum(y)
    n = len(y)
    if n < min_points:
        return {"file": path, "word": os.path.splitext(os.path.basename(path))[0], "status": f"too_few_points: {n} < {min_points}"}
    t = np.arange(n, dtype=float)
    split,_ = make_train_test_split(n, train_frac)
    t_train, t_test = t[:split], t[split:]
    y_train, y_test = y[:split], y[split:]
    N_train, N_test = N[:split], N[split:]
    metrics = {}; preds = {}

    log_pars = None
    if scipy_ok:
        K0 = max(N_train[-1], 1.0); r0=0.1; t00 = float(np.median(t_train))
        popt = fit_curve(logistic_cum, t_train, N_train, p0=[K0,r0,t00], bounds=([1e-6,1e-6,-np.inf],[np.inf,10.0,np.inf]), maxfev=80000)
        if popt is not None:
            K,r,t0 = popt
            Nhat_tr = logistic_cum(t_train, K,r,t0); Nhat_ts = logistic_cum(t_test, K,r,t0)
            yhat_tr = logistic_inc_from_params(t_train, K,r,t0); yhat_ts = logistic_inc_from_params(t_test, K,r,t0)
            preds["Nhat_logistic"] = (Nhat_tr, Nhat_ts); preds["yhat_logistic"] = (yhat_tr, yhat_ts)
            metrics[("logistic_increments","test","RMSE")] = rmse(y_test, yhat_ts)
            metrics[("logistic_increments","test","MAE")]  = mae(y_test, yhat_ts)
            metrics[("logistic_cumulative","test","RMSE")] = rmse(N_test, Nhat_ts)
            log_pars = {"K":K,"r":r,"t0":t0}

    gom_pars = None
    if scipy_ok:
        K0 = max(N_train[-1], 1.0); b0=0.1; m0 = float(np.median(t_train))
        popt = fit_curve(gompertz_cum, t_train, N_train, p0=[K0,b0,m0], bounds=([1e-6,1e-6,-np.inf],[np.inf,10.0,np.inf]), maxfev=80000)
        if popt is not None:
            Kb,b,m = popt
            Nhat_tr = gompertz_cum(t_train, Kb,b,m); Nhat_ts = gompertz_cum(t_test, Kb,b,m)
            yhat_tr = gompertz_inc_from_params(t_train, Kb,b,m); yhat_ts = gompertz_inc_from_params(t_test, Kb,b,m)
            preds["Nhat_gompertz"] = (Nhat_tr, Nhat_ts); preds["yhat_gompertz"] = (yhat_tr, yhat_ts)
            metrics[("gompertz_increments","test","RMSE")] = rmse(y_test, yhat_ts)
            metrics[("gompertz_increments","test","MAE")]  = mae(y_test, yhat_ts)
            metrics[("gompertz_cumulative","test","RMSE")] = rmse(N_test, Nhat_ts)
            gom_pars = {"K":Kb,"b":b,"m":m}

    log2_pars = None
    if scipy_ok:
        Ksum0 = max(N_train[-1], 1.0)
        tmed = float(np.median(t_train))
        p0 = [0.6*Ksum0, 0.1, tmed-5.0, 0.4*Ksum0, 0.1, tmed+5.0]
        bounds = ([1e-6,1e-6,-np.inf, 1e-6,1e-6,-np.inf],
                  [np.inf, 10.0, np.inf,  np.inf, 10.0, np.inf])
        popt = fit_curve(logistic2_cum, t_train, N_train, p0=p0, bounds=bounds, maxfev=120000)
        if popt is not None:
            K1,r1,t01,K2,r2,t02 = popt
            Nhat_tr = logistic2_cum(t_train, K1,r1,t01,K2,r2,t02)
            Nhat_ts = logistic2_cum(t_test,  K1,r1,t01,K2,r2,t02)
            yhat_tr = logistic2_inc_from_params(t_train, K1,r1,t01,K2,r2,t02)
            yhat_ts = logistic2_inc_from_params(t_test,  K1,r1,t01,K2,r2,t02)
            preds["Nhat_logistic2"] = (Nhat_tr, Nhat_ts); preds["yhat_logistic2"] = (yhat_tr, yhat_ts)
            metrics[("logistic2_increments","test","RMSE")] = rmse(y_test, yhat_ts)
            metrics[("logistic2_increments","test","MAE")]  = mae(y_test, yhat_ts)
            metrics[("logistic2_cumulative","test","RMSE")] = rmse(N_test, Nhat_ts)
            log2_pars = {"K1":K1,"r1":r1,"t01":t01,"K2":K2,"r2":r2,"t02":t02}

    nai_test = naive_forecast_increments(y_train, len(y_test))
    metrics[("naive_increments","test","RMSE")] = rmse(y_test, nai_test)
    metrics[("naive_increments","test","MAE")]  = mae(y_test, nai_test)
    try:
        ari_test = arima_forecast_increments(y_train, len(y_test))
    except Exception:
        ari_test = nai_test.copy()
    metrics[("arima_increments","test","RMSE")] = rmse(y_test, ari_test)
    metrics[("arima_increments","test","MAE")]  = mae(y_test, ari_test)

    logisticness_score, rel_rmse_imp, I_AIC, baseline_rmse = compute_logisticness(metrics, preds, N_test, y_train, y_test)
    lo_ci, hi_ci = bootstrap_logisticness(preds, y_train, y_test, baseline_rmse, I_AIC, B=bootstrap_B)

    rmse_map = {
        "Logistic-1": metrics.get(("logistic_increments","test","RMSE"), np.nan),
        "Gompertz":   metrics.get(("gompertz_increments","test","RMSE"), np.nan),
        "Logistic-2": metrics.get(("logistic2_increments","test","RMSE"), np.nan),
        "ARIMA":      metrics.get(("arima_increments","test","RMSE"), np.nan),
        "Naive":      metrics.get(("naive_increments","test","RMSE"), np.nan),
    }
    finite_items = [(k,v) for k,v in rmse_map.items() if np.isfinite(v)]
    if finite_items:
        best_label, best_rmse = min(finite_items, key=lambda kv: kv[1])
    else:
        best_label, best_rmse = None, np.nan
    probs = soft_preferences(rmse_map)
    confidence = probs.get(best_label, 0.0) if best_label is not None else 0.0

    word = os.path.splitext(os.path.basename(path))[0]
    wdir = ensure_dir(os.path.join(out_dir, word))

    if make_plots:
        plt.figure()
        plt.plot(time, N, label="Observed cumulative")
        for key, lbl in [("Nhat_logistic","Logistic-1"), ("Nhat_gompertz","Gompertz"), ("Nhat_logistic2","Logistic-2")]:
            if key in preds:
                Ntr,Nts = preds[key]
                plt.plot(time[:split], Ntr, label=f"{lbl} (train)")
                plt.plot(time[split:], Nts, label=f"{lbl} (test)")
        plt.axvline(time[split], linestyle="--", label="Train/Test split")
        plt.legend(); plt.title(f"{word}: cumulative fits"); plt.tight_layout()
        plt.savefig(os.path.join(wdir,"cumulative_fit.png"), dpi=150); plt.close()

        plt.figure()
        plt.plot(time, y, label="Observed increments")
        for key, lbl in [("yhat_logistic","Logistic-1"), ("yhat_gompertz","Gompertz"), ("yhat_logistic2","Logistic-2")]:
            if key in preds:
                ytr,yts = preds[key]
                plt.plot(time[:split], ytr, label=f"{lbl} (train)")
                plt.plot(time[split:], yts, label=f"{lbl} (test)")
        plt.plot(time[split:], nai_test, label="Naive (test)")
        plt.plot(time[split:], ari_test, label=("ARIMA (test)" if arima_ok else "Naive (ARIMA NA)"))
        plt.axvline(time[split], linestyle="--", label="Train/Test split")
        plt.legend(); plt.title(f"{word}: increments & forecasts"); plt.tight_layout()
        plt.savefig(os.path.join(wdir,"increments_fit.png"), dpi=150); plt.close()

    with open(os.path.join(wdir,"fit_params.json"), "w", encoding="utf-8") as f:
        json.dump({
            "logistic_params": log_pars,
            "gompertz_params": gom_pars,
            "logistic2_params": log2_pars,
            "n_points": int(n),
            "best_label": best_label,
            "best_rmse_test": float(best_rmse) if np.isfinite(best_rmse) else None,
            "rmse_map": rmse_map,
            "model_probs": probs,
            "confidence": confidence,
            "logisticness_score": float(logisticness_score),
            "logisticness_ci": [lo_ci, hi_ci],
            "rel_rmse_imp": float(rel_rmse_imp),
            "I_AIC": float(I_AIC)
        }, f, ensure_ascii=False, indent=2)

    rows = []
    for (m, split_tag, metric_name), val in metrics.items():
        rows.append({"file": path, "word": word, "model": m, "split": split_tag, "metric": metric_name, "value": val})

    out = {
        "file": path, "word": word, "status": "ok",
        "n_points": int(n), "train_points": int(split), "test_points": int(n - split),
        "best_model_label": best_label, "best_model_rmse": float(best_rmse) if np.isfinite(best_rmse) else None,
        "prob_logistic1": probs.get("Logistic-1", 0.0),
        "prob_gompertz": probs.get("Gompertz", 0.0),
        "prob_logistic2": probs.get("Logistic-2", 0.0),
        "prob_arima": probs.get("ARIMA", 0.0),
        "prob_naive": probs.get("Naive", 0.0),
        "tendency_confidence": confidence,
        "logisticness_score": float(logisticness_score),
        "logisticness_ci_lo": lo_ci, "logisticness_ci_hi": hi_ci,
        "logistic_params": log_pars, "gompertz_params": gom_pars, "logistic2_params": log2_pars,
        "rows": rows
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--min_points", type=int, default=10)
    ap.add_argument("--numeric_is_cumulative", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=200, help="Bootstrap B for logisticness CI")
    args = ap.parse_args()

    in_dir = args.input_dir; out_dir = ensure_dir(args.output_dir)
    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.lower().endswith(".csv")]
    files.sort()
    if not files:
        print("No CSV found under", in_dir); return

    all_rows = []; long_rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] processing:", os.path.basename(path))
        res = process_file(
            path,
            out_dir=out_dir,
            train_frac=args.train_frac,
            min_points=args.min_points,
            numeric_is_cumulative=args.numeric_is_cumulative,
            make_plots=args.plots,
            bootstrap_B=args.bootstrap
        )
        if res.get("status") != "ok":
            all_rows.append({
                "file": res.get("file"),
                "word": os.path.splitext(os.path.basename(res.get("file","")))[0],
                "status": res.get("status"),
                "n_points": np.nan, "train_points": np.nan, "test_points": np.nan,
                "best_model_label": None, "best_model_rmse": np.nan,
                "prob_logistic1": 0.0, "prob_gompertz": 0.0, "prob_logistic2": 0.0, "prob_arima": 0.0, "prob_naive": 0.0,
                "tendency_confidence": 0.0, "logisticness_score": np.nan,
                "logistic_params": None, "gompertz_params": None, "logistic2_params": None
            })
            continue

        all_rows.append({
            "file": res["file"], "word": res["word"], "status": res["status"],
            "n_points": res["n_points"], "train_points": res["train_points"], "test_points": res["test_points"],
            "best_model_label": res["best_model_label"], "best_model_rmse": res["best_model_rmse"],
            "prob_logistic1": res["prob_logistic1"], "prob_gompertz": res["prob_gompertz"], "prob_logistic2": res["prob_logistic2"], "prob_arima": res["prob_arima"], "prob_naive": res["prob_naive"],
            "tendency_confidence": res["tendency_confidence"], "logisticness_score": res["logisticness_score"],
            "logistic_params": json.dumps(res["logistic_params"], ensure_ascii=False),
            "gompertz_params": json.dumps(res["gompertz_params"], ensure_ascii=False),
            "logistic2_params": json.dumps(res["logistic2_params"], ensure_ascii=False),
        })
        long_rows.extend(res["rows"])

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False, encoding="utf-8-sig")
    scores_long = pd.DataFrame(long_rows)
    scores_long.to_csv(os.path.join(out_dir, "scores_long.csv"), index=False, encoding="utf-8-sig")
    print("Done. Wrote:", os.path.join(out_dir,"summary.csv"), "and", os.path.join(out_dir,"scores_long.csv"))

if __name__ == "__main__":
    main()
