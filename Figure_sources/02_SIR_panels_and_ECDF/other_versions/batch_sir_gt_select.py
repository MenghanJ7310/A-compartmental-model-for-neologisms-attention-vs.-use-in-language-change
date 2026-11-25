#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch SIR fits for Google Trends data with per-series model selection.

- Input: CSV files with columns (time, freq). 'freq' is the GT 0–100 index.
- Preprocess: monthly average, centered MA smoothing (default MA-3).
- Window: peak-centered ±half_span (default 8), min_len=20.
- Models:
    M0 (nobase):   yhat = A * I(t; beta, gamma, i0)
    M1 (baseline): yhat = B + A * I(t; beta, gamma, i0), B >= 0
- Fitting: OLS (scipy.optimize.least_squares, loss='linear'), bounds beta,gamma <= 40 by default.
- Selection (unified rules):
    Step 1) Fit both models on TRAIN subset (last val_k points held out).
            Compute RSS_train and AICc_train for each.
    Step 2) Prefer M1 if ΔAICc = AICc(M0) - AICc(M1) >= aicc_delta  (default 4.0);
            Else if val MSE(M1) <= (1 - val_improve) * val MSE(M0) (default val_improve=0.20);
            Else choose M0.
    Step 3) Refit the chosen model on FULL window for final parameters & R^2/adj-R^2, and for plotting.
- Outputs:
    - summary CSV with all metrics (train AICc, val MSE, full-window R^2/adjR^2, params for both models, chosen_model, decision_reason).
    - per-series overlay plot (both models over FULL window, with R^2 in legend; chosen model marked in title).
    - ECDF of chosen-model R^2 across all items.

Usage (Windows):
python batch_sir_gt_select.py --input_dir "C:\path\GT_data" --output_dir "C:\path\GT_SIR_selected" \
  --ma_window 3 --half_span 8 --min_len 20 --beta_max 40 --gamma_max 40 --val_k 3 --aicc_delta 4 --val_improve 0.2

"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from scipy.integrate import odeint

# --------------------- IO & Preprocess ---------------------

def read_gt_series(csv_path: Path):
    """Robust reader for GT CSV: columns time,freq; monthly AVERAGE; drop NaT rows; fallback formats."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = "time" if "time" in df.columns else df.columns[0]
    vcol = "freq" if "freq" in df.columns else df.columns[1]
    # initial wide parsing
    t = pd.to_datetime(df[tcol], errors="coerce", infer_datetime_format=True)
    y = pd.to_numeric(df[vcol], errors="coerce")
    # fallback formats if all NaT
    if t.notna().sum() == 0:
        for fmt in ["%b-%y", "%Y-%m", "%Y/%m", "%m/%d/%Y", "%Y/%m/%d"]:
            try:
                t_try = pd.to_datetime(df[tcol].astype(str), format=fmt, errors="coerce")
                if t_try.notna().sum() > 0:
                    t = t_try
                    break
            except Exception:
                pass
    mask = t.notna() & y.notna()
    t = t[mask]; y = y[mask]
    if len(t) == 0:
        raise ValueError("no valid timestamps after parsing; please check time formats")
    s = pd.Series(y.values, index=t)
    # Google Trends: intensity index → average within month
    m = s.resample("MS").mean().fillna(0.0)
    return m.index.to_period("M").to_timestamp(), m.values.astype(float)

def ma(x, w=3):
    s = pd.Series(x)
    return s.rolling(w, center=True, min_periods=1).mean().values

def peak_window(y_sm, half_span=8, min_len=20):
    """Window is centered at the global peak (in smoothed series)."""
    if len(y_sm) == 0 or np.nanmax(y_sm) <= 0:
        return slice(0, len(y_sm))
    p = int(np.nanargmax(y_sm))
    start = max(0, p - half_span)
    end = min(len(y_sm) - 1, p + half_span)
    if end - start + 1 < min_len:
        need = min_len - (end - start + 1)
        start = max(0, start - need//2)
        end = min(len(y_sm)-1, end + (need - need//2))
    return slice(start, end+1)

# --------------------- SIR Dynamics ---------------------

def sir_rhs(state, t, beta, gamma):
    S, I, R = state
    dS = -beta*S*I
    dI = beta*S*I - gamma*I
    dR = gamma*I
    return [dS, dI, dR]

def sir_I(t, beta, gamma, i0):
    s0 = 1.0 - i0
    y0 = [s0, i0, 0.0]
    sol = odeint(sir_rhs, y0, t, args=(beta, gamma))
    return sol[:,1]

def model_nobase(p, t):
    A, beta, gamma, i0 = p
    return A * sir_I(t, beta, gamma, i0)

def model_baseline(p, t):
    B, A, beta, gamma, i0 = p
    return B + A * sir_I(t, beta, gamma, i0)

# --------------------- Fitting utils ---------------------

def fit_least_squares(y, version=0, beta_max=40.0, gamma_max=40.0, max_nfev=30000):
    """Fit model on provided y (1D numpy) using OLS (no robust loss), small multi-start grid."""
    t = np.arange(len(y), dtype=float)
    ymax = max(1.0, float(np.nanmax(y)))
    if version == 0:
        # M0: nobase (A, beta, gamma, i0)
        lb = [1e-8, 1e-8, 1e-8, 1e-8]
        ub = [1e12, beta_max, gamma_max, 0.2]
        starts = [(ymax, 0.6, 0.4, 1e-3),
                  (ymax, 2.0, 1.5, 1e-2),
                  (2*ymax, 6.0, 4.0, 5e-2)]
        best = None; best_ss = np.inf
        for p0 in starts:
            try:
                res = least_squares(lambda p: model_nobase(p, t) - y, x0=p0, bounds=(lb,ub), loss='linear', max_nfev=max_nfev)
                if res.success:
                    yhat = model_nobase(res.x, t)
                    ss = float(np.sum((y - yhat)**2))
                    if ss < best_ss:
                        best_ss, best = ss, (res.x, yhat)
            except Exception:
                pass
        return best
    else:
        # M1: baseline (B, A, beta, gamma, i0)
        lb = [0.0, 1e-8, 1e-8, 1e-8, 1e-8]
        ub = [1e6, 1e12, beta_max, gamma_max, 0.2]
        B0 = max(0.0, float(np.nanmedian(y[-min(10, len(y)):]])))
        starts = [(B0, ymax, 0.6, 0.4, 1e-3),
                  (B0, ymax, 2.0, 1.5, 1e-2),
                  (B0, 2*ymax, 6.0, 4.0, 5e-2)]
        best = None; best_ss = np.inf
        for p0 in starts:
            try:
                res = least_squares(lambda p: model_baseline(p, t) - y, x0=p0, bounds=(lb,ub), loss='linear', max_nfev=max_nfev+10000)
                if res.success:
                    yhat = model_baseline(res.x, t)
                    ss = float(np.sum((y - yhat)**2))
                    if ss < best_ss:
                        best_ss, best = ss, (res.x, yhat)
            except Exception:
                pass
        return best

def metrics_train(y_train, fit):
    """Return RSS_train, AICc_train, k, n for a fitted model (params known)."""
    if fit is None:
        return np.inf, np.inf, 0, len(y_train)
    params, yhat = fit
    n = len(y_train)
    k = len(params)
    rss = float(np.sum((y_train - yhat)**2))
    aicc = np.inf
    if n > (k + 1):
        aic = n * np.log(rss / n) + 2 * k
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    return rss, aicc, k, n

def r2_and_adj(y_full, yhat_full, k):
    y = np.asarray(y_full, float); yhat = np.asarray(yhat_full, float)
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    n = len(y)
    adj = np.nan
    if n > (k + 1) and not np.isnan(r2):
        adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
    return r2, adj, ss_res

# --------------------- Pipeline per series ---------------------

def process_one(csv_path: Path, ma_window=3, half_span=8, min_len=20,
                beta_max=40.0, gamma_max=40.0, val_k=3, aicc_delta=4.0, val_improve=0.20):
    idx, y = read_gt_series(csv_path)
    y_sm = ma(y, ma_window)
    sl = peak_window(y_sm, half_span=half_span, min_len=min_len)
    y_win = y_sm[sl]

    if len(y_win) < 10 or np.allclose(y_win, 0):
        return {"file": csv_path.name, "word": csv_path.stem, "status": "insufficient",
                "n_window": int(len(y_win)), "window_start": int(sl.start), "window_end": int(sl.stop)}

    # Split train/val for selection
    n = len(y_win)
    vk = int(val_k)
    if vk >= n - 6:   # ensure enough DOF for both models
        vk = max(0, n - 6)
    train = slice(0, n - vk) if vk > 0 else slice(0, n)
    val = slice(n - vk, n) if vk > 0 else slice(n, n)
    y_tr = y_win[train]
    y_va = y_win[val]

    # Fit both on TRAIN
    fit0_tr = fit_least_squares(y_tr, version=0, beta_max=beta_max, gamma_max=gamma_max)
    fit1_tr = fit_least_squares(y_tr, version=1, beta_max=beta_max, gamma_max=gamma_max)
    rss0, aicc0, k0, n0 = metrics_train(y_tr, fit0_tr)
    rss1, aicc1, k1, n1 = metrics_train(y_tr, fit1_tr)

    # Validation MSE if val_k>0
    mse0 = mse1 = np.nan
    if val_k > 0 and len(y_va) > 0:
        # predict on VAL using train fits
        if fit0_tr is not None:
            p0 = fit0_tr[0]
            t_val = np.arange(len(y_va), dtype=float)
            from_params0 = model_nobase(p0, t_val)
            mse0 = float(np.mean((y_va - from_params0)**2))
        if fit1_tr is not None:
            p1 = fit1_tr[0]
            t_val = np.arange(len(y_va), dtype=float)
            from_params1 = model_baseline(p1, t_val)
            mse1 = float(np.mean((y_va - from_params1)**2))

    # Selection logic
    decision_reason = "nobase_default"
    choose = "nobase"
    dAICc = aicc0 - aicc1
    if (not np.isinf(aicc0)) and (not np.isinf(aicc1)) and (dAICc >= aicc_delta):
        choose = "baseline"; decision_reason = f"AICc≥{aicc_delta} (Δ={dAICc:.2f})"
    elif np.isfinite(mse0) and np.isfinite(mse1) and (mse1 <= (1.0 - val_improve) * mse0):
        choose = "baseline"; decision_reason = f"val_improve≥{int(val_improve*100)}% (MSE0={mse0:.3g}, MSE1={mse1:.3g})"

    # Final fits on FULL window for reporting/plots
    fit0 = fit_least_squares(y_win, version=0, beta_max=beta_max, gamma_max=gamma_max)
    fit1 = fit_least_squares(y_win, version=1, beta_max=beta_max, gamma_max=gamma_max)

    # Metrics on FULL window
    if fit0 is not None:
        p0, yhat0 = fit0; r2_0, adj_0, ss_0 = r2_and_adj(y_win, yhat0, k=len(p0))
    else:
        p0, yhat0, r2_0, adj_0, ss_0 = [np.nan]*4, np.zeros_like(y_win), np.nan, np.nan, np.nan
    if fit1 is not None:
        p1, yhat1 = fit1; r2_1, adj_1, ss_1 = r2_and_adj(y_win, yhat1, k=len(p1))
    else:
        p1, yhat1, r2_1, adj_1, ss_1 = [np.nan]*5, np.zeros_like(y_win), np.nan, np.nan, np.nan

    result = {
        "file": csv_path.name, "word": csv_path.stem, "status": "ok",
        "window_start": int(sl.start), "window_end": int(sl.stop), "n_window": int(n),
        "ma_window": int(ma_window), "half_span": int(half_span),
        "val_k": int(vk), "aicc_delta": float(aicc_delta), "val_improve": float(val_improve),
        # train metrics
        "RSS_train_nobase": float(rss0), "AICc_train_nobase": float(aicc0),
        "RSS_train_baseline": float(rss1), "AICc_train_baseline": float(aicc1),
        "val_MSE_nobase": float(mse0), "val_MSE_baseline": float(mse1),
        # full metrics
        "R2_nobase_full": float(r2_0), "adjR2_nobase_full": float(adj_0), "SSres_nobase_full": float(ss_0),
        "R2_baseline_full": float(r2_1), "adjR2_baseline_full": float(adj_1), "SSres_baseline_full": float(ss_1),
        # params
        "A_nobase": float(p0[0]) if fit0 else np.nan,
        "beta_nobase": float(p0[1]) if fit0 else np.nan,
        "gamma_nobase": float(p0[2]) if fit0 else np.nan,
        "i0_nobase": float(p0[3]) if fit0 else np.nan,
        "B_baseline": float(p1[0]) if fit1 else np.nan,
        "A_baseline": float(p1[1]) if fit1 else np.nan,
        "beta_baseline": float(p1[2]) if fit1 else np.nan,
        "gamma_baseline": float(p1[3]) if fit1 else np.nan,
        "i0_baseline": float(p1[4]) if fit1 else np.nan,
    }

    # final choice
    result["chosen_model"] = choose
    result["decision_reason"] = decision_reason
    return result, (y_sm, sl, y_win, (fit0, fit1))

# --------------------- Plotting ---------------------

def plot_overlay(word, y_sm, sl, y_win, fit0, fit1, out_png, chosen="nobase"):
    plt.figure(figsize=(7.8, 4.6), dpi=200)
    plt.plot(np.arange(len(y_sm)), y_sm, lw=1.2, label="Smoothed freq (full)")
    plt.scatter(np.arange(sl.start, sl.stop), y_win, s=16, label="Fit window (data)")

    legend_items = []
    if fit0 is not None:
        p0, yhat0 = fit0
        r2_0, _, _ = r2_and_adj(y_win, yhat0, len(p0))
        plt.plot(np.arange(sl.start, sl.stop), yhat0, lw=2.2, label=f"SIR (no baseline)  R$^2$={r2_0:.2f}")
    if fit1 is not None:
        p1, yhat1 = fit1
        r2_1, _, _ = r2_and_adj(y_win, yhat1, len(p1))
        plt.plot(np.arange(sl.start, sl.stop), yhat1, lw=2.2, label=f"SIR + baseline  R$^2$={r2_1:.2f}")

    plt.axvspan(sl.start, sl.stop-1, color="0.92", zorder=0)
    tag = "baseline" if chosen == "baseline" else "no baseline"
    plt.title(f"{word} — Selected: {tag} (peak±window)")
    plt.xlabel("Time (regular steps from first non-zero)")
    plt.ylabel("Frequency (smoothed)")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with Google Trends CSVs (time,freq)")
    ap.add_argument("--output_dir", required=True, help="Output folder")
    ap.add_argument("--ma_window", type=int, default=3)
    ap.add_argument("--half_span", type=int, default=8)
    ap.add_argument("--min_len", type=int, default=20)
    ap.add_argument("--beta_max", type=float, default=40.0)
    ap.add_argument("--gamma_max", type=float, default=40.0)
    ap.add_argument("--val_k", type=int, default=3, help="Hold-out size (points at end of window) for selection")
    ap.add_argument("--aicc_delta", type=float, default=4.0, help="Minimum AICc improvement to prefer baseline")
    ap.add_argument("--val_improve", type=float, default=0.20, help="Minimum relative MSE improvement on hold-out to prefer baseline")
    ap.add_argument("--save_plots", action="store_true", help="Save per-series overlay plots")
    args = ap.parse_args()

    in_dir = Path(args.input_dir); out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"; out_dir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    files = sorted(list(in_dir.glob("*.csv")))
    for fp in files:
        try:
            out, aux = process_one(fp, ma_window=args.ma_window, half_span=args.half_span, min_len=args.min_len,
                                   beta_max=args.beta_max, gamma_max=args.gamma_max,
                                   val_k=args.val_k, aicc_delta=args.aicc_delta, val_improve=args.val_improve)
            if isinstance(out, dict) and out.get("status") == "insufficient":
                rows.append(out)
                continue
            res, (y_sm, sl, y_win, (fit0, fit1)) = out, aux
            rows.append(res)
            if args.save_plots:
                plot_overlay(res["word"], y_sm, sl, y_win, fit0, fit1, plots_dir / f"{res['word']}_select_overlay.png", res["chosen_model"])
        except Exception as e:
            rows.append({"file": fp.name, "word": fp.stem, "status": f"error: {e}"})

    summary = pd.DataFrame(rows)
    out_csv = out_dir / "GT_SIR_selected_summary.csv"
    summary.to_csv(out_csv, index=False)

    # ECDF of chosen-model R^2 (use per-row chosen column to pick R2)
    def chosen_r2(row):
        if row.get("chosen_model") == "baseline":
            return row.get("R2_baseline_full")
        elif row.get("chosen_model") == "nobase":
            return row.get("R2_nobase_full")
        return np.nan

    r2vals = summary.apply(chosen_r2, axis=1).astype(float)
    r2vals = r2vals[np.isfinite(r2vals)]
    if r2vals.size > 0:
        r2s = np.sort(r2vals); ecdf = np.arange(1, len(r2s)+1)/len(r2s)
        plt.figure(figsize=(6.8, 4.2), dpi=180)
        plt.step(r2s, ecdf, where="post", lw=2)
        for x, ls in [(0.0, ":"), (0.6, "--"), (0.8, "--")]:
            plt.axvline(x, ls=ls, lw=1.5, color="0.3")
        plt.xlim(min(0.0, float(r2s.min())), 1.0); plt.ylim(0, 1.0)
        plt.xlabel(r"$R^2$")
        plt.ylabel("Cumulative share of items")
        plt.title("ECDF of chosen-model SIR fit quality (GT)")
        plt.tight_layout()
        plt.savefig(out_dir / "GT_SIR_selected_R2_ECDF.png", bbox_inches="tight")
        plt.close()

    print(f"[Done] Summary: {out_csv}")
    print(f"[Done] Plots:   {plots_dir} (only if --save_plots)")

if __name__ == "__main__":
    main()
