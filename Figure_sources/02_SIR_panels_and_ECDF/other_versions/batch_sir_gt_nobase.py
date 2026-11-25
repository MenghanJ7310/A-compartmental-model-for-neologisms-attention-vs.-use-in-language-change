#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch SIR fits for Google Trends series (time,freq), NO baseline.
- Regularize to monthly average (GT is an index).
- Smooth with centered moving average (window configurable).
- Fit window = peak-centered ± half_span (default 8), min_len 20.
- Fit ŷ(t) = A * I(t; β, γ, i0) with OLS (least_squares, loss='linear').
- Bounds: β,γ ∈ (1e-8, beta_max/gamma_max], i0 ∈ (1e-8, 0.2], A > 0.
- Multi-start (3 starts by default) to avoid local minima.

Usage (Windows):
python batch_sir_gt_nobase.py --input_dir "C:\path\GT_data" --output_dir "C:\path\gt_sir_nobase" --ma_window 3 --half_span 8 --min_len 20 --beta_max 40 --gamma_max 40
"""

import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import odeint

def read_gt_series(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = "time" if "time" in df.columns else df.columns[0]
    vcol = "freq" if "freq" in df.columns else df.columns[1]
    t = pd.to_datetime(df[tcol], errors="coerce")
    y = pd.to_numeric(df[vcol], errors="coerce")
    s = pd.Series(y.values, index=t).dropna()
    m = s.resample("MS").mean().fillna(0.0)  # monthly average for GT
    return m.index.to_period("M").to_timestamp(), m.values.astype(float)

def ma(x, w=3):
    s = pd.Series(x)
    return s.rolling(w, center=True, min_periods=1).mean().values

def peak_window(y_sm, half_span=8, min_len=20):
    if len(y_sm) == 0 or np.nanmax(y_sm) <= 0:
        return slice(0, len(y_sm))
    p = int(np.nanargmax(y_sm))
    start = max(0, p - half_span)
    end = min(len(y_sm) - 1, p + half_span)
    if end - start + 1 < min_len:
        need = min_len - (end - start + 1)
        start = max(0, start - need//2)
        end = min(len(y_sm) - 1, end + (need - need//2))
    return slice(start, end+1)

# ----- SIR (no baseline) -----
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

def model(params, t):
    A, beta, gamma, i0 = params
    return A * sir_I(t, beta, gamma, i0)

def fit_sir_nobase(y, beta_max=40.0, gamma_max=40.0):
    t = np.arange(len(y), dtype=float)
    best = None; best_ss = np.inf
    ymax = max(1.0, float(np.nanmax(y)))
    lb = [1e-8, 1e-8, 1e-8, 1e-8]
    ub = [1e12, beta_max, gamma_max, 0.2]
    # small multi-start set
    starts = [(ymax, 0.6, 0.4, 1e-3),
              (ymax, 2.0, 1.5, 1e-2),
              (2*ymax, 6.0, 4.0, 5e-2)]
    for p0 in starts:
        try:
            res = least_squares(lambda p: model(p, t) - y, x0=p0, bounds=(lb,ub), loss='linear', max_nfev=30000)
            if res.success:
                yhat = model(res.x, t); ss = float(np.sum((yhat - y)**2))
                if ss < best_ss: best_ss, best = ss, (res.x, yhat)
        except Exception:
            pass
    return best  # (params, yhat)

def r2_score(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

def plot_one(word, y_sm, sl, y_win, yhat, r2, out_png):
    plt.figure(figsize=(7.5, 4.6), dpi=200)
    plt.plot(np.arange(len(y_sm)), y_sm, lw=1.2, label="Smoothed freq (full)")
    plt.scatter(np.arange(sl.start, sl.stop), y_win, s=16, label="Fit window (data)")
    plt.plot(np.arange(sl.start, sl.stop), yhat, lw=2.2, label=f"SIR (no baseline)  R$^2$={r2:.2f}")
    plt.axvspan(sl.start, sl.stop-1, color="0.92", zorder=0)
    plt.xlabel("Time (regular steps from first non-zero)")
    plt.ylabel("Frequency (smoothed)")
    plt.title(f"{word} — SIR fit (no baseline, peak±window)")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

def process_dir(input_dir: Path, output_dir: Path, ma_window=3, half_span=8, min_len=20, beta_max=40.0, gamma_max=40.0):
    input_dir = Path(input_dir); output_dir = Path(output_dir)
    plots = output_dir / "plots"; plots.mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(list(input_dir.glob("*.csv")))
    for fp in files:
        try:
            idx, y = read_gt_series(fp)
            y_sm = ma(y, ma_window)
            sl = peak_window(y_sm, half_span=half_span, min_len=min_len)
            y_win = y_sm[sl]
            if len(y_win) < 10 or np.allclose(y_win, 0):
                rows.append({"file": fp.name, "word": fp.stem, "status": "insufficient", "r2": np.nan})
                continue
            fit = fit_sir_nobase(y_win, beta_max=beta_max, gamma_max=gamma_max)
            if not fit:
                rows.append({"file": fp.name, "word": fp.stem, "status": "fit_fail", "r2": np.nan})
                continue
            params, yhat = fit
            A,beta,gamma,i0 = [float(x) for x in params]
            r2 = float(r2_score(y_win, yhat))
            plot_one(fp.stem, y_sm, sl, y_win, yhat, r2, plots / f"{fp.stem}_sir_nobase.png")
            rows.append({
                "file": fp.name, "word": fp.stem, "status": "ok",
                "r2": r2, "A": A, "beta": beta, "gamma": gamma, "i0": i0,
                "window_start": int(sl.start), "window_end": int(sl.stop), "n_points": int(len(y_win))
            })
        except Exception as e:
            rows.append({"file": fp.name, "word": fp.stem, "status": f"error: {e}", "r2": np.nan})
    summary = pd.DataFrame(rows)
    out_csv = output_dir / "GT_SIR_nobase_summary.csv"
    summary.to_csv(out_csv, index=False)

    # ECDF of R2
    r2vals = summary["r2"].dropna().values
    if r2vals.size > 0:
        r2s = np.sort(r2vals); ecdf = np.arange(1, len(r2s)+1)/len(r2s)
        plt.figure(figsize=(6.8, 4.2), dpi=180)
        plt.step(r2s, ecdf, where="post", lw=2)
        for x, ls in [(0.0, ":"), (0.6, "--"), (0.8, "--")]:
            plt.axvline(x, ls=ls, lw=1.5, color="0.3")
        plt.xlim(min(0.0, float(r2s.min())), 1.0); plt.ylim(0, 1.0)
        plt.xlabel(r"$R^2$"); plt.ylabel("Cumulative share of items")
        plt.title("ECDF of SIR fit quality (no baseline)")
        plt.tight_layout()
        plt.savefig(output_dir / "GT_SIR_nobase_R2_ECDF.png", bbox_inches="tight")
        plt.close()
    return str(out_csv), str(plots)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ma_window", type=int, default=3)
    ap.add_argument("--half_span", type=int, default=8)
    ap.add_argument("--min_len", type=int, default=20)
    ap.add_argument("--beta_max", type=float, default=40.0)
    ap.add_argument("--gamma_max", type=float, default=40.0)
    args = ap.parse_args()
    out_csv, plots = process_dir(args.input_dir, args.output_dir, args.ma_window, args.half_span, args.min_len, args.beta_max, args.gamma_max)
    print(f"[Done] Summary: {out_csv}")
    print(f"[Done] Plots:   {plots}")

if __name__ == "__main__":
    main()
