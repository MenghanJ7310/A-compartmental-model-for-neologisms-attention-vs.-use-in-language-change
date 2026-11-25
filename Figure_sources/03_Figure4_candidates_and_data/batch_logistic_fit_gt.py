# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 15:17:32 2025

@author: xiang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_series(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = "time" if "time" in df.columns else df.columns[0]
    ycol = "freq" if "freq" in df.columns else df.columns[1]
    t = pd.to_datetime(df[tcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce").fillna(0.0)
    out = (pd.DataFrame({"time": t, "freq": y})
             .dropna(subset=["time"])
             .sort_values("time")
             .set_index("time")
             .asfreq("MS", fill_value=0.0)
             .rename_axis("time")
             .reset_index())
    return out

def smooth_ma(y: np.ndarray, win: int = 3) -> np.ndarray:
    if win <= 1: return y
    s = pd.Series(y, dtype=float).rolling(win, center=True, min_periods=1).mean().values
    return np.asarray(s, dtype=float)

def logistic(t, L, k, t0):
    return L / (1.0 + np.exp(-k * (t - t0)))

def fit_logistic(time_index: np.ndarray,
                 y_flow: np.ndarray,
                 ma_window: int = 3,
                 norm_to_one: bool = True):
    # 1) smooth flow
    y_sm = smooth_ma(y_flow, ma_window).clip(min=0.0)
    # 2) cumulative
    C = np.cumsum(y_sm)
    if C[-1] <= 0:
        return None

    # 3) choose adoption window (1%~99% of final)
    C_end = C[-1]
    lo = np.argmax(C >= 0.01 * C_end)
    hi = len(C) - 1 - np.argmax(C[::-1] <= 0.99 * C_end)
    lo = max(lo - 2, 0); hi = min(hi + 2, len(C)-1)
    if hi - lo + 1 < 20:
        lo, hi = 0, len(C)-1
    t = np.arange(len(C), dtype=float)

    t_fit = t[lo:hi+1]
    C_fit = C[lo:hi+1]

    if norm_to_one:
        L0 = 1.0
        Cn = C_fit / C_fit[-1]
        # two-parameter fit: fix L=1
        def f2(t, k, t0): return logistic(t, 1.0, k, t0)
        p0 = [0.05, t_fit.mean()]
        popt, _ = curve_fit(f2, t_fit, Cn, p0=p0, bounds=([1e-6, -1e6], [10.0, 1e6]), maxfev=20000)
        k, t0 = [float(v) for v in popt]
        C_hat = logistic(t_fit, 1.0, k, t0) * C_fit[-1]  # scale back for plotting
        L = C_fit[-1]
    else:
        # three-parameter fit
        L0 = max(C_fit[-1], 1.0)
        p0 = [L0, 0.05, t_fit.mean()]
        bounds = ([1e-6, 1e-6, -1e6], [1e12, 10.0, 1e6])
        popt, _ = curve_fit(logistic, t_fit, C_fit, p0=p0, bounds=bounds, maxfev=20000)
        L, k, t0 = [float(v) for v in popt]
        C_hat = logistic(t_fit, L, k, t0)

    # R^2 on the adoption window
    ybar = float(np.mean(C_fit))
    ss_res = float(np.sum((C_fit - C_hat)**2))
    ss_tot = float(np.sum((C_fit - ybar)**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    return {
        "lo": int(lo), "hi": int(hi),
        "L": float(L), "k": float(k), "t0": float(t0),
        "r2": float(r2),
        "t_fit": t_fit, "C_fit": C_fit, "C_hat": C_hat,
        "t_all": t, "C_all": C, "y_sm": y_sm
    }

def plot_one(name: str, fitres: dict, out_png: Path):
    t, C = fitres["t_all"], fitres["C_all"]
    t_fit, C_hat = fitres["t_fit"], fitres["C_hat"]
    lo, hi = fitres["lo"], fitres["hi"]

    plt.figure(figsize=(8,5))
    # cumulative curve
    plt.plot(t, C, linewidth=2, label="Cumulative (from smoothed flow)")
    # points in fit window
    plt.scatter(np.arange(lo, hi+1), C[lo:hi+1], s=16, label="Fit window (data)")
    # logistic fit
    plt.plot(t_fit, C_hat, linewidth=2, label="Logistic fit (C(t))")
    plt.xlabel("Time (regular steps from start)")
    plt.ylabel("Cumulative index (arbitrary units)")
    plt.title(f"{name} — Logistic fit on cumulative GT")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Batch logistic fits on Google Trends time series (time,freq).")
    ap.add_argument("--input_dir", required=True, help="Folder with CSVs (time,freq)")
    ap.add_argument("--output_dir", required=True, help="Where to save summary & plots")
    ap.add_argument("--ma_window", type=int, default=3, help="Centered moving-average window")
    ap.add_argument("--norm_to_one", action="store_true", help="Normalize cumulative to [0,1] and fix L=1")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plots = out_dir / "plots"; plots.mkdir(exist_ok=True)
    rows = []

    files = sorted(list(in_dir.glob("*.csv")))
    for fp in files:
        name = fp.stem
        try:
            df = read_series(fp)
            fitres = fit_logistic(df["time"].values, df["freq"].values, ma_window=args.ma_window, norm_to_one=args.norm_to_one)
            if fitres is None:
                rows.append({"file": fp.name, "word": name, "status": "fail"})
                continue
            plot_one(name, fitres, plots / f"{name}_logistic_fit.png")
            rows.append({
                "file": fp.name, "word": name, "status": "ok",
                "L": fitres["L"], "k": fitres["k"], "t0": fitres["t0"], "r2": fitres["r2"],
                "n_points": int(len(fitres["t_all"]))
            })
        except Exception as e:
            rows.append({"file": fp.name, "word": name, "status": f"error: {e}"})

    pd.DataFrame(rows).to_csv(out_dir / "logistic_fit_summary_results.csv", index=False)
    print("[Done] Summary ->", out_dir / "logistic_fit_summary_results.csv")
    print("[Done] Plots   ->", plots)

if __name__ == "__main__":
    main()
