
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

def _sir_rhs(state, t, beta, gamma):
    S, I, R = state
    dS = -beta * S * I
    dI = beta * S * I - gamma * I
    dR = gamma * I
    return [dS, dI, dR]

def _sir_prop(y0, t, beta, gamma):
    sol = odeint(_sir_rhs, y0, t, args=(beta, gamma))
    S, I, R = sol.T
    return S, I, R

def _I_obs_model(t, A, beta, gamma, i0_prop):
    i0 = float(np.clip(i0_prop, 1e-8, 0.2))
    s0 = 1.0 - i0
    r0 = 0.0
    S, I, R = _sir_prop([s0, i0, r0], t, beta, gamma)
    return A * I

SIZE_CANDIDATES = [
    "tokens","token","n_tokens","total_tokens","corpus_tokens","corpus_size","size","n","N","docs"
]

def _infer_freq_idx(times: pd.Series) -> str:
    t = pd.to_datetime(times, errors="coerce")
    t = t.dropna().sort_values()
    if len(t) < 3:
        return "D"
    deltas = t.diff().dropna().dt.days.values
    med = np.median(deltas) if len(deltas) else 1
    if 25 <= med <= 35: return "MS"
    if 5  <= med <= 9:  return "W"
    return "D"

def _read_timeseries(path: Path, normalize: str = "auto", per_million: float = 1_000_000.0):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    time_col = "time" if "time" in df.columns else df.columns[0]
    count_col = "count" if "count" in df.columns else df.columns[1]
    t = pd.to_datetime(df[time_col], errors="coerce")
    y_count = pd.to_numeric(df[count_col], errors="coerce")
    out = pd.DataFrame({"time": t, "count": y_count})
    rate = None
    size_col_found = None
    if normalize.lower() != "none":
        for cand in SIZE_CANDIDATES:
            if cand.lower() in df.columns:
                size_col_found = cand
                size = pd.to_numeric(df[cand.lower()], errors="coerce")
                with np.errstate(divide="ignore", invalid="ignore"):
                    rate = (y_count / size) * per_million
                break
    if normalize.lower() == "auto" and rate is None:
        rate = y_count.astype(float)
    elif normalize.lower() == "pmw" and rate is None:
        rate = y_count.astype(float)
    out["rate"] = rate
    out = out.dropna(subset=["time","rate"]).sort_values("time")
    out.attrs["size_col"] = size_col_found
    return out

def _apply_smoother(y, method="ma", ma_window=5, sg_window=11, sg_poly=2):
    y = np.asarray(y, float)
    if method == "sgolay":
        w = sg_window if sg_window % 2 == 1 else sg_window + 1
        w = max(w, sg_poly + 3 if (sg_poly + 3) % 2 == 1 else sg_poly + 4)
        w = min(w, len(y)-1 if (len(y)-1) % 2 == 1 else len(y)-2)
        if w < sg_poly + 2 or w < 5:
            return pd.Series(y).rolling(ma_window, center=True, min_periods=1).mean().values
        return savgol_filter(y, window_length=w, polyorder=sg_poly, mode="interp")
    return pd.Series(y).rolling(ma_window, center=True, min_periods=1).mean().values

def _regularize_and_smooth(df: pd.DataFrame, smoother: str, ma_window: int, sg_window: int, sg_poly: int, eps: float):
    """Regularize to inferred frequency with proper aggregation (sum for W/MS; fill for D), smooth, and trim leading ~zeros."""
    freq_alias = _infer_freq_idx(df["time"])
    s = pd.Series(df["rate"].values, index=pd.to_datetime(df["time"])).sort_index()
    if freq_alias == "MS":
        s = s.resample("MS").sum()
    elif freq_alias == "W":
        s = s.resample("W").sum()
    else:  # "D" or fallback
        s = s.resample("D").sum()  # sum to be safe; identical to filling when data are daily events
    s = s.fillna(0.0)
    sm = _apply_smoother(s.values, method=smoother, ma_window=ma_window, sg_window=sg_window, sg_poly=sg_poly)
    df_reg = pd.DataFrame({"time": s.index, "rate_sm": sm, "rate": s.values})
    arr = df_reg["rate_sm"].values
    idx = np.argmax(arr > eps) if np.any(arr > eps) else 0
    df_trim = df_reg.iloc[idx:].reset_index(drop=True)
    return df_trim, freq_alias

def _crop_by_threshold(y, frac=0.05, pad=5, min_len=20):
    y = np.asarray(y, float)
    if not np.isfinite(y).any() or np.nanmax(y) <= 0:
        return slice(0, len(y))
    thr = float(frac) * float(np.nanmax(y))
    idx = np.where(y >= thr)[0]
    if idx.size == 0:
        return slice(0, len(y))
    start = max(int(idx[0]) - int(pad), 0)
    end   = min(int(idx[-1]) + int(pad), len(y)-1)
    if end - start + 1 < int(min_len):
        pad2 = (int(min_len) - (end - start + 1)) // 2 + 1
        start = max(0, start - pad2)
        end   = min(len(y)-1, end + pad2)
    return slice(start, end+1)

def _fit_sir_rate(df_trim: pd.DataFrame,
                  crop_frac: float = 0.05, crop_pad: int = 5, crop_min_len: int = 20,
                  beta_max: float = 20.0, gamma_max: float = 20.0,
                  min_points: int = 10):
    t_full = np.arange(len(df_trim), dtype=float)
    y_full = df_trim["rate_sm"].values.astype(float)
    sl = _crop_by_threshold(y_full, frac=crop_frac, pad=crop_pad, min_len=crop_min_len)
    t = t_full[sl]; y = y_full[sl]
    if len(y) < int(min_points) or np.nansum(y) == 0:
        return None, None, np.nan, sl
    lb = np.array([1e-8, 1e-8, 1e-8, 1e-8], dtype=float)
    ub = np.array([1e12, beta_max, gamma_max, 0.2], dtype=float)
    def _residuals(p):
        A, beta, gamma, i0 = p
        yhat = _I_obs_model(t, A, beta, gamma, i0)
        return yhat - y
    A0 = max(1.0, float(np.nanmax(y)))
    cand_beta  = [0.1, 0.25, 0.4]
    cand_gamma = [0.05, 0.15, 0.3]
    cand_i0    = [1e-4, 0.005, 0.02]
    best = None
    scale = np.median(np.abs(y - np.median(y))) + 1e-9
    for b0 in cand_beta:
        for g0 in cand_gamma:
            for i0_0 in cand_i0:
                p0 = np.array([A0, b0, g0, i0_0], dtype=float)
                res = least_squares(_residuals, p0, bounds=(lb, ub),
                                    loss='soft_l1', f_scale=scale, max_nfev=120000)
                p = res.x
                yhat = _I_obs_model(t, *p)
                ss_res = float(np.nansum((y - yhat)**2))
                ss_tot = float(np.nansum((y - np.nanmean(y))**2))
                r2_try = 1 - ss_res/ss_tot if ss_tot > 0 else -np.inf
                if (best is None) or (r2_try > best["r2"]):
                    best = {"p": p, "r2": r2_try, "yhat": yhat}
    if best is None:
        return None, None, np.nan, sl
    A_fit, beta_fit, gamma_fit, i0_fit = [float(v) for v in best["p"]]
    yhat = best["yhat"]; r2 = float(best["r2"])
    return (A_fit, beta_fit, gamma_fit, i0_fit), (t, y, yhat), r2, sl

def _plot_fit(df_trim: pd.DataFrame, packed, title: str, out_png: Path, sl: slice):
    t_full = np.arange(len(df_trim), dtype=float)
    y_full = df_trim["rate_sm"].values.astype(float)
    t, y, yhat = packed
    plt.figure(figsize=(6.6, 4.2))
    plt.plot(t_full, y_full, linewidth=1, label="Smoothed rate (full)")
    plt.scatter(t, y, s=16, label="Fit window (data)")
    plt.plot(t, yhat, linewidth=2, label="SIR fit (I(t))")
    plt.xlabel("Time (regular steps from first non-zero)")
    plt.ylabel("Rate (counts/pmw, smoothed)")
    plt.title(title); plt.grid(True, linestyle="--", alpha=0.3); plt.legend()
    plt.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight", dpi=300); plt.close()

def process_directory(input_dir: Path, output_dir: Path,
                      normalize: str = "auto", per_million: float = 1_000_000.0,
                      recursive: bool = False,
                      smoother: str = "ma", ma_window: int = 5, sg_window: int = 11, sg_poly: int = 2,
                      eps: float = 0.0,
                      crop_frac: float = 0.05, crop_pad: int = 5, crop_min_len: int = 20,
                      beta_max: float = 10.0, gamma_max: float = 10.0,
                      min_points: int = 10):
    input_dir = Path(input_dir); output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    glb = input_dir.rglob if recursive else input_dir.glob
    files = sorted(list(glb("*.csv")) + list(glb("*.xlsx")))
    rows = []
    for fp in files:
        name = fp.stem
        try:
            df = _read_timeseries(fp, normalize=normalize, per_million=per_million)
            size_col = df.attrs.get("size_col", None)
            df_trim, freq_alias = _regularize_and_smooth(df, smoother, ma_window, sg_window, sg_poly, eps)
            params, packed, r2, sl = _fit_sir_rate(df_trim,
                                                   crop_frac=crop_frac, crop_pad=crop_pad, crop_min_len=crop_min_len,
                                                   beta_max=beta_max, gamma_max=gamma_max, min_points=min_points)
            if params is None or packed is None or not np.isfinite(r2):
                rows.append({"file": fp.name, "word": name, "status": "fail",
                             "freq_alias": freq_alias, "normalized_by": size_col or "",
                             "r2": np.nan, "A": np.nan, "beta": np.nan, "gamma": np.nan, "i0": np.nan,
                             "n_points": int(len(df_trim))})
                continue
            A_fit, beta_fit, gamma_fit, i0_fit = params
            title = f"{name} — SIR fit (monitor v2, monthly/weekly aggregated)"
            out_png = plots_dir / f"{name}_sir_fit.png"
            _plot_fit(df_trim, packed, title, out_png, sl)
            rows.append({"file": fp.name, "word": name, "status": "ok",
                         "freq_alias": freq_alias, "normalized_by": size_col or "",
                         "r2": float(r2), "A": float(A_fit), "beta": float(beta_fit), "gamma": float(gamma_fit), "i0": float(i0_fit),
                         "n_points": int(len(df_trim)), "crop_start": int(sl.start), "crop_end": int(sl.stop-1)})
        except Exception as e:
            rows.append({"file": fp.name, "word": name, "status": f"error: {e}",
                         "freq_alias": "", "normalized_by": "", "r2": np.nan,
                         "A": np.nan, "beta": np.nan, "gamma": np.nan, "i0": np.nan, "n_points": 0})
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "sir_fit_summary.csv"
    summary.to_csv(summary_path, index=False)
    okmask = (summary["status"].astype(str).str.lower() == "ok")
    r2vals = summary.loc[okmask, "r2"].dropna().astype(float).values
    if r2vals.size > 0:
        r2_sorted = np.sort(r2vals); ecdf = np.arange(1, len(r2_sorted)+1) / len(r2_sorted)
        plt.figure(figsize=(6.4, 4.2))
        plt.step(r2_sorted, ecdf, where="post", linewidth=2)
        xmin = min(0.0, float(r2_sorted.min()))
        plt.xlim(xmin, 1.0); plt.ylim(0, 1.0)
        plt.xlabel(r"$R^2$"); plt.ylabel("Cumulative share of items")
        plt.title("ECDF of SIR fit quality — monitor v2 (aggregated)")
        r2_med = float(np.median(r2_sorted))
        for x, ls in [(0.0, ":"), (0.6, "--"), (0.8, "--"), (r2_med, "-.")]:
            plt.axvline(x, linestyle=ls, linewidth=1.5)
        def share_le(val): return float((r2_sorted <= val).sum()/len(r2_sorted))
        txt = (f"≤0.00: {share_le(0.0):.2%}\n"
               f"≤0.60: {share_le(0.6):.2%}\n"
               f"≤0.80: {share_le(0.8):.2%}\n"
               f"Median: {r2_med:.2f}")
        plt.text(0.02, 0.05, txt, transform=plt.gca().transAxes, va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.95))
        plt.tight_layout(); plt.savefig(output_dir / "sir_r2_ecdf.png", bbox_inches="tight", dpi=300); plt.close()
    return summary_path, plots_dir

def main():
    ap = argparse.ArgumentParser(description="Batch SIR fits — Monitor corpus (v2 fixed aggregation)")
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--normalize", type=str, default="auto", choices=["auto","none","pmw"])
    ap.add_argument("--per_million", type=float, default=1_000_000.0)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--smoother", type=str, default="ma", choices=["ma","sgolay"])
    ap.add_argument("--ma_window", type=int, default=5)
    ap.add_argument("--sg_window", type=int, default=11)
    ap.add_argument("--sg_poly", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--crop_frac", type=float, default=0.05)
    ap.add_argument("--crop_pad", type=int, default=5)
    ap.add_argument("--crop_min_len", type=int, default=20)
    ap.add_argument("--beta_max", type=float, default=10.0)
    ap.add_argument("--gamma_max", type=float, default=10.0)
    ap.add_argument("--min_points", type=int, default=10)
    args = ap.parse_args()
    summary_path, plots_dir = process_directory(
        input_dir=args.input_dir, output_dir=args.output_dir,
        normalize=args.normalize, per_million=args.per_million,
        recursive=args.recursive, smoother=args.smoother, ma_window=args.ma_window, sg_window=args.sg_window, sg_poly=args.sg_poly,
        eps=args.eps, crop_frac=args.crop_frac, crop_pad=args.crop_pad, crop_min_len=args.crop_min_len,
        beta_max=args.beta_max, gamma_max=args.gamma_max, min_points=args.min_points
    )
    print(f"[Done] Summary: {summary_path}"); print(f"[Done] Plots: {plots_dir}")

if __name__ == "__main__":
    main()
