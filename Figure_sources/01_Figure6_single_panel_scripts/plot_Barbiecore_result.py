
# -*- coding: utf-8 -*-
# Nature Human Behaviour–style plotting for a single word.
# Usage:
#   python plot_Barbiecore_result.py
# Requirements: pandas, numpy, matplotlib, (and statsmodels for ARIMA where used)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Unified style ---
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10
})
deep_green = "#007200"   # increments
deep_red   = "#B00020"   # prediction

from scipy.optimize import curve_fit

CSV_NAME = "Barbiecore_result.csv"
OUT_PNG  = "fig_Barbiecore_result_final.png"
TITLE    = "Barbiecore_result: Increment — Logistic-derived forecast (test)"

df = pd.read_csv(CSV_NAME)
# Detect numeric series
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    for c in df.columns:
        try: df[c] = pd.to_numeric(df[c], errors="coerce")
        except: pass
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
series_col = max(num_cols, key=lambda c: df[c].var(skipna=True))

y = pd.to_numeric(df[series_col], errors="coerce").fillna(0).astype(float).values
y[y<0] = 0.0
n = len(y); t = np.arange(n, dtype=float); i = int(round(n*0.7))
N = np.cumsum(y)

def logistic_cum(t, K, r, t0): return K/(1.0 + np.exp(-r*(t - t0)))
def logistic_inc(t, K, r, t0): return logistic_cum(t, K, r, t0) - logistic_cum(t-1.0, K, r, t0)

# Fit logistic on cumulative (train), predict increments on test
K0 = max(N[:i][-1], 1.0); r0 = 0.1; t00 = float(np.median(t[:i]))
popt, _ = curve_fit(logistic_cum, t[:i], N[:i],
                    p0=[K0, r0, t00],
                    bounds=([1e-6, 1e-6, -np.inf],[np.inf, 10.0, np.inf]),
                    maxfev=20000)
K, r, t0 = [float(x) for x in popt]
yhat = logistic_inc(t[i:], K, r, t0)

# --- Plot ---
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(t, y, width=6, align="center", color=deep_green, alpha=0.4, label="Increment (observed)")
ax.plot(t[i:], yhat, color=deep_red, linewidth=2.4, label="Logistic-derived forecast (test)")
ax.axvline(i, linestyle="--", color="gray", linewidth=1)
ax.set_xlabel("Time index")
ax.set_ylabel("Increment")
ax.set_title(TITLE)
ax.legend(frameon=False, loc="upper left")
ax.grid(False)
for spine in ax.spines.values(): spine.set_visible(True)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print("Saved:", OUT_PNG)
