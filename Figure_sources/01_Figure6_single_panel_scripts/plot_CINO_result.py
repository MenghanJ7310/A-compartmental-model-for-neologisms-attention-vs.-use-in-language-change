
# -*- coding: utf-8 -*-
# Nature Human Behaviour–style plotting for a single word.
# Usage:
#   python plot_CINO_result.py
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

from statsmodels.tsa.arima.model import ARIMA

CSV_NAME = "CINO_result.csv"
OUT_PNG  = "fig_CINO_result_final.png"
TITLE    = "CINO_result: Increment — ARIMA forecast (test)"

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
n = len(y); t = np.arange(n); i = int(round(n*0.7))

# ARIMA(1,0,0) on train, forecast test
res = ARIMA(y[:i], order=(1,0,0)).fit()
yhat = res.forecast(steps=n-i)

# --- Plot ---
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(t, y, width=6, align="center", color=deep_green, alpha=0.4, label="Increment (observed)")
ax.plot(t[i:], yhat, color=deep_red, linewidth=2.4, label="ARIMA forecast (test)")
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
