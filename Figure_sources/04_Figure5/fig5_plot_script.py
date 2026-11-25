
# Auto-generated plotting script for NHB-style logisticness distribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

scores_csv = "v3lite_logisticness_scores.csv"
df = pd.read_csv(scores_csv)
scores = df["logisticness_score"].dropna().values

n = len(scores)
mean = float(np.mean(scores))
median = float(np.median(scores))
q25 = float(np.quantile(scores, 0.25))
q75 = float(np.quantile(scores, 0.75))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

fig, ax = plt.subplots(figsize=(7.2, 4.0))

deep_blue = (0.10, 0.20, 0.50)
kde_blue  = (0.20, 0.40, 0.80)

ax.hist(scores, bins=20, density=True, alpha=0.6, color=deep_blue, label="Histogram")

try:
    kde = gaussian_kde(scores[np.isfinite(scores)])
    xs = np.linspace(0, 1, 300)
    ax.plot(xs, kde(xs), linewidth=1.5, color=kde_blue, label="Kernel density")
except:
    pass

ax.axvline(mean, linestyle="--", linewidth=1.0, color="black", label=f"Mean = {mean:.2f}")
ax.axvline(median, linestyle="-", linewidth=1.0, color="black", label=f"Median = {median:.2f}")

ax.set_xlabel("Logisticness score (0–1)")
ax.set_ylabel("Density")

ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color("black")

leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)

plt.subplots_adjust(top=0.80, bottom=0.28, left=0.12, right=0.98)
caption = f"Distribution of logisticness score (n = {n}). Quartiles: Q1 = {q25:.2f}, median = {median:.2f}, Q3 = {q75:.2f}."
fig.text(0.5, 0.06, caption, ha="center", va="center")

plt.savefig("fig5_logisticness_distribution_NHB_style.png", dpi=300, bbox_inches="tight")
