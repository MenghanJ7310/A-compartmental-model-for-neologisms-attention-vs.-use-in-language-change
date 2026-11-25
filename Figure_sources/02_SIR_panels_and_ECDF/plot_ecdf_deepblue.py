
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_MONITOR = "/mnt/data/sir_fit_monitor_summary_v4.csv"
DATA_GT = "/mnt/data/sir_fit_for GT_summary_v4_pro.csv"

OUT_MONITOR = "/mnt/data/ECDF_SIR_R2_Monitor_deepblue.png"
OUT_GT = "/mnt/data/ECDF_SIR_R2_GT_deepblue.png"

LINE_COLOR = "#0B3D91"  # deep blue
DASH_COLOR = "#666666"
THRESHOLDS = [0.60, 0.80, 0.98]
FIGSIZE = (10, 6)
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13

def find_r2_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lower_map = {c: c.lower() for c in cols}
    patterns = [
        r"updated[_\s]*r2",
        r"updated[_\s]*r\^?2",
        r"r2[_\s]*updated",
        r"r\^?2[_\s]*updated",
        r"\br2\b",
        r"r\^?2",
        r"r[_\s]*squared",
    ]
    for pat in patterns:
        for c in cols:
            if re.search(pat, lower_map[c]):
                return c
    raise ValueError(f"Could not find an R^2 column in columns: {cols}")

def load_r2_values(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    r2_col = find_r2_column(df)
    r2 = pd.to_numeric(df[r2_col], errors="coerce").dropna().values
    return r2

def ecdf(values: np.ndarray):
    x = np.sort(values)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return x, y

def share_ge(values: np.ndarray, threshold: float) -> float:
    if len(values) == 0:
        return float("nan")
    return (values >= threshold).mean()

def plot_ecdf_deepblue(values: np.ndarray, title: str, outfile: str):
    x, y = ecdf(values)
    xmin = min(-0.2, np.nanmin(x) - 0.02)
    xmax = 1.0

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=160)
    ax.plot(x, y, linewidth=2.5, color=LINE_COLOR)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)

    for t in THRESHOLDS:
        ax.axvline(t, linestyle="--", color=DASH_COLOR, linewidth=1.2)

    ax.set_xlabel(r"$R^2$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Cumulative share of items", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=16)

    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(False)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.0)

    ge60 = share_ge(values, 0.60)
    ge80 = share_ge(values, 0.80)
    med = np.nanmedian(values) if len(values) else float("nan")
    avg = np.nanmean(values) if len(values) else float("nan")

    text = (
        f"≥0.60: {ge60*100:5.2f}%\n"
        f"≥0.80: {ge80*100:5.2f}%\n"
        f"Median: {med:0.2f}\n"
        f"Average: {avg:0.2f}"
    )

    ax.text(
        xmin + 0.02, 0.12, text,
        fontsize=12, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#333333", alpha=0.9)
    )

    plt.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

def main():
    r2_monitor = load_r2_values(DATA_MONITOR)
    plot_ecdf_deepblue(
        r2_monitor,
        r"ECDF of SIR fit quality ($R^2$) — Monitor Corpus",
        OUT_MONITOR
    )

    r2_gt = load_r2_values(DATA_GT)
    plot_ecdf_deepblue(
        r2_gt,
        r"ECDF of SIR fit quality ($R^2$) — Google Trends",
        OUT_GT
    )

if __name__ == "__main__":
    main()
