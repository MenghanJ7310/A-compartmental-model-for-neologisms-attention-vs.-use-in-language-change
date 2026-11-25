#!/usr/bin/env python3
"""
Figure 1 layout prototype for the neologism manuscript.

IMPORTANT
---------
This script creates a deterministic schematic from basic vector elements.
For submission to a journal that prohibits AI-created graphics, treat the
output as an internal layout prototype and reconstruct the final figure in
PowerPoint, Illustrator, Inkscape, or another non-generative drawing tool.

The script writes:
    figure1_layout_prototype.png
    figure1_layout_prototype.pdf
    figure1_layout_prototype.svg

Compatible with Python 3.7+, NumPy, and Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
BLUE = "#2F5D7C"
ORANGE = "#D97732"
DARK = "#263238"
MID_GREY = "#6B7280"
LIGHT_GREY = "#D7DCE0"
PALE_BLUE = "#EAF1F5"
PALE_ORANGE = "#FAEEE5"
PALE_GREY = "#F4F5F6"


def set_style():
    """Apply a clean journal-style appearance."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def rounded_box(ax, xy, width, height, text, facecolor, edgecolor,
                fontsize=9, fontweight="normal", textcolor=DARK,
                radius=0.025, linewidth=1.1):
    """Draw a rounded text box in axes coordinates."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size={}".format(radius),
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2.0,
        y + height / 2.0,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        linespacing=1.25,
    )
    return box


def arrow(ax, start, end, color=DARK, linewidth=1.5,
          mutation_scale=12, connectionstyle="arc3,rad=0.0"):
    """Draw an arrow in axes coordinates."""
    arr = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
    )
    ax.add_patch(arr)
    return arr


def clean_small_axis(ax, xlabel=None, ylabel=None):
    """Format a compact schematic axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False,
                   labelleft=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    if xlabel:
        ax.set_xlabel(xlabel, color=DARK, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, color=DARK, labelpad=3)


def panel_a(ax):
    """Panel A: emergence and two complementary observations."""
    ax.set_axis_off()
    ax.text(-0.04, 1.02, "(a)", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=DARK, va="top")
    ax.text(0.04, 0.98, "Neologism emergence", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color=DARK, va="top")

    rounded_box(
        ax, (0.18, 0.69), 0.64, 0.13,
        "Emergence of\na new word",
        facecolor=PALE_GREY, edgecolor=MID_GREY,
        fontsize=10, fontweight="bold"
    )

    # Split the emerging word into two complementary observational paths.
    arrow(ax, (0.50, 0.68), (0.27, 0.49), color=BLUE,
          connectionstyle="arc3,rad=0.12")
    arrow(ax, (0.50, 0.68), (0.73, 0.49), color=ORANGE,
          connectionstyle="arc3,rad=-0.12")

    rounded_box(
        ax, (0.02, 0.27), 0.46, 0.20,
        "Reactive\nperception\nGoogle Trends",
        facecolor=PALE_BLUE, edgecolor=BLUE,
        fontsize=8.7, fontweight="bold", textcolor=BLUE
    )
    rounded_box(
        ax, (0.52, 0.27), 0.46, 0.20,
        "Adaptive\nproduction\nOxford Monitor Corpus",
        facecolor=PALE_ORANGE, edgecolor=ORANGE,
        fontsize=8.2, fontweight="bold", textcolor=ORANGE
    )

    ax.text(
        0.50, 0.12,
        "Two complementary time-series observations",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=8.5, color=DARK
    )


def panel_b(ax):
    """Panel B: schematic temporal patterns and models."""
    ax.set_axis_off()
    ax.text(-0.03, 1.02, "(b)", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=DARK, va="top")
    ax.text(0.03, 0.98, "Temporal trajectories", transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color=DARK, va="top")

    top = ax.inset_axes([0.10, 0.56, 0.84, 0.27])
    bottom = ax.inset_axes([0.10, 0.12, 0.84, 0.27])

    # A normalized rise-peak-decay curve used only as a schematic SIR pattern.
    t = np.linspace(0, 1, 500)
    sir_curve = (t ** 5.0) * np.exp(-9.5 * t)
    sir_curve = sir_curve / sir_curve.max()
    top.plot(t, sir_curve, color=BLUE, linewidth=2.4)
    top.fill_between(t, 0, sir_curve, color=BLUE, alpha=0.09)
    clean_small_axis(top, xlabel="Time", ylabel="Relative search interest")
    top.text(0.00, 1.12, "Reactive perception", transform=top.transAxes,
             color=BLUE, fontweight="bold", fontsize=9.3, va="bottom")
    top.text(1.00, 1.12, "SIR", transform=top.transAxes,
             color=BLUE, fontweight="bold", fontsize=9.3,
             ha="right", va="bottom")
    top.text(0.7, 0.97, "rise–peak–decay", transform=top.transAxes,
             color=DARK, fontsize=8)

    # Multi-pulse schematic for heterogeneous production dynamics.
    prod_curve = (
        0.18
        + 0.48 * np.exp(-((t - 0.16) / 0.055) ** 2)
        + 0.32 * np.exp(-((t - 0.35) / 0.085) ** 2)
        + 0.60 * np.exp(-((t - 0.60) / 0.050) ** 2)
        + 0.27 * np.exp(-((t - 0.79) / 0.080) ** 2)
        + 0.06 * np.sin(20 * np.pi * t) * np.exp(-0.6 * t)
    )
    prod_curve = np.clip(prod_curve, 0, None)
    prod_curve = prod_curve / prod_curve.max()
    bottom.plot(t, prod_curve, color=ORANGE, linewidth=2.2)
    bottom.fill_between(t, 0, prod_curve, color=ORANGE, alpha=0.09)
    clean_small_axis(bottom, xlabel="Ordered production observations",
                     ylabel="Occurrence count")
    bottom.text(0.00, 1.12, "Adaptive production",
                transform=bottom.transAxes, color=ORANGE,
                fontweight="bold", fontsize=9.3, va="bottom")
    bottom.text(1.00, 1.12, "Model comparison",
                transform=bottom.transAxes, color=ORANGE,
                fontweight="bold", fontsize=9.3,
                ha="right", va="bottom")

    ax.text(
        0.52, 0.045,
        "ARIMA   |   Naive   |   single-wave logistic",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=8.4, color=DARK
    )


def panel_c(ax):
    """Panel C: cumulative versus incremental representations."""
    ax.set_axis_off()
    ax.text(-0.03, 1.02, "(c)", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=DARK, va="top")
    ax.text(0.03, 0.98, "Cumulative and incremental views",
            transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color=DARK, va="top")

    top = ax.inset_axes([0.12, 0.56, 0.82, 0.25])
    bottom = ax.inset_axes([0.12, 0.15, 0.82, 0.25])

    t = np.linspace(0, 1, 120)
    increments = (
        0.015
        + 0.22 * np.exp(-((t - 0.18) / 0.035) ** 2)
        + 0.10 * np.exp(-((t - 0.30) / 0.060) ** 2)
        + 0.30 * np.exp(-((t - 0.48) / 0.040) ** 2)
        + 0.13 * np.exp(-((t - 0.68) / 0.055) ** 2)
        + 0.08 * np.exp(-((t - 0.82) / 0.040) ** 2)
    )
    cumulative = np.cumsum(increments)
    cumulative = cumulative / cumulative.max()
    increments_scaled = increments / increments.max()

    top.plot(t, cumulative, color=DARK, linewidth=2.3)
    clean_small_axis(top, xlabel="Time", ylabel="Cumulative usage")
    top.text(0.00, 1.10, "Accumulated series", transform=top.transAxes,
             fontsize=9.1, fontweight="bold", color=DARK, va="bottom")

    width = 0.007
    bottom.bar(t, increments_scaled, width=width, color=ORANGE,
               edgecolor=ORANGE, alpha=0.88)
    clean_small_axis(bottom, xlabel="Time", ylabel="Production increment")
    bottom.text(0.00, 1.10, "Corresponding increments",
                transform=bottom.transAxes, fontsize=9.1,
                fontweight="bold", color=ORANGE, va="bottom")

    arrow(ax, (0.92, 0.45), (0.92, 0.53), color=DARK,
          linewidth=1.2, mutation_scale=10)
    ax.text(0.86, 0.49, "accumulation", transform=ax.transAxes,
            rotation=90, ha="center", va="center", fontsize=7,
            color=DARK)

    ax.text(
        0.52, 0.045,
        "A smooth cumulative trajectory may obscure\nvariation in incremental production.",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=8.4, color=DARK, linespacing=1.25
    )


def build_figure():
    """Construct and return the complete three-panel figure."""
    set_style()
    fig = plt.figure(figsize=(12.6, 4.8), facecolor="white")
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[0.95, 1.28, 1.12],
        left=0.035, right=0.985, bottom=0.08, top=0.95,
        wspace=0.20,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    # Light separators make the panel structure clear without boxes.
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    x_sep_1 = (pos_a.x1 + pos_b.x0) / 2.0
    pos_c = ax_c.get_position()
    x_sep_2 = (pos_b.x1 + pos_c.x0) / 2.0
    for x_sep in (x_sep_1, x_sep_2):
        fig.add_artist(plt.Line2D(
            [x_sep, x_sep], [0.10, 0.91],
            transform=fig.transFigure,
            color=LIGHT_GREY, linewidth=0.8
        ))

    return fig


def main():
    fig = build_figure()
    base = "figure1_layout_prototype"
    fig.savefig(base + ".png", dpi=600, bbox_inches="tight",
                pad_inches=0.04)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Saved:")
    print("  " + base + ".png")
    print("  " + base + ".pdf")
    print("  " + base + ".svg")


if __name__ == "__main__":
    main()
