"""Optional matplotlib figures (``pip install radsafe[plot]``).

Colour follows the entity: each shielding model keeps the same colour in
every figure (categorical slots 1-3 of a palette validated for colour-vision
deficiency); the no-shield reference is neutral grey. Light surface only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from radsafe.montecarlo import UNSHIELDED, SimulationResult
from radsafe.report import _pretty
from radsafe.sensitivity import spearman_sensitivity

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES_COLOR = {
    "narrow_beam": "#2a78d6",  # blue
    "linear_buildup": "#eb6834",  # orange
    "practical_hvl": "#1baf7a",  # aqua
    UNSHIELDED: INK_MUTED,
}


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="both", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_summary(result: SimulationResult, path: str | Path) -> Path:
    """Two panels: exceedance curves P(dose > x), and what drives the spread."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path = Path(path)
    series = [s for s in result.series if s != UNSHIELDED] or [UNSHIELDED]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, constrained_layout=True)

    # --- Panel 1: exceedance curves --------------------------------------
    all_msv = np.concatenate([result.dose_usv[s] for s in series]) / 1000.0
    positive = all_msv[all_msv > 0]
    lo, hi = np.percentile(positive, [0.1, 99.9])
    grid = np.logspace(np.log10(lo), np.log10(hi), 300)
    for s in series:
        d = np.sort(result.dose_usv[s] / 1000.0)
        p_exceed = 1.0 - np.searchsorted(d, grid, side="right") / d.size
        ax1.plot(grid, p_exceed, color=SERIES_COLOR[s], linewidth=2, label=s)
    # Limit labels sit above the axes in staggered rows so they never collide with the curves.
    row_step = 0.075
    limits = sorted(result.scenario.limits, key=lambda lim: lim.value_msv)
    for i, limit in enumerate(limits):
        y_label = 1.0 + row_step * (1 + i % 2)
        ax1.plot(
            [limit.value_msv] * 2,
            [0, y_label],
            transform=ax1.get_xaxis_transform(),
            color=INK_MUTED,
            linestyle=(0, (4, 3)),
            linewidth=1,
            clip_on=False,
        )
        ax1.text(
            limit.value_msv,
            y_label,
            f"{limit.name}, {limit.value_msv:g} mSv",
            transform=ax1.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("Dose over the exposure period (mSv)", color=INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("P(dose > x)", color=INK_SECONDARY, fontsize=10)
    title_pad = 8 + (46 if limits else 0)
    ax1.set_title("Exceedance probability", loc="left", color=INK_PRIMARY, fontsize=11, pad=title_pad)
    _style_axes(ax1)
    if len(series) > 1:
        ax1.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right")

    # --- Panel 2: sensitivity ---------------------------------------------
    sens = {s: {t.input: t.share for t in spearman_sensitivity(result, s)} for s in series}
    peak = {k: max(d.get(k, 0.0) for d in sens.values()) for d in sens.values() for k in d}
    inputs = sorted(peak, key=lambda k: -peak[k])
    y = np.arange(len(inputs))
    bar_h = 0.8 / len(series)
    for j, s in enumerate(series):
        shares = [sens[s].get(k, 0.0) for k in inputs]
        offset = (j - (len(series) - 1) / 2) * bar_h
        ax2.barh(y + offset, shares, height=bar_h * 0.85, color=SERIES_COLOR[s], label=s)
    ax2.set_yticks(y, [_pretty(k) for k in inputs])
    ax2.invert_yaxis()
    ax2.set_xlabel("Share of explained rank variance (Spearman rho$^2$)", color=INK_SECONDARY, fontsize=10)
    ax2.set_title("What drives the uncertainty", loc="left", color=INK_PRIMARY, fontsize=11, pad=title_pad)
    _style_axes(ax2)
    ax2.grid(axis="y", visible=False)
    for label in ax2.get_yticklabels():
        label.set_color(INK_SECONDARY)
    if len(series) > 1:
        ax2.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")

    fig.suptitle(result.scenario.name, x=0.01, ha="left", color=INK_PRIMARY, fontsize=12)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
