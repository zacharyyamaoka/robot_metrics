"""One look for every gallery figure: white page, hairlines, one accent; empty is white, so a hole reads at a glance.

WHY white-to-one-colour (Zach, Daily Note 2026-11-19): "for the color it would be nice if I could just see it at a
glance. perhaps you can use the heat map color? or just white to a solid color) so its clear that white means empty".
`EMPTY_WHITE` is white to the accent; `HEAT` is white through yellow to red, for the cards that compare the two.
Unreachable cells are a light grey hatch, never white, so "can't go there" never reads as "didn't go there".
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402

ACCENT = "#1f5fa8"
ACCENT_DARK = "#0d2f5c"
INK = "#1d1d1f"
MUTED = "#6e6e73"
HAIR = "#d2d2d7"
UNREACHABLE = "#ececf0"
LIMIT = "#c0392b"     # a limit line is the one warm mark on a cool page: it is the thing you compare against

# WHY the ramps start at a visible tint, not at white (datashader's "undersaturation" pitfall): zero cells are masked to
# the white page, so ONE sample must already read as a colour; a ramp that starts at white makes 1 and 0 look alike.
EMPTY_WHITE = LinearSegmentedColormap.from_list("empty_white", ["#d6e4f5", "#8fb3df", ACCENT, ACCENT_DARK])
HEAT = LinearSegmentedColormap.from_list("heat_white", ["#fff1a8", "#fdb44b", "#e8590c", "#9c1c1c"])
for _ramp in (EMPTY_WHITE, HEAT):
    _ramp.set_bad("#ffffff")
VISITED = ListedColormap([ACCENT])          # a binary grid: 0 is masked to the page, 1 is the accent
VISITED.set_bad("#ffffff")
TWO_TONE = ListedColormap(["#ffffff", ACCENT])
CATEGORY = ["#1f5fa8", "#e8590c", "#2b9348", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d", "#d4a017"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 8.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": HAIR, "axes.labelcolor": INK, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 7.5, "axes.grid": False, "figure.facecolor": "white",
})


def figure(width: float = 4.4, height: float = 3.2, **options):
    return plt.subplots(figsize=(width, height), **options)


def heatmap(axes, counts: np.ndarray, x_edges, y_edges, reachable: np.ndarray | None = None, cmap=EMPTY_WHITE,
            vmax: float | None = None, norm=None, colorbar_label: str | None = "samples per cell", figure=None):
    """counts[i, j] over x bin i, y bin j. Zero is exactly white; unreachable cells hatched grey on top."""

    shown = np.ma.masked_where(counts <= 0, counts.astype(float))      # empty = the white page, never a colour
    if norm is None:
        vmax = vmax if vmax is not None else max(float(np.percentile(counts[counts > 0], 98)) if (counts > 0).any() else 1.0, 1.0)
        mesh = axes.pcolormesh(x_edges, y_edges, shown.T, cmap=cmap, vmin=min(1.0, vmax), vmax=vmax, shading="flat")
    else:
        mesh = axes.pcolormesh(x_edges, y_edges, shown.T, cmap=cmap, norm=norm, shading="flat")
    if reachable is not None:
        blocked = np.ma.masked_where(reachable.T, np.ones_like(reachable.T, float))
        axes.pcolormesh(x_edges, y_edges, blocked, cmap=ListedColormap([UNREACHABLE]), shading="flat")
        axes.pcolor(x_edges, y_edges, blocked, hatch="////", alpha=0.0, edgecolor=HAIR, linewidth=0)
    if colorbar_label and figure is not None:
        bar = figure.colorbar(mesh, ax=axes, fraction=0.046, pad=0.03)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7, colors=MUTED)
        bar.set_label(colorbar_label, fontsize=7.5, color=MUTED)
    return mesh


def to_data_uri(fig, colors: int = 160) -> str:
    """PNG, palette-quantized (these are flat-colour charts, so 160 colours is visually lossless and ~3x smaller)."""

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def caption_axes(axes, text: str):
    axes.text(0.0, 1.02, text, transform=axes.transAxes, fontsize=7.5, color=MUTED, va="bottom")
