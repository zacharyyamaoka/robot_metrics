"""The Matplotlib column of "same view, four tools": each core type drawn as a static figure.

Each function's source is shown verbatim in the report as "the Matplotlib implementation". It draws only
`by_id(<type>).data()`, so its picture differs from the Rerun, Viser and W&B ones by the tool, never the data.

    taskset -c 2-5,16,17 /home/bam/robot_metrics/.venv/bin/python -c "import tools_matplotlib as t; print(t.capture())"
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from core_types import by_id  # noqa: E402

MEDIA = Path("/home/bam/robot_metrics/reports/media/coverage-gallery-2026-11-19/tools/matplotlib")
DEG = 180.0 / np.pi
BLUES = LinearSegmentedColormap.from_list("empty_white", ["#d6e4f5", "#8fb3df", "#1f5fa8", "#0d2f5c"])
BLUES.set_bad("white")                                    # zero cells are masked: empty reads as the white page
LIMIT_RED = "#c0392b"


def limit_bars():
    payload = by_id("limit-bars").data()
    figure, rows = plt.subplots(4, 1, figsize=(7, 3.4))
    for row, name, unit, values, limit in zip(rows, payload["names"], payload["units_display"], payload["values"],
                                             payload["limits"]):
        values_deg = values * DEG
        low, high = np.percentile(values_deg, [0.1, 99.9])
        span_low, span_high = (limit[0] * DEG, limit[1] * DEG) if limit else (1.15 * low, 1.15 * high)
        edges = np.linspace(span_low - 0.1 * (span_high - span_low), span_high + 0.1 * (span_high - span_low), 121)
        counts, _ = np.histogram(values_deg, edges)
        row.fill_between(edges[:-1], 0, np.sqrt(counts / counts.max()), step="post", color="#1f5fa8", linewidth=0)
        if limit:
            row.axvspan(span_low, span_high, color="#ececf0", zorder=-1)
            row.axvline(span_low, color=LIMIT_RED)
            row.axvline(span_high, color=LIMIT_RED)
        row.set_yticks([])
        row.set_ylabel(name, rotation=0, ha="right", va="center")
        row.set_xlabel(unit, fontsize=7)
    figure.tight_layout()
    return figure


def coverage_heatmap():
    payload = by_id("coverage-heatmap").data()
    figure, axis = plt.subplots(figsize=(5.4, 3.4))
    runs_per_cell = np.ma.masked_where(payload["runs"] <= 0, payload["runs"])
    mesh = axis.pcolormesh(payload["q_edges_deg"], payload["qd_edges_deg"], runs_per_cell.T, cmap=BLUES, vmin=1)
    unreachable = np.ma.masked_where(payload["reachable"].T, np.ones(payload["reachable"].T.shape))
    axis.pcolormesh(payload["q_edges_deg"], payload["qd_edges_deg"], unreachable, cmap=ListedColormap(["#ececf0"]))
    figure.colorbar(mesh, ax=axis, label="runs through the cell")
    axis.set_xlabel("q from bottom (deg)")
    axis.set_ylabel("qd (deg/s)")
    axis.set_title(f"{payload['visited_pct']:.0f} % of reachable cells visited", loc="left")
    return figure


def histogram_over_runs():
    payload = by_id("histogram-over-runs").data()
    edges = np.linspace(-payload["qd_limit"], payload["qd_limit"], 81) * DEG
    histograms = np.array([np.histogram(qd * DEG, edges)[0] for qd in payload["qd_per_run"]], float)
    every_tenth = histograms[::10]
    figure, axis = plt.subplots(figsize=(5.4, 4.2))
    for index, counts in enumerate(every_tenth):
        base = -index * 0.35
        height = counts / max(counts.max(), 1)
        colour = plt.cm.Oranges(0.3 + 0.65 * index / len(every_tenth))
        axis.fill_between(edges[:-1], base, base + height, step="post", color=colour, zorder=len(every_tenth) - index)
        axis.plot(edges[:-1], base + height, drawstyle="steps-post", color="white", linewidth=0.5,
                  zorder=len(every_tenth) - index)
    axis.set_yticks([-index * 0.35 for index in range(0, len(every_tenth), 5)],
                    [f"run {index * 10}" for index in range(0, len(every_tenth), 5)])
    axis.set_xlabel("qd (deg/s)")
    axis.set_title("qd per run, every 10th run, oldest on top", loc="left")
    return figure


def coverage_growth():
    payload = by_id("coverage-growth").data()
    runs = np.arange(1, len(payload["in_order_pct"]) + 1)
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    axis.fill_between(runs, payload["random_band_pct"][0], payload["random_band_pct"][1], color="#1f5fa8", alpha=0.15,
                      label="random order, 10-90 %")
    axis.plot(runs, payload["random_mean_pct"], color="#1f5fa8", label="random order, mean")
    axis.plot(runs, payload["in_order_pct"], color="#0d2f5c", linewidth=2, label="as recorded")
    axis.axhline(payload["chao2_pct"], color="#0d2f5c", linestyle=":", label="Chao2 ceiling")
    axis.axhline(100, color=LIMIT_RED)
    axis.set_xlabel("runs recorded")
    axis.set_ylabel("% of reachable (q, qd, qdd) cells")
    axis.legend(loc="center right", fontsize=7)
    return figure


def phase_space_3d():
    payload = by_id("phase-space-3d").data()
    visited = (payload["counts"] > 0) & payload["reachable"]
    empty = (payload["counts"] == 0) & payload["reachable"]
    figure = plt.figure(figsize=(5.4, 4.4))
    axis = figure.add_subplot(projection="3d")
    centres = payload["centers_deg"]
    axis.scatter(*centres[empty].T, s=4, color="#c8c8d0", alpha=0.25, label="reachable, empty")
    axis.scatter(*centres[visited].T, s=10, c=np.log10(payload["counts"][visited]), cmap=BLUES, label="visited")
    axis.set_xlabel(payload["axes"][0])
    axis.set_ylabel(payload["axes"][1])
    axis.set_zlabel(payload["axes"][2])
    axis.set_title(f"{payload['visited_pct']:.0f} % of reachable cells visited", loc="left")
    axis.legend(loc="upper left", fontsize=7)
    return figure


def tool_paths():
    payload = by_id("tool-paths").data()
    figure, (top, side) = plt.subplots(1, 2, figsize=(8, 3.4))
    for path in payload["paths_m"]:
        top.plot(path[:, 0] * 1000, path[:, 1] * 1000, color="#0d2f5c", alpha=0.03, linewidth=0.6)
        side.plot(path[:, 0] * 1000, path[:, 2] * 1000, color="#0d2f5c", alpha=0.03, linewidth=0.6)
    top.set(xlabel="x (mm)", ylabel="y (mm)", title="top view", aspect="equal")
    side.set(xlabel="x (mm)", ylabel="z (mm)", title="side view", aspect="equal")
    figure.tight_layout()
    return figure


def grasp_poses():
    payload = by_id("grasp-poses").data()
    figure = plt.figure(figsize=(5.4, 4.4))
    axis = figure.add_subplot(projection="3d")
    finger_m, palm_back_m = 0.04, 0.05
    colours = plt.cm.viridis(payload["opening_rad"] / payload["opening_limit_rad"])
    for position, rotation, opening, colour in zip(payload["positions_m"], payload["rotations"], payload["opening_rad"],
                                                  colours):
        half_width = 0.035 * opening / payload["opening_limit_rad"]
        jaw, approach = rotation[:, 0], rotation[:, 2]
        left, right = position + half_width * jaw, position - half_width * jaw
        gripper = np.array([left - finger_m * approach, left, left, right, right, right - finger_m * approach])
        axis.plot(*(gripper[[1, 0]] * 1000).T, color=colour, alpha=0.25, linewidth=0.6)
        axis.plot(*(gripper[[4, 5]] * 1000).T, color=colour, alpha=0.25, linewidth=0.6)
        axis.plot(*(np.array([left, right]) * 1000).T, color=colour, alpha=0.25, linewidth=0.6)
        palm = (left + right) / 2
        axis.plot(*(np.array([palm, palm - palm_back_m * approach]) * 1000).T, color=colour, alpha=0.25, linewidth=0.6)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")
    axis.set_title("every real repick grasp; colour = claw opening / limit", loc="left", fontsize=8)
    return figure


def direction_sphere():
    payload = by_id("direction-sphere").data()
    rings, sectors = payload["rings"], payload["sectors"]
    longitude = np.linspace(-np.pi, np.pi, sectors + 1)
    latitude = np.arcsin(np.linspace(-1, 1, rings + 1))     # rings uniform in z = equal area
    figure = plt.figure(figsize=(6, 3.4))
    axis = figure.add_subplot(projection="mollweide")
    counts = np.ma.masked_where(payload["counts"] <= 0, payload["counts"])
    mesh = axis.pcolormesh(longitude, latitude, counts, cmap=BLUES, vmin=1)
    axis.grid(color="#d2d2d7", linewidth=0.5)
    figure.colorbar(mesh, ax=axis, orientation="horizontal", fraction=0.05, label="grasps per equal-area cell")
    axis.set_title(f"approach directions: {payload['visited_pct']:.1f} % of cells used (bottom = straight down)",
                   fontsize=8)
    return figure


IMPLEMENTATIONS = {"limit-bars": limit_bars, "coverage-heatmap": coverage_heatmap, "histogram-over-runs": histogram_over_runs,
                   "coverage-growth": coverage_growth, "phase-space-3d": phase_space_3d, "tool-paths": tool_paths,
                   "grasp-poses": grasp_poses, "direction-sphere": direction_sphere}


def capture() -> dict:
    MEDIA.mkdir(parents=True, exist_ok=True)
    out = {}
    for type_id, draw in IMPLEMENTATIONS.items():
        figure = draw()
        path = MEDIA / f"{type_id}.png"
        figure.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        out[type_id] = {"image": str(path), "notes": ""}
    return out


if __name__ == "__main__":
    print(capture())
