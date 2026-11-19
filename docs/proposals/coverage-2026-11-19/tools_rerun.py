"""The Rerun column of the "same coverage view, four tools" gallery: one function per core TYPE, each writing one .rrd.

    import tools_rerun
    tools_rerun.coverage_heatmap(Path("coverage-heatmap.rrd"))     # then: rerun coverage-heatmap.rrd
    tools_rerun.capture()                                           # all 8 .rrd + a headless 1400 x 850 screenshot each

Each function draws `core_types.by_id(<type>).data()` and nothing else (so a difference between columns is the tool),
logs it the most Rerun-native way, and sends a blueprint so the viewer opens on the right view with no clicking. Its
source is shown verbatim in the gallery as "the Rerun implementation", so it stays short and reads top to bottom.

WHY degrees in the .rrd (Zach's rule: degrees in every human view, SI in stored data): the .rrd is a viewer projection;
`core_types` stays SI. WHY one application id per type: Rerun keeps one blueprint per application id, so opening two
of these files in one viewer never swaps their layouts.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import rerun_cli
from matplotlib.colors import to_rgba

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_types import TYPES, by_id  # noqa: E402
from style import ACCENT, ACCENT_DARK, CATEGORY, EMPTY_WHITE, HAIR, INK, LIMIT, MUTED, UNREACHABLE  # noqa: E402

DEG = 180.0 / np.pi
WHITE = [255, 255, 255]
THIN = rr.Radius.ui_points(0.8)          # line width in screen points, so it stays a hairline at any zoom
MEDIA = Path("/home/bam/robot_metrics/reports/media/coverage-gallery-2026-11-19/tools/rerun")


def rgba(color: str, alpha: float = 1.0) -> list[int]:
    """A style.py hex colour as the RGBA bytes Rerun takes."""
    return [round(255 * channel) for channel in to_rgba(color, alpha)]


def ramp(counts: np.ndarray, log: bool = False, top: float | None = None) -> np.ndarray:
    """The gallery's white-to-accent ramp (style.EMPTY_WHITE) as RGBA bytes: linear from 1 to the 98th percentile, as
    the Matplotlib heatmaps; `log` from 1 to the maximum, for 3-D views where lit faces wash a pale cell out."""
    top = top or (counts.max() if log else np.percentile(counts[counts > 0], 98))
    share = np.log(counts) / np.log(max(top, 2.0)) if log else (counts - 1.0) / max(top - 1.0, 1.0)
    return (EMPTY_WHITE(np.clip(share, 0.0, 1.0)) * 255).astype(np.uint8)


def text(positions, labels) -> rr.Points3D:
    """Axis and tick labels in a 3-D view (Rerun draws a label in its point's colour on a light chip)."""
    return rr.Points3D(positions, labels=labels, colors=rgba(INK), radii=rr.Radius.ui_points(0.01), show_labels=True)


# --- 1-D --------------------------------------------------------------------------------------------------------------

def limit_bars(path: Path) -> Path:
    """One native BarChart per derivative; each limit is a thin red bar at ±limit (a BarChartView has no rules)."""
    payload = by_id("limit-bars").data()
    with rr.RecordingStream("coverage-limit-bars", recording_id="limit-bars") as recording:
        recording.save(path)
        views = []
        for name, unit, values, limit, planner in zip(payload["names"], payload["units_display"], payload["values"],
                                                      payload["limits"], payload["planner_limits"]):
            degrees = values * DEG
            marks = [bound * DEG for bound in (limit or ()) + (planner or ())]
            low, high = np.percentile(degrees, [0.1, 99.9])
            low, high = min([low, *marks]), max([high, *marks])
            counts, edges = np.histogram(degrees, bins=80, range=(low, high))
            centres, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
            recording.log(f"{name}/samples", rr.BarChart(counts, abscissa=centres, widths=width, color=rgba(ACCENT)),
                          static=True)
            for label, bounds, color in (("limit", limit, LIMIT), ("planner_limit", planner, CATEGORY[1])):
                if bounds is not None:
                    recording.log(f"{name}/{label}", rr.BarChart([counts.max() * 1.08] * 2, widths=width * 0.6,
                                                                 abscissa=np.array(bounds) * DEG, color=rgba(color)),
                                  static=True)
            outside = int(((degrees < low) | (degrees > high)).sum())
            stated = f"red = limits {limit[0] * DEG:,.0f} / {limit[1] * DEG:,.0f}" if limit else "no limit stated"
            views.append(rrb.BarChartView(origin=name, name=f"{name} [{unit}] · samples per bin · {stated}"
                                          + (" · orange = planner limit" if planner else "")
                                          + (f" · {outside} tail samples off-chart" if outside else ""),
                                          plot_legend=rrb.PlotLegend(visible=False)))
        recording.send_blueprint(rrb.Blueprint(rrb.Grid(*views, grid_columns=2), collapse_panels=True))
    return path


# --- 2-D --------------------------------------------------------------------------------------------------------------

def coverage_heatmap(path: Path) -> Path:
    """Flat solid Boxes3D cells seen straight down (Rerun's 2-D boxes are outline-only); empty cells are not drawn."""
    payload = by_id("coverage-heatmap").data()
    q_edges, qd_edges = payload["q_edges_deg"], payload["qd_edges_deg"]
    q_low, qd_low = (grid.ravel() for grid in np.meshgrid(q_edges[:-1], qd_edges[:-1], indexing="ij"))
    corners = np.column_stack([q_low, qd_low, np.zeros_like(q_low)])
    cell = [q_edges[1] - q_edges[0], qd_edges[1] - qd_edges[0], 0.0]        # one size, broadcast to every box
    with rr.RecordingStream("coverage-coverage-heatmap", recording_id="coverage-heatmap") as recording:
        recording.save(path)
        for quantity in ("samples", "runs"):
            counts = payload[quantity].ravel()
            visited = counts > 0                                   # empty cells are simply not drawn: the white page
            recording.log(f"{quantity}/visited", rr.Boxes3D(
                mins=corners[visited], sizes=[cell], colors=ramp(counts[visited]), fill_mode="solid",
                labels=[f"{count:,.0f} {quantity}" for count in counts[visited]], show_labels=False), static=True)
        blocked = ~payload["reachable"].ravel() & (payload["samples"].ravel() == 0)
        recording.log("frame/unreachable", rr.Boxes3D(mins=corners[blocked], sizes=[cell], colors=rgba(UNREACHABLE),
                                                      fill_mode="solid"), static=True)
        box = np.column_stack([q_edges[[0, -1, -1, 0, 0]], qd_edges[[0, 0, -1, -1, 0]], np.full(5, 0.1)])
        recording.log("frame/limits", rr.LineStrips3D([box], colors=rgba(LIMIT), radii=rr.Radius.ui_points(1.5)),
                      static=True)
        q_ticks = np.arange(np.ceil(q_edges[0] / 60) * 60, q_edges[-1], 60)
        qd_ticks = np.arange(np.ceil(qd_edges[0] / 120) * 120, qd_edges[-1], 120)
        recording.log("frame/ticks", text(
            [[q, qd_edges[0] - 25, 0] for q in q_ticks] + [[q_edges[0] - 30, qd, 0] for qd in qd_ticks]
            + [[np.mean(q_edges), qd_edges[0] - 60, 0], [q_edges[0] - 30, qd_edges[-1] + 30, 0]],
            [f"{q:.0f}°" for q in q_ticks] + [f"{qd:.0f}°/s" for qd in qd_ticks]
            + ["q from bottom (deg) →", "qd (deg/s) ↑"]), static=True)
        centre = [np.mean(q_edges), np.mean(qd_edges) - 15, 0.0]           # nudged down to fit the q axis title
        top_down = rrb.EyeControls3D(kind="Orbital", look_target=centre, eye_up=[0, 1, 0],
                                     position=[centre[0], centre[1], 1.2 * np.ptp(qd_edges)])
        views = [rrb.Spatial3DView(origin="/", contents=[f"/{quantity}/**", "/frame/**"], eye_controls=top_down,
                                   background=WHITE, line_grid=False,
                                   name=f"{title} · white = empty · grey = unreachable · red = limits")
                 for quantity, title in (("samples", "Samples per cell"), ("runs", "Runs per cell"))]
        recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(*views), collapse_panels=True))
    return path


# --- series -----------------------------------------------------------------------------------------------------------

def histogram_over_runs(path: Path) -> Path:
    """The run IS Rerun's timeline: percentile bands as Scalars per run, and a BarChart that re-draws as you scrub."""
    payload = by_id("histogram-over-runs").data()
    speeds = [run * DEG for run in payload["qd_per_run"]]
    limit = payload["qd_limit"] * DEG
    percentiles = [0, 7, 16, 31, 50, 69, 84, 93, 100]               # TensorBoard's distribution bands
    shades = ["#d6e4f5", "#b3cbe9", "#8fb3df", "#5b8ccb", ACCENT_DARK, "#5b8ccb", "#8fb3df", "#b3cbe9", "#d6e4f5"]
    bands = np.array([np.percentile(speed, percentiles) for speed in speeds])
    runs = rr.TimeColumn("run", sequence=np.arange(len(speeds)))
    edges = np.linspace(-1.2 * limit, 1.2 * limit, 49)
    centres, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    histograms = np.array([np.histogram(speed, edges)[0] for speed in speeds])
    with rr.RecordingStream("coverage-histogram-over-runs", recording_id="histogram-over-runs") as recording:
        recording.save(path)
        for index, (percent, shade) in enumerate(zip(percentiles, shades)):
            recording.log(f"bands/p{percent}", rr.SeriesLines(colors=rgba(shade), names=f"p{percent}"), static=True)
            recording.send_columns(f"bands/p{percent}", indexes=[runs],
                                   columns=rr.Scalars.columns(scalars=bands[:, index]))
        for sign in (1, -1):
            recording.log(f"bands/limit{sign:+d}", rr.SeriesLines(colors=rgba(LIMIT), names="limit"), static=True)
            recording.send_columns(f"bands/limit{sign:+d}", indexes=[runs],
                                   columns=rr.Scalars.columns(scalars=np.full(len(speeds), sign * limit)))
        recording.log("histogram/limit", rr.BarChart([histograms.max()] * 2, abscissa=[-limit, limit],
                                                     widths=width / 2, color=rgba(LIMIT)), static=True)
        for run, counts in enumerate(histograms):
            recording.set_time("run", sequence=run)
            recording.log("histogram/qd", rr.BarChart(counts, abscissa=centres, widths=width, color=rgba(ACCENT)))
        recording.send_blueprint(rrb.Blueprint(
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/bands", name="qd per run [deg/s] · bands = 0/7/16/31/50/69/84/93/100th "
                                   "percentile (TensorBoard distributions) · red = ±limit",
                                   plot_legend=rrb.PlotLegend(visible=False)),
                rrb.BarChartView(origin="/histogram", name="qd histogram of the run under the time cursor [deg/s] · "
                                 "scrub 'run' below · red = ±limit", plot_legend=rrb.PlotLegend(visible=False)),
                row_shares=[3, 2]),
            rrb.BlueprintPanel(state="collapsed"), rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed", timeline="run")))
    return path


def coverage_growth(path: Path) -> Path:
    """Scalars on a 'run' sequence timeline (Rerun has no fill-between, so the random-order band is two hairlines)."""
    payload = by_id("coverage-growth").data()
    in_order = payload["in_order_pct"]
    runs = rr.TimeColumn("run", sequence=np.arange(1, len(in_order) + 1))
    curves = {"in recording order": (in_order, ACCENT, 2.5),
              "random order, mean": (payload["random_mean_pct"], MUTED, 1.5),
              "random order, 10th pct": (payload["random_band_pct"][0], HAIR, 1.0),
              "random order, 90th pct": (payload["random_band_pct"][1], HAIR, 1.0),
              "Chao2 estimate of the reachable total": (np.full(len(in_order), payload["chao2_pct"]), CATEGORY[1], 1.5)}
    with rr.RecordingStream("coverage-coverage-growth", recording_id="coverage-growth") as recording:
        recording.save(path)
        for name, (percent, color, width) in curves.items():
            entity = "visited/" + name.replace(" ", "_").replace(",", "")
            recording.log(entity, rr.SeriesLines(colors=rgba(color), widths=width, names=name), static=True)
            recording.send_columns(entity, indexes=[runs], columns=rr.Scalars.columns(scalars=percent))
        top = 1.15 * max(in_order.max(), payload["chao2_pct"])
        recording.send_blueprint(rrb.Blueprint(
            rrb.TimeSeriesView(origin="/visited", name=f"Visited % of {payload['reachable_cells']:,} reachable "
                               "(q, qd, qdd) cells after each run · x = run (step)",
                               axis_y=rrb.ScalarAxis(range=(0.0, top), zoom_lock=False),
                               plot_legend=rrb.PlotLegend(corner="LeftTop")),
            rrb.BlueprintPanel(state="collapsed"), rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed", timeline="run")))
    return path


# --- 3-D --------------------------------------------------------------------------------------------------------------

def phase_space_3d(path: Path) -> Path:
    """Boxes3D per visited cell beside the raw samples, which arrive run by run on the 'run' timeline."""
    payload = by_id("phase-space-3d").data()
    centres, size, counts = payload["centers_deg"], payload["cell_size_deg"], payload["counts"]
    low, high = centres.min(axis=0) - size / 2, centres.max(axis=0) + size / 2
    visited, empty = counts > 0, payload["reachable"] & (counts == 0)
    samples, sample_run = payload["samples_deg"], payload["sample_run"]
    run_colors = (EMPTY_WHITE(0.25 + 0.75 * sample_run / sample_run.max()) * 255).astype(np.uint8)  # later = darker
    with rr.RecordingStream("coverage-phase-space-3d", recording_id="phase-space-3d") as recording:
        recording.save(path)
        # WHY one scale per axis: qdd spans ~20x q's degrees. The logged values stay in true degrees; the transform
        # only fits each axis to one display unit, so the cube is not a pancake.
        recording.log("phase", rr.Transform3D(scale=1.0 / (high - low)), static=True)
        recording.log("phase/cells/visited", rr.Boxes3D(
            centers=centres[visited], sizes=[size * 0.9], colors=ramp(counts[visited], log=True), fill_mode="solid",
            labels=[f"{count:,.0f} samples" for count in counts[visited]], show_labels=False), static=True)
        recording.log("phase/cells/reachable_empty", rr.Boxes3D(
            centers=centres[empty], sizes=[size * 0.9], colors=rgba(UNREACHABLE, 0.12), fill_mode="solid"), static=True)
        edges = [(low, [high[0], low[1], low[2]]), ([high[0], low[1], low[2]], [high[0], high[1], low[2]]),
                 (low, [low[0], low[1], high[2]])]                  # the three cube edges nearest the camera
        recording.log("phase/axes", rr.LineStrips3D(edges, colors=rgba(INK), radii=THIN), static=True)
        recording.log("phase/axes/labels", text(
            [np.mean(edge, axis=0) for edge in edges],
            [f"{name}: {start:,.0f} → {end:,.0f}" for name, start, end in zip(payload["axes"], low, high)]),
            static=True)
        for run in np.unique(sample_run):
            recording.set_time("run", sequence=int(run))
            in_run = sample_run == run
            recording.log("phase/samples", rr.Points3D(samples[in_run], colors=run_colors[in_run],
                                                       radii=rr.Radius.ui_points(1.2)))
        eye = rrb.EyeControls3D(kind="Orbital", look_target=[0, 0, 0], eye_up=[0, 0, 1], position=[1.5, -1.9, 1.2])
        so_far = rrb.VisibleTimeRange("run", start=rrb.TimeRangeBoundary.infinite(),
                                      end=rrb.TimeRangeBoundary.cursor_relative(seq=0))
        recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["/phase/cells/**", "/phase/axes/**"], eye_controls=eye,
                              background=WHITE, line_grid=False,
                              name=f"Visited cells, colour = samples (log) · grey = reachable, never visited · "
                                   f"{payload['visited_pct']:.0f} % of reachable visited"),
            rrb.Spatial3DView(origin="/", contents=["/phase/samples", "/phase/axes/**"], eye_controls=eye,
                              background=WHITE, line_grid=False, time_ranges=so_far,
                              name="Raw samples up to the time cursor (darker = later run) · scrub 'run' to replay")),
            rrb.BlueprintPanel(state="collapsed"), rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed", timeline="run")))
    return path


def tool_paths(path: Path) -> Path:
    """Every TCP path as LineStrips3D, one cycle per timeline tick, beside the 2 cm voxels it passed through."""
    payload = by_id("tool-paths").data()
    edges, counts = payload["voxel_edges_m"], payload["voxel_counts"]
    occupied = np.argwhere(counts > 0)
    voxel_centres = np.column_stack([(edges[axis][occupied[:, axis]] + edges[axis][occupied[:, axis] + 1]) / 2
                                     for axis in range(3)])
    voxel_of = lambda points: tuple(  # noqa: E731
        np.clip(np.searchsorted(edges[axis], points[:, axis]) - 1, 0, counts.shape[axis] - 1) for axis in range(3))
    every = np.concatenate(payload["paths_m"])
    centre, span = (every.min(axis=0) + every.max(axis=0)) / 2, np.ptp(every, axis=0).max()
    with rr.RecordingStream("coverage-tool-paths", recording_id="tool-paths") as recording:
        recording.save(path)
        recording.log("base", rr.TransformAxes3D(axis_length=0.1), static=True)
        recording.log("voxels", rr.Boxes3D(centers=voxel_centres, sizes=[[0.0184] * 3], fill_mode="solid",
                                           colors=ramp(counts[counts > 0], log=True),
                                           labels=[f"{count:,.0f} path samples" for count in counts[counts > 0]],
                                           show_labels=False), static=True)
        # WHY segments coloured by voxel density, not alpha: Rerun's lines do not blend additively, so 1131 faint
        # paths just paint over each other. Colouring each segment by how busy its voxel is gives Strava's worn path.
        for cycle, tcp in enumerate(payload["paths_m"]):         # one cycle per tick: scrub 'cycle' to watch it fill
            points = tcp[::4]
            density = counts[voxel_of((points[:-1] + points[1:]) / 2)]
            recording.set_time("cycle", sequence=cycle)
            recording.log("paths", rr.LineStrips3D(np.stack([points[:-1], points[1:]], axis=1), radii=THIN,
                                                   colors=ramp(density, log=True, top=counts.max())))
        eye = rrb.EyeControls3D(kind="Orbital", look_target=centre, eye_up=[0, 0, 1],
                                position=centre + span * np.array([0.7, -1.0, 0.6]))
        so_far = rrb.VisibleTimeRange("cycle", start=rrb.TimeRangeBoundary.infinite(),
                                      end=rrb.TimeRangeBoundary.cursor_relative(seq=0))
        recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["/paths", "/base"], eye_controls=eye, time_ranges=so_far,
                              background=WHITE, line_grid=False,
                              name=f"{len(payload['paths_m']):,} planned TCP paths, darker = busier voxel (Strava) · "
                                   "base frame, m · scrub 'cycle' to replay"),
            rrb.Spatial3DView(origin="/", contents=["/voxels", "/base"], eye_controls=eye,
                              background=WHITE, line_grid=False,
                              name=f"{len(occupied):,} occupied 2 cm voxels · colour = path samples (log) · "
                                   "empty = not drawn")),
            rrb.BlueprintPanel(state="collapsed"), rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed", timeline="cycle")))
    return path


# WHY a nominal claw: the repick plans carry the claw's opening angle, not its finger geometry. Each jaw is drawn as a
# 50 mm finger hinged at the palm, so tip separation = 2 * 50 mm * sin(opening): the shape is honest, the scale nominal.
FINGER_M, STEM_M = 0.05, 0.04


def grasp_poses(path: Path) -> Path:
    """Each grasp as a two-finger LineStrips3D wireframe (jaws along x, approach +z), coloured by its opening."""
    payload = by_id("grasp-poses").data()
    positions, rotations, opening = payload["positions_m"], payload["rotations"], payload["opening_rad"]
    limit = payload["opening_limit_rad"]
    half, zero = FINGER_M * np.sin(opening), np.zeros_like(opening)
    # gripper frame: jaws open along x, approach is +z, the fingertips sit on the TCP
    corner = lambda x, z: np.column_stack([x, zero, zero + z])  # noqa: E731
    jaws = np.stack([corner(-half, 0), corner(-half, -FINGER_M), corner(half, -FINGER_M), corner(half, 0)], axis=1)
    stem = np.stack([corner(zero, -FINGER_M), corner(zero, -FINGER_M - STEM_M)], axis=1)
    to_base = lambda local: positions[:, None, :] + np.einsum("gij,gpj->gpi", rotations, local)  # noqa: E731
    colors = (EMPTY_WHITE(0.2 + 0.8 * (opening - opening.min()) / (limit - opening.min())) * 255).astype(np.uint8)
    colors[opening >= limit - 1e-6] = rgba(LIMIT)
    labels = [f"opening {angle * DEG:.1f}° of {limit * DEG:.1f}°" for angle in opening]
    centre, span = (positions.min(axis=0) + positions.max(axis=0)) / 2, np.ptp(positions, axis=0).max()
    counts, edges = np.histogram(opening * DEG, bins=40, range=(opening.min() * DEG - 1, limit * DEG + 1))
    with rr.RecordingStream("coverage-grasp-poses", recording_id="grasp-poses") as recording:
        recording.save(path)
        recording.log("grasps/jaws", rr.LineStrips3D(to_base(jaws), colors=colors, radii=THIN, labels=labels,
                                                     show_labels=False), static=True)
        recording.log("grasps/stems", rr.LineStrips3D(to_base(stem), colors=colors, radii=THIN), static=True)
        recording.log("opening/grasps", rr.BarChart(counts, abscissa=(edges[:-1] + edges[1:]) / 2,
                                                    widths=edges[1] - edges[0], color=rgba(ACCENT)), static=True)
        recording.log("opening/limit", rr.BarChart([counts.max() * 1.08], abscissa=[limit * DEG],
                                                   widths=(edges[1] - edges[0]) / 2, color=rgba(LIMIT)), static=True)
        eye = rrb.EyeControls3D(kind="Orbital", look_target=centre, eye_up=[0, 0, 1],   # from +x: the jaws face you
                                position=centre + span * np.array([0.9, -0.36, 0.65]))
        table = rrb.LineGrid3D(True, spacing=0.05, color=rgba(HAIR),
                               plane=rr.components.Plane3D([0, 0, 1], positions[:, 2].min()))
        recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(
            rrb.Spatial3DView(origin="/grasps", background=WHITE, line_grid=table, eye_controls=eye,
                              name=f"{len(positions):,} real repick grasps as two-finger grippers · base frame, "
                                   "5 cm grid at the lowest fingertip · darker = wider · red = at the limit"),
            rrb.BarChartView(origin="/opening", name="Claw opening [deg] · red = limit",
                             plot_legend=rrb.PlotLegend(visible=False)),
            column_shares=[3, 1]), collapse_panels=True))
    return path


def direction_sphere(path: Path) -> Path:
    """The equal-area bin grid as hairlines, visited bins as Mesh3D patches, every approach direction as a point."""
    payload = by_id("direction-sphere").data()
    rings, sectors, counts = payload["rings"], payload["sectors"], payload["counts"]
    z_edges, azimuth_edges = np.linspace(-1, 1, rings + 1), np.linspace(-np.pi, np.pi, sectors + 1)
    on_sphere = lambda z, azimuth: np.stack(  # noqa: E731
        [np.sqrt(1 - z**2) * np.cos(azimuth), np.sqrt(1 - z**2) * np.sin(azimuth), z], axis=-1)
    with rr.RecordingStream("coverage-direction-sphere", recording_id="direction-sphere") as recording:
        recording.save(path)
        sweep = np.linspace(-np.pi, np.pi, 97)
        circles = [on_sphere(np.full_like(sweep, z), sweep) for z in z_edges[1:-1]]
        meridians = [on_sphere(np.linspace(-1, 1, 49), np.full(49, azimuth)) for azimuth in azimuth_edges[:-1]]
        recording.log("sphere/bins", rr.LineStrips3D(circles + meridians, colors=rgba(HAIR), radii=THIN), static=True)
        vertices, triangles, colors = [], [], []              # each visited bin: a 5 x 5 vertex patch on the sphere
        for (ring, sector), color in zip(np.argwhere(counts > 0), ramp(counts[counts > 0], log=True)):
            z, azimuth = np.meshgrid(np.linspace(*z_edges[ring:ring + 2], 5),
                                     np.linspace(*azimuth_edges[sector:sector + 2], 5))
            first = len(vertices) * 25
            vertices.append(1.002 * on_sphere(z, azimuth).reshape(-1, 3))
            grid = np.arange(25).reshape(5, 5)[:-1, :-1].ravel() + first
            triangles += [np.column_stack([grid, grid + 1, grid + 5]), np.column_stack([grid + 1, grid + 6, grid + 5])]
            colors.append(np.repeat([color], 25, axis=0))
        recording.log("sphere/visited", rr.Mesh3D(vertex_positions=np.concatenate(vertices),
                                                  triangle_indices=np.concatenate(triangles),
                                                  vertex_colors=np.concatenate(colors)), static=True)
        recording.log("sphere/approaches", rr.Points3D(1.01 * payload["directions"], colors=rgba(INK),
                                                       radii=rr.Radius.ui_points(2.0)), static=True)
        recording.log("sphere/axes", rr.Arrows3D(vectors=[[1.35, 0, 0], [0, 1.35, 0], [0, 0, 1.35], [0, 0, -1.35]],
                                                 labels=["+x", "+y", "+z (up)", "−z (straight down)"],
                                                 colors=[rgba(INK)] * 4, radii=THIN), static=True)
        eye = rrb.EyeControls3D(kind="Orbital", look_target=[0, 0, -0.4], eye_up=[0, 0, 1], position=[0.8, -1.1, -3.0])
        recording.send_blueprint(rrb.Blueprint(
            rrb.Spatial3DView(origin="/sphere", background=WHITE, line_grid=False, eye_controls=eye,
                              name=f"Approach direction of {len(payload['directions']):,} real grasps · "
                                   f"{rings} x {sectors} equal-area bins · colour = grasps (log) · "
                                   f"{payload['visited_pct']:.1f} % of bins visited"),
            collapse_panels=True))
    return path


# --- headless capture (not part of any tool's implementation) ---------------------------------------------------------

WEB_PORT, GRPC_PORT = 9931, 9932
# WHY pinned, lowest priority, one process at a time: the real arm may be running a pick loop on this host, and a
# loaded host makes its motion gate refuse (orchestrator, 2026-11-19). Swiftshader is CPU-hungry: see the browser flags.
PINNED = ["taskset", "-c", "2-5,16,17", "nice", "-n", "19"]
# WHY the bundled binary, not the venv's `rerun`: that is a Python wrapper that spawns this binary as a child, so
# stopping the wrapper's PID left the real server holding both ports (seen 2026-11-19).
RERUN = Path(rerun_cli.__file__).with_name("rerun")
CHROME = "/home/bam/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome"
NOTES = {
    "limit-bars": "Native BarChart per derivative (abscissa + widths, so bins sit at their true degrees). A "
                  "BarChartView has no reference lines and no axis titles, so each limit is a thin red bar at ±limit "
                  "and the units ride in the view name. The qdd panel's x axis shows only its 0 tick (egui thins "
                  "wide labels).",
    "coverage-heatmap": "Rerun has no 2-D heatmap with axes: Spatial2DView boxes are outline-only and Image/Tensor "
                        "cells live in pixel space without units. So the cells are flat solid Boxes3D seen straight "
                        "down (a plane facing the camera has no perspective distortion) and the ticks are labels.",
    "histogram-over-runs": "Rerun's own idiom: the run IS the timeline. TensorBoard's distribution bands become 9 "
                           "shaded SeriesLines (no fill-between); the BarChart below re-draws the histogram of the run "
                           "under the time cursor, so scrubbing 'run' plays the histogram over steps.",
    "coverage-growth": "Scalars on a 'run' sequence timeline: the native fit, as good as any plotting tool. No band "
                       "fill, so the random-order 10-90 % band is two hairlines.",
    "phase-space-3d": "Boxes3D per visited cell (log colour), reachable-but-empty cells translucent grey; a per-axis "
                      "scale transform keeps the logged values in degrees. Samples arrive run by run on the timeline. "
                      "3-D views have no tick marks, so the axes carry their range in the label.",
    "tool-paths": "LineStrips3D, one cycle per timeline tick so scrubbing replays the campaign; voxel Boxes3D beside "
                  "it with the same camera. Rerun lines do not blend additively (low alpha just paints over), so the "
                  "Strava 'worn path' is each segment coloured by its voxel's density (log).",
    "grasp-poses": "Every grasp as a two-finger LineStrips3D wireframe (jaws along x, approach +z; nominal 50 mm "
                   "fingers, width from the opening angle) on a 5 cm table grid, coloured by opening, red = at the "
                   "limit. 1131 grippers in a 0.5 m patch overlap heavily; the opening histogram beside it carries "
                   "the distribution.",
    "direction-sphere": "Equal-area bin grid as hairline LineStrips3D, visited bins as Mesh3D patches (log colour), "
                        "every approach as a point, camera from below. Honest result: every approach is within a few "
                        "bins of straight down.",
}

SHOOT = r"""
import sys
from playwright.sync_api import sync_playwright
url, out, chrome = sys.argv[1:4]
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, executable_path=chrome, args=[
        "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars",
        "--renderer-process-limit=1", "--num-raster-threads=1"])
    page = browser.new_page(viewport={"width": 1400, "height": 850})
    # offline: the viewer's update check otherwise drops a toast over a view title
    page.route("**/*", lambda route: route.continue_() if "127.0.0.1" in route.request.url else route.abort())
    page.goto(url)
    page.wait_for_timeout(9000)
    previous = None
    for _ in range(15):                   # done when two shots 2 s apart are identical (the data has streamed in)
        shot = page.screenshot()
        if shot == previous:
            break
        previous = shot
        page.wait_for_timeout(2000)
    open(out, "wb").write(shot)
    browser.close()
"""


def _answers(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=1)
        return True
    except OSError:
        return False


def _serve(rrd: Path) -> subprocess.Popen:
    if _answers(f"http://127.0.0.1:{WEB_PORT}/"):
        raise RuntimeError(f"port {WEB_PORT} is already serving something that is not ours; not touching it")
    log = rrd.with_suffix(".serve.log")
    server = subprocess.Popen([*PINNED, str(RERUN), "--serve-web", "--bind", "127.0.0.1",
                               "--web-viewer-port", str(WEB_PORT), "--port", str(GRPC_PORT), str(rrd)],
                              stdout=log.open("w"), stderr=subprocess.STDOUT)
    for _ in range(100):
        if server.poll() is not None:
            raise RuntimeError(f"rerun exited: {log.read_text()[-800:]}")
        if _answers(f"http://127.0.0.1:{WEB_PORT}/"):
            return server
        time.sleep(0.2)
    _stop(server)
    raise RuntimeError("rerun web viewer never answered")


def _stop(server: subprocess.Popen) -> None:
    """Stop exactly the process this module started, by its PID (never by port or name)."""
    if server.poll() is None:
        os.kill(server.pid, signal.SIGTERM)
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            os.kill(server.pid, signal.SIGKILL)
            server.wait(5)


def screenshot(rrd: Path, image: Path) -> Path:
    """Serve `rrd` on 127.0.0.1, screenshot it in headless swiftshader Chromium, stop the server we started."""
    server = _serve(rrd)
    try:
        url = f"http://127.0.0.1:{WEB_PORT}/?url=" + urllib.parse.quote(f"rerun+http://127.0.0.1:{GRPC_PORT}/proxy",
                                                                        safe="")
        # WHY a session of its own: on a timeout the whole uv -> python -> chrome tree is ours to stop, as one group
        shooter = subprocess.Popen([*PINNED, "uv", "run", "--no-project", "--with", "playwright==1.65.2",
                                    "python", "-c", SHOOT, url, str(image), CHROME], start_new_session=True,
                                   env={key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"})
        try:
            if shooter.wait(timeout=180) != 0:
                raise RuntimeError(f"screenshot of {rrd.name} failed")
        except subprocess.TimeoutExpired:
            os.killpg(shooter.pid, signal.SIGKILL)
            shooter.wait()
            raise
    finally:
        _stop(server)
    return image


def capture(only: list[str] | None = None) -> dict:
    """All 8 (or `only` these) types: write each .rrd, serve it headlessly, screenshot 1400 x 850, stop the server.
    Returns {type_id: {"image": png, "rrd": rrd, "notes": str}}."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    results = {}
    for core in TYPES:
        if only and core.id not in only:
            continue
        rrd = globals()[core.id.replace("-", "_")](MEDIA / f"{core.id}.rrd")
        image = screenshot(rrd, MEDIA / f"{core.id}.png")
        results[core.id] = {"image": image, "rrd": rrd, "notes": NOTES[core.id]}
    return results


if __name__ == "__main__":
    print(capture(sys.argv[1:] or None))
