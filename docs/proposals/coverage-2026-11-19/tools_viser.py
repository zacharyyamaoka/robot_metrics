"""The Viser column of the coverage gallery: each of the eight core TYPES drawn the most Viser-native way.

WHY this file (Zach, 2026-11-19): "lets have like Type of visualization, and then show the implementation in these
different visualization tools". Every scene function below is shown verbatim in the report as "the Viser
implementation" and draws `by_id(...).data()` and nothing else, so a difference from the other columns is the tool.

WHY colour-coded cells are square POINTS, not boxes (measured 2026-11-19 in headless Chrome): Viser renders meshes and
line segments through three.js' ACES tone mapping, so the white-to-accent ramp washes out (#0d2f5c draws as #5e9bcb,
#d6e4f5 as near-grey #e7e9ee, indistinguishable from "unreachable"). Point clouds bypass tone mapping and draw the exact
sRGB colour, so a cell's colour matches its legend. Translucent boxes are kept only for colour-free context.
Two more Viser 1.1.1 quirks, both measured: a point's on-screen width is TWICE `point_size` in world units (the shader
divides by tan(fov/2), not 2 tan(fov/2)), so a cell of width w gets `point_size=w / 2`; and `gui.add_plotly(aspect=)`
is applied as height / width (`height = width * aspect`), the inverse of its docstring and of `add_uplot`.

Capture all eight from a fresh shell (headless Chrome, port 8931, nothing left running):

    taskset -c 2-5,16,17 nice -n 19 uv run --no-project --python /home/bam/robot_metrics/.venv/bin/python --with viser==1.1.1 --with plotly==6.3.0 --with playwright==1.65.2 python /home/bam/robot_metrics/docs/proposals/coverage-2026-11-19/tools_viser.py

Explore one live (prints its URL; Ctrl-C stops it):

    taskset -c 2-5,16,17 nice -n 19 uv run --no-project --python /home/bam/robot_metrics/.venv/bin/python --with viser==1.1.1 --with plotly==6.3.0 --with playwright==1.65.2 python /home/bam/robot_metrics/docs/proposals/coverage-2026-11-19/tools_viser.py --serve phase-space-3d
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from matplotlib.colors import to_hex, to_rgb
from matplotlib.ticker import MaxNLocator
from playwright.sync_api import sync_playwright
from plotly.subplots import make_subplots
from viser import uplot

from core_types import TYPES, by_id
from style import ACCENT, EMPTY_WHITE, HAIR, INK, LIMIT, MUTED, UNREACHABLE

ACCENT_RGB, LIMIT_RGB, GREY_RGB, HAIR_RGB, MUTED_RGB, INK_RGB = (
    tuple(round(channel * 255) for channel in to_rgb(colour)) for colour in (ACCENT, LIMIT, UNREACHABLE, HAIR, MUTED, INK))
PLANNER = "#e8590c"                            # the planner's own, softer limit: amber, so it never reads as the hard one
BOX = trimesh.creation.box()                   # unit cube, instanced for context cells and voxels
DEG = 180.0 / np.pi
OUT = Path("/home/bam/robot_metrics/reports/media/coverage-gallery-2026-11-19/tools/viser")
CHROME = "/home/bam/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome"
PORT = 8931


# --- three tiny helpers: Viser has instancing, but no colour bar and no axes widget ----------------------------------

def boxes(server, name, centres, sizes, colour, **options):
    """Instanced boxes: thousands of cells are ONE batched mesh (one draw call), so the scene stays interactive."""
    count = len(centres)
    return server.scene.add_batched_meshes_simple(
        name, BOX.vertices, BOX.faces, batched_wxyzs=np.tile((1.0, 0.0, 0.0, 0.0), (count, 1)), batched_positions=centres,
        batched_scales=np.broadcast_to(sizes, (count, 3)).copy(), batched_colors=colour, **options)


def colour_by(server, values, label, log=True):
    """White-to-accent colours for `values` (all > 0) AND their legend in the GUI, so no scene shows colour without key."""
    low, high = float(values.min()), max(float(np.percentile(values, 98)), float(values.min()) * 1.0001)
    ramp = (np.log(values / low) / np.log(high / low)) if log else (values - low) / (high - low)
    rgb = (EMPTY_WHITE(np.clip(ramp, 0, 1))[:, :3] * 255).astype(np.uint8)
    stops = ", ".join(to_hex(EMPTY_WHITE(x)) for x in np.linspace(0, 1, 6))
    server.gui.add_html(
        f'<div style="padding:2px 12px 6px;font-size:12px;color:{MUTED}">{label}{" (log)" if log else ""}'
        f'<div style="height:10px;border-radius:2px;margin:3px 0;background:linear-gradient(90deg,{stops})"></div>'
        f'<div style="display:flex;justify-content:space-between"><span>{low:.3g}</span><span>{high:.3g}+</span></div></div>')
    return rgb


def axes_box(server, low, high, scale, titles):
    """A hairline box with nice-number ticks and axis titles in DISPLAY units (scene = display x scale)."""
    low, high, scale = (np.asarray(v, float) for v in (low, high, scale))
    corners = np.array(list(itertools.product(*zip(low * scale, high * scale))))
    edges = [(a, b) for a, b in itertools.combinations(range(8), 2) if bin(a ^ b).count("1") == 1]
    server.scene.add_line_segments("/axes/box", corners[np.array(edges)], HAIR_RGB, thickness=1.0, thickness_units="screen")
    # ticks on three edges that share no corner: x along the front, y along the right (in 3-D), z up the left
    on_high = {0: (0, 0, 0), 1: (int(high[2] > low[2]), 0, 0), 2: (0, 0, 0)}
    for axis in np.flatnonzero(high > low):
        side = np.array(on_high[axis])
        edge = np.where(side, high, low) * scale
        outward = np.where(np.arange(3) == axis, 0.0, np.where(side, 1, -1) * 0.05 * (high - low) * scale)
        for tick in MaxNLocator(5).tick_values(low[axis], high[axis]):
            if low[axis] <= tick <= high[axis]:
                at = edge + outward
                at[axis] = tick * scale[axis]
                server.scene.add_label(f"/axes/{axis}/{tick:g}", f"{tick:g}", position=at, anchor="center-center",
                                       font_screen_scale=0.85)
        at = edge + 2.8 * outward
        at[axis] = (low[axis] + high[axis]) / 2 * scale[axis]
        server.scene.add_label(f"/axes/{axis}/title", titles[axis], position=at, anchor="center-center")


def look(server, position, look_at, up=(0.0, 0.0, 1.0), fov_deg=45.0):
    server.initial_camera.position, server.initial_camera.look_at = position, look_at
    server.initial_camera.up, server.initial_camera.fov = up, np.radians(fov_deg)


# --- 1-D: a Plotly figure in the GUI (Viser's scene has no chart primitives) -----------------------------------------

def limit_bars(server: viser.ViserServer) -> None:
    payload = by_id("limit-bars").data()
    rows = list(zip(payload["names"], payload["units_display"], payload["values"], payload["limits"], payload["planner_limits"]))
    figure = make_subplots(rows=2, cols=2, subplot_titles=[f"{name} ({unit})" for name, unit, *_ in rows],
                           horizontal_spacing=0.08, vertical_spacing=0.16)
    peaks = []
    for index, (name, unit, values, limit, planner) in enumerate(rows):
        at = dict(row=index // 2 + 1, col=index % 2 + 1)
        values = DEG * values                                                 # SI inside, degrees at display
        reach = 1.15 * DEG * max(np.abs(limit)) if limit else np.abs(values).max()
        counts, edges = np.histogram(values, np.linspace(-reach, reach, 81))
        centres = (edges[:-1] + edges[1:]) / 2
        over = (centres < DEG * limit[0]) | (centres > DEG * limit[1]) if limit else np.zeros(len(centres), bool)
        figure.add_bar(x=centres, y=np.where(counts > 0, counts, None), width=np.diff(edges),          # empty = no bar
                       marker_color=np.where(over, LIMIT, ACCENT if limit else MUTED).tolist(), **at)
        for end in limit or ():
            figure.add_vline(x=DEG * end, line_color=LIMIT, line_width=2, opacity=1, annotation_text=f"limit {DEG * end:+.0f}", **at)
        for end in planner or ():
            figure.add_vline(x=DEG * end, line_color=PLANNER, line_dash="dash", opacity=1, annotation_text="planner", **at)
        share = np.where(values < 0, values / (DEG * limit[0]), values / (DEG * limit[1])) if limit else None
        peaks.append((f"{name} peak, % of its limit", round(100 * float(share.max()))) if limit else
                     (f"{name} peak ({unit}), no limit", round(float(np.abs(values).max()))))
    figure.update_yaxes(type="log", dtick=1, title_text="samples (log)", title_font_size=11)
    figure.update_layout(template="simple_white", showlegend=False, bargap=0, margin=dict(l=60, r=20, t=40, b=30),
                         font=dict(size=11))
    server.gui.main_panel.float(x=150, y=12, width=1100, height=826)   # the plot IS the view: give it the room
    server.gui.add_plotly(figure, aspect=1 / 1.85)                       # Viser applies it as height / width
    for label, value in peaks:
        server.gui.add_number(label, value, disabled=True)


# --- 2-D: one exact-colour square point per cell, flat in the scene --------------------------------------------------

def coverage_heatmap(server: viser.ViserServer) -> None:
    payload = by_id("coverage-heatmap").data()
    q_edges, qd_edges = payload["q_edges_deg"], payload["qd_edges_deg"]
    scale = np.array([1 / np.diff(q_edges)[0], 1 / np.diff(qd_edges)[0], 1.0])     # one cell = one scene unit
    q, qd = np.meshgrid((q_edges[:-1] + q_edges[1:]) / 2, (qd_edges[:-1] + qd_edges[1:]) / 2, indexing="ij")
    centres = np.column_stack([q.ravel(), qd.ravel(), 0 * q.ravel()]) * scale
    samples, reachable = payload["samples"].ravel(), payload["reachable"].ravel()
    visited = samples > 0
    cells = dict(point_size=0.92 / 2, point_shape="square", point_shading="flat")   # 0.92 wide: the gap reads as a grid
    server.scene.add_point_cloud("/visited cells", centres[visited],
                                 colors=colour_by(server, samples[visited], "samples per cell"), **cells)
    server.scene.add_point_cloud("/unreachable cells", centres[~reachable & ~visited], colors=GREY_RGB, **cells)
    axes_box(server, (q_edges[0], qd_edges[0], 0), (q_edges[-1], qd_edges[-1], 0), scale,
             ("q from bottom (deg)", "qd (deg/s)", ""))
    server.gui.add_number("visited, % of reachable", round(payload["visited_pct"], 1), disabled=True)
    server.gui.add_number("≥ 50 samples, % of reachable", round(payload["sufficient_pct"], 1), disabled=True)
    server.gui.add_markdown("grey = unreachable · white = reachable, never visited")
    middle = (q_edges[0] + q_edges[-1]) / 2 * scale[0]
    look(server, position=(middle + 3.6, -0.6, 41.0), look_at=(middle + 3.6, -0.6, 0.0), up=(0.0, 1.0, 0.0))


# --- series ---------------------------------------------------------------------------------------------------------

def histogram_over_runs(server: viser.ViserServer) -> None:
    payload = by_id("histogram-over-runs").data()
    limit, runs = DEG * payload["qd_limit"], len(payload["qd_per_run"])
    edges = np.linspace(-1.15 * limit, 1.15 * limit, 47)
    share = np.array([np.histogram(DEG * qd, edges)[0] / max(len(qd), 1) for qd in payload["qd_per_run"]]) * 100
    scale = np.array([4.0 / runs, 2.4 / np.ptp(edges), 1.4 / share.max()])          # run, deg/s, % -> scene units
    run, speed = np.meshgrid(np.arange(runs), (edges[:-1] + edges[1:]) / 2, indexing="ij")
    vertices = np.stack([run, speed, share], axis=-1).reshape(-1, 3) * scale       # TensorBoard's ridges, joined up
    bins = share.shape[1]
    corner = (np.arange(runs - 1)[:, None] * bins + np.arange(bins - 1)).ravel()
    quads = np.stack([corner, corner + bins, corner + bins + 1, corner + 1], axis=1)
    quads = quads[share.ravel()[quads].max(axis=1) > 0]                            # empty bins: no surface at all
    server.scene.add_mesh_simple("/share of each run", vertices, np.concatenate([quads[:, :3], quads[:, [0, 2, 3]]]),
                                 color=ACCENT_RGB, side="double")
    for side in (-1, 1):
        server.scene.add_line_segments(f"/limit{side:+d}", np.array([[[0, side * limit, 0], [runs, side * limit, 0]]]) * scale,
                                       LIMIT_RGB, thickness=2.5, thickness_units="screen")
    axes_box(server, (0, edges[0], 0), (runs, edges[-1], share.max()), scale,
             ("run (recording order)", "qd (deg/s)", "% of the run"))
    touching = sum(bool(np.any(np.abs(DEG * qd) >= limit)) for qd in payload["qd_per_run"])
    server.gui.add_number("runs", runs, disabled=True)
    server.gui.add_number(f"runs reaching the ±{limit:.0f} deg/s limit", touching, disabled=True)
    server.gui.add_markdown("height = share of the run's samples at that speed · red = the qd limit")
    look(server, position=(7.0, -3.2, 3.0), look_at=(1.75, 0.0, 0.2))


def coverage_growth(server: viser.ViserServer) -> None:
    payload = by_id("coverage-growth").data()
    runs = np.arange(1, len(payload["in_order_pct"]) + 1, dtype=float)
    low, high = payload["random_band_pct"]                                       # 10th-90th pct of random run orders
    chao2 = np.full_like(runs, payload["chao2_pct"])
    server.gui.add_uplot(
        data=(runs, payload["in_order_pct"], payload["random_mean_pct"], low, high, chao2),
        series=(uplot.Series(label="run"),
                uplot.Series(label="recording order", stroke=ACCENT, width=2.5),
                uplot.Series(label="random order (mean)", stroke=MUTED, width=1.5, dash=(6, 4)),
                uplot.Series(label="random, 10th pct", stroke="rgba(0,0,0,0)"),
                uplot.Series(label="random, 90th pct", stroke="rgba(0,0,0,0)"),
                uplot.Series(label="Chao2 asymptote", stroke=INK, width=1.5, dash=(2, 4))),
        bands=(uplot.Band(series=(4, 3), fill="rgba(110,110,115,0.18)"),),
        scales={"x": uplot.Scale(time=False), "y": uplot.Scale(range=(0.0, float(max(chao2[0], high.max())) * 1.15))},
        axes=(uplot.Axis(label="runs, in recording order"), uplot.Axis(label="visited % of reachable cells", size=60)),
        aspect=2.05)
    server.gui.add_number("visited % after all runs", round(float(payload["in_order_pct"][-1]), 1), disabled=True)
    server.gui.add_number("Chao2: % ever expected to be visited", round(float(payload["chao2_pct"]), 1), disabled=True)
    server.gui.add_number("reachable (q, qd, qdd) cells", int(payload["reachable_cells"]), disabled=True)
    server.gui.main_panel.float(x=150, y=12, width=1100, height=826)   # the plot IS the view: give it the room


# --- 3-D ------------------------------------------------------------------------------------------------------------

def phase_space_3d(server: viser.ViserServer) -> None:
    payload = by_id("phase-space-3d").data()
    centres, size, counts, reachable = payload["centers_deg"], payload["cell_size_deg"], payload["counts"], payload["reachable"]
    scale = 1.0 / size                                   # one cell = one scene unit on every axis: a cube per cell
    visited = counts > 0
    server.scene.add_point_cloud("/visited cells", centres[visited] * scale,
                                 colors=colour_by(server, counts[visited], "samples per cell"),
                                 point_size=0.8 / 2, point_shape="square", point_shading="flat")
    boxes(server, "/reachable, never visited", centres[reachable & ~visited] * scale, 0.7, GREY_RGB, opacity=0.1)
    samples = server.scene.add_point_cloud("/raw samples", payload["samples_deg"] * scale, colors=INK_RGB,
                                           point_size=0.03, visible=False)
    show = server.gui.add_checkbox("show 40 k raw samples", False)
    show.on_update(lambda _: setattr(samples, "visible", show.value))
    low, high = (centres[reachable] - size / 2).min(axis=0), (centres[reachable] + size / 2).max(axis=0)
    axes_box(server, low, high, scale, payload["axes"])
    server.gui.add_number("visited, % of reachable", round(float(payload["visited_pct"]), 1), disabled=True)
    server.gui.add_markdown("grey ghost = reachable, never visited · unreachable = absent")
    middle = (low + high) / 2 * scale
    look(server, position=middle + (-17.0, -24.0, 15.0), look_at=middle)


def tool_paths(server: viser.ViserServer) -> None:
    payload = by_id("tool-paths").data()
    edges, counts = payload["voxel_edges_m"], payload["voxel_counts"]
    visited = counts > 0
    palette = np.zeros(counts.shape + (3,), np.uint8)
    palette[visited] = colour_by(server, counts[visited], "path samples per 2 cm voxel")
    points = np.concatenate(payload["paths_m"])
    voxel = tuple(np.searchsorted(edges[axis], points[:, axis], side="right") - 1 for axis in range(3))
    # WHY points coloured by voxel density, not translucent lines: Viser lines have no opacity and are tone-mapped, so
    # the Strava "where paths pile up" read is carried by exact colour on every path sample instead
    server.scene.add_point_cloud("/paths", points, colors=palette[voxel], point_size=0.0008, point_shape="circle",
                                 point_shading="flat")
    middles = np.stack(np.meshgrid(*[(edge[:-1] + edge[1:]) / 2 for edge in edges], indexing="ij"), axis=-1)
    voxels = boxes(server, "/occupied voxels", middles[visited], 0.019, ACCENT_RGB, opacity=0.06)
    show = server.gui.add_checkbox("show occupied 2 cm voxels", True)
    show.on_update(lambda _: setattr(voxels, "visible", show.value))
    server.scene.add_frame("/base", axes_length=0.06, axes_radius=0.0025)
    server.scene.add_label("/base/name", "base", position=(0, 0, 0.03))
    low, high = np.array([edge[0] for edge in edges]), np.array([edge[-1] for edge in edges])
    axes_box(server, 1000 * low, 1000 * high, (0.001,) * 3, ("x (mm)", "y (mm)", "z (mm)"))
    server.gui.add_number("planned TCP paths", len(payload["paths_m"]), disabled=True)
    server.gui.add_number("occupied 2 cm voxels", int(visited.sum()), disabled=True)
    middle = (low + high) / 2
    look(server, position=middle + (-0.6, -1.15, 0.62), look_at=middle + (0.06, 0.0, -0.02))


def grasp_poses(server: viser.ViserServer) -> None:
    payload = by_id("grasp-poses").data()
    positions, rotations = payload["positions_m"], payload["rotations"]
    share = payload["opening_rad"] / payload["opening_limit_rad"]          # 1.0 = claw at its limit
    # robot_msgs' gripper glyph in the tool frame: fingertips at the TCP, jaws along x, approach +z, body behind (-z)
    glyph = np.array([[(-1, 0, 0), (-1, 0, -1)], [(1, 0, 0), (1, 0, -1)], [(-1, 0, -1), (1, 0, -1)], [(0, 0, -1), (0, 0, -2.2)]])
    half_width = 0.018 * share                    # jaw spacing ∝ claw angle: the claw's geometry is not in the data
    local = glyph[None] * np.stack([half_width, 0 * share + 1, 0 * share + 0.016], axis=1)[:, None, None, :]
    world = positions[:, None, None, :] + np.einsum("nij,nsej->nsei", rotations, local)
    colours = colour_by(server, share, "claw opening at the TCP dot, share of its limit", log=False)
    colours[share >= 0.999] = LIMIT_RGB
    line_colours = np.where((share >= 0.999)[:, None], LIMIT_RGB, MUTED_RGB).astype(np.uint8)
    server.scene.add_line_segments("/grippers", world.reshape(-1, 2, 3), np.repeat(line_colours, 8, axis=0).reshape(-1, 2, 3),
                                   thickness=1.2, thickness_units="screen")
    # WHY the colour sits on the TCP dot, not the gripper lines: Viser tone-maps lines, points keep the exact colour
    server.scene.add_point_cloud("/tcp", positions, colors=colours, point_size=0.002, point_shape="circle",
                                 point_shading="flat")
    low, high = positions.min(axis=0) - 0.03, positions.max(axis=0) + 0.03
    axes_box(server, 1000 * low, 1000 * high, (0.001,) * 3, ("x (mm)", "y (mm)", "z (mm)"))
    server.gui.add_number("grasps", len(positions), disabled=True)
    server.gui.add_number("runs", len(set(payload["runs"])), disabled=True)
    server.gui.add_number("median opening, % of limit", round(100 * float(np.median(share)), 1), disabled=True)
    server.gui.add_number("at the claw limit (red)", int((share >= 0.999).sum()), disabled=True)
    middle = (low + high) / 2
    look(server, position=middle + (-0.1, -0.52, 0.36), look_at=middle + (0.03, 0.0, 0.0))


def direction_sphere(server: viser.ViserServer) -> None:
    payload = by_id("direction-sphere").data()
    counts, rings, sectors = payload["counts"], payload["rings"], payload["sectors"]
    colours = np.zeros(counts.shape + (3,), np.uint8)
    colours[counts > 0] = colour_by(server, counts[counts > 0].astype(float), "grasps per equal-area cell")
    server.scene.configure_environment_map(hdri=None)              # the default HDRI tints a white sphere pink-grey
    server.scene.configure_default_lights(enabled=False)
    server.scene.add_light_ambient("/light/ambient", intensity=4.2)
    server.scene.add_light_directional("/light/key", intensity=1.2, wxyz=(0.92, 0.25, -0.3, 0.0))
    server.scene.add_icosphere("/sphere", radius=0.99, color=(255, 255, 255), subdivisions=5)   # empty = white
    # fill each visited cell with exact-colour points; rings are uniform in z (Archimedes: equal z bands, equal area)
    ring, sector = np.flatnonzero(counts > 0) // sectors, np.flatnonzero(counts > 0) % sectors
    rise, turn = np.meshgrid(np.linspace(0.02, 0.98, 30), np.linspace(0.02, 0.98, 24), indexing="ij")
    bottom, top = np.arccos(-1 + 2 * ring[:, None] / rings), np.arccos(-1 + 2 * (ring[:, None] + 1) / rings)
    z = np.cos(bottom + (top - bottom) * rise.ravel())                 # even in arc length, so no gaps near the pole
    azimuth = -np.pi + 2 * np.pi * (sector[:, None] + turn.ravel()) / sectors
    fill = np.stack([np.sqrt(1 - z**2) * np.cos(azimuth), np.sqrt(1 - z**2) * np.sin(azimuth), z], axis=-1)
    server.scene.add_point_cloud("/visited cells", fill.reshape(-1, 3), colors=np.repeat(colours[ring, sector], rise.size, 0),
                                 point_size=0.013, point_shape="square", point_shading="flat")
    circle = np.linspace(0, 2 * np.pi, 97)
    for edge in -1 + 2 * np.arange(1, rings) / rings:                      # the equal-area bin edges, as hairlines
        loop = np.column_stack([np.sqrt(1 - edge**2) * np.cos(circle), np.sqrt(1 - edge**2) * np.sin(circle), 0 * circle + edge])
        server.scene.add_line_segments(f"/bins/z{edge:+.2f}", np.stack([loop[:-1], loop[1:]], axis=1) * 1.003, HAIR_RGB,
                                       thickness=1.0, thickness_units="screen")
    height = np.linspace(-1, 1, 49)
    for boundary in -np.pi + 2 * np.pi * np.arange(sectors) / sectors:
        meridian = np.column_stack([np.sqrt(1 - height**2) * np.cos(boundary), np.sqrt(1 - height**2) * np.sin(boundary), height])
        server.scene.add_line_segments(f"/bins/a{boundary:+.2f}", np.stack([meridian[:-1], meridian[1:]], axis=1) * 1.003,
                                       HAIR_RGB, thickness=1.0, thickness_units="screen")
    server.scene.add_point_cloud("/approaches", payload["directions"] * 1.012, colors=INK_RGB, point_size=0.003,
                                 point_shape="circle", point_shading="flat")
    for name, at in {"+x": (1.3, 0, 0), "+y": (0, 1.3, 0), "+z (up)": (0, 0, 1.25), "−z (straight down)": (0, 0, -1.3)}.items():
        server.scene.add_label(f"/axis/{name}", name, position=at, anchor="center-center", depth_test=True)
    server.gui.add_number("bins visited, % of sphere", round(float(payload["visited_pct"]), 1), disabled=True)
    server.gui.add_number("grasps", len(payload["directions"]), disabled=True)
    server.gui.add_markdown("black dots = each grasp's approach direction")
    look(server, position=(-1.85, -0.75, -2.55), look_at=(0.0, 0.0, -0.25))


# --- serving and headless capture -----------------------------------------------------------------------------------

SCENES = {core.id: globals()[core.id.replace("-", "_")] for core in TYPES}

# Written after looking at each capture (2026-11-19, headless Chrome 1400x850): what Viser did well and where it is weak.
NOTES = {
    "limit-bars": (
        "Viser's 3-D scene has no chart primitives, so the honest Viser answer for 1-D is a Plotly figure in Viser's GUI "
        "panel, floated wide; Viser adds only the panel and the peak-%-of-limit numbers. Gotcha: Viser 1.1.1 applies "
        "add_plotly(aspect=) as height/width, the inverse of its docstring. The page logs one Plotly console error "
        "('_redrawFromAutoMarginCount': Viser calls Plotly.react once at width 0 before the panel sizes); the final "
        "render is correct."),
    "coverage-heatmap": (
        "A flat grid in the 3-D scene: one exact-colour square point per cell. Boxes and lines go through three.js' ACES "
        "tone mapping, which washed the ramp until 1-sample cells looked like the grey unreachable ones; point clouds "
        "skip it. Viser has no axes or colour-bar widget, so both are hand-built (scene labels, an HTML gradient in the "
        "GUI). Pan/zoom is free, but it adds nothing over a static heatmap."),
    "histogram-over-runs": (
        "TensorBoard's offset histograms as one lit surface (run x qd x share of the run), with red limit rails along the "
        "run axis. 423 separate ridge lines merged into one mass, so they are joined into a single mesh. Meshes are "
        "tone-mapped, so the surface is one colour: height carries density, lighting gives shape. You can see that rounds "
        "differ; exact values are hard to read. A flat heat strip (W&B style) reads faster; orbiting is what Viser adds."),
    "coverage-growth": (
        "A series, so it goes in the GUI: uPlot (bundled with Viser, no extra dependency) floated wide, with the 10th-90th "
        "percentile random-order band and the Chao2 asymptote. The 3-D canvas is unused. Viser adds nothing over "
        "Matplotlib or W&B here, except that the uPlot could update live as runs arrive."),
    "phase-space-3d": (
        "Viser's home turf. Visited (q, qd, qdd) cells are exact-colour square points, one cell = one scene unit on every "
        "axis. Reachable-but-empty cells are faint instanced boxes; unreachable cells are absent. A GUI checkbox shows "
        "the 40 k raw samples. Orbiting is the point: a still shows one side of the blob. Points always face the camera, "
        "so cells read as tiles rather than cubes."),
    "tool-paths": (
        "The Strava look comes from colour, not alpha: Viser line segments have no opacity and are tone-mapped. So every "
        "planned TCP sample is an exact-colour point, coloured by its 2 cm voxel's sample count (dark = the shared "
        "corridor). Occupied voxels are faint instanced boxes (GUI toggle), and the base frame is drawn. Reads at a "
        "glance; better live, where you can orbit."),
    "grasp-poses": (
        "1131 robot_msgs-style two-finger glyphs (jaws along x, approach +z, fingertips at the TCP) drawn as one batched "
        "line set. Jaw spacing follows the commanded claw angle, because the claw's geometry isn't in the data. Opening "
        "is coloured on the TCP dot, since points keep exact colour and tone-mapped lines would wash it; the one "
        "at-limit grasp is red. Weak in a still: openings barely vary (median 81 % of limit), so the colour carries "
        "little, and the dense pick zone turns into a grey thicket. Zoom and orbit in the live view to read it."),
    "direction-sphere": (
        "A white unit sphere lit by neutral lights; the default HDRI tinted it pink-grey. The 12 x 24 equal-area bin "
        "edges are hairlines, visited bins are filled with exact-colour points spaced evenly in arc length, and each "
        "approach is a black dot. Reads at a glance: every grasp approaches within about 34 deg of straight down, 1.4 % "
        "of bins. add_mesh_simple has no per-vertex colour, hence the point fill."),
}


def build(server: viser.ViserServer, type_id: str) -> None:
    core = by_id(type_id)
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(show_logo=False, show_share_button=False, brand_color=ACCENT_RGB)
    server.gui.add_markdown(f"**{core.title}**\n\n*{core.data()['source']}*")
    SCENES[type_id](server)


def _settled_screenshot(page, timeout_s: float = 120.0) -> bytes:
    """Viser streams the scene after the socket opens: wait until two consecutive frames are pixel-identical."""
    page.wait_for_selector("canvas", timeout=timeout_s * 1000)
    # the "software WebGL" toast is about the headless renderer, not the scene: hide it from the picture
    page.add_style_tag(content=".mantine-Notifications-root { display: none !important; }")
    page.wait_for_timeout(2500)
    deadline, previous = time.monotonic() + timeout_s, None
    while time.monotonic() < deadline:
        shot = page.screenshot(type="png")
        digest = hashlib.sha256(shot).hexdigest()
        if digest == previous:
            return shot
        previous = digest
        page.wait_for_timeout(1200)
    raise TimeoutError("the Viser page never settled")


def capture(type_ids: list[str] | None = None) -> dict:
    """Start each scene on port 8931 (one at a time), screenshot it in isolated headless Chrome at 1400x850, stop it."""
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=CHROME, headless=True,
                                             args=["--enable-webgl", "--enable-unsafe-swiftshader", "--use-angle=swiftshader",
                                                   # WHY: the real arm shares this host and its motion gate refuses on load
                                                   "--renderer-process-limit=1", "--num-raster-threads=1"])
        try:
            for type_id in type_ids or [core.id for core in TYPES]:
                server = viser.ViserServer(host="127.0.0.1", port=PORT, verbose=False)
                try:
                    build(server, type_id)
                    page = browser.new_page(viewport={"width": 1400, "height": 850})
                    problems = []
                    page.on("console", lambda message: problems.append(message.text) if message.type == "error" else None)
                    page.on("pageerror", lambda error: problems.append(str(error)))
                    page.goto(f"http://127.0.0.1:{server.get_port()}/", wait_until="domcontentloaded")
                    image = OUT / f"{type_id}.png"
                    image.write_bytes(_settled_screenshot(page))
                    page.close()
                finally:
                    server.stop()
                results[type_id] = {"image": str(image), "notes": NOTES[type_id]}
                print(f"{type_id}: {image}" + (f"  browser errors: {problems}" if problems else ""), flush=True)
        finally:
            browser.close()
    return results


def serve(type_id: str, port: int = 0) -> None:
    server = viser.ViserServer(host="127.0.0.1", port=port)
    build(server, type_id)
    print(f"http://127.0.0.1:{server.get_port()}  ({type_id}; Ctrl-C stops it)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serve", choices=list(SCENES), help="serve one type live instead of capturing all")
    parser.add_argument("--only", nargs="*", choices=list(SCENES), help="capture only these types")
    arguments = parser.parse_args()
    if arguments.serve:
        serve(arguments.serve)
    else:
        capture(arguments.only)
