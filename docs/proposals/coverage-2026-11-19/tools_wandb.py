"""The Weights & Biases column of "same view, four tools": each core type logged the most W&B-native way.

Each type function's source is shown verbatim in the report as "the W&B implementation". It logs only
`by_id(<type>).data()` into the run it is handed, using W&B's own panels (histograms over steps, scalars over steps,
Tables + `wandb.plot.*` charts, `wandb.Object3D` point clouds), never a matplotlib image (that is the Matplotlib column).

One W&B run per type (`log_all`), WHY: two types use the step axis for "recorded run" (histogram-over-runs,
coverage-growth) and the rest log once, so one shared run would mix step meanings; a run per type also opens on a
workspace that holds only that type's keys, and a report section can filter to it by job type.

    env -u VIRTUAL_ENV taskset -c 2-5,16,17 uv run --no-project --python /home/bam/robot_metrics/.venv/bin/python \
        --with wandb==0.31.0 --with wandb-workspaces==0.4.13 python -c "import tools_wandb as t; print(t.log_all())"

Re-check (or re-report) what is already on W&B without logging again: `t.publish(t.logged_in_project())`.
Fix one panel in place (resume the run, log only that key): `t.relog("grasp-poses", ["grasps/grippers"])`.
Logged once on 2026-11-19 07:05 (8 runs, ~8 MB of files; heatmap spec zach-yamaoka-independent/coverage-heatmap);
07:3x: heatmap spec made linear, and grasps/grippers, tool_paths/* (base_link data, cm, exact boxes), phase_space/cells
re-logged into their own runs at step 1.
Registered as a W&B tracker before its first run: bam_ws tools/wandb_trackers/trackers/robot_dataset_coverage_gallery.py.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import time
from urllib.parse import quote

import matplotlib
import numpy as np
import wandb

from core_types import CACHE, TYPES, by_id

ENTITY = "zach-yamaoka-independent"          # the entity every BAM project logs to (bam_trackio.wandb_view.ENTITY)
PROJECT = "Robot dataset coverage gallery"
TAG = "coverage-gallery"
WANDB_DIR = CACHE / "wandb"                  # W&B's local run files: a cache next to the ledger, never the repo
DEG = 180.0 / np.pi
MAX_POINTS = 90_000                          # per Object3D: the Free plan is near its 5 GB cap, keep the project small

# WHY a custom chart: W&B ships no heatmap / 2-D histogram preset (wandb 0.31 has 17 presets; wandb/heatmap/v0 and
# friends do not exist on the server, probed 2026-11-19). This Vega-Lite spec is saved PRIVATE under the entity, as the
# Panel Gallery's gate-heatmap is, and every heatmap here is a plain wandb.plot_table against it.
HEATMAP_CHART = f"{ENTITY}/coverage-heatmap"
HEATMAP_SPEC = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "title": "${string:title}",
    "data": {"name": "wandb"},
    "width": "container", "height": 320,
    "encoding": {
        "x": {"field": "${field:x_low}", "type": "quantitative", "bin": {"binned": True}, "title": "${string:x_title}"},
        "x2": {"field": "${field:x_high}"},
        "y": {"field": "${field:y_low}", "type": "quantitative", "bin": {"binned": True}, "title": "${string:y_title}"},
        "y2": {"field": "${field:y_high}"},
        "tooltip": [{"field": "${field:count}", "title": "${string:color_title}"},
                    {"field": "${field:x_low}", "title": "x from"}, {"field": "${field:y_low}", "title": "y from"}],
    },
    "layer": [   # white = reachable but empty, grey = unreachable, blue = visited (linear: 1 = the lightest tint)
        {"mark": {"type": "rect", "stroke": "#e5e5ea", "strokeWidth": 0.5},
         "encoding": {"color": {"condition": {"test": "datum['${field:reachable}'] > 0", "value": "white"},
                                "value": "#d2d2d7"}}},
        {"transform": [{"filter": "datum['${field:count}'] > 0"}],
         "mark": {"type": "rect", "stroke": "white", "strokeWidth": 0.5},
         "encoding": {"color": {"field": "${field:count}", "type": "quantitative", "title": "${string:color_title}",
                                # WHY linear (Zach, 2026-11-19: "better to just have a linear scale"); zero
                                # stays white because the filter leaves empty cells to the layer below
                                "scale": {"type": "linear", "zero": False, "range": ["#d6e4f5", "#0d2f5c"]}}}},
    ],
}

GREY = (150, 150, 150)


def rgb(fraction, colormap: str = "plasma") -> np.ndarray:
    """0..1 -> (N, 3) RGB in 0..255, the per-point colour a W&B point cloud takes ([x, y, z, r, g, b] rows)."""
    return (np.asarray(matplotlib.colormaps[colormap](np.clip(fraction, 0, 1)))[..., :3] * 255).round()


def box(centre, size, rotation=np.eye(3), colour=(255, 0, 0)) -> dict:
    """A W&B Box3D: 8 corners in wandb.box3d's order and an RGB colour in 0..255, like a point's. WHY 0..255:
    wandb.box3d's docstring and the last public weave-js say 0..1, but W&B's live viewer (Forge) draws 0..1 box
    colours near-black and 0..255 ones correctly (seen in Zach's browser, 2026-11-19). WHY not wandb.box3d, and why cm
    (weave-js SdkPointCloudToBabylon.ts + render_babylon.ts, read 2026-11-19):
    - wandb 0.31's box3d takes abs() of each quaternion component, which mirrors most rotations;
    - the viewer finds a box's edges by testing corner triples for |dot| < 0.003 (absolute units), so a box whose edges
      are under ~5 units passes the wrong triple (diagonals drawn as edges) or throws "Provided corners do not form a
      box"; corners are therefore never rounded and small scenes are logged in cm;
    - a box with score None prints "(null)" beside it, so label and score are left out."""
    unit = np.array(list(itertools.product((-1, 1), repeat=3)), float)
    corners = np.asarray(centre, float) + (0.5 * np.asarray(size, float) * unit) @ np.asarray(rotation, float).T
    return {"corners": corners.tolist(), "color": [int(c) for c in colour]}


def cloud(xyz, colours) -> np.ndarray:
    return np.column_stack([np.round(xyz, 4), colours])


# --- the eight types ---------------------------------------------------------------------------------------------------

def limit_bars(run) -> None:
    payload = by_id("limit-bars").data()
    bars = wandb.Table(columns=["limit", "used_pct"])
    for name, unit, values, limit, planner in zip(payload["names"], payload["units_display"], payload["values"],
                                                 payload["limits"], payload["planner_limits"]):
        values_display = values * DEG
        if limit:
            # the histogram's x-range IS the limit: samples past it are counted in the end bins, never dropped
            low, high = limit[0] * DEG, limit[1] * DEG
            used_pct = 100 * max(values.min() / limit[0], values.max() / limit[1])
            bars.add_data(f"{name} (limit {high:.0f} {unit})", used_pct)
            run.summary[f"limits/{name}_used_pct"] = used_pct
        else:
            low, high = 1.15 * np.percentile(values_display, [0.1, 99.9])
        if planner:
            planner_pct = 100 * max(values.min() / planner[0], values.max() / planner[1])
            bars.add_data(f"{name} (planner {planner[1] * DEG:.0f} {unit})", planner_pct)
            run.summary[f"limits/{name}_planner_used_pct"] = planner_pct
        edges = np.linspace(low, high, 121)
        histogram = np.histogram(np.clip(values_display, low, high), edges)
        run.log({f"limits/{name}_histogram": wandb.Histogram(np_histogram=histogram)}, commit=False)
    run.log({"limits/used_pct": wandb.plot.bar(bars, "limit", "used_pct",
                                               title="Share of each limit the data used (%; 100 = at the limit)")})


def coverage_heatmap(run) -> None:
    payload = by_id("coverage-heatmap").data()
    q_edges, qd_edges = payload["q_edges_deg"], payload["qd_edges_deg"]
    cells = wandb.Table(columns=["q_low_deg", "q_high_deg", "qd_low_deg", "qd_high_deg", "runs", "samples", "reachable"])
    for i, j in itertools.product(range(len(q_edges) - 1), range(len(qd_edges) - 1)):
        cells.add_data(float(q_edges[i]), float(q_edges[i + 1]), float(qd_edges[j]), float(qd_edges[j + 1]),
                       int(payload["runs"][i, j]), int(payload["samples"][i, j]), int(payload["reachable"][i, j]))
    fields = {"x_low": "q_low_deg", "x_high": "q_high_deg", "y_low": "qd_low_deg", "y_high": "qd_high_deg",
              "reachable": "reachable"}
    axes = {"x_title": "q from bottom (deg)", "y_title": "qd (deg/s)"}
    run.log({
        "coverage/runs_per_cell": wandb.plot_table(HEATMAP_CHART, cells, {**fields, "count": "runs"}, {
            **axes, "color_title": "runs through the cell",
            "title": f"q x qd: {payload['visited_pct']:.0f} % of reachable cells visited (white = empty, grey = unreachable)"}),
        "coverage/samples_per_cell": wandb.plot_table(HEATMAP_CHART, cells, {**fields, "count": "samples"}, {
            **axes, "color_title": "samples in the cell", "title": "q x qd: samples per cell"}),
        "coverage/visited_pct": payload["visited_pct"],
        "coverage/sufficient_pct": payload["sufficient_pct"],
    })


def histogram_over_runs(run) -> None:
    payload = by_id("histogram-over-runs").data()
    limit_deg = payload["qd_limit"] * DEG
    edges = np.linspace(-limit_deg, limit_deg, 81)          # the panel's x-range is the speed limit
    runs = wandb.Table(columns=["step", "round", "samples", "max_abs_qd_deg_s"])
    for step, (qd, round_name) in enumerate(zip(payload["qd_per_run"], payload["rounds"])):
        qd_deg = qd * DEG
        # step = recorded run, so W&B's histogram panel stacks one distribution per run (TensorBoard's histograms)
        run.log({"qd/histogram": wandb.Histogram(np_histogram=np.histogram(np.clip(qd_deg, -limit_deg, limit_deg), edges)),
                 "qd/max_abs_deg_s": float(np.abs(qd_deg).max()),
                 "qd/p99_abs_deg_s": float(np.percentile(np.abs(qd_deg), 99)),
                 "qd/limit_deg_s": limit_deg}, step=step)
        runs.add_data(step, round_name, len(qd), float(np.abs(qd_deg).max()))
    run.log({"qd/runs": runs}, step=len(payload["qd_per_run"]) - 1)


def coverage_growth(run) -> None:
    payload = by_id("coverage-growth").data()
    in_order, mean = payload["in_order_pct"], payload["random_mean_pct"]
    p10, p90 = payload["random_band_pct"]
    chao2 = float(payload["chao2_pct"])
    run.define_metric("runs_recorded")
    run.define_metric("coverage/*", step_metric="runs_recorded")
    for step in range(len(in_order)):
        run.log({"runs_recorded": step + 1,
                 "coverage/in_order_pct": in_order[step],
                 "coverage/random_mean_pct": mean[step],
                 "coverage/random_p10_pct": p10[step],
                 "coverage/random_p90_pct": p90[step],
                 "coverage/chao2_ceiling_pct": chao2}, step=step)
    runs_recorded = list(range(1, len(in_order) + 1))
    run.log({"coverage_chart/growth": wandb.plot.line_series(
        xs=runs_recorded, ys=[in_order.tolist(), mean.tolist(), p10.tolist(), p90.tolist(), [chao2] * len(in_order)],
        keys=["as recorded", "random order, mean", "random order, p10", "random order, p90", "Chao2 ceiling"],
        title=f"% of {payload['reachable_cells']} reachable (q, qd, qdd) cells visited as runs accumulate",
        xname="runs recorded")}, step=len(in_order) - 1)
    run.summary["coverage/reachable_cells"] = int(payload["reachable_cells"])


def phase_space_3d(run) -> None:
    payload = by_id("phase-space-3d").data()
    size = payload["cell_size_deg"]
    origin = payload["centers_deg"].min(axis=0) - size / 2

    def in_cells(points_deg):
        # one unit = one cell on every axis, else qdd's ±6000 deg/s² squashes q and qd into a flat sheet
        return (points_deg - origin) / size

    counts, reachable = payload["counts"], payload["reachable"]
    visited, empty = (counts > 0) & reachable, (counts == 0) & reachable
    shade = np.log10(counts[visited]) / np.log10(counts.max())
    holes = [box(centre, (0.8, 0.8, 0.8)) for centre in in_cells(payload["centers_deg"][empty])]
    samples_order = payload["sample_run"] / payload["sample_run"].max()
    run.log({
        # visited cells as points coloured by sample count, every reachable-but-empty cell as a red wire cube
        "phase_space/cells": wandb.Object3D.from_point_cloud(
            points=cloud(in_cells(payload["centers_deg"][visited]), rgb(shade)), boxes=holes),
        # the raw samples, coloured by recording order (dark = first run, yellow = last)
        "phase_space/samples": wandb.Object3D(cloud(in_cells(payload["samples_deg"]), rgb(samples_order, "viridis"))),
        "phase_space/visited_pct": float(payload["visited_pct"]),
        "phase_space/empty_reachable_cells": int(empty.sum()),
    })
    run.config.update({"phase_space_axes": payload["axes"], "phase_space_cell_size_deg": size.round(3).tolist()})


def tool_paths(run) -> None:
    payload = by_id("tool-paths").data()
    counts = payload["voxel_counts"]
    edges = [axis_edges * 100 for axis_edges in payload["voxel_edges_m"]]       # cm: see box()
    every = np.concatenate(payload["paths_m"]) * 100
    points = every[::int(np.ceil(len(every) / MAX_POINTS))]
    voxel_of = tuple(np.clip(np.searchsorted(edges[axis], points[:, axis]) - 1, 0, counts.shape[axis] - 1)
                     for axis in range(3))
    heat = np.log1p(counts[voxel_of]) / np.log1p(counts.max())
    occupied = np.argwhere(counts > 0)
    voxel_size = [edges[axis][1] - edges[axis][0] for axis in range(3)]
    centres = np.array([[edges[axis][index[axis]] + voxel_size[axis] / 2 for axis in range(3)] for index in occupied])
    occupancy = np.log1p(counts[tuple(occupied.T)]) / np.log1p(counts.max())
    run.log({
        # Strava in 3-D: W&B point clouds have no opacity, so busy places glow by their voxel's count instead
        "tool_paths/strava": wandb.Object3D(cloud(points, rgb(heat))),
        # every occupied 2 cm voxel as a wire cube, dark = visited once ... yellow = the busiest
        "tool_paths/voxels": wandb.Object3D.from_point_cloud(
            points=cloud(centres, rgb(occupancy)),
            boxes=[box(centre, voxel_size, colour=rgb(share)) for centre, share in zip(centres, occupancy)]),
        "tool_paths/paths": len(payload["paths_m"]),
        "tool_paths/occupied_voxels": len(occupied),
    })


def grasp_poses(run) -> None:
    payload = by_id("grasp-poses").data()
    limit = payload["opening_limit_rad"]
    positions_cm = payload["positions_m"] * 100                # cm: see box()
    finger_cm, finger_width_cm, jaw_cm = 4.0, 1.2, 0.3
    jaws = []
    for position, rotation, opening in zip(positions_cm, payload["rotations"], payload["opening_rad"]):
        half_width = 3.5 * opening / limit                    # jaw gap drawn in proportion to the claw angle
        jaw, approach = rotation[:, 0], rotation[:, 2]        # box axes = (opening, binormal, approach)
        colour = rgb(opening / limit, "viridis")              # the TCP points' colour: yellow = at the claw limit
        for side in (1, -1):                                  # two fingers reaching back from the TCP
            centre = position + side * half_width * jaw - finger_cm / 2 * approach
            jaws.append(box(centre, (jaw_cm, finger_width_cm, finger_cm), rotation, colour))
        palm = position - finger_cm * approach
        jaws.append(box(palm, (2 * half_width + jaw_cm, finger_width_cm, jaw_cm), rotation, colour))
    openings = wandb.Table(columns=["opening_deg", "run"],
                           data=[[float(o * DEG), r] for o, r in zip(payload["opening_rad"], payload["runs"])])
    at_limit_pct = 100 * float(np.mean(payload["opening_rad"] >= limit - np.radians(0.5)))
    run.log({
        "grasps/grippers": wandb.Object3D.from_point_cloud(
            points=cloud(positions_cm, rgb(payload["opening_rad"] / limit, "viridis")), boxes=jaws),
        "grasps/opening_histogram": wandb.plot.histogram(
            openings, "opening_deg", title=f"Claw opening at each grasp (deg); the limit is {limit * DEG:.1f}°"),
        "grasps/opening_limit_deg": limit * DEG,
        "grasps/at_claw_limit_pct": at_limit_pct,
        "grasps/count": len(payload["positions_m"]),
    })


def direction_sphere(run) -> None:
    payload = by_id("direction-sphere").data()
    rings, sectors, counts = payload["rings"], payload["sectors"], payload["counts"]
    z_edges, azimuth_edges = np.linspace(-1, 1, rings + 1), np.linspace(-180, 180, sectors + 1)
    # a dense sphere surface, each point painted with its equal-area bin's count (rings uniform in z)
    k = np.arange(20_000) + 0.5
    z, azimuth = 1 - 2 * k / len(k), (np.pi * (1 + 5 ** 0.5) * k) % (2 * np.pi) - np.pi
    surface = np.column_stack([np.sqrt(1 - z ** 2) * np.cos(azimuth), np.sqrt(1 - z ** 2) * np.sin(azimuth), z])
    ring = np.clip(((z + 1) / 2 * rings).astype(int), 0, rings - 1)
    sector = np.clip(((azimuth + np.pi) / (2 * np.pi) * sectors).astype(int), 0, sectors - 1)
    bin_count = counts[ring, sector]
    colours = np.where(bin_count[:, None] > 0, rgb(np.log1p(bin_count) / np.log1p(counts.max())), GREY)
    approaches = [{"start": [0, 0, 0], "end": np.round(d, 4).tolist()} for d in payload["directions"]]
    cells = wandb.Table(columns=["azimuth_low", "azimuth_high", "z_low", "z_high", "grasps", "reachable"])
    for r, s in itertools.product(range(rings), range(sectors)):
        cells.add_data(float(azimuth_edges[s]), float(azimuth_edges[s + 1]), float(z_edges[r]), float(z_edges[r + 1]),
                       int(counts[r, s]), 1)
    run.log({
        "directions/sphere": wandb.Object3D.from_point_cloud(points=cloud(surface, colours), boxes=[], vectors=approaches),
        # the same bins unrolled: x = azimuth, y = z is Lambert's cylindrical equal-area map, so every cell is one area
        "directions/equal_area_map": wandb.plot_table(HEATMAP_CHART, cells, {
            "x_low": "azimuth_low", "x_high": "azimuth_high", "y_low": "z_low", "y_high": "z_high", "count": "grasps",
            "reachable": "reachable"}, {
            "x_title": "approach azimuth (deg)", "y_title": "approach z (-1 = straight down)",
            "color_title": "grasps in the cell",
            "title": f"Approach directions on equal-area cells: {payload['visited_pct']:.1f} % used"}),
        "directions/visited_pct": float(payload["visited_pct"]),
    })


IMPLEMENTATIONS = {"limit-bars": limit_bars, "coverage-heatmap": coverage_heatmap, "histogram-over-runs": histogram_over_runs,
                   "coverage-growth": coverage_growth, "phase-space-3d": phase_space_3d, "tool-paths": tool_paths,
                   "grasp-poses": grasp_poses, "direction-sphere": direction_sphere}

# what to open in each run, and how native W&B's version is (honest)
PANEL_HINTS = {
    "limit-bars": "custom chart limits/used_pct (bars, % of each limit) + histograms limits/q_histogram, qd, qdd, qddd",
    "coverage-heatmap": "custom chart coverage/runs_per_cell (and coverage/samples_per_cell); scalar coverage/visited_pct",
    "histogram-over-runs": "histogram panel qd/histogram (one distribution per step = recorded run); "
                           "line qd/max_abs_deg_s vs qd/limit_deg_s",
    "coverage-growth": "line charts coverage/* over runs_recorded; one-panel version: custom chart coverage_chart/growth",
    "phase-space-3d": "3-D panel phase_space/cells (red wire cubes = reachable but empty); phase_space/samples",
    "tool-paths": "3-D panel tool_paths/strava; tool_paths/voxels",
    "grasp-poses": "3-D panel grasps/grippers; custom chart grasps/opening_histogram",
    "direction-sphere": "3-D panel directions/sphere; custom chart directions/equal_area_map",
}
NOTES = {
    "limit-bars": "Native bar chart + native histograms. No panel draws a limit line on a histogram, so each histogram's "
                  "x-range is set to exactly the limit (samples past it land in the end bins) and the bar chart gives "
                  "the share of each limit used; qddd has no limit, so its range is 1.15 x its p0.1/p99.9.",
    "coverage-heatmap": "W&B has no heatmap preset (probed: none of wandb/heatmap/v0 etc. exist; 17 SDK presets, none a "
                        "heatmap). This is a custom Vega-Lite chart saved privately under the entity "
                        f"({HEATMAP_CHART}) and logged with wandb.plot_table: a native custom-chart panel, our spec.",
    "histogram-over-runs": "Fully native: wandb.Histogram once per step. W&B draws steps as a heat strip (step on one "
                           "axis, qd on the other), not TensorBoard's offset ridgeline. Range = ±the qd limit.",
    "coverage-growth": "Fully native scalars over a custom x (runs_recorded). W&B line plots cannot shade a band, so the "
                       "random-order 10-90 % band is two lines (p10, p90); Chao2 is a constant line.",
    "phase-space-3d": "Native Object3D point cloud. No axis labels or ticks in W&B's 3-D viewer, and axes are rescaled "
                      "to one unit per cell (else qdd flattens the cloud); sizes in the run config.",
    "tool-paths": "Native Object3D. Point clouds have no opacity, so the Strava look is a density colour (plasma) on a "
                  "<= 90 k-point subsample of every planned TCP path; voxels are wire boxes. Centimetres, base frame.",
    "grasp-poses": "Native Object3D boxes as thin two-finger grippers (2 fingers + palm per grasp, coloured by claw "
                   "opening / limit). Corners built by hand: wandb 0.31's box3d mirrors most rotations, and W&B's "
                   "viewer needs boxes of several units, so the scene is in cm. Jaw gap drawn in proportion to the "
                   "claw angle, not the claw's kinematics.",
    "direction-sphere": "Native Object3D: a 20 k-point sphere painted by bin count plus one vector per grasp. The flat "
                        "map is the same custom Vega-Lite heatmap (Lambert equal-area: x = azimuth, y = z).",
}


# --- logging, verification, report -------------------------------------------------------------------------------------

def publish_heatmap_chart(api) -> str:
    """Create or update HEATMAP_SPEC as a PRIVATE custom chart under the entity (the Panel Gallery's vega.publish recipe:
    wandb 0.31's Api.create_custom_chart cannot update, so a re-run uses the two GraphQL mutations)."""
    from wandb_workspaces._graphql import execute_graphql

    name = HEATMAP_CHART.split("/", 1)[1]
    chart_input = {"entity": ENTITY, "name": name, "displayName": "Coverage heatmap (binned x/x2, y/y2; log colour)",
                   "type": "vega2", "access": "PRIVATE", "spec": json.dumps(HEATMAP_SPEC, indent=1)}
    existing = execute_graphql(api, "query ($id: ID!) { customChart(id: $id) { id } }", {"id": HEATMAP_CHART})["customChart"]
    verb = "updateCustomChart" if existing else "createCustomChart"
    mutation = (f"mutation ($input: {verb[0].upper() + verb[1:]}Input!) "
                f"{{ {verb}(input: $input) {{ chart {{ id access }} }} }}")
    chart = execute_graphql(api, mutation, {"input": chart_input})[verb]["chart"]
    if chart["access"] != "PRIVATE":
        raise SystemExit(f"FAIL: {chart['id']} saved as {chart['access']}")
    return chart["id"]


def run_id(*parts: str) -> str:
    """A stable W&B run id, the house rule (bam_trackio.wandb_view.wandb_id_text): 16 hex characters of a sha1."""
    return hashlib.sha1("|".join((TAG, *parts)).encode()).hexdigest()[:16]


def browser_url(url: str) -> str:
    """W&B's app moved under forge.coreweave.com/wandb/; hand out the wandb.ai form, which redirects there
    (sim2real.tracking.wandb_log.browser_url, 2026-11-11)."""
    url = re.sub(r"^https://(?:forge\.coreweave\.com/(?:wandb/)?|wandb\.ai/)", "https://wandb.ai/", url)
    return quote(url, safe=":/?=%#")   # WHY: the project name has spaces and the report slug "·" and "&"


EXPECTED = {   # key -> how many history rows it must have (None = summary only)
    "limit-bars": {"limits/q_histogram": 1, "limits/qddd_histogram": 1, "limits/used_pct_table": 1, "limits/qd_used_pct": None},
    "coverage-heatmap": {"coverage/runs_per_cell_table": 1, "coverage/samples_per_cell_table": 1, "coverage/visited_pct": 1},
    "histogram-over-runs": {"qd/histogram": "runs", "qd/max_abs_deg_s": "runs", "qd/runs": None},
    "coverage-growth": {"coverage/in_order_pct": "runs", "coverage/random_p90_pct": "runs", "coverage_chart/growth_table": None},
    "phase-space-3d": {"phase_space/cells": 1, "phase_space/samples": 1, "phase_space/visited_pct": 1},
    "tool-paths": {"tool_paths/strava": 1, "tool_paths/voxels": 1},
    "grasp-poses": {"grasps/grippers": 1, "grasps/opening_histogram_table": 1, "grasps/at_claw_limit_pct": 1},
    "direction-sphere": {"directions/sphere": 1, "directions/equal_area_map_table": 1},
}


def verify(api, logged: dict, patience_s: float = 300.0) -> dict:
    """Read every run back through wandb.Api(): each expected key exists (with one history row per recorded run where
    the step is a run), and the project's stored bytes. Raises on a missing key.
    WHY poll: Forge ingests history and summary asynchronously; right after finish() a run reads back empty (2026-11-19:
    empty for ~30 s, then complete)."""
    recorded_runs = len(by_id("histogram-over-runs").data()["qd_per_run"])
    sizes = {}
    for type_id, entry in logged.items():
        deadline = time.time() + patience_s
        while True:
            remote = api.run(f"{ENTITY}/{PROJECT}/{entry['run_id']}")
            missing = [key for key in EXPECTED[type_id] if key not in remote.summary]
            if not missing or time.time() > deadline:
                break
            time.sleep(10)
        if missing:
            raise SystemExit(f"FAIL {type_id}: {missing} not in the run's summary after {patience_s:.0f} s")
        for key, rows in EXPECTED[type_id].items():
            if rows is not None:
                found = sum(1 for _ in remote.scan_history(keys=[key], page_size=1000))
                expected = recorded_runs if rows == "runs" else rows
                if found != expected:
                    raise SystemExit(f"FAIL {type_id}: {key} has {found} history rows, expected {expected}")
        sizes[type_id] = sum(file.size for file in remote.files())
    return {"bytes_per_run": sizes, "project_mb": round(sum(sizes.values()) / 1e6, 2)}


def log_runs() -> dict:
    """Log one run per type (see the module docstring). Returns {type_id: {run_id, run_url, panel_hint, notes}}."""
    publish_heatmap_chart(wandb.Api(timeout=60))
    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    logged = {}
    for core in TYPES:
        run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id(core.id, stamp), name=re.split(r"[:(]", core.title)[0].strip(),
                         group="coverage gallery", job_type=core.id, tags=[TAG, core.dims], notes=core.question,
                         config={"type_id": core.id, "title": core.title, "question": core.question, "dims": core.dims,
                                 "source": core.data()["source"]},
                         dir=str(WANDB_DIR), resume="never", reinit="create_new")
        IMPLEMENTATIONS[core.id](run)
        logged[core.id] = {"run_id": run.id, "run_url": browser_url(run.url), "panel_hint": PANEL_HINTS[core.id],
                           "notes": NOTES[core.id]}
        run.finish()
    return logged


def logged_in_project() -> dict:
    """The newest gallery run per type already in the project, in log_runs()'s shape (to verify or report without
    logging again)."""
    logged = {}
    for remote in wandb.Api(timeout=60).runs(f"{ENTITY}/{PROJECT}", order="-created_at", per_page=200):
        if TAG in remote.tags and remote.job_type in IMPLEMENTATIONS and remote.job_type not in logged:
            logged[remote.job_type] = {"run_id": remote.id, "run_url": browser_url(remote.url),
                                       "panel_hint": PANEL_HINTS[remote.job_type], "notes": NOTES[remote.job_type]}
    return {type_id: logged[type_id] for type_id in IMPLEMENTATIONS if type_id in logged}


def publish(logged: dict) -> dict:
    """Verify the runs through the API, then replace this gallery's own earlier runs (only after the new ones verify)
    and save the report. Returns {type_id: {run_url, panel_hint, notes, report_url}, "_checks": {...}}."""
    api = wandb.Api(timeout=60)
    checks = verify(api, logged)
    keep = {entry["run_id"] for entry in logged.values()}
    replaced = [old.id for old in api.runs(f"{ENTITY}/{PROJECT}", per_page=200) if TAG in old.tags and old.id not in keep]
    for old_id in replaced:
        api.run(f"{ENTITY}/{PROJECT}/{old_id}").delete()
    try:
        report_url = save_report(logged)
    except Exception as error:   # the runs are the deliverable; a report the API refuses never undoes them
        report_url = f"(report not saved: {error})"
    for entry in logged.values():
        entry["report_url"] = report_url
    return {**logged, "_checks": {**checks, "replaced_runs": replaced, "heatmap_chart": HEATMAP_CHART}}


def log_all() -> dict:
    """Log every type (one run each), read them back, replace older gallery runs, save the report."""
    return publish(log_runs())


class _OnlyKeys:
    """Stands in for a run so a type function re-logs only `keys` into its existing run (one fixed panel)."""

    def __init__(self, run, keys):
        self._run, self._keys = run, set(keys)

    def log(self, data, **_):
        kept = {key: value for key, value in data.items() if key in self._keys}
        if kept:
            self._run.log(kept)      # a resumed run continues at its next step; the panel shows the newest
        return None

    def __getattr__(self, name):
        return getattr(self._run, name)


def relog(type_id: str, keys: list[str]) -> str:
    """Re-log `keys` of one type into its existing run (resume="must"), from the current code and data: fixes a panel
    without a new run. The key's old version stays one step back on the panel's step slider."""
    existing = logged_in_project()[type_id]["run_id"]
    run = wandb.init(entity=ENTITY, project=PROJECT, id=existing, resume="must", dir=str(WANDB_DIR), reinit="create_new")
    IMPLEMENTATIONS[type_id](_OnlyKeys(run, keys))
    run.finish()
    return existing


def _report_panels(wr, type_id: str) -> list:
    """The panels each type's report section shows (its run's keys)."""
    L = wr.Layout

    def chart(key, spec, fields, title, x=0, w=24):
        panel = wr.CustomChart.from_table(f"{key}_table", chart_fields=fields, chart_strings={"title": title})
        panel.chart_name = spec
        panel.layout = L(x=x, y=0, w=w, h=10)
        return panel

    heat_fields = {"x_low": "q_low_deg", "x_high": "q_high_deg", "y_low": "qd_low_deg", "y_high": "qd_high_deg",
                   "reachable": "reachable"}
    heat_axes = {"x_title": "q from bottom (deg)", "y_title": "qd (deg/s)"}
    return {
        "limit-bars": [chart("limits/used_pct", "wandb/bar/v0", {"label": "limit", "value": "used_pct"},
                             "Share of each limit the data used (%)", w=8)]
                      + [wr.LinePlot(title=f"{name} (x-range = the limit)", y=[f"limits/{name}_histogram"],
                                     layout=L(x=8 + 4 * index, y=0, w=4, h=10))
                         for index, name in enumerate(["q", "qd", "qdd", "qddd"])],
        "coverage-heatmap": [
            chart("coverage/runs_per_cell", HEATMAP_CHART, {**heat_fields, "count": "runs"}, "Runs through each q x qd cell", w=12),
            chart("coverage/samples_per_cell", HEATMAP_CHART, {**heat_fields, "count": "samples"}, "Samples per cell", x=12, w=12)],
        "histogram-over-runs": [
            wr.LinePlot(title="qd per recorded run (step = run)", y=["qd/histogram"], layout=L(x=0, y=0, w=14, h=12)),
            wr.LinePlot(title="max |qd| per run vs the limit (deg/s)", y=["qd/max_abs_deg_s", "qd/p99_abs_deg_s", "qd/limit_deg_s"],
                        layout=L(x=14, y=0, w=10, h=12))],
        "coverage-growth": [
            wr.LinePlot(title="% of reachable (q, qd, qdd) cells visited", x="runs_recorded",
                        y=["coverage/in_order_pct", "coverage/random_mean_pct", "coverage/random_p10_pct",
                           "coverage/random_p90_pct", "coverage/chao2_ceiling_pct"], layout=L(x=0, y=0, w=24, h=10))],
        "phase-space-3d": [wr.MediaBrowser(title="cells: points = visited, red cubes = reachable but empty",
                                           media_keys=["phase_space/cells"], layout=L(x=0, y=0, w=12, h=14)),
                           wr.MediaBrowser(title="samples, coloured by recording order", media_keys=["phase_space/samples"],
                                           layout=L(x=12, y=0, w=12, h=14))],
        "tool-paths": [wr.MediaBrowser(title="every planned TCP path, coloured by voxel density", media_keys=["tool_paths/strava"],
                                       layout=L(x=0, y=0, w=12, h=14)),
                       wr.MediaBrowser(title="occupied 2 cm voxels", media_keys=["tool_paths/voxels"], layout=L(x=12, y=0, w=12, h=14))],
        "grasp-poses": [wr.MediaBrowser(title="every real repick grasp as a thin gripper", media_keys=["grasps/grippers"],
                                        layout=L(x=0, y=0, w=14, h=14)),
                        chart("grasps/opening_histogram", "wandb/histogram/v0", {"value": "opening_deg"},
                              "Claw opening at each grasp (deg)", x=14, w=10)],
        "direction-sphere": [wr.MediaBrowser(title="approach directions on the bin-painted sphere", media_keys=["directions/sphere"],
                                             layout=L(x=0, y=0, w=12, h=14)),
                             chart("directions/equal_area_map", HEATMAP_CHART,
                                   {"x_low": "azimuth_low", "x_high": "azimuth_high", "y_low": "z_low", "y_high": "z_high",
                                    "count": "grasps", "reachable": "reachable"}, "Equal-area map", x=12, w=12)],
    }[type_id]


def save_report(logged: dict) -> str:
    """One W&B report, a section per type: its question, its source, how native W&B's version is, and its panels over
    a runset filtered to that type's run. Best effort: the runs' own workspaces are the fallback surface."""
    try:
        import wandb_workspaces.reports.v2 as wr
        from wandb_workspaces import expr
    except ImportError:
        return ""
    blocks = [wr.MarkdownBlock(text="The W&B column of the robot dataset coverage gallery: the same eight coverage views "
                                    "drawn in Matplotlib, Rerun, Viser and W&B from one dataset (robot_metrics "
                                    "`docs/proposals/coverage-2026-11-19/core_types.py`). One run per view type."),
              wr.TableOfContents()]
    for core in TYPES:
        blocks += [wr.H2(text=core.title),
                   wr.MarkdownBlock(text=f"**{core.question}**  \n{core.data()['source']}  \n*How native:* {NOTES[core.id]}"),
                   wr.PanelGrid(runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name=core.id,
                                                   filters=[expr.Config("type_id") == core.id])],
                                panels=_report_panels(wr, core.id))]
    report = wr.Report(entity=ENTITY, project=PROJECT, title="Robot dataset coverage gallery · W&B column",
                       description="Eight coverage-view types, logged the W&B-native way; one run each", blocks=blocks)
    report.width = "fluid"
    try:
        report.save()
    except Exception as error:   # the runs are the deliverable; a report the API refuses never undoes them
        return f"(report not saved: {error})"
    return browser_url(report.url)
