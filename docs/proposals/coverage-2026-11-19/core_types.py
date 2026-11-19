"""The eight core TYPES of coverage view, each with the one dataset every tool draws (Zach, 2026-11-19: "lets have like
Type of visualization, and then show the implementation in these different visualization tools": Matplotlib, Rerun,
Viser, Weights & Biases).

Every tool module (`tools_matplotlib.py`, `tools_rerun.py`, `tools_viser.py`, `tools_wandb.py`) draws `TYPES[i].data()`
and nothing else, so a difference between two tools' pictures is the tool, never the data. Plain numpy out; SI inside,
degrees and mm only at display.

    from core_types import TYPES, by_id
    payload = by_id("coverage-heatmap").data()      # dict of numpy arrays + a few floats/strings
"""

from __future__ import annotations

import functools
import glob
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import data
import pendulum as pend
from coverage import Grid, rarefaction, sphere_cell

DEG = 180.0 / np.pi
CACHE = data.CACHE
# WHY one coarse 3-D grid for every 3-D number (20° x 58°/s x 19 rad/s²): a visited % only means something with its
# grid, and a fine 3-D grid has more cells than 423 plays can fill; figures_dataset uses the same one.
GRID3 = {"q_bins": 18, "qd_bins": 10, "qdd_bins": 12}
REPICKS = Path("/archive/datasets/bam_rig/rack_2026-11-10/fivebar_real")


@dataclass(frozen=True)
class CoreType:
    id: str
    title: str            # the type of view, tool-agnostic
    question: str         # what it answers at a glance
    dims: str             # "1-D", "2-D", "3-D", "series"
    data: Callable[[], dict]


def _cached(name: str):
    def wrap(build):
        @functools.cache
        def load() -> dict:
            path = CACHE / f"core_{name}.pkl"
            if path.is_file():
                return pickle.loads(path.read_bytes())
            payload = build()
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            return payload
        return load
    return wrap


@functools.cache
def _can11():
    env = pend.envelope()
    runs = pend.can11_runs()
    ordered = runs["campaign"] + runs["envelope"]
    rows, run_of, tau = pend.samples(ordered)
    return env, ordered, rows, run_of


# --- 1-D: limit bars ---------------------------------------------------------------------------------------------------

@_cached("limit_bars")
def limit_bars() -> dict:
    """CAN 11's q, qd, qdd, qddd samples (all 425 real plays) and each one's limit (None = no limit stated)."""

    env, runs, rows, _ = _can11()
    step = max(1, len(rows) // 60000)                     # 60 k samples is plenty for a histogram in any tool
    return {"names": ["q", "qd", "qdd", "qddd"], "units_display": ["deg", "deg/s", "deg/s²", "deg/s³"],
            "values": [rows[::step, 0] - env["bottom"], rows[::step, 1], rows[::step, 2], rows[::step, 3]],
            "limits": [(env["low"] - env["bottom"], env["high"] - env["bottom"]), (-env["qd"], env["qd"]),
                       (-env["qdd"], env["qdd"]), None],
            "planner_limits": [None, None, (-env["qdd_plan"], env["qdd_plan"]), None],
            "source": f"CAN 11, {len(runs)} real plays (rounds 0-9 + Nov 18 full-turn), every {step}th sample"}


# --- 2-D: coverage heatmap ----------------------------------------------------------------------------------------------

@_cached("coverage_heatmap")
def coverage_heatmap() -> dict:
    """CAN 11's q x qd occupancy (samples AND runs per cell) on 36 x 20 linear cells, with the reachable mask."""

    env, runs, rows, run_of = _can11()
    grid = pend.grid(rows, env, run_of)
    return {"samples": grid.counts, "runs": grid.runs_per_cell, "reachable": grid.reachable,
            "q_edges_deg": (grid.axes[0].edges() - env["bottom"]) * DEG, "qd_edges_deg": grid.axes[1].edges() * DEG,
            "visited_pct": grid.visited_pct(), "sufficient_pct": grid.sufficient_pct(50),
            "source": f"CAN 11, {len(runs)} real plays; 36 x 20 linear cells (10° x 29°/s)"}


# --- series: histogram over runs (TensorBoard / W&B histograms over steps) ---------------------------------------------

@_cached("histogram_over_runs")
def histogram_over_runs() -> dict:
    """qd of every CAN 11 play in recording order: one histogram per step = per run (the TensorBoard/W&B histogram)."""

    env, runs, rows, run_of = _can11()
    per_run = [rows[run_of == index, 1] for index in range(len(runs))]
    return {"qd_per_run": per_run, "rounds": [run["round"] for run in runs], "qd_limit": env["qd"],
            "source": f"CAN 11, {len(runs)} real plays in recording order (step = run)"}


# --- series: coverage growth ---------------------------------------------------------------------------------------------

@_cached("coverage_growth")
def coverage_growth() -> dict:
    """Visited % of reachable (q, qd, qdd) cells after each run, in recording order, plus the random-order band and Chao2."""

    env, runs, rows, run_of = _can11()
    full = pend.axes(env, **GRID3)
    grid = Grid.of(rows[:, :3], full, reachable=pend.reachable(env, full), run_ids=run_of)
    seen = np.zeros(grid.reachable.size, bool)
    reach = grid.reachable.ravel()
    in_order = []
    for cells in grid.run_cells:
        seen[cells] = True
        in_order.append(100.0 * (seen & reach).sum() / reach.sum())
    k, mean, band = rarefaction(grid)
    return {"in_order_pct": np.array(in_order), "random_mean_pct": mean, "random_band_pct": band,
            "chao2_pct": min(100.0, 100.0 * grid.chao2_reachable_estimate() / grid.reachable_cells),
            "rounds": [run["round"] for run in runs], "reachable_cells": grid.reachable_cells,
            "source": f"CAN 11, {len(runs)} real plays; 18 x 10 x 12 (q, qd, qdd) cells"}


# --- 3-D: phase-space occupancy -----------------------------------------------------------------------------------------

@_cached("phase_space_3d")
def phase_space_3d() -> dict:
    """(q, qd, qdd) cell centres with sample counts, and the reachable-but-empty cells: the 3-D grid itself, plus a
    subsample of the raw samples for point-cloud views."""

    env, runs, rows, run_of = _can11()
    full = pend.axes(env, **GRID3)
    reach = pend.reachable(env, full)
    counts, _ = np.histogramdd(rows[:, :3], bins=[axis.edges() for axis in full])
    q, qd, qdd = np.meshgrid(*(axis.centers() for axis in full), indexing="ij")
    centers = np.column_stack([(q.ravel() - env["bottom"]) * DEG, qd.ravel() * DEG, qdd.ravel() * DEG])
    sizes = np.array([(full[0].edges()[1] - full[0].edges()[0]) * DEG, (full[1].edges()[1] - full[1].edges()[0]) * DEG,
                      (full[2].edges()[1] - full[2].edges()[0]) * DEG])
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), min(40000, len(rows)), replace=False)
    samples = np.column_stack([(rows[pick, 0] - env["bottom"]) * DEG, rows[pick, 1] * DEG, rows[pick, 2] * DEG])
    return {"centers_deg": centers, "cell_size_deg": sizes, "counts": counts.ravel(), "reachable": reach.ravel(),
            "samples_deg": samples, "sample_run": run_of[pick],
            "visited_pct": 100.0 * ((counts.ravel() > 0) & reach.ravel()).sum() / reach.sum(),
            "axes": ["q from bottom (deg)", "qd (deg/s)", "qdd (deg/s²)"],
            "source": f"CAN 11, {len(runs)} real plays; 18 x 10 x 12 cells; 40 k of {len(rows) // 1000} k samples as points"}


# --- 3-D: tool paths and grasps from the real full-arm repicks --------------------------------------------------------

def _repick_plans(limit: int = 1500) -> list[Path]:
    plans = sorted(glob.glob(str(REPICKS / "pick_loop_full_arm_repick_real_*" / "plans" / "c*.json")))
    return [Path(path) for path in plans[:limit]]


@_cached("repicks")
def _repicks() -> dict:
    paths, grasps = [], []
    for plan in _repick_plans():
        try:
            meta = json.loads(plan.read_text())
            arrays = np.load(plan.with_suffix(".npz"))
        except Exception:
            continue
        grasp = meta.get("grasp") or {}
        frame = meta.get("plane_frame") or {}
        if "tcp" in arrays.files and frame.get("B2P_rows"):
            # WHY the transform: a plan's tcp is in the arm's plane frame, its grasp in base_link; the plan states
            # p_base = B2P.T @ p_plane + origin, so both land in base_link and the tools can overlay them
            base_to_plane = np.array(frame["B2P_rows"], float)
            paths.append(np.asarray(arrays["tcp"], float) @ base_to_plane + np.array(frame["origin_in_base_m"], float))
        if grasp.get("tcp_base_m") and grasp.get("approach_base") and grasp.get("opening_dir_base"):
            grasps.append({"position": np.array(grasp["tcp_base_m"], float), "approach": np.array(grasp["approach_base"], float),
                           "opening": np.array(grasp["opening_dir_base"], float),
                           "theta_open": float(grasp.get("theta_open_rad", np.nan)),
                           "claw_max": float((meta.get("limits") or {}).get("claw_max_rad", np.nan)),
                           "run": plan.parent.parent.name})
    return {"paths": paths, "grasps": grasps}


@_cached("tool_paths")
def tool_paths() -> dict:
    """Every real repick cycle's planned TCP path in the base frame (m), and its occupancy on 2 cm voxels."""

    repicks = _repicks()
    paths = [path for path in repicks["paths"] if len(path) > 2 and np.all(np.isfinite(path))]
    every = np.concatenate(paths)
    low, high = every.min(axis=0) - 0.02, every.max(axis=0) + 0.02
    edges = [np.arange(low[i], high[i] + 0.02, 0.02) for i in range(3)]
    counts, _ = np.histogramdd(every, bins=edges)
    runs = len({p for p in repicks["grasps"] for p in [p["run"]]})
    return {"paths_m": paths, "voxel_edges_m": edges, "voxel_counts": counts,
            "source": f"Full arm (five-bar + wrist + claw), {len(paths)} planned TCP paths of real repick cycles, base_link (from each plan's plane frame)"}


@_cached("grasp_poses")
def grasp_poses() -> dict:
    """Every real repick's 6-DOF grasp: TCP position (m), rotation (columns: opening x, binormal y, approach z) and the
    claw's commanded opening against its limit."""

    grasps = _repicks()["grasps"]
    positions = np.array([g["position"] for g in grasps])
    rotations = []
    for g in grasps:
        z = g["approach"] / np.linalg.norm(g["approach"])
        x = g["opening"] - np.dot(g["opening"], z) * z
        x = x / np.linalg.norm(x)
        rotations.append(np.column_stack([x, np.cross(z, x), z]))
    opening = np.array([g["theta_open"] for g in grasps])
    claw_max = float(np.nanmax([g["claw_max"] for g in grasps]))
    return {"positions_m": positions, "rotations": np.array(rotations), "opening_rad": opening, "opening_limit_rad": claw_max,
            "runs": [g["run"] for g in grasps],
            "source": f"Full arm real repicks: {len(grasps)} planned grasps (TCP, approach, opening direction, claw angle)"}


@_cached("direction_sphere")
def direction_sphere() -> dict:
    """The approach direction of every real repick grasp on the unit sphere, and its counts on equal-area bins."""

    poses = grasp_poses()
    directions = poses["rotations"][:, :, 2]
    rings, sectors = 12, 24
    cells = sphere_cell(directions, rings, sectors)
    counts = np.bincount(cells, minlength=rings * sectors).reshape(rings, sectors)
    return {"directions": directions, "counts": counts, "rings": rings, "sectors": sectors,
            "visited_pct": 100.0 * (counts > 0).mean(),
            "source": poses["source"] + "; 12 x 24 equal-area bins (rings uniform in z)"}


TYPES = [
    CoreType("limit-bars", "Limit bars: each derivative's distribution against its limit",
             "How much of each limit did the data use?", "1-D", limit_bars),
    CoreType("coverage-heatmap", "2-D coverage heatmap: q × qd cells, white = empty, grey = unreachable",
             "Where in (angle, speed) are the holes?", "2-D", coverage_heatmap),
    CoreType("histogram-over-runs", "Histogram over runs (TensorBoard / W&B histograms, step = run)",
             "How did the speed distribution change run by run?", "series", histogram_over_runs),
    CoreType("coverage-growth", "Coverage growth: visited % as runs accumulate",
             "Is more data still buying coverage?", "series", coverage_growth),
    CoreType("phase-space-3d", "3-D phase-space occupancy: (q, qd, qdd) cells and samples",
             "Which (angle, speed, acceleration) states has the joint held, all at once?", "3-D", phase_space_3d),
    CoreType("tool-paths", "Tool-path heatmap (Strava): every TCP path at low opacity + voxel occupancy",
             "Where in the workspace has the tool been, and where never?", "3-D", tool_paths),
    CoreType("grasp-poses", "Grasp poses as thin grippers, opening shown",
             "Which 6-DOF picks, with which gripper openings, are in the data?", "3-D", grasp_poses),
    CoreType("direction-sphere", "Approach directions on an equal-area sphere",
             "From which directions have we approached?", "3-D", direction_sphere),
]


def by_id(type_id: str) -> CoreType:
    return next(t for t in TYPES if t.id == type_id)


if __name__ == "__main__":
    for core in TYPES:
        payload = core.data()
        print(core.id, {k: (getattr(v, "shape", None) or (len(v) if isinstance(v, list) else v)) for k, v in payload.items()
                        if k != "source"}, "|", payload["source"])
