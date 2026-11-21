"""The Weights & Biases column of "same view, four tools": each core type's summary statistics, as W&B scalars.

Summary statistics only, since 2026-11-21. WHY (Zach, 2026-11-21: "to help limit W and B storage I think we need to just
be using it for summary statistics; for the media-rich things we should be using rerun!"): the entity was at 17 GB
against the Free plan's 5 GB cap. Until then this column logged each type the most W&B-native way: histograms over steps,
Tables + `wandb.plot.*` / `wandb.plot_table` custom charts (a private Vega-Lite heatmap spec) and `wandb.Object3D` point
clouds, ~8 MB of files for 8 runs. Now each type function logs only numbers W&B can chart from a run's history and
summary (counts, coverage shares, ranges, percentiles, the share of each limit used) and puts the path of the type's
.rrd in the gallery's Rerun column (`tools_rerun.py`, written by `tools_rerun.capture()`) in the run config: the
point clouds, heatmaps and histograms live there. The gallery page built on 2026-11-19 still shows the W&B screenshots
and notes of the media version (`reports/media/coverage-gallery-2026-11-19/tools/wandb/`); this file is what runs now.

Each type function's source is shown verbatim in the report as "the W&B implementation". It logs only
`by_id(<type>).data()` into the run it is handed, never a matplotlib image (that is the Matplotlib column).

One W&B run per type (`log_all`), WHY: two types use the step axis for "recorded run" (percentiles-over-runs,
coverage-growth) and the rest log once, so one shared run would mix step meanings; a run per type also opens on a
workspace that holds only that type's keys, and a report section can filter to it by job type.

    env -u VIRTUAL_ENV taskset -c 2-5,16,17 uv run --no-project --python /home/bam/robot_metrics/.venv/bin/python \
        --with wandb==0.31.0 --with wandb-workspaces==0.4.13 python -c "import tools_wandb as t; print(t.log_all())"

Re-check (or re-report) what is already on W&B without logging again: `t.publish(t.logged_in_project())`.
Fix one panel in place (resume the run, log only that key): `t.relog("grasp-poses", ["grasps/at_claw_limit_pct"])`.
Media version logged once on 2026-11-19 07:05 (8 runs); summary-only version not logged yet (2026-11-21). `publish()`
replaces this gallery's older runs only after the new ones verify, so the next `log_all()` also retires the media runs.
Registered as a W&B tracker before its first run: bam_ws tools/wandb_trackers/trackers/robot_dataset_coverage_gallery.py.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import wandb

from core_types import CACHE, TYPES, by_id

ENTITY = "zach-yamaoka-independent"          # the entity every BAM project logs to (bam_trackio.wandb_view.ENTITY)
PROJECT = "Robot dataset coverage gallery"
TAG = "coverage-gallery"
WANDB_DIR = CACHE / "wandb"                  # W&B's local run files: a cache next to the ledger, never the repo
DEG = 180.0 / np.pi
# The gallery's Rerun column writes one .rrd per type here (= tools_rerun.MEDIA; not imported, WHY: tools_rerun pulls
# rerun-sdk into this column's venv). W&B gets this path, never the picture.
RERUN_MEDIA = Path("/home/bam/robot_metrics/reports/media/coverage-gallery-2026-11-19/tools/rerun")


def spread(key: str, values) -> dict:
    """min, p1, p50, p99, max of `values` as five scalars named `key.format(stat)` (NaN ignored): a histogram's summary."""
    values = np.asarray(values, float)
    stats = dict(zip(("p01", "p50", "p99"), np.nanpercentile(values, [1, 50, 99])))
    return {key.format(stat): float(value) for stat, value in
            {"min": np.nanmin(values), **stats, "max": np.nanmax(values)}.items()}


def rerun_pointer(run, type_id: str) -> None:
    """The picture of this type lives in the gallery's Rerun column: its .rrd path and writer go in the run config."""
    run.config.update({"rerun_rrd": str(RERUN_MEDIA / f"{type_id}.rrd"),
                       "rerun_writer": f"docs/proposals/coverage-2026-11-19/tools_rerun.py::{type_id.replace('-', '_')}"})


# --- the eight types ---------------------------------------------------------------------------------------------------

def limit_bars(run) -> None:
    payload = by_id("limit-bars").data()
    row = {}
    for name, values, limit, planner in zip(payload["names"], payload["values"], payload["limits"],
                                            payload["planner_limits"]):
        row.update(spread(f"limits/{name}_{{}}", values * DEG))      # deg, deg/s, deg/s², deg/s³: see config
        if limit:
            row[f"limits/{name}_used_pct"] = 100 * float(max(values.min() / limit[0], values.max() / limit[1]))
            row[f"limits/{name}_past_limit_pct"] = 100 * float(np.mean((values < limit[0]) | (values > limit[1])))
        if planner:
            row[f"limits/{name}_planner_used_pct"] = 100 * float(max(values.min() / planner[0], values.max() / planner[1]))
    run.log(row)
    run.config.update({"limits_units": dict(zip(payload["names"], payload["units_display"]))})
    rerun_pointer(run, "limit-bars")


def coverage_heatmap(run) -> None:
    payload = by_id("coverage-heatmap").data()
    samples, runs, reachable = payload["samples"], payload["runs"], payload["reachable"]
    visited = (samples > 0) & reachable
    run.log({
        "coverage/visited_pct": float(payload["visited_pct"]),
        "coverage/sufficient_pct": float(payload["sufficient_pct"]),          # cells with >= 50 samples
        "coverage/reachable_cells": int(reachable.sum()),
        "coverage/visited_cells": int(visited.sum()),
        "coverage/empty_reachable_cells": int((reachable & ~visited).sum()),
        "coverage/samples": int(samples.sum()),
        "coverage/samples_per_visited_cell_p50": float(np.median(samples[visited])),
        "coverage/runs_per_visited_cell_p50": float(np.median(runs[visited])),
        "coverage/runs_per_cell_max": int(runs.max()),
    })
    rerun_pointer(run, "coverage-heatmap")


def histogram_over_runs(run) -> None:
    payload = by_id("histogram-over-runs").data()
    limit_deg = payload["qd_limit"] * DEG
    for step, qd in enumerate(payload["qd_per_run"]):
        qd_deg = qd * DEG
        # step = recorded run: p1 / p50 / p99 over steps are the histogram-over-runs as three lines
        run.log({**spread("qd/{}_deg_s", qd_deg),
                 "qd/max_abs_deg_s": float(np.abs(qd_deg).max()),
                 "qd/p99_abs_deg_s": float(np.percentile(np.abs(qd_deg), 99)),
                 "qd/limit_deg_s": limit_deg,
                 "qd/samples": len(qd)}, step=step)
    run.summary["qd/recorded_runs"] = len(payload["qd_per_run"])
    run.summary["qd/rounds"] = len(set(payload["rounds"]))
    rerun_pointer(run, "histogram-over-runs")


def coverage_growth(run) -> None:
    payload = by_id("coverage-growth").data()
    in_order, mean = payload["in_order_pct"], payload["random_mean_pct"]
    p10, p90 = payload["random_band_pct"]
    chao2 = float(payload["chao2_pct"])
    run.define_metric("runs_recorded")
    run.define_metric("coverage/*", step_metric="runs_recorded")
    for step in range(len(in_order)):
        run.log({"runs_recorded": step + 1,
                 "coverage/in_order_pct": float(in_order[step]),
                 "coverage/random_mean_pct": float(mean[step]),
                 "coverage/random_p10_pct": float(p10[step]),
                 "coverage/random_p90_pct": float(p90[step]),
                 "coverage/chao2_ceiling_pct": chao2}, step=step)
    run.summary["coverage/reachable_cells"] = int(payload["reachable_cells"])
    rerun_pointer(run, "coverage-growth")


def phase_space_3d(run) -> None:
    payload = by_id("phase-space-3d").data()
    counts, reachable = payload["counts"], payload["reachable"]
    visited, empty = (counts > 0) & reachable, (counts == 0) & reachable
    row = {
        "phase_space/visited_pct": float(payload["visited_pct"]),
        "phase_space/reachable_cells": int(reachable.sum()),
        "phase_space/visited_cells": int(visited.sum()),
        "phase_space/empty_reachable_cells": int(empty.sum()),
        "phase_space/samples_per_visited_cell_p50": float(np.median(counts[visited])),
        "phase_space/samples_per_cell_max": int(counts.max()),
    }
    for axis, (name, unit) in enumerate([("q", "deg"), ("qd", "deg_s"), ("qdd", "deg_s2")]):
        row.update(spread(f"phase_space/{name}_{{}}_{unit}", payload["samples_deg"][:, axis]))
    run.log(row)
    run.config.update({"phase_space_axes": payload["axes"],
                       "phase_space_cell_size_deg": payload["cell_size_deg"].round(3).tolist()})
    rerun_pointer(run, "phase-space-3d")


def tool_paths(run) -> None:
    payload = by_id("tool-paths").data()
    counts = payload["voxel_counts"]
    every = np.concatenate(payload["paths_m"]) * 100                       # cm, base_link
    lengths_cm = [100 * float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) for path in payload["paths_m"]]
    occupied = np.sort(counts[counts > 0])[::-1]
    busiest_tenth = occupied[:math.ceil(len(occupied) / 10)]
    row = {
        "tool_paths/paths": len(payload["paths_m"]),
        "tool_paths/points": len(every),
        "tool_paths/occupied_voxels": len(occupied),                         # 2 cm voxels
        "tool_paths/occupied_pct": 100 * len(occupied) / counts.size,         # of the paths' bounding box
        "tool_paths/busiest_tenth_share_pct": 100 * float(busiest_tenth.sum() / occupied.sum()),
        "tool_paths/busiest_voxel_points": int(occupied[0]),
        **spread("tool_paths/path_length_{}_cm", lengths_cm),
    }
    for axis, name in enumerate("xyz"):
        row[f"tool_paths/{name}_min_cm"] = float(every[:, axis].min())
        row[f"tool_paths/{name}_max_cm"] = float(every[:, axis].max())
    run.log(row)
    rerun_pointer(run, "tool-paths")


def grasp_poses(run) -> None:
    payload = by_id("grasp-poses").data()
    limit = payload["opening_limit_rad"]
    positions_cm = payload["positions_m"] * 100
    row = {
        "grasps/count": len(payload["positions_m"]),
        "grasps/runs": len(set(payload["runs"])),
        "grasps/opening_limit_deg": limit * DEG,
        "grasps/at_claw_limit_pct": 100 * float(np.mean(payload["opening_rad"] >= limit - np.radians(0.5))),
        **spread("grasps/opening_{}_deg", payload["opening_rad"] * DEG),
    }
    for axis, name in enumerate("xyz"):
        row.update(spread(f"grasps/tcp_{name}_{{}}_cm", positions_cm[:, axis]))
    run.log(row)
    rerun_pointer(run, "grasp-poses")


def direction_sphere(run) -> None:
    payload = by_id("direction-sphere").data()
    counts, approach_z = payload["counts"], payload["directions"][:, 2]
    run.log({
        "directions/visited_pct": float(payload["visited_pct"]),             # of 12 x 24 equal-area cells
        "directions/cells": int(counts.size),
        "directions/used_cells": int((counts > 0).sum()),
        "directions/grasps": len(approach_z),
        "directions/busiest_cell_grasps": int(counts.max()),
        "directions/within_30deg_of_down_pct": 100 * float(np.mean(approach_z <= -np.cos(np.radians(30)))),
        **spread("directions/approach_z_{}", approach_z),                    # -1 = straight down
    })
    rerun_pointer(run, "direction-sphere")


IMPLEMENTATIONS = {"limit-bars": limit_bars, "coverage-heatmap": coverage_heatmap, "histogram-over-runs": histogram_over_runs,
                   "coverage-growth": coverage_growth, "phase-space-3d": phase_space_3d, "tool-paths": tool_paths,
                   "grasp-poses": grasp_poses, "direction-sphere": direction_sphere}

# what to open in each run, and how much of the view W&B still holds (honest)
PANEL_HINTS = {
    "limit-bars": "bar chart limits/*_used_pct (% of each limit); scalars limits/<q|qd|qdd|qddd>_p01 .. _p99",
    "coverage-heatmap": "scalars coverage/visited_pct, coverage/sufficient_pct, coverage/empty_reachable_cells",
    "histogram-over-runs": "lines qd/p01_deg_s, qd/p50_deg_s, qd/p99_deg_s per recorded run; qd/max_abs_deg_s vs "
                           "qd/limit_deg_s",
    "coverage-growth": "line charts coverage/* over runs_recorded",
    "phase-space-3d": "scalars phase_space/visited_pct, phase_space/empty_reachable_cells; the 3-D view is the .rrd",
    "tool-paths": "scalars tool_paths/occupied_voxels, tool_paths/busiest_tenth_share_pct; the 3-D view is the .rrd",
    "grasp-poses": "scalars grasps/at_claw_limit_pct, grasps/opening_p01 .. _p99_deg; the grippers are the .rrd",
    "direction-sphere": "scalars directions/visited_pct, directions/within_30deg_of_down_pct; the sphere is the .rrd",
}
NOTES = {
    "limit-bars": "Summary statistics only (2026-11-21): the share of each limit used and past it, and each "
                  "derivative's min / p1 / p50 / p99 / max as scalars, one bar chart over them. The histograms against "
                  "the limit are the Rerun column's .rrd.",
    "coverage-heatmap": "Summary statistics only (2026-11-21): visited and sufficient % of reachable cells, cell "
                        "counts, samples and runs per visited cell. The q x qd heatmap is the Rerun column's .rrd.",
    "histogram-over-runs": "Summary statistics only (2026-11-21): p1, p50 and p99 of qd per step (step = recorded run) "
                           "as three native lines, which is the histogram-over-runs read as a band; max |qd| against "
                           "the limit. The per-run histograms are the Rerun column's .rrd.",
    "coverage-growth": "Fully native scalars over a custom x (runs_recorded), unchanged by the 2026-11-21 cut. W&B line "
                       "plots cannot shade a band, so the random-order 10-90 % band is two lines (p10, p90); Chao2 is a "
                       "constant line.",
    "phase-space-3d": "Summary statistics only (2026-11-21): visited / empty reachable (q, qd, qdd) cells and each "
                      "axis' range. The 3-D cells and samples are the Rerun column's .rrd.",
    "tool-paths": "Summary statistics only (2026-11-21): paths, occupied 2 cm voxels, how much of the traffic the "
                  "busiest tenth of voxels carries, path lengths, workspace extents (cm, base_link). The Strava view "
                  "is the Rerun column's .rrd.",
    "grasp-poses": "Summary statistics only (2026-11-21): grasp count, claw opening percentiles and the share at the "
                   "claw limit, TCP extents (cm, base_link). The grippers are the Rerun column's .rrd.",
    "direction-sphere": "Summary statistics only (2026-11-21): used equal-area cells (12 x 24), the busiest cell, the "
                        "share of approaches within 30° of straight down. The sphere is the Rerun column's .rrd.",
}


# --- logging, verification, report -------------------------------------------------------------------------------------

def run_id(*parts: str) -> str:
    """A stable W&B run id, the house rule (bam_trackio.wandb_view.wandb_id_text): 16 hex characters of a sha1."""
    return hashlib.sha1("|".join((TAG, *parts)).encode()).hexdigest()[:16]


def browser_url(url: str) -> str:
    """W&B's app moved under forge.coreweave.com/wandb/; hand out the wandb.ai form, which redirects there
    (sim2real.tracking.wandb_log.browser_url, 2026-11-11)."""
    url = re.sub(r"^https://(?:forge\.coreweave\.com/(?:wandb/)?|wandb\.ai/)", "https://wandb.ai/", url)
    return quote(url, safe=":/?=%#")   # WHY: the project name has spaces and the report slug "·" and "&"


EXPECTED = {   # key -> how many history rows it must have (None = summary only)
    "limit-bars": {"limits/q_used_pct": 1, "limits/qddd_p99": 1, "limits/qdd_planner_used_pct": 1},
    "coverage-heatmap": {"coverage/visited_pct": 1, "coverage/empty_reachable_cells": 1},
    "histogram-over-runs": {"qd/p50_deg_s": "runs", "qd/max_abs_deg_s": "runs", "qd/recorded_runs": None},
    "coverage-growth": {"coverage/in_order_pct": "runs", "coverage/random_p90_pct": "runs", "coverage/reachable_cells": None},
    "phase-space-3d": {"phase_space/visited_pct": 1, "phase_space/empty_reachable_cells": 1},
    "tool-paths": {"tool_paths/occupied_voxels": 1, "tool_paths/path_length_p50_cm": 1},
    "grasp-poses": {"grasps/at_claw_limit_pct": 1, "grasps/opening_p50_deg": 1},
    "direction-sphere": {"directions/visited_pct": 1, "directions/used_cells": 1},
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
    return {**logged, "_checks": {**checks, "replaced_runs": replaced}}


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
    without a new run. The key's old value stays one step back in the run's history."""
    existing = logged_in_project()[type_id]["run_id"]
    run = wandb.init(entity=ENTITY, project=PROJECT, id=existing, resume="must", dir=str(WANDB_DIR), reinit="create_new")
    IMPLEMENTATIONS[type_id](_OnlyKeys(run, keys))
    run.finish()
    return existing


def _report_panels(wr, type_id: str) -> list:
    """The panels each type's report section shows: charts over its run's scalars, and where its picture lives.
    WHY no MediaBrowser / custom chart (Zach, 2026-11-21: "to help limit W and B storage I think we need to just be
    using it for summary statistics; for the media-rich things we should be using rerun!"): the run holds no media or
    table keys any more, so the picture is a pointer to the Rerun column's .rrd."""
    L = wr.Layout

    def bars(title, metrics, x=0, w=12):
        return wr.BarPlot(title=title, metrics=metrics, orientation="h", layout=L(x=x, y=0, w=w, h=8))

    def number(title, metric, x, w=6):
        return wr.ScalarChart(title=title, metric=metric, layout=L(x=x, y=0, w=w, h=8))

    def in_rerun(what, x=12, w=12):
        rrd = RERUN_MEDIA / f"{type_id}.rrd"
        return wr.MarkdownPanel(markdown=f"**{what}** is kept off W&B (summary statistics only, since 2026-11-21): it is "
                                         f"the gallery's Rerun column. On the workstation:\n\n`rerun {rrd}`\n\n(written "
                                         f"by `tools_rerun.py::{type_id.replace('-', '_')}`)", layout=L(x=x, y=0, w=w, h=8))

    return {
        "limit-bars": [bars("Share of each limit the data used (%; 100 = at the limit)",
                            ["limits/q_used_pct", "limits/qd_used_pct", "limits/qdd_used_pct", "limits/qdd_planner_used_pct"]),
                       in_rerun("Each derivative's histogram against its limit")],
        "coverage-heatmap": [number("Reachable q x qd cells visited (%)", "coverage/visited_pct", x=0),
                             number("Cells with >= 50 samples (%)", "coverage/sufficient_pct", x=6),
                             in_rerun("The q x qd heatmap (runs and samples per cell)")],
        "histogram-over-runs": [
            wr.LinePlot(title="qd per recorded run: p1, p50, p99 (deg/s; step = run)",
                        y=["qd/p01_deg_s", "qd/p50_deg_s", "qd/p99_deg_s"], layout=L(x=0, y=0, w=9, h=10)),
            wr.LinePlot(title="max |qd| per run vs the limit (deg/s)", y=["qd/max_abs_deg_s", "qd/p99_abs_deg_s", "qd/limit_deg_s"],
                        layout=L(x=9, y=0, w=9, h=10)),
            in_rerun("One qd histogram per recorded run", x=18, w=6)],
        "coverage-growth": [
            wr.LinePlot(title="% of reachable (q, qd, qdd) cells visited", x="runs_recorded",
                        y=["coverage/in_order_pct", "coverage/random_mean_pct", "coverage/random_p10_pct",
                           "coverage/random_p90_pct", "coverage/chao2_ceiling_pct"], layout=L(x=0, y=0, w=24, h=10))],
        "phase-space-3d": [number("Reachable (q, qd, qdd) cells visited (%)", "phase_space/visited_pct", x=0),
                           number("Reachable cells never visited", "phase_space/empty_reachable_cells", x=6),
                           in_rerun("The 3-D cells and samples")],
        "tool-paths": [number("Occupied 2 cm voxels", "tool_paths/occupied_voxels", x=0),
                       number("Traffic in the busiest tenth of voxels (%)", "tool_paths/busiest_tenth_share_pct", x=6),
                       in_rerun("Every planned TCP path and the voxel occupancy")],
        "grasp-poses": [bars("Claw opening at each grasp (deg)", ["grasps/opening_p01_deg", "grasps/opening_p50_deg",
                                                                  "grasps/opening_p99_deg", "grasps/opening_limit_deg"]),
                        in_rerun("Every real repick grasp as a thin gripper")],
        "direction-sphere": [number("Equal-area direction cells used (%)", "directions/visited_pct", x=0),
                             number("Approaches within 30° of straight down (%)", "directions/within_30deg_of_down_pct", x=6),
                             in_rerun("The approach directions on the bin-painted sphere")],
    }[type_id]


def save_report(logged: dict) -> str:
    """One W&B report, a section per type: its question, its source, what W&B holds of it, and its panels over a
    runset filtered to that type's run. Best effort: the runs' own workspaces are the fallback surface."""
    try:
        import wandb_workspaces.reports.v2 as wr
        from wandb_workspaces import expr
    except ImportError:
        return ""
    blocks = [wr.MarkdownBlock(text="The W&B column of the robot dataset coverage gallery: the same eight coverage views "
                                    "in Matplotlib, Rerun, Viser and W&B from one dataset (robot_metrics "
                                    "`docs/proposals/coverage-2026-11-19/core_types.py`). One run per view type. Since "
                                    "2026-11-21 W&B holds each view's summary statistics only; the pictures are the "
                                    "Rerun column's .rrd files."),
              wr.TableOfContents()]
    for core in TYPES:
        blocks += [wr.H2(text=core.title),
                   wr.MarkdownBlock(text=f"**{core.question}**  \n{core.data()['source']}  \n*On W&B:* {NOTES[core.id]}"),
                   wr.PanelGrid(runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name=core.id,
                                                   filters=[expr.Config("type_id") == core.id])],
                                panels=_report_panels(wr, core.id))]
    report = wr.Report(entity=ENTITY, project=PROJECT, title="Robot dataset coverage gallery · W&B column",
                       description="Eight coverage-view types as W&B summary statistics; one run each", blocks=blocks)
    report.width = "fluid"
    try:
        report.save()
    except Exception as error:   # the runs are the deliverable; a report the API refuses never undoes them
        return f"(report not saved: {error})"
    return browser_url(report.url)
