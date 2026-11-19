"""Sections 2 (many joints) and 3 (Cartesian): the rack's multi-joint robots, every axis and the tool, against their limits.

Three real robots (arm_data): the five-bar on the rack (two motors, 53 speed-ladder rungs, Nov 10-11, plus their sim twin), the UR
wrist rig (three joints, 44 multi-joint plays, Nov 11-12) and the full arm's repick loop (J2, C1, J1, W1-W3 and the claw, 453 real runs,
Nov 13-18). Every number is relative to a committed limit: the joint descriptions and reachable masks of 2026-11-11, the plan's limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize

import arm_data
import style
from cards import Card
from coverage import Axis, Grid, limit_use, sphere_cell
from style import ACCENT, CATEGORY, HAIR, INK, LIMIT, MUTED, UNREACHABLE

DEG = arm_data.DEG
SMOOTH = 7                      # samples in the moving average before each derivative (70 ms at 100 Hz, 140 ms at 50 Hz)
# WHY 1.2 rad/s for the full arm's J2 and C1: each repick summary.json's drives.rule_deg_s (68.75 deg/s), the speed the arm is held to;
# the rack five-bar's 6 / 12 rad/s are its description's caps, a lighter end mass on a different gain set
ARM_FIVEBAR_RULE_RAD_S = 68.7549354156988 / DEG
CLAW_SPAN_MM = 70.0             # attempts.jsonl look.item.claw_span_mm: the open claw's span
GROUPS = {"five-bar rack": CATEGORY[0], "UR wrist rig": CATEGORY[2], "full arm": CATEGORY[1]}


@dataclass
class Joint:
    group: str
    name: str
    low: float
    high: float
    qd_limit: float | None
    qdd_limit: float | None
    q: np.ndarray
    qd: np.ndarray | None
    qdd: np.ndarray | None
    run: np.ndarray
    unit: str = "deg"
    scale: float = DEG
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} · {self.group}"


def _smooth(x: np.ndarray) -> np.ndarray:
    return np.convolve(x, np.ones(SMOOTH) / SMOOTH, mode="same")


def _rate(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """d/dt of a moving average (the boards report q and qd; qdd and the arm's qd are differentiated). Ends are NaN: biased."""

    out = np.gradient(_smooth(np.asarray(x, float)), t) if len(t) > 2 * SMOOTH else np.full(len(t), np.nan)
    out[:SMOOTH] = np.nan
    out[-SMOOTH:] = np.nan
    return out


# --- the context: every joint of every robot, once ----------------------------------------------------------------------

def _rack(ladders, limits) -> list[Joint]:
    q, qd, qdd, run = [], [], [], []
    for index, rung in enumerate(ladders):
        real = rung["real"]
        t = real["t"].astype(float)
        motors = real["measured_motors"].astype(float)
        rate = real["measured_motor_rate"].astype(float)
        q.append(np.column_stack([motors[:, 0], arm_data.wrap_c1(motors[:, 1], limits["joints"]["C1"]["low"])]))
        qd.append(rate)
        qdd.append(np.column_stack([_rate(t, rate[:, 0]), _rate(t, rate[:, 1])]))
        run.append(np.full(len(t), index))
    q, qd, qdd, run = (np.concatenate(v) for v in (q, qd, qdd, run))
    out = []
    for column, name in enumerate(("J2", "C1")):
        j = limits["joints"][name]
        out.append(Joint("five-bar rack", name, j["low"], j["high"], j["qd"], j["qdd"], q[:, column], qd[:, column],
                         qdd[:, column], run))
    return out


def _wrist(runs, limits) -> list[Joint]:
    q, qd, qdd, run = [], [], [], []
    for index, entry in enumerate(runs):
        t = entry["t"]
        q.append(entry["q"].astype(float))
        qd.append(entry["qd"].astype(float))
        qdd.append(np.column_stack([_rate(t, entry["qd"][:, i].astype(float)) for i in range(3)]))
        run.append(np.full(len(t), index))
    q, qd, qdd, run = (np.concatenate(v) for v in (q, qd, qdd, run))
    out = []
    for column, can in enumerate(arm_data.WRIST_CANS):
        j = limits["joints"][can]
        out.append(Joint("UR wrist rig", f"CAN {can}", j["low"], j["high"], j["qd"], j["qdd"], q[:, column], qd[:, column],
                         qdd[:, column], run))
    return out


def _arm(repicks, fivebar, wrist_limits) -> dict:
    """The full arm's measured configurations, one row per 50 Hz timeline tick: J2, C1, J1, W1, W2, W3, claw gap, and their rates."""

    low_c1 = fivebar["joints"]["C1"]["low"]
    q, qd, run, grasp_run, tool, tool_speed, segment = [], [], [], [], [], [], []
    for index, entry in enumerate(repicks):
        if entry["w"] is None:          # 15 runs have no wrist trace: they cannot give a 6-joint configuration
            continue
        gap = entry["gap_mm"] if entry["gap_mm"] is not None else np.full(len(entry["t"]), np.nan)
        rows = np.column_stack([entry["j2"], arm_data.wrap_c1(entry["c1"], low_c1), entry["j1"], entry["w"], gap])
        t = entry["t"]
        rates = np.full((len(t), 6), np.nan)
        xy, _ = arm_data.fivebar_fk(entry["j2"], entry["c1"])
        speed = np.full(len(t), np.nan)
        for move in np.unique(entry["move"]):
            pick = np.nonzero(entry["move"] == move)[0]
            if len(pick) <= 3 * SMOOTH:
                continue
            for column in range(6):
                rates[pick, column] = _rate(t[pick], rows[pick, column])
            speed[pick] = np.hypot(_rate(t[pick], xy[pick, 0]), _rate(t[pick], xy[pick, 1]))
        closed = any(a.get("closed") for a in entry["attempt"].values())
        keep = np.isfinite(rows[:, :6]).all(axis=1)      # the claw gap may be NaN: hover-only runs mostly never read the claw
        q.append(rows[keep])
        qd.append(rates[keep])
        run.append(np.full(keep.sum(), index))
        grasp_run.append(np.full(keep.sum(), closed))
        tool.append(xy[keep])
        tool_speed.append(speed[keep])
        segment.append(entry["move"][keep].astype(np.int64) + 1000 * index)
    q, qd, run, grasp_run, tool, tool_speed, segment = (np.concatenate(v) for v in (q, qd, run, grasp_run, tool, tool_speed, segment))
    plan_limits = next(p["limits"] for entry in repicks for p in entry["plans"] if p.get("limits"))
    f, w = fivebar["joints"], wrist_limits["joints"]
    # (name, low, high, qd limit): J2 / C1 the five-bar's committed range at the arm's drive rule; J1 and W1-W3 the plan files' limits
    # (CAD, homed frame + offsets) at the wrist's own speed caps (W1 = revolute_11 = CAN 14, W2 = CAN 15, W3 = CAN 16)
    spec = [("J2", f["J2"]["low"], f["J2"]["high"], ARM_FIVEBAR_RULE_RAD_S),
            ("C1", f["C1"]["low"], f["C1"]["high"], ARM_FIVEBAR_RULE_RAD_S),
            ("J1", plan_limits["j1_range_rad"][0], plan_limits["j1_range_rad"][1], None),
            ("W1", -plan_limits["w1_abs_max_rad"], plan_limits["w1_abs_max_rad"], w[14]["qd"]),
            ("W2", -plan_limits["w23_abs_max_rad"], plan_limits["w23_abs_max_rad"], w[15]["qd"]),
            ("W3", -plan_limits["w23_abs_max_rad"], plan_limits["w23_abs_max_rad"], w[16]["qd"])]
    joints = [Joint("full arm", name, low, high, qd_limit, None, q[:, i], qd[:, i], None, run)
              for i, (name, low, high, qd_limit) in enumerate(spec)]
    joints.append(Joint("full arm", "claw gap", 0.0, CLAW_SPAN_MM, None, None, q[:, 6], None, None, run, unit="mm", scale=1.0))
    return {"q": q, "qd": qd, "run": run, "grasp_run": grasp_run, "tool": tool, "tool_speed": tool_speed, "segment": segment,
            "joints": joints, "plan_limits": plan_limits, "runs": len(np.unique(run))}


def _context() -> dict:
    fivebar = arm_data.fivebar_limits()
    wrist_limits = arm_data.wrist_limits()
    ladders = arm_data.ladder_rungs()
    wrist = arm_data.wrist_runs()
    repicks = arm_data.repicks()
    arm = _arm(repicks, fivebar, wrist_limits)
    rack = _rack(ladders, fivebar)
    wrist_joints = _wrist(wrist, wrist_limits)
    return {"fivebar": fivebar, "wrist_limits": wrist_limits, "ladders": ladders, "wrist": wrist, "repicks": repicks,
            "arm": arm, "joints": rack + wrist_joints + arm["joints"], "rack": rack, "wrist_joints": wrist_joints}


# --- sources (what each card drew, how much) ----------------------------------------------------------------------------

def _src_rack(ctx) -> str:
    n = sum(len(r["real"]["t"]) for r in ctx["ladders"])
    ladders = len({r["ladder"] for r in ctx["ladders"]})
    return (f"five-bar on the rack (CAN 12 J2, CAN 13 C1): {len(ctx['ladders'])} real speed-ladder rungs from {ladders} ladders "
            f"(Nov 10-11), {n / 1000:.0f} k samples at 100 Hz")


def _src_wrist(ctx) -> str:
    n = sum(len(r["t"]) for r in ctx["wrist"])
    return (f"UR wrist rig (CAN 14/15/16): {len(ctx['wrist'])} real multi-joint plays (multi3_pick / dwell / wide, batched_proof, "
            f"Nov 11-12), {n / 1000:.0f} k samples at ~90 Hz")


def _src_arm(ctx) -> str:
    arm = ctx["arm"]
    grasp = len(np.unique(arm["run"][arm["grasp_run"]]))
    return (f"full arm, real repick loop (Nov 13-18): {arm['runs']} of {len(ctx['repicks'])} readable runs (the rest have no "
            f"wrist trace; {grasp} with real closes, the rest hover-only), "
            f"{len(arm['q']) / 1000:.0f} k measured 50 Hz ticks; W1-W3 through each run's pick_loop.json wrist maps")


def _limits_note() -> str:
    return ("limits: bam_fivebar and bam_ur_wrist_coverage reachable_2026-11-11 joint descriptions; full arm J1 / W1-W3 from the plan "
            "files' limits, J2 / C1 the five-bar's range at the arm's 68.8 deg/s drive rule")


# --- shared drawing helpers ---------------------------------------------------------------------------------------------

def _normalized(joint: Joint, which: str) -> tuple[np.ndarray, float | None]:
    """Values as a share of the limit, -1 = low limit, +1 = high (q), or of the +-limit (qd, qdd)."""

    if which == "q":
        q = joint.q[np.isfinite(joint.q)]
        return 2.0 * (q - joint.low) / (joint.high - joint.low) - 1.0, 1.0
    values, limit = (joint.qd, joint.qd_limit) if which == "qd" else (joint.qdd, joint.qdd_limit)
    if values is None or limit is None:
        return np.array([]), None
    values = values[np.isfinite(values)]
    return values / limit, limit


def _span_pct(joint: Joint) -> float:
    """Share of the q range used, between the p0.1 and p99.9 extremes (one glitch never sets it)."""

    q = joint.q[np.isfinite(joint.q)]
    lo, hi = np.percentile(q, [0.1, 99.9])
    return limit_use(np.clip(q, lo, hi), joint.low, joint.high)["span_pct"]


def _edges_outline(ax, pixels: np.ndarray, size: float, color=INK, linewidth=0.9):
    """The staircase outline of a set of (i, j) pixels (each pixel [i, i+1) x [j, j+1) times `size`): exact, never smoothed."""

    cells = {tuple(p) for p in pixels.tolist()}
    segments = []
    for i, j in cells:
        x0, y0, x1, y1 = i * size, j * size, (i + 1) * size, (j + 1) * size
        if (i - 1, j) not in cells:
            segments.append([(x0, y0), (x0, y1)])
        if (i + 1, j) not in cells:
            segments.append([(x1, y0), (x1, y1)])
        if (i, j - 1) not in cells:
            segments.append([(x0, y0), (x1, y0)])
        if (i, j + 1) not in cells:
            segments.append([(x0, y1), (x1, y1)])
    ax.add_collection(LineCollection(segments, colors=color, linewidths=linewidth, zorder=5))


def _eq_hist(counts: np.ndarray) -> np.ndarray:
    """Histogram equalization of the non-empty cells (Strava's per-tile CDF; datashader's how='eq_hist'): rank / count, 0 stays 0."""

    out = np.zeros(counts.shape)
    nonzero = counts > 0
    values = counts[nonzero]
    ranks = np.searchsorted(np.sort(values), values, side="right")
    out[nonzero] = ranks / len(values)
    return out


def _resample_path(xy: np.ndarray, step: float) -> np.ndarray:
    """Points every `step` metres along a polyline: a path's density is then its length, not how long it dwelt (the line-density view)."""

    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 2:
        return xy
    lengths = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    if lengths[-1] < step:
        return xy[:1]
    s = np.arange(0.0, lengths[-1], step)
    return np.column_stack([np.interp(s, lengths, xy[:, 0]), np.interp(s, lengths, xy[:, 1])])


def _plane_paths(ctx):
    """Every real L2-tip path in the five-bar plane: each ladder rung (recorded tool), each full-arm move (FK of measured J2, C1)."""

    if "plane_paths" in ctx:
        return ctx["plane_paths"]
    paths, speeds = [], []
    for rung in ctx["ladders"]:
        real = rung["real"]
        xy = real["measured_tool"].astype(float)
        t = real["t"].astype(float)
        paths.append(("rack", xy))
        speeds.append((xy, np.hypot(_rate(t, xy[:, 0]), _rate(t, xy[:, 1]))))
    arm = ctx["arm"]
    # the segment ids (move + 1000 x run) only ever grow down the rows, so each move is one contiguous slice
    starts = np.concatenate([[0], np.flatnonzero(np.diff(arm["segment"])) + 1, [len(arm["segment"])]])
    for start, stop in zip(starts[:-1], starts[1:]):
        if stop - start > 3:
            paths.append(("arm", arm["tool"][start:stop]))
            speeds.append((arm["tool"][start:stop], arm["tool_speed"][start:stop]))
    ctx["plane_paths"] = (paths, speeds)
    return paths, speeds


def _pixel_box(pixels: np.ndarray, size: float, pad: int = 1):
    low = pixels.min(axis=0) - pad
    high = pixels.max(axis=0) + pad + 1
    return (Axis("x", low[0] * size, high[0] * size, int(high[0] - low[0]), "m", 1000.0, "mm"),
            Axis("y", low[1] * size, high[1] * size, int(high[1] - low[1]), "m", 1000.0, "mm"))


def _pixel_mask(pixels: np.ndarray, axes) -> np.ndarray:
    mask = np.zeros((axes[0].bins, axes[1].bins), bool)
    i = np.round(pixels[:, 0] - axes[0].low / (axes[0].edges()[1] - axes[0].edges()[0])).astype(int)
    j = np.round(pixels[:, 1] - axes[1].low / (axes[1].edges()[1] - axes[1].edges()[0])).astype(int)
    mask[i, j] = True
    return mask


# --- section 2: many joints ---------------------------------------------------------------------------------------------

def fleet_ridgeline(ctx) -> Card:
    joints = ctx["joints"]
    fig, axes = style.figure(9.8, 5.6, ncols=3, sharey=True)
    edges = np.linspace(-1.3, 1.3, 105)
    spacing = 1.0
    numbers = {}
    for ax, which, title in zip(axes, ("q", "qd", "qdd"), ("q: position in its range", "qd: speed / speed limit",
                                                            "qdd: acceleration / its limit")):
        for row, joint in enumerate(joints):
            base = -row * spacing
            values, limit = _normalized(joint, which)
            color = GROUPS[joint.group]
            if limit is None or len(values) == 0:
                ax.text(0, base + 0.25, "no limit on file", ha="center", va="center", fontsize=6.5, color=MUTED)
                continue
            beyond = np.mean(np.abs(values) > 1.3)
            counts, _ = np.histogram(np.clip(values, -1.299, 1.299), edges)
            height = 0.92 * counts / max(counts.max(), 1)
            ax.fill_between(edges[:-1], base, base + height, step="post", color=color, linewidth=0, alpha=0.9)
            # the reached extremes as a hairline under the ridge: a linear ridge hides a rare excursion, this never does
            lo, hi = np.clip(np.percentile(values, [0.1, 99.9]), -1.3, 1.3)
            ax.plot([lo, hi], [base - 0.07] * 2, color=INK, linewidth=0.8, solid_capstyle="butt")
            ax.plot([lo, lo, np.nan, hi, hi], [base - 0.13, base - 0.01, np.nan, base - 0.13, base - 0.01], color=INK, linewidth=0.8)
            if beyond > 0.001:
                ax.text(1.32, base + 0.1, f"{100 * beyond:.1f} %▸", fontsize=6, color=LIMIT, va="bottom")
        for limit in (-1, 1):
            ax.axvline(limit, color=LIMIT, linewidth=1, zorder=10)
        ax.set_xlim(-1.3, 1.3)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["−100 %", "−50", "0", "50", "+100 %"])
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([-row * spacing + 0.3 for row in range(len(joints))], [j.label for j in joints], fontsize=7)
    for tick, joint in zip(axes[0].get_yticklabels(), joints):
        tick.set_color(GROUPS[joint.group])
    axes[0].set_ylim(-(len(joints) - 1) * spacing - 0.15, 1.0)
    axes[0].set_xlabel("−100 % = low limit, +100 % = high limit")
    axes[1].set_xlabel("share of ± speed limit")
    axes[2].set_xlabel("share of ± acceleration limit")
    fig.tight_layout(w_pad=0.6)
    spans = {j.label: _span_pct(j) for j in joints}
    peaks = {j.label: 100 * np.nanpercentile(np.abs(j.qd), 99.9) / j.qd_limit for j in joints if j.qd is not None and j.qd_limit}
    least = min(spans, key=spans.get)
    numbers["q range used, median joint"] = f"{np.median(list(spans.values())):.0f} % (least: {least} {spans[least]:.0f} %)"
    fastest, slowest = max(peaks, key=peaks.get), min(peaks, key=peaks.get)
    numbers["peak speed / limit (p99.9)"] = f"{fastest} {peaks[fastest]:.0f} %; {slowest} {peaks[slowest]:.0f} %"
    return Card("fleet-ridgeline", "jointN", "Fleet ridgeline: every joint's q, qd, qdd on its own limit, one row each",
                "Zach's 'single time of q or qd or qdd of all axis': which joints use their range, which stay in the middle?",
                style.to_data_uri(fig), f"{_src_rack(ctx)}; {_src_wrist(ctx)}; {_src_arm(ctx)}. {_limits_note()}. "
                "Height is linear count per row, each row scaled to its own peak; the black hairline under a ridge spans its "
                "p0.1-p99.9 extremes. The claw gap reads past its 70 mm span (p99.9 78 mm, max 112 mm): a gap-calibration "
                "question, not coverage",
                "TensorBoard Histograms (offset mode) / joy plots, normalized per axis like a robot's limit dashboard",
                numbers, "N-DOF arm", wide=True)


def limit_scorecard(ctx) -> Card:
    joints = ctx["joints"]
    columns = ["q span used", "toward low limit", "toward high limit", "time within 5 % of a q limit", "peak |qd| (p99.9)",
               "peak |qdd| (p99.9)"]
    table = np.full((len(joints), len(columns)), np.nan)
    for row, joint in enumerate(joints):
        q = joint.q[np.isfinite(joint.q)]
        lo, hi = np.percentile(q, [0.1, 99.9])
        middle = 0.5 * (joint.low + joint.high)
        # WHY centred: limit_use's min/max shares are against zero; centred on the range's middle they say how far toward each end
        centred = limit_use(np.clip(q, lo, hi) - middle, joint.low - middle, joint.high - middle)
        table[row, 0] = limit_use(np.clip(q, lo, hi), joint.low, joint.high)["span_pct"]
        table[row, 1] = centred["min_pct_of_low"]
        table[row, 2] = centred["max_pct_of_high"]
        table[row, 3] = limit_use(q, joint.low, joint.high)["time_near_limit_pct"]
        if joint.qd is not None and joint.qd_limit:
            table[row, 4] = 100 * np.nanpercentile(np.abs(joint.qd), 99.9) / joint.qd_limit
        if joint.qdd is not None and joint.qdd_limit:
            table[row, 5] = 100 * np.nanpercentile(np.abs(joint.qdd), 99.9) / joint.qdd_limit
    fig, ax = style.figure(7.4, 4.6)
    shown = np.ma.masked_where(~np.isfinite(table) | (table <= 0), np.clip(table, 0, 100))
    ax.pcolormesh(np.arange(len(columns) + 1), np.arange(len(joints) + 1), shown, cmap=style.EMPTY_WHITE, vmin=0, vmax=100,
                  edgecolors="white", linewidth=1.5)
    for row in range(len(joints)):
        for column in range(len(columns)):
            value = table[row, column]
            text = "—" if not np.isfinite(value) else (f"{value:.1f}" if value < 1 else f"{value:.0f}")
            dark = np.isfinite(value) and value > 55
            ax.text(column + 0.5, row + 0.5, text, ha="center", va="center", fontsize=7.5,
                    color="white" if dark else (MUTED if not np.isfinite(value) else INK))
    ax.set_xticks(np.arange(len(columns)) + 0.5, [c.replace(" (", "\n(").replace("within 5 % of", "within 5 %\nof") for c in columns],
                  fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(joints)) + 0.5, [j.label for j in joints], fontsize=7.5)
    for tick, joint in zip(ax.get_yticklabels(), joints):
        tick.set_color(GROUPS[joint.group])
    ax.invert_yaxis()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("% of the limit (white = 0, dark = all of it; — = no limit on file)", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    unused = [j.label for j, row in zip(joints, table) if row[0] < 50]
    numbers = {"joints using under half their q range": f"{len(unused)} of {len(joints)}",
               "closest to a speed limit": f"{joints[int(np.nanargmax(table[:, 4]))].label} ({np.nanmax(table[:, 4]):.0f} %)"}
    return Card("limit-scorecard", "jointN", "Limit-use scorecard: one row per joint, one number per limit",
                "Which limit did each joint actually get near: its range ends, its speed, its acceleration?",
                style.to_data_uri(fig), f"{_src_rack(ctx)}; {_src_wrist(ctx)}; {_src_arm(ctx)}. {_limits_note()}. "
                "Extremes are p0.1 / p99.9 (qdd and the arm's qd are differentiated, so one noisy sample never sets them)",
                "TFDV / Facets Overview's per-feature statistics table, with the schema range as the limit",
                numbers, "N-DOF arm")


def _joint_axis(joint: Joint, bins: int) -> Axis:
    return Axis(joint.name, joint.low, joint.high, bins, "rad", DEG, "deg")


def _pair_reachable(ctx, a: Joint, b: Joint, axes) -> np.ndarray | None:
    """The closed chain's coupled limit for (J2, C1): any allowed 1-degree pose of the committed mask inside the cell."""

    if {a.name, b.name} != {"J2", "C1"}:
        return None
    fb = ctx["fivebar"]
    j2, c1 = np.meshgrid(fb["axis_j2_deg"] / DEG, fb["axis_c1_deg"] / DEG, indexing="ij")
    poses = np.column_stack([j2[fb["allowed"]], c1[fb["allowed"]]])
    if a.name == "C1":
        poses = poses[:, ::-1]
    counts, _ = np.histogramdd(poses, bins=[axis.edges() for axis in axes])
    return counts > 0


def corner_plot(ctx) -> Card:
    arm = ctx["arm"]
    joints = arm["joints"][:6]
    bins = 18
    n = len(joints)
    fig, grid = style.figure(8.2, 7.8, nrows=n, ncols=n)
    pair_pct = {}
    for i, row_joint in enumerate(joints):
        for j, column_joint in enumerate(joints):
            ax = grid[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(HAIR)
            if i == j:
                values = (row_joint.q - row_joint.low) / (row_joint.high - row_joint.low)
                counts, edges = np.histogram(values, np.linspace(-0.08, 1.08, 59))
                ax.fill_between(edges[:-1], 0, counts / counts.max(), step="post", color=ACCENT, linewidth=0)
                ax.axvspan(-0.08, 0, color=UNREACHABLE, zorder=-1)
                ax.axvspan(1, 1.08, color=UNREACHABLE, zorder=-1)
                for limit in (0, 1):
                    ax.axvline(limit, color=LIMIT, linewidth=1)
                ax.set_xlim(-0.08, 1.08)
                ax.set_ylim(0, 1.1)
                ax.text(0.5, 1.02, row_joint.name, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                        fontweight="bold", color=INK)
                continue
            a, b = (column_joint, row_joint) if i > j else (row_joint, column_joint)
            axes = (_joint_axis(a, bins), _joint_axis(b, bins))
            reach = _pair_reachable(ctx, a, b, axes)
            pair = Grid.of(np.column_stack([a.q, b.q]), axes, reachable=reach)
            if i > j:
                style.heatmap(ax, pair.counts, np.linspace(0, 1, bins + 1), np.linspace(0, 1, bins + 1), reach,
                              colorbar_label=None)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                pair_pct[(a.name, b.name)] = pair.visited_pct()
            else:
                value = pair.visited_pct()
                ax.set_facecolor(style.EMPTY_WHITE(min(value / 100, 1)) if value > 0 else "white")
                ax.text(0.5, 0.5, f"{value:.0f} %", transform=ax.transAxes, ha="center", va="center", fontsize=10,
                        fontweight="bold", color="white" if value > 45 else INK)
                if reach is not None:
                    ax.text(0.5, 0.18, "of the coupled mask", transform=ax.transAxes, ha="center", fontsize=5.5,
                            color="white" if value > 45 else MUTED)
            if i == n - 1:
                ax.set_xticks([0, 1], [f"{a.low * DEG:.0f}°", f"{a.high * DEG:.0f}°"], fontsize=6)
            if j == 0 and i > 0:
                ax.set_yticks([0, 1], [f"{b.low * DEG:.0f}°", f"{b.high * DEG:.0f}°"], fontsize=6)
    for j, joint in enumerate(joints):
        grid[n - 1, j].set_xlabel(joint.name, fontsize=8)
    for i, joint in enumerate(joints[1:], start=1):
        grid[i, 0].set_ylabel(joint.name, fontsize=8)
    fig.suptitle("lower: 2-D occupancy, each axis spanning its limits (white = empty, grey = beyond the coupled mask);  "
                 "diagonal: histogram, red = limit;  upper: % of the pair's cells visited", fontsize=7.5, color=MUTED, y=0.995)
    fig.tight_layout(h_pad=0.25, w_pad=0.25, rect=(0, 0, 1, 0.985))
    worst = min(pair_pct, key=pair_pct.get)
    numbers = {"mean pairwise visited": f"{np.mean(list(pair_pct.values())):.0f} % of {bins} × {bins} cells",
               "emptiest pair": f"{worst[0]} × {worst[1]}: {pair_pct[worst]:.0f} %",
               "J2 × C1 (vs the closed chain's own mask)": f"{pair_pct.get(('J2', 'C1'), pair_pct.get(('C1', 'J2'), float('nan'))):.0f} %"}
    return Card("corner-plot", "jointN", "Corner plot of the full arm's six joints, each axis spanning its limits",
                "Which pairs of joints were moved together, and which combinations were never tried?",
                style.to_data_uri(fig), _src_arm(ctx) + f". {_limits_note()}. {bins} × {bins} linear cells per pair; the J2 × C1 "
                "panel's denominator is the five-bar's committed 1° reachable mask, every other pair a box",
                "corner.py / seaborn pairplot (SPLOM, Hartigan 1975); upper triangle as a pairwise-coverage matrix (combinatorial "
                "pairwise testing's 2-way coverage)", numbers, "N-DOF arm", wide=True)


PC_LOW, PC_HIGH = -0.14, 1.20          # the drawn band around each axis's [low limit, high limit] = [0, 1]


def _pc_density(normalized: np.ndarray, weights: np.ndarray, columns: int = 36, rows: int = 150) -> list[np.ndarray]:
    """Every configuration's polyline rasterized between each pair of neighbouring axes (columns x rows pixels per gap), weighted:
    the parallel-coordinates plot as a line-density image, so a million lines read as one picture and empty stays white."""

    u = (np.arange(columns) + 0.5) / columns
    images = []
    for k in range(normalized.shape[1] - 1):
        a, b = normalized[:, k], normalized[:, k + 1]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b, w = a[ok], b[ok], weights[ok]
        image = np.zeros(rows * columns)
        for start in range(0, len(a), 50000):
            y = a[start:start + 50000, None] + (b - a)[start:start + 50000, None] * u[None, :]
            v = np.clip(((y - PC_LOW) / (PC_HIGH - PC_LOW) * rows).astype(int), 0, rows - 1)
            flat = (v * columns + np.arange(columns)[None, :]).ravel()
            image += np.bincount(flat, weights=np.repeat(w[start:start + 50000], columns), minlength=rows * columns)
        images.append(image.reshape(rows, columns))
    return images


def parallel_coordinates(ctx) -> Card:
    arm = ctx["arm"]
    joints = arm["joints"]
    normalized = np.column_stack([(j.q - j.low) / (j.high - j.low) for j in joints])
    # WHY each run weighted equally: the 168 multi-cycle runs hold 94 % of the ticks; per-tick weight would hide the hover-only runs
    _, inverse, per_run = np.unique(arm["run"], return_inverse=True, return_counts=True)
    weights = 1.0 / per_run[inverse]
    fig, panels = style.figure(8.8, 4.4, nrows=2, sharex=True)
    numbers = {}
    for ax, closed, name in ((panels[0], True, "runs with real closes"), (panels[1], False, "hover-only runs")):
        rows = arm["grasp_run"] == closed
        images = _pc_density(normalized[rows], weights[rows])
        stacked = np.concatenate([image.ravel() for image in images])
        ranks = _eq_hist(stacked)
        offset = 0
        for k, image in enumerate(images):
            shown = ranks[offset:offset + image.size].reshape(image.shape)
            offset += image.size
            ax.pcolormesh(np.linspace(k, k + 1, image.shape[1] + 1), np.linspace(PC_LOW, PC_HIGH, image.shape[0] + 1),
                          np.ma.masked_where(shown <= 0, shown), cmap=style.EMPTY_WHITE, vmin=0, vmax=1, shading="flat")
        runs = len(np.unique(arm["run"][rows]))
        ax.set_title(f"{name}: {runs} runs, {rows.sum() / 1000:.0f} k configurations", loc="left", fontsize=8.5)
        numbers[f"{name}"] = f"{runs} runs, {rows.sum() / 1000:.0f} k ticks"
        for index, joint in enumerate(joints):
            ax.plot([index, index], [0, 1], color=INK, linewidth=0.7, zorder=3)
            for end, value in ((0, joint.low), (1, joint.high)):
                ax.plot([index - 0.07, index + 0.07], [end, end], color=LIMIT, linewidth=1.5, zorder=4)
                if ax is panels[0]:
                    text = f"{value * joint.scale:.0f}{'°' if joint.unit == 'deg' else ' mm'}"
                    ax.text(index + 0.09, end, text, ha="left", va="center", fontsize=6.5, color=LIMIT, zorder=5)
        ax.set_ylim(PC_LOW, PC_HIGH)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    panels[1].set_xticks(np.arange(len(joints)), [j.name for j in joints], fontsize=8)
    panels[1].set_xlim(-0.15, len(joints) - 0.85)
    panels[1].set_xlabel("each axis runs from its low limit (red, bottom) to its high limit (red, top); colour = line density, "
                         "equalized, white = no configuration passed", fontsize=7.5, color=MUTED)
    fig.tight_layout(h_pad=0.6)
    span = {j.name: _span_pct(j) for j in joints}
    numbers["range used per joint (p0.1-p99.9)"] = ", ".join(f"{k} {v:.0f} %" for k, v in span.items())
    return Card("parallel-coordinates", "jointN", "Parallel coordinates as a line-density image: every configuration the full arm held",
                "Which joint combinations make up the dataset, and which axes never leave a narrow band?",
                style.to_data_uri(fig), _src_arm(ctx) + f". Every tick's 7-axis polyline rasterized (36 × 150 px per gap), each run "
                f"weighted equally, histogram-equalized. {_limits_note()}; claw gap 0-70 mm (the open claw's span; hover-only runs "
                "mostly never read the claw, so their last gap is sparse)",
                "Inselberg's parallel coordinates (1985); density / 'saturation brushing' parallel coordinates (Wegman 1990; "
                "Heinrich & Weiskopf 2009); HiPlot", numbers, "N-DOF arm", wide=True)


def _braking_reachable(joint: Joint, axes) -> np.ndarray:
    """(q, qd) cells inside the speed limit from which the joint can brake at its acceleration limit before its range ends."""

    q, qd = np.meshgrid(axes[0].centers(), axes[1].centers(), indexing="ij")
    room = np.where(qd > 0, joint.high - q, q - joint.low)
    return (np.abs(qd) <= joint.qd_limit) & (qd ** 2 / (2 * joint.qdd_limit) <= room + 1e-12)


def conditional_speed(ctx) -> Card:
    joints = ctx["rack"] + ctx["wrist_joints"]
    fig, axes = style.figure(10.0, 2.5, ncols=len(joints), sharey=True)
    numbers = {}
    for ax, joint in zip(axes, joints):
        grid_axes = (Axis("q", joint.low, joint.high, 24, "rad", DEG, "deg"),
                     Axis("qd", -joint.qd_limit, joint.qd_limit, 16, "rad/s", DEG, "deg/s"))
        reach = _braking_reachable(joint, grid_axes)
        grid = Grid.of(np.column_stack([joint.q, joint.qd]), grid_axes, reachable=reach, run_ids=joint.run)
        per_q = grid.conditional(given=0)
        x = grid_axes[0].centers() * DEG
        width = (x[1] - x[0]) * 0.9
        ax.bar(x, np.full(len(x), 100.0), width=width, color="#f1f1f4", linewidth=0)
        ax.bar(x, 100 * np.nan_to_num(per_q), width=width, color=GROUPS[joint.group], linewidth=0)
        for limit in (joint.low, joint.high):
            ax.axvline(limit * DEG, color=LIMIT, linewidth=1)
        ax.set_title(f"{joint.label}\nmean {100 * np.nanmean(per_q):.0f} %", loc="left", fontsize=7.5)
        ax.set_xlabel("q (deg, board / Craig)", fontsize=7)
        numbers[joint.label] = f"mean {100 * np.nanmean(per_q):.0f} %, worst {100 * np.nanmin(per_q):.0f} %, " \
                               f"{grid.visited_pct():.0f} % of (q, qd)"
    axes[0].set_ylabel("% of reachable qd visited")
    axes[0].set_ylim(0, 105)
    fig.tight_layout(w_pad=0.4)
    return Card("conditional-speed-per-joint", "jointN", "Speeds covered per angle, every rig joint: % of reachable qd per q bin",
                "Are we going fast in various configurations on every joint, or only mid-range on some?",
                style.to_data_uri(fig), f"{_src_rack(ctx)}; {_src_wrist(ctx)}. 24 q × 16 qd linear cells; reachable = inside the "
                "description's speed limit and able to brake at its acceleration limit before the range ends (grey bar = 100 %)",
                "Conditional coverage (stratified by configuration); TOPP-RA's admissible velocity region", numbers, "N-DOF arm",
                wide=True)


def closed_chain_plane(ctx) -> Card:
    fb = ctx["fivebar"]
    rack = ctx["rack"]
    arm = ctx["arm"]
    j2_low, j2_high = fb["joints"]["J2"]["low"], fb["joints"]["J2"]["high"]
    c1_low, c1_high = fb["joints"]["C1"]["low"], fb["joints"]["C1"]["high"]
    axes = (Axis("J2", j2_low, j2_high, 46, "rad", DEG, "deg"), Axis("C1", c1_low, c1_high, 45, "rad", DEG, "deg"))
    j2, c1 = np.meshgrid(fb["axis_j2_deg"] / DEG, fb["axis_c1_deg"] / DEG, indexing="ij")
    mask_counts, _ = np.histogramdd(np.column_stack([j2[fb["allowed"]], c1[fb["allowed"]]]), bins=[a.edges() for a in axes])
    reach = mask_counts > 0
    samples = np.concatenate([np.column_stack([rack[0].q, rack[1].q]), arm["q"][:, :2]])
    runs = np.concatenate([rack[0].run, arm["run"] + rack[0].run.max() + 1])
    box = Grid.of(samples, axes, run_ids=runs)
    masked = Grid.of(samples, axes, reachable=reach, run_ids=runs)
    fig, (left, right) = style.figure(8.8, 3.4, ncols=2)
    edges_x, edges_y = axes[0].edges() * DEG, axes[1].edges() * DEG
    style.heatmap(left, box.runs_per_cell.astype(float), edges_x, edges_y, None, colorbar_label=None)
    left.set_title(f"as a box: {box.visited_pct():.0f} % of {box.reachable_cells} cells visited", loc="left")
    style.heatmap(right, masked.runs_per_cell.astype(float), edges_x, edges_y, reach, colorbar_label="runs through the cell",
                  figure=fig)
    right.set_title(f"against the coupled mask: {masked.visited_pct():.0f} % of {masked.reachable_cells}", loc="left")
    holes = masked.holes(top=3)
    for rank, (centre, size) in enumerate(holes, start=1):
        right.annotate(f"{rank}", (centre[0] * DEG, centre[1] * DEG), fontsize=8, fontweight="bold", color=LIMIT, ha="center",
                       va="center")
    for ax in (left, right):
        ax.set_xlabel("J2 (deg, Craig)")
        ax.set_ylabel("C1 (deg, Craig)")
        for x in (j2_low, j2_high):
            ax.axvline(x * DEG, color=LIMIT, linewidth=1)
        for y in (c1_low, c1_high):
            ax.axhline(y * DEG, color=LIMIT, linewidth=1)
    fig.tight_layout()
    estimate = 100 * masked.chao2_reachable_estimate() / masked.reachable_cells
    numbers = {"visited, box denominator": f"{box.visited_pct():.0f} %",
               "visited, coupled-mask denominator": f"{masked.visited_pct():.0f} % of {masked.reachable_cells} cells",
               "mask's share of the box": f"{100 * reach.mean():.0f} %",
               "largest hole (red 1)": f"{holes[0][1]} cells ≈ {holes[0][1] * 3.0 * 6.0:.0f} deg² near J2 {holes[0][0][0] * DEG:.0f}°, "
                                       f"C1 {holes[0][0][1] * DEG:.0f}°" if holes else "none",
               "Chao2: this kind of run tops out near": f"{min(estimate, 100):.0f} % of the mask"}
    return Card("closed-chain-joint-plane", "jointN", "Coupled joint limits: J2 × C1 against a box vs against the closed chain's mask",
                "Is the joint box the right denominator? (A closed chain can't reach most of it, so a box understates coverage.)",
                style.to_data_uri(fig), f"{_src_rack(ctx)}; {_src_arm(ctx)}. Mask: bam_fivebar reachable_2026-11-11 'allowed' "
                "(closes, no collision, folds ≥ 0.05, holdable vertical, j2 ≥ 20° keep-out) at 1°, a cell counts when any allowed "
                "pose falls in it; 46 × 45 cells (3.0° × 6.0°)",
                "Capability / reachability maps (Zacharias 2007); configuration-space obstacles (Lozano-Pérez 1983)",
                numbers, "N-DOF arm", wide=True)


def sim_vs_real(ctx) -> Card:
    fb = ctx["fivebar"]
    low_c1 = fb["joints"]["C1"]["low"]
    fig, axes = style.figure(8.8, 3.1, ncols=2)
    numbers = {}
    diverging = LinearSegmentedColormap.from_list("sim_real", [CATEGORY[0], "#c9d9ee", "#ececec", "#f7c9a8", CATEGORY[1]])
    diverging.set_bad("#ffffff")
    for ax, (column, name) in zip(axes, enumerate(("J2", "C1"))):
        joint = fb["joints"][name]
        grid_axes = (Axis(name, joint["low"], joint["high"], 30, "rad", DEG, "deg"),
                     Axis("qd", -2.4, 2.4, 24, "rad/s", DEG, "deg/s"))
        grids = {}
        for which in ("real", "sim"):
            rows = []
            for rung in ctx["ladders"]:
                data = rung[which]
                if data is None:
                    continue
                q = data["measured_motors"][:, column].astype(float)
                q = arm_data.wrap_c1(q, low_c1) if name == "C1" else q
                rows.append(np.column_stack([q, data["measured_motor_rate"][:, column]]))
            grids[which] = Grid.of(np.concatenate(rows), grid_axes)
        real, sim = grids["real"].counts, grids["sim"].counts
        share_real, share_sim = real / real.sum(), sim / sim.sum()
        either = (real > 0) | (sim > 0)
        # WHY a signed share of the larger density in [-1, 1] and white only where neither went: white keeps meaning empty
        difference = np.where(either, (share_real - share_sim) / np.maximum(np.maximum(share_real, share_sim), 1e-12), np.nan)
        mesh = ax.pcolormesh(grid_axes[0].edges() * DEG, grid_axes[1].edges() * DEG, np.ma.masked_invalid(difference).T,
                             cmap=diverging, vmin=-1, vmax=1, shading="flat")
        for limit in (joint["low"], joint["high"]):
            ax.axvline(limit * DEG, color=LIMIT, linewidth=1)
        margin = 0.04 * (joint["high"] - joint["low"]) * DEG
        ax.set_xlim(joint["low"] * DEG - margin, joint["high"] * DEG + margin)
        only_real = int(((real > 0) & (sim == 0)).sum())
        only_sim = int(((sim > 0) & (real == 0)).sum())
        both = int(((sim > 0) & (real > 0)).sum())
        ax.set_title(f"{name}: {both} cells both, {only_real} real only, {only_sim} sim only", loc="left", fontsize=8.5)
        ax.set_xlabel(f"{name} (deg, Craig)")
        ax.set_ylabel("measured qd (deg/s)")
        numbers[f"{name} (q, qd) cells"] = f"{both} both · {only_real} real only · {only_sim} sim only"
    bar = fig.colorbar(mesh, ax=axes, fraction=0.03, pad=0.02, ticks=[-1, 0, 1])
    bar.ax.set_yticklabels(["sim only", "same", "real only"], fontsize=7)
    bar.outline.set_visible(False)
    rungs = sum(r["sim"] is not None for r in ctx["ladders"])
    return Card("sim-vs-real-fivebar", "jointN", "Sim twin vs real, same tapes: where the twin's (q, qd) coverage differs",
                "Does the simulated dataset cover the same states as the real one, or does the twin move differently?",
                style.to_data_uri(fig), f"{_src_rack(ctx)}; each rung's twin replay (simulated, bam_fivebar's MuJoCo twin, "
                f"{rungs} rungs on the same tapes). Colour: (real share − sim share) / the larger, per 30 × 24 cell; white = neither. "
                "q spans the committed range (red); qd is zoomed to ±137 deg/s, the speed limits (344 / 688 deg/s) lie outside it",
                "Train/serve skew (TFDV); sim-to-real gap maps in domain-randomization papers", numbers, "N-DOF arm", wide=True)


def wrist_cube(ctx) -> Card:
    wl = ctx["wrist_limits"]
    axis_deg = wl["axis_cad_deg"]
    step = axis_deg[1] - axis_deg[0]
    edges = np.concatenate([axis_deg - step / 2, [axis_deg[-1] + step / 2]]) / DEG
    allowed = wl["allowed"]
    q_cad = np.concatenate([run["q"].astype(float) for run in ctx["wrist"]]) + np.array([wl["offset_rad"][c] for c in arm_data.WRIST_CANS])
    run_of = np.concatenate([np.full(len(run["q"]), i) for i, run in enumerate(ctx["wrist"])])
    axes = tuple(Axis(f"CAN {can}", edges[0], edges[-1], len(axis_deg), "rad", DEG, "deg") for can in arm_data.WRIST_CANS)
    grid = Grid.of(q_cad, axes, reachable=allowed, run_ids=run_of)
    visited = (grid.counts > 0) & allowed
    coarse = Grid.of(q_cad, tuple(Axis(a.name, -2.96442, 2.96442, 4, "rad") for a in axes))
    fig, panels = style.figure(9.8, 3.0, ncols=3)
    fig.subplots_adjust(left=0.07, right=0.9, wspace=0.42, bottom=0.17, top=0.9)
    pairs = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
    for ax, (a, b, collapsed) in zip(panels, pairs):
        reach_column = allowed.sum(axis=collapsed)
        visit_column = visited.sum(axis=collapsed)
        share = np.where(reach_column > 0, visit_column / np.maximum(reach_column, 1), np.nan)
        shown = np.ma.masked_where(~(reach_column > 0) | (visit_column == 0), share)
        mesh = ax.pcolormesh(edges * DEG, edges * DEG, shown.T, cmap=style.EMPTY_WHITE, vmin=0, vmax=1, shading="flat")
        blocked = np.ma.masked_where(reach_column.T > 0, np.ones_like(reach_column.T, float))
        ax.pcolormesh(edges * DEG, edges * DEG, blocked, cmap=ListedColormap([UNREACHABLE]), shading="flat")
        for k in (a, b):
            joint = wl["joints"][arm_data.WRIST_CANS[k]]
            lo, hi = (joint["low"] + wl["offset_rad"][arm_data.WRIST_CANS[k]]) * DEG, (joint["high"] + wl["offset_rad"][arm_data.WRIST_CANS[k]]) * DEG
            draw = ax.axvline if k == a else ax.axhline
            draw(lo, color=LIMIT, linewidth=0.9)
            draw(hi, color=LIMIT, linewidth=0.9)
        ax.set_xlabel(f"CAN {arm_data.WRIST_CANS[a]} (deg, CAD)")
        ax.set_ylabel(f"CAN {arm_data.WRIST_CANS[b]} (deg, CAD)")
        ax.set_title(f"collapsed over CAN {arm_data.WRIST_CANS[collapsed]}", loc="left", fontsize=8.5)
        ax.set_aspect("equal")
    bar = fig.colorbar(mesh, ax=panels, fraction=0.02, pad=0.02)
    bar.set_label("share of the column's reachable cells visited", fontsize=7, color=MUTED)
    bar.outline.set_visible(False)
    numbers = {"visited, 29³ grid (12° cells)": f"{grid.visited_pct():.1f} % of {grid.reachable_cells} allowed cells",
               "visited, the 4³ cube the gates use": f"{coarse.visited_pct():.0f} % of 64",
               "≥ 2 runs through the cell": f"{100 * ((grid.runs_per_cell >= 2) & allowed).sum() / grid.reachable_cells:.1f} %"}
    return Card("wrist-cube-collapsed", "jointN", "Three-joint reachable mask, collapsed: each pixel = share of its column visited",
                "Is the 3-joint space really covered, or does a coarse 4 × 4 × 4 cube flatter it?", style.to_data_uri(fig),
                _src_wrist(ctx) + ". Mask: bam_ur_wrist_coverage reachable_2026-11-11 'allowed' (5 mm mesh clearance + the holder / "
                "forearm keep-out) on its 29³ CAD grid; CAD = board + the file's frames (CAN 14 +3.74°, 15 −4.97°, 16 +3.92°); grey = a "
                "column with no reachable cell", "Projections of a 3-D occupancy grid (OctoMap-style), the coverage version of a "
                "maximum-intensity projection", numbers, "N-DOF arm", wide=True)


# --- section 3: Cartesian -----------------------------------------------------------------------------------------------

def strava_plane(ctx) -> Card:
    fb = ctx["fivebar"]
    pixels = fb["pixels_allowed"]
    size = fb["pixel_m"]
    paths, _ = _plane_paths(ctx)
    fine = 0.004
    low = (pixels.min(axis=0) - 1) * size
    high = (pixels.max(axis=0) + 2) * size
    x_edges = np.arange(low[0], high[0] + fine, fine)
    y_edges = np.arange(low[1], high[1] + fine, fine)
    counts = np.zeros((len(x_edges) - 1, len(y_edges) - 1))
    for _, xy in paths:
        points = _resample_path(xy, 0.001)
        if len(points):
            counts += np.histogram2d(points[:, 0], points[:, 1], [x_edges, y_edges])[0] > 0   # one path counts once per pixel
    fig, (left, right) = style.figure(8.8, 3.9, ncols=2)
    style.heatmap(left, counts, x_edges * 1000, y_edges * 1000, None, colorbar_label=None)
    left.set_title("linear (clipped at p98)", loc="left")
    style.heatmap(right, _eq_hist(counts), x_edges * 1000, y_edges * 1000, None, norm=Normalize(0.0, 1.0), colorbar_label=None)
    right.set_title("histogram-equalized (Strava / eq_hist)", loc="left")
    for ax in (left, right):
        _edges_outline(ax, pixels, size * 1000, color=INK, linewidth=0.7)
        ax.set_aspect("equal")
        ax.set_xlabel("x in the arm's plane (mm, Craig)")
        ax.set_ylabel("y (mm)")
    fig.tight_layout()
    rack_paths = sum(1 for kind, _ in paths if kind == "rack")
    numbers = {"paths drawn": f"{len(paths)} ({rack_paths} ladder rungs + {len(paths) - rack_paths} full-arm moves)",
               "pixels any path crossed": f"{int((counts > 0).sum())} of {counts.size} (4 mm)",
               "busiest pixel": f"{int(counts.max())} paths"}
    return Card("strava-plane", "cartesian", "Strava-style tool-path heatmap in the arm's plane: linear vs histogram-equalized",
                "Which tool paths are worn in, which are walked once, and where inside the outline has nothing gone?",
                style.to_data_uri(fig), f"{_src_rack(ctx)} (recorded L2 tip); {_src_arm(ctx)} (L2 tip by the five-bar's FK of the "
                "measured motors). Each path resampled every 1 mm and counted once per 4 mm pixel; black outline = the committed "
                "reachable workspace (2 cm pixels)",
                "Strava Global Heatmap (per-tile CDF normalization); datashader's eq_hist; DenseLines (Moritz & Fisher 2018)",
                numbers, "Cartesian", wide=True)


def workspace_holes(ctx) -> Card:
    fb = ctx["fivebar"]
    pixels = fb["pixels_allowed"]
    size = fb["pixel_m"]
    axes = _pixel_box(pixels, size)
    reach = _pixel_mask(pixels, axes)
    paths, _ = _plane_paths(ctx)
    resampled = [_resample_path(xy, 0.005) for _, xy in paths]
    samples = np.concatenate([points for points in resampled if len(points)])
    run_of = np.concatenate([np.full(len(points), i) for i, points in enumerate(resampled) if len(points)])
    grid = Grid.of(samples, axes, reachable=reach, run_ids=run_of)
    fig, ax = style.figure(5.4, 4.2)
    shown = Grid(axes, grid.runs_per_cell.astype(float), reach)
    style.heatmap(ax, shown.counts, axes[0].edges() * 1000, axes[1].edges() * 1000, reach, colorbar_label="paths through the pixel",
                  figure=fig)
    holes = grid.holes(top=3)
    empty = (grid.counts == 0) & reach
    labels = np.full(empty.shape, -1)
    # outline each of the three largest holes: its empty reachable pixels, found by the same flood fill as Grid.holes
    from collections import deque
    region = 0
    sizes = []
    for start in zip(*np.nonzero(empty)):
        if labels[start] >= 0:
            continue
        queue, members = deque([start]), []
        labels[start] = region
        while queue:
            cell = queue.popleft()
            members.append(cell)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (cell[0] + di, cell[1] + dj)
                if 0 <= n[0] < empty.shape[0] and 0 <= n[1] < empty.shape[1] and empty[n] and labels[n] < 0:
                    labels[n] = region
                    queue.append(n)
        sizes.append((len(members), region, members))
        region += 1
    offset = np.array([round(axes[0].low / size), round(axes[1].low / size)])
    for rank, (count, _, members) in enumerate(sorted(sizes, key=lambda s: -s[0])[:3], start=1):
        cells = np.array(members) + offset
        _edges_outline(ax, cells, size * 1000, color=LIMIT, linewidth=1.3)
        centre = (cells.mean(axis=0) + 0.5) * size * 1000
        ax.text(centre[0], centre[1], str(rank), color=LIMIT, fontsize=9, fontweight="bold", ha="center", va="center")
    _edges_outline(ax, pixels, size * 1000, color=INK, linewidth=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("x in the arm's plane (mm, Craig)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"{grid.visited_pct():.0f} % of {grid.reachable_cells} reachable 2 cm pixels visited", loc="left")
    estimate = 100 * grid.chao2_reachable_estimate() / grid.reachable_cells
    numbers = {"visited": f"{grid.visited_pct():.0f} % of {grid.reachable_cells} pixels",
               "≥ 5 paths": f"{100 * ((grid.runs_per_cell >= 5) & reach).sum() / grid.reachable_cells:.0f} %",
               "largest hole (red 1)": f"{holes[0][1]} pixels = {holes[0][1] * 4} cm²" if holes else "none",
               "Chao2 ceiling for these kinds of path": f"{min(estimate, 100):.0f} %"}
    return Card("workspace-holes", "cartesian", "Workspace occupancy vs the committed reachable mask, the largest holes outlined",
                "Where can the tool go that it never went? (White inside the outline = a hole; grey = can't reach.)",
                style.to_data_uri(fig), f"{_src_rack(ctx)}; {_src_arm(ctx)}. Denominator: bam_fivebar reachable_2026-11-11 "
                f"workspace_pixels_allowed ({len(pixels)} L2-tip pixels of 2 cm); colour = distinct paths through the pixel",
                "Occupancy grids (Elfes 1989); coverage path planning's uncovered-area maps", numbers, "Cartesian")


def max_speed(ctx) -> Card:
    fb = ctx["fivebar"]
    pixels = fb["pixels_allowed"]
    size = fb["pixel_m"]
    axes = _pixel_box(pixels, size)
    reach = _pixel_mask(pixels, axes)
    _, speeds = _plane_paths(ctx)
    xy = np.concatenate([s[0] for s in speeds])
    speed = np.concatenate([s[1] for s in speeds])
    keep = np.isfinite(speed) & np.isfinite(xy).all(axis=1)
    xy, speed = xy[keep], speed[keep]
    grid = Grid.of(xy, axes, reachable=reach)
    index = grid.cell_index(xy)
    fastest = np.zeros(grid.counts.size)
    inside = index >= 0
    np.maximum.at(fastest, index[inside], speed[inside])
    fastest = fastest.reshape(grid.counts.shape)
    fig, ax = style.figure(5.4, 4.2)
    style.heatmap(ax, np.where(grid.counts > 0, np.maximum(fastest, 1e-6), 0), axes[0].edges() * 1000, axes[1].edges() * 1000,
                  reach, cmap=style.HEAT, norm=Normalize(0.0, max(0.4, float(np.percentile(fastest[fastest > 0], 99)))),
                  colorbar_label="fastest L2-tip speed (m/s)", figure=fig)
    _edges_outline(ax, pixels, size * 1000, color=INK, linewidth=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("x in the arm's plane (mm, Craig)")
    ax.set_ylabel("y (mm)")
    fast = (fastest >= 0.2) & reach
    ax.set_title(f"≥ 0.2 m/s in {100 * fast.sum() / reach.sum():.0f} % of reachable pixels", loc="left")
    numbers = {"pixels where the tip went ≥ 0.2 m/s": f"{100 * fast.sum() / reach.sum():.0f} % of {int(reach.sum())}",
               "≥ 0.1 m/s": f"{100 * ((fastest >= 0.1) & reach).sum() / reach.sum():.0f} %",
               "fastest anywhere (p99.9)": f"{np.percentile(speed, 99.9):.2f} m/s",
               "median of the per-pixel max (visited)": f"{np.median(fastest[(grid.counts > 0) & reach]):.2f} m/s"}
    return Card("max-speed-per-cell", "cartesian", "Fastest tool speed reached in each workspace pixel (linear, white = never visited)",
                "Zach's 'going fast in various configurations', in Cartesian form: where has the tool only ever crawled?",
                style.to_data_uri(fig), f"{_src_rack(ctx)}; {_src_arm(ctx)}. Speed = |d/dt| of the L2 tip after a 7-sample "
                "moving average; ladder rungs ran 0.05-0.35 m/s along the tape, full-arm moves at the plan's pace",
                "Speed-coloured heatmaps (Strava's speed overlays); TOPP-RA's velocity envelope, mapped onto the workspace",
                numbers, "Cartesian")


def _mollweide_mesh(ax, counts: np.ndarray, rings: int, sectors: int, cmap=style.EMPTY_WHITE, vmax=None):
    lon = np.linspace(-np.pi, np.pi, sectors + 1)
    lat = np.arcsin(np.linspace(-1, 1, rings + 1))      # equal-z rings are equal-area (Archimedes); on Mollweide they stay equal-area
    shown = np.ma.masked_where(counts <= 0, counts.astype(float))
    vmax = vmax or float(np.percentile(counts[counts > 0], 98))
    return ax.pcolormesh(lon, lat, shown, cmap=cmap, vmin=1, vmax=max(vmax, 1), shading="flat")


def orientation_sphere(ctx) -> Card:
    holder = arm_data.wrist_holder_samples()
    normals = holder["normals"].astype(float)
    rings, sectors = 12, 24
    cells = sphere_cell(normals, rings, sectors)
    counts = np.bincount(cells, minlength=rings * sectors).reshape(rings, sectors)
    runs_per = np.zeros(rings * sectors)
    for run in np.unique(holder["run"]):
        runs_per[np.unique(cells[holder["run"] == run])] += 1
    fig = style.plt.figure(figsize=(6.6, 3.5))
    ax = fig.add_subplot(projection="mollweide")
    mesh = _mollweide_mesh(ax, counts, rings, sectors)
    ax.grid(color=HAIR, linewidth=0.5)
    ax.tick_params(labelsize=6.5, colors=MUTED)
    ax.set_xlabel("azimuth of the holder board's normal (deg, base_link)", fontsize=7.5)
    ax.set_ylabel("elevation (deg)", fontsize=7.5)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.03)
    bar.set_label("samples (25 Hz)", fontsize=7, color=MUTED)
    bar.outline.set_visible(False)
    visited = int((counts > 0).sum())
    ax.set_title(f"{visited} of {rings * sectors} equal-area direction cells visited", fontsize=8.5, loc="left")
    numbers = {"direction cells visited": f"{100 * visited / counts.size:.0f} % of {counts.size} (the committed facing mask "
                                          "says every direction is reachable)",
               "≥ 50 samples": f"{100 * (counts >= 50).sum() / counts.size:.0f} %",
               "seen in ≥ 3 runs": f"{100 * (runs_per >= 3).sum() / counts.size:.0f} %"}
    return Card("orientation-sphere", "cartesian", "Tool-orientation coverage on an equal-area sphere map (Mollweide)",
                "Which directions has the tool faced, and which never? (Equal-area, so every cell is the same share of the sphere.)",
                style.to_data_uri(fig), _src_wrist(ctx) + f". The holder board #4-7's normal by MuJoCo FK of the rig's bundle "
                f"(3f7ef84) at every 4th row ({len(normals) / 1000:.0f} k poses); 12 × 24 equal-area cells (coverage.sphere_cell)",
                "HEALPix / equal-area sky maps; grasp-approach coverage on the sphere (GraspNet-1B's 300 approach views)",
                numbers, "Cartesian", wide=True)


def orientations_per_voxel(ctx) -> Card:
    wl = ctx["wrist_limits"]
    holder = arm_data.wrist_holder_samples()
    size = wl["voxel_m"]
    points = holder["points"].astype(float)
    cells = sphere_cell(holder["normals"].astype(float), 12, 24)
    voxel = np.floor(points / size).astype(int)
    allowed = {tuple(v) for v in wl["voxels_allowed"].tolist()}
    by_voxel: dict[tuple, set] = {}
    for v, c in zip(map(tuple, voxel.tolist()), cells.tolist()):
        by_voxel.setdefault(v, set()).add(c)
    directions = {v: len(s) for v, s in by_voxel.items() if v in allowed}
    va = wl["voxels_allowed"]
    lo, hi = va.min(axis=0), va.max(axis=0)
    fig, (top, side) = style.figure(8.6, 3.3, ncols=2, gridspec_kw={"width_ratios": [1.0, 1.0]})
    numbers = {}
    for ax, (a, b), name in ((top, (0, 1), "top view (x, y): most directions over height"),
                             (side, (0, 2), "side view (x, z): most directions over depth")):
        shape = (hi[a] - lo[a] + 1, hi[b] - lo[b] + 1)
        best = np.zeros(shape)
        reachable = np.zeros(shape, bool)
        for v in allowed:
            reachable[v[a] - lo[a], v[b] - lo[b]] = True
        for v, n in directions.items():
            i, j = v[a] - lo[a], v[b] - lo[b]
            best[i, j] = max(best[i, j], n)
        x_edges = (np.arange(lo[a], hi[a] + 2)) * size * 1000
        y_edges = (np.arange(lo[b], hi[b] + 2)) * size * 1000
        style.heatmap(ax, best, x_edges, y_edges, reachable, vmax=float(max(best.max(), 1)),
                      colorbar_label="direction cells (of 288)" if ax is side else None, figure=fig)
        ax.set_aspect("equal")
        ax.set_title(name, loc="left", fontsize=8.5)
        ax.set_xlabel(f"{'xyz'[a]} (mm, base_link)")
        ax.set_ylabel(f"{'xyz'[b]} (mm)")
    counts = np.array(list(directions.values()))
    numbers["reachable voxels visited"] = f"{100 * len(directions) / len(allowed):.0f} % of {len(allowed)} (2 cm)"
    numbers["direction cells per visited voxel"] = f"median {np.median(counts):.0f}, max {counts.max()} of 288"
    numbers["visited voxels with ≥ 10 directions"] = f"{100 * np.mean(counts >= 10):.0f} %"
    fig.tight_layout()
    return Card("orientations-per-voxel", "cartesian", "Orientations per position voxel (reachability-map style)",
                "At each place the tool went, did it go there facing many ways, or only one?", style.to_data_uri(fig),
                _src_wrist(ctx) + ". Holder board #4-7 centre and normal by MuJoCo FK; 2 cm voxels, denominator = the committed "
                f"workspace_voxels_allowed ({len(allowed)}); directions = equal-area 12 × 24 cells of the normal",
                "Reachability / capability maps (Zacharias 2007; Vahrenkamp 2013); OctoMap voxels", numbers, "Cartesian", wide=True)


def grasp_close_points(ctx) -> Card:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

    closes, hovers, rectangles = [], [], []
    for entry in ctx["repicks"]:
        for plan in entry["plans"]:
            point = plan.get("grasp_tcp_base")
            if point is None:
                continue
            attempt = entry["attempt"].get(plan["cycle"], {})
            if attempt.get("closed"):
                closes.append((point, attempt.get("outcome") == "success"))
            elif "close" not in plan["legs"]:
                hovers.append(point)
        area, frame = entry.get("area"), next((p.get("plane_frame") for p in entry["plans"] if p.get("plane_frame")), None)
        if area and frame and area.get("x_m") and area.get("z_m"):
            rows, origin = np.asarray(frame["B2P_rows"], float), np.asarray(frame["origin_in_base_m"], float)
            corners = np.array([rows.T @ np.array([x, area["surface_y_m"], z]) + origin for x in area["x_m"] for z in area["z_m"]])
            rectangles.append((corners.min(axis=0), corners.max(axis=0)))
    hovers = np.asarray(hovers, float).reshape(-1, 3)
    points = np.asarray([p for p, _ in closes], float)
    success = np.asarray([s for _, s in closes], bool)
    every = np.concatenate([points, hovers])
    mm = 1000.0
    fig = style.plt.figure(figsize=(9.8, 3.9))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    top = fig.add_subplot(1, 2, 2)
    for target in (ax, top):
        xyz = (lambda a: (a[:, 0] * mm, a[:, 1] * mm, a[:, 2] * mm)) if target is ax else (lambda a: (a[:, 0] * mm, a[:, 1] * mm))
        options = {"depthshade": False} if target is ax else {}
        if len(hovers):
            target.scatter(*xyz(hovers), s=5, color="#aebbd0", label=f"hovered over, never closed ({len(hovers)})", **options)
        target.scatter(*xyz(points[success]), s=10, color=ACCENT, label=f"closed, picked ({success.sum()})", **options)
        target.scatter(*xyz(points[~success]), s=16, color=CATEGORY[1], marker="x", linewidths=1.0,
                       label=f"closed, failed ({(~success).sum()})", **options)
    # the denominator: every 2 cm belt cell some run offered an item in (each run's pick area, plane frame -> base)
    size = 0.02
    low = np.min([r[0] for r in rectangles], axis=0) if rectangles else every.min(axis=0) - size
    high = np.max([r[1] for r in rectangles], axis=0) if rectangles else every.max(axis=0) + size
    axes = (Axis("x", np.floor(low[0] / size) * size, np.ceil(high[0] / size) * size,
                 int(round((np.ceil(high[0] / size) - np.floor(low[0] / size)))), "m", 1000.0, "mm"),
            Axis("y", np.floor(low[1] / size) * size, np.ceil(high[1] / size) * size,
                 int(round((np.ceil(high[1] / size) - np.floor(low[1] / size)))), "m", 1000.0, "mm"))
    cx, cy = np.meshgrid(axes[0].centers(), axes[1].centers(), indexing="ij")
    offered = np.zeros(cx.shape, bool)
    for lo, hi in rectangles:
        offered |= (cx >= lo[0]) & (cx <= hi[0]) & (cy >= lo[1]) & (cy <= hi[1])
    grid = Grid.of(points[:, :2], axes, reachable=offered)
    blocked = np.ma.masked_where(offered.T, np.ones_like(offered.T, float))
    top.pcolormesh(axes[0].edges() * mm, axes[1].edges() * mm, blocked, cmap=ListedColormap([UNREACHABLE]), shading="flat", zorder=0)
    filled = np.ma.masked_where(~((grid.counts > 0) & offered).T, np.ones_like(offered.T, float))
    top.pcolormesh(axes[0].edges() * mm, axes[1].edges() * mm, filled, cmap=ListedColormap(["#dbe6f4"]), shading="flat", zorder=0)
    top.set_aspect("equal")
    top.set_xlabel("x (mm, arm base)")
    top.set_ylabel("y (mm)")
    top.set_title(f"top view: closes in {grid.visited_pct():.0f} % of the {grid.reachable_cells} offered 2 cm cells", loc="left",
                  fontsize=8.5)
    top.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    z_low, z_high = every[:, 2].min() - 0.01, every[:, 2].max() + 0.03
    box = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (z_low, z_high)]) * mm
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(box[i] - box[j]) > 1e-9) == 1:
                ax.plot(*box[[i, j]].T, color=MUTED, linewidth=0.6)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
        pane._axinfo["grid"]["color"] = HAIR
    ax.set_xlabel("x (mm)", fontsize=7, labelpad=-4)
    ax.set_ylabel("y (mm)", fontsize=7, labelpad=-4)
    ax.set_zlabel("z (mm)", fontsize=7, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-3)
    ax.view_init(elev=26, azim=-60)
    ax.set_title("3-D, base frame; box = the offered pick area", loc="left", fontsize=8.5)
    fig.subplots_adjust(left=0.0, right=0.83, wspace=0.3, bottom=0.14, top=0.9)
    span = (points.max(axis=0) - points.min(axis=0)) * mm
    numbers = {"closes": f"{len(points)} ({100 * success.mean():.0f} % picked)",
               "offered 2 cm belt cells with a close": f"{grid.visited_pct():.0f} % of {grid.reachable_cells}",
               "most closes in one cell": f"{int(grid.counts.max())} ({100 * grid.counts.max() / len(points):.0f} % of all)",
               "spread of the closes": f"{span[0]:.0f} × {span[1]:.0f} × {span[2]:.0f} mm (x, y, z)"}
    return Card("grasp-close-points", "cartesian", "Where the gripper closed: every real close in the base frame (DROID-style)",
                "Did the grasps spread over the pick area, or keep returning to one spot?", style.to_data_uri(fig),
                f"{_src_arm(ctx)}. Points = each cycle's planned grasp centre (plans/cNNN.json grasp.tcp_base_m, where the claw "
                "was driven before it closed); outcome from attempts.jsonl; offered area = the union of every run's pick_loop.json "
                "area (item ± 60 mm, plane → base), grey outside it",
                "DROID (Khazatsky et al. 2024) scene-diversity figure of gripper-close positions; Open X-Embodiment workspace scatters",
                numbers, "Cartesian", wide=True)


def cards() -> list[Card]:
    ctx = _context()
    builders = [fleet_ridgeline, limit_scorecard, corner_plot, parallel_coordinates, conditional_speed, closed_chain_plane,
                wrist_cube, sim_vs_real, strava_plane, workspace_holes, max_speed, orientation_sphere, orientations_per_voxel,
                grasp_close_points]
    return [build(ctx) for build in builders]


CONTEXT = _context
