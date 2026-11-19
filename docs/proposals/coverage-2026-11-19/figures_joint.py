"""Sections 0 (limits) and 1 (one joint): CAN 11's pendulum, the campaign's 425 real plays and the Nov 18 full-turn runs.

Every card draws real recordings (data.pendulum_runs) against the envelope pendulum.envelope() states.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm

import pendulum as pend
import style
from cards import Card
from coverage import Grid, limit_use
from style import ACCENT, CATEGORY, HAIR, INK, LIMIT, MUTED, UNREACHABLE

DEG = pend.DEG


def _context():
    env = pend.envelope()
    runs = pend.can11_runs()
    campaign, campaign_run, campaign_tau = pend.samples(runs["campaign"])
    envelope, envelope_run, envelope_tau = pend.samples(runs["envelope"])
    every = np.concatenate([campaign, envelope])
    every_run = np.concatenate([campaign_run, envelope_run + campaign_run.max() + 1])
    every_tau = np.concatenate([campaign_tau, envelope_tau])
    return {"env": env, "runs": runs, "campaign": (campaign, campaign_run), "envelope": (envelope, envelope_run),
            "every": (every, every_run), "tau": every_tau}


def _q_edges(env, axis):
    return (axis.edges() - env["bottom"]) * DEG


def _source(ctx, which="every"):
    runs = ctx["runs"]
    n = {"campaign": len(runs["campaign"]), "envelope": len(runs["envelope"])}
    rows = ctx[which][0]
    if which == "campaign":
        return f"CAN 11, rounds 0-9 (Nov 2026), {n['campaign']} real plays, {len(rows) / 1000:.0f} k samples at 100 Hz"
    if which == "envelope":
        return f"CAN 11, Nov 18 full-turn envelope runs, {n['envelope']} real plays, {len(rows) / 1000:.0f} k samples"
    return (f"CAN 11, all {n['campaign'] + n['envelope']} real plays (rounds 0-9 + Nov 18 full-turn), "
            f"{len(rows) / 1000:.0f} k samples")


def _plane(ax, fig, grid: Grid, env, x_is_q=True, cmap=style.EMPTY_WHITE, norm=None, vmax=None, label="samples per cell",
           x_scale=DEG, y_scale=DEG):
    x_axis, y_axis = grid.axes
    x_edges = _q_edges(env, x_axis) if x_is_q else x_axis.edges() * x_scale
    y_edges = y_axis.edges() * y_scale
    style.heatmap(ax, grid.counts, x_edges, y_edges, grid.reachable, cmap=cmap, norm=norm, vmax=vmax,
                  colorbar_label=label, figure=fig)
    ax.set_xlabel(f"{x_axis.name} from bottom (deg)" if x_is_q else f"{x_axis.name} ({x_axis.display_unit})")
    ax.set_ylabel(f"{y_axis.name} ({y_axis.display_unit})")


def _numbers(grid: Grid, enough: int = 50) -> dict[str, str]:
    return {"visited": f"{grid.visited_pct():.0f} % of {grid.reachable_cells} reachable cells",
            f"≥ {enough} samples": f"{grid.sufficient_pct(enough):.0f} %",
            "evenness": f"{grid.effective_pct():.0f} % (exp entropy)"}


# --- section 0: limits ------------------------------------------------------------------------------------------------

def limit_bars(ctx) -> Card:
    env = ctx["env"]
    every = ctx["every"][0]
    campaign = ctx["campaign"][0]
    # (name, physical low/high, planner low/high or None, unit, column)
    rows = [("q", env["low"] - env["bottom"], env["high"] - env["bottom"], None, "deg", 0),
            ("qd", -env["qd"], env["qd"], None, "deg/s", 1),
            ("qdd", -env["qdd"], env["qdd"], env["qdd_plan"], "deg/s²", 2),
            ("qddd", None, None, None, "deg/s³", 3)]
    fig, axes = style.figure(6.8, 3.3, nrows=4, sharex=False)
    numbers = {}
    for ax, (name, low, high, plan, unit, column) in zip(axes, rows):
        values_all = every[:, column] - (env["bottom"] if column == 0 else 0.0)
        values_campaign = campaign[:, column] - (env["bottom"] if column == 0 else 0.0)
        # WHY robust extremes: qdd and qddd are differentiated from the board's qd, so a single noisy sample would set
        # the max; p0.1 / p99.9 is where the data really went
        lowest, highest = np.percentile(values_all, [0.1, 99.9])
        if low is None:
            low, high = lowest * 1.15, highest * 1.15
        span = high - low
        edges = np.linspace(low - 0.08 * span, high + 0.08 * span, 121)
        for values, color in ((values_all, ACCENT), (values_campaign, "#9db8d9")):
            counts, _ = np.histogram(values, edges)
            height = np.sqrt(counts / max(counts.max(), 1))
            ax.fill_between(edges[:-1] * DEG, 0, height, step="post", color=color, linewidth=0)
        if rows[[r[0] for r in rows].index(name)][1] is not None:
            ax.axvspan(low * DEG, high * DEG, color=UNREACHABLE, zorder=-2)
            for limit in (low, high):
                ax.axvline(limit * DEG, color=LIMIT, linewidth=1.2)
            used = limit_use(np.clip(values_all, lowest, highest), low, high)
            note = f"{used['span_pct']:.0f} % of limit"
            numbers[f"{name}: limit used"] = note
        else:
            note = "no limit stated"
            numbers[f"{name}: limit"] = "none stated (p0.1-p99.9 shown)"
        if plan is not None:
            for limit in (-plan, plan):
                ax.axvline(limit * DEG, color=LIMIT, linewidth=1, linestyle=(0, (3, 2)))
        for extreme in (lowest, highest):
            ax.plot([extreme * DEG] * 2, [0, 1.05], color=INK, linewidth=0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, 1.1)
        ax.text(-0.01, 0.45, name, transform=ax.transAxes, ha="right", va="center", fontsize=9, color=INK, fontweight="bold")
        ax.text(1.01, 0.45, note, transform=ax.transAxes, ha="left", va="center", fontsize=7.5, color=MUTED)
        ax.set_xlabel(unit, fontsize=7, color=MUTED, labelpad=1)
    axes[0].set_title("grey box + red lines = the limit (dashed: the planner's); black ticks = p0.1 / p99.9 of the data",
                      loc="left", fontsize=8.5)
    fig.tight_layout(h_pad=0.6)
    return Card("limit-bars", "limits", "Limit bars: q, qd, qdd, qddd against their limits",
                "How much of each limit did the data use, and where did it pile up?", style.to_data_uri(fig),
                _source(ctx) + ". Light: rounds 0-9 alone; dark: everything. Height is √count. The qdd limit is the "
                "torque limit's (11 N·m over the inertia, gravity helping); dashed is the planner's 20 rad/s².",
                "TFDV / Facets Overview: one row per feature with min, max and a histogram, the schema range = the limit",
                numbers, "1-DOF joint", wide=True)


def reachable_envelope(ctx) -> Card:
    env = ctx["env"]
    full = pend.axes(env, 72, 50, 60)
    reach = pend.reachable(env, full)
    fig, (left, right) = style.figure(7.6, 2.9, ncols=2)
    q_qd = reach.any(axis=2)
    q_qdd = reach.any(axis=1)
    left.pcolormesh(_q_edges(env, full[0]), full[1].edges() * DEG, q_qd.T, cmap=style.TWO_TONE, vmin=0, vmax=1,
                    alpha=0.3, shading="flat")
    right.pcolormesh(_q_edges(env, full[0]), full[2].edges() * DEG, q_qdd.T, cmap=style.TWO_TONE, vmin=0, vmax=1,
                     alpha=0.3, shading="flat")
    right.axhline(env["qdd_plan"] * DEG, color=LIMIT, linewidth=1, linestyle=(0, (3, 2)))
    right.axhline(-env["qdd_plan"] * DEG, color=LIMIT, linewidth=1, linestyle=(0, (3, 2)))
    left.set_title("(q, qd): can it stop before the window ends?", loc="left")
    left.set_xlabel("q from bottom (deg)")
    left.set_ylabel("qd (deg/s)")
    right.set_title("(q, qdd): gravity eats the torque budget", loc="left")
    right.set_xlabel("q from bottom (deg)")
    right.set_ylabel("qdd (deg/s²)")
    fig.tight_layout()
    cells = int(reach.sum())
    return Card("reachable-envelope", "limits", "The denominator: reachable (q, qd) and (q, qdd) from the rig model",
                "Which cells can the joint physically hold, so a hole means 'didn't go' and never 'can't go'?",
                style.to_data_uri(fig),
                "CAN 11 envelope (challenge2 config: window, 5 rad/s, 11 N·m, the rig model J, G, Fc, b; braking at the "
                "planner's 20 rad/s², dashed); no samples, only the rule",
                "sim2real.schedule.joint.reachable_cells (torque at the cell centre + braking room); Zacharias 2007 "
                "capability maps", {"reachable (q, qd, qdd) cells": f"{cells} of {reach.size} ({100 * cells / reach.size:.0f} %)"},
                "1-DOF joint", wide=True)


def runs_per_cell(ctx) -> Card:
    """Episode-weighted: how many distinct runs passed through each cell, so a long dwell at rest counts once."""

    env = ctx["env"]
    rows, run_of = ctx["every"]
    grid = pend.grid(rows, env, run_of)
    fig, axes = style.figure(8.6, 3.0, ncols=2)
    _plane(axes[0], fig, grid, env, label="samples per cell")
    axes[0].set_title("samples per cell: the rest pose dominates", loc="left")
    runs = Grid(grid.axes, grid.runs_per_cell.astype(float), grid.reachable)
    _plane(axes[1], fig, runs, env, label="runs through the cell")
    axes[1].set_title("runs per cell: each run counts once", loc="left")
    fig.tight_layout()
    one_run = int(((grid.runs_per_cell == 1) & grid.reachable).sum())
    return Card("runs-per-cell", "joint1", "Samples per cell vs runs per cell (episode-weighted)",
                "Is a cell well covered, or did one long dwell fill it? Count episodes, not frames.",
                style.to_data_uri(fig), _source(ctx),
                "Episode-weighted density / DenseLines (Moritz & Fisher 2018); effective sample size for correlated samples",
                {"cells only one run visited": f"{one_run}", "median runs per visited cell":
                 f"{np.median(grid.runs_per_cell[grid.runs_per_cell > 0]):.0f}"}, "1-DOF joint", wide=True)


def acceleration_envelope(ctx) -> Card:
    """Max |qdd| the torque limit allows at each q (gravity eats the budget), against the max the data reached there."""

    env = ctx["env"]
    every = ctx["every"][0]
    q_axis = pend.axes(env, 36)[0]
    edges = q_axis.edges()
    centers = q_axis.centers()
    fig, ax = style.figure(5.2, 3.0)
    q_line = np.linspace(env["low"], env["high"], 400)
    gravity = env["gravity"] * np.sin(q_line - env["bottom"])
    up = np.minimum((env["torque_limit"] - gravity) / env["inertia"], env["qdd"])
    down = np.maximum((-env["torque_limit"] - gravity) / env["inertia"], -env["qdd"])
    ax.fill_between((q_line - env["bottom"]) * DEG, down * DEG, up * DEG, color="#e3ecf7", linewidth=0, label="allowed")
    ax.plot((q_line - env["bottom"]) * DEG, up * DEG, color=LIMIT, linewidth=1)
    ax.plot((q_line - env["bottom"]) * DEG, down * DEG, color=LIMIT, linewidth=1)
    for limit in (env["qdd_plan"], -env["qdd_plan"]):
        ax.axhline(limit * DEG, color=LIMIT, linewidth=1, linestyle=(0, (3, 2)))
    index = np.clip(np.digitize(every[:, 0], edges) - 1, 0, len(centers) - 1)
    reached_up = np.array([every[index == i, 2].max() if (index == i).any() else np.nan for i in range(len(centers))])
    reached_down = np.array([every[index == i, 2].min() if (index == i).any() else np.nan for i in range(len(centers))])
    x = (centers - env["bottom"]) * DEG
    ax.fill_between(x, reached_down * DEG, reached_up * DEG, color=ACCENT, alpha=0.75, step="mid", linewidth=0,
                    label="reached")
    ax.set_xlabel("q from bottom (deg)")
    ax.set_ylabel("qdd (deg/s²)")
    ax.legend(loc="upper right")
    allowed_span = np.interp(centers, q_line, up - down)
    fill = np.nanmean(np.clip((reached_up - reached_down) / allowed_span, 0, 1))
    return Card("acceleration-envelope", "limits", "Acceleration envelope fill: max qdd allowed vs reached, per q",
                "Did we accelerate as hard as the motor allows in every configuration?", style.to_data_uri(fig),
                _source(ctx), "TOPP-RA admissible acceleration bounds; actuator torque-speed envelopes",
                {"mean envelope fill": f"{100 * fill:.0f} %"}, "1-DOF joint")


def torque_speed(ctx) -> Card:
    env = ctx["env"]
    every = ctx["every"][0]
    tau = ctx["tau"]
    keep = np.isfinite(tau)
    fig, ax = style.figure(5.0, 3.1)
    qd_edges = np.linspace(-1.2 * env["qd"], 1.2 * env["qd"], 61)
    tau_edges = np.linspace(-1.3 * env["torque_limit"], 1.3 * env["torque_limit"], 53)
    counts, _, _ = np.histogram2d(every[keep, 1], tau[keep], [qd_edges, tau_edges])
    style.heatmap(ax, counts, qd_edges * DEG, tau_edges, None, figure=fig)
    for limit in (env["torque_limit"], -env["torque_limit"]):
        ax.axhline(limit, color=LIMIT, linewidth=1)
    for limit in (env["qd"], -env["qd"]):
        ax.axvline(limit * DEG, color=LIMIT, linewidth=1)
    ax.set_xlabel("qd (deg/s)")
    ax.set_ylabel("measured torque (N·m)")
    peak = np.nanmax(np.abs(tau[keep]))
    return Card("torque-speed", "limits", "Operating points: measured torque × speed against the torque and speed limits",
                "Did the data use the actuator's torque-speed box, or only its middle?", style.to_data_uri(fig),
                _source(ctx) + "; torque is the board's own reading",
                "Motor torque-speed (T-N) curves; datasheet operating envelopes",
                {"peak |torque|": f"{peak:.1f} N·m of {env['torque_limit']:.0f}",
                 "share of samples above half the limit": f"{100 * np.mean(np.abs(tau[keep]) > env['torque_limit'] / 2):.1f} %"},
                "1-DOF joint")


# --- section 1: one joint ---------------------------------------------------------------------------------------------

def q_qd_linear(ctx) -> Card:
    env = ctx["env"]
    fig, axes = style.figure(8.6, 3.0, ncols=2)
    numbers = {}
    for ax, which, name in ((axes[0], "campaign", "rounds 0-9"), (axes[1], "every", "+ 8 full-turn runs")):
        rows, run_of = ctx[which]
        grid = pend.grid(rows, env, run_of)
        _plane(ax, fig, grid, env)
        ax.set_title(f"{name}: {grid.visited_pct():.0f} % of reachable cells", loc="left")
        numbers[name] = f"{grid.visited_pct():.0f} % visited, {grid.sufficient_pct(50):.0f} % with ≥ 50 samples"
    fig.tight_layout()
    return Card("q-qd-linear", "joint1", "q × qd heatmap, linear bins and scale, white = empty, grey = unreachable",
                "Where in (angle, speed) has the joint been, and where are the holes?", style.to_data_uri(fig),
                _source(ctx) + ". 36 × 20 linear cells (10° × 29 deg/s)",
                "sim2real's coverage plane, redrawn per Zach's Nov 19 notes (linear, white = empty)", numbers, "1-DOF joint",
                wide=True)


def four_scales(ctx) -> Card:
    env = ctx["env"]
    rows, run_of = ctx["every"]
    grid = pend.grid(rows, env, run_of)
    fig, axes = style.figure(9.6, 2.6, ncols=4)
    vmax = float(np.percentile(grid.counts[grid.counts > 0], 98))
    _plane(axes[0], fig, grid, env, vmax=vmax, label=None)
    axes[0].set_title("linear (clipped at p98)", loc="left")
    _plane(axes[1], fig, grid, env, norm=LogNorm(vmin=1, vmax=grid.counts.max()), label=None)
    axes[1].set_title("log", loc="left")
    visited = Grid(grid.axes, (grid.counts > 0).astype(float), grid.reachable)
    _plane(axes[2], fig, visited, env, cmap=style.VISITED, vmax=1, label=None)
    axes[2].set_title("visited / not", loc="left")
    enough = Grid(grid.axes, (grid.counts >= 50).astype(float), grid.reachable)
    _plane(axes[3], fig, enough, env, cmap=style.VISITED, vmax=1, label=None)
    axes[3].set_title("≥ 50 samples / not", loc="left")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    return Card("four-scales", "joint1", "One grid, four colour scales: linear, log, visited, enough",
                "Log or linear? (Log hides piles and flatters thin cells; binary shows holes; ≥ k shows fit-worthy cells.)",
                style.to_data_uri(fig), _source(ctx),
                "datashader's how='linear'|'log'|'eq_hist' comparison; ML class-count plots on log vs linear axes",
                _numbers(grid), "1-DOF joint", wide=True)


def heat_vs_mono(ctx) -> Card:
    env = ctx["env"]
    rows, run_of = ctx["every"]
    grid = pend.grid(rows, env, run_of, dims=(0, 1))
    fig, axes = style.figure(8.6, 2.9, ncols=2)
    _plane(axes[0], fig, grid, env, cmap=style.EMPTY_WHITE)
    axes[0].set_title("white → one colour", loc="left")
    _plane(axes[1], fig, grid, env, cmap=style.HEAT)
    axes[1].set_title("white → heat (yellow, orange, red)", loc="left")
    fig.tight_layout()
    return Card("heat-vs-mono", "joint1", "White-to-one-colour vs white-to-heat, same cells",
                "Which reads faster at a glance? (Both keep empty = white.)", style.to_data_uri(fig), _source(ctx),
                "ColorBrewer sequential schemes; Strava's heat ramp", {}, "1-DOF joint", wide=True)


def qd_qdd(ctx) -> Card:
    env = ctx["env"]
    rows, run_of = ctx["every"]
    grid = pend.grid(rows, env, run_of, dims=(1, 2))
    fig, ax = style.figure(4.6, 3.2)
    _plane(ax, fig, grid, env, x_is_q=False)
    ax.set_title(f"{grid.visited_pct():.0f} % of reachable (qd, qdd) cells", loc="left")
    return Card("qd-qdd", "joint1", "qd × qdd heatmap: accelerations at each speed",
                "Are we accelerating hard at every speed, or only from rest?", style.to_data_uri(fig), _source(ctx),
                "sim2real's second coverage plane, linear", _numbers(grid), "1-DOF joint")


def conditional_curves(ctx) -> Card:
    env = ctx["env"]
    fig, (top, bottom) = style.figure(6.6, 3.4, nrows=2)
    numbers = {}
    for which, color, name in (("every", ACCENT, "all"), ("campaign", "#9db8d9", "rounds 0-9 only")):
        rows, run_of = ctx[which]
        grid_q = pend.grid(rows, env, run_of, dims=(0, 1))
        per_q = grid_q.conditional(given=0)
        x = (grid_q.axes[0].centers() - env["bottom"]) * DEG
        top.bar(x, 100 * per_q, width=(x[1] - x[0]) * 0.9, color=color, label=name)
        grid_qd = pend.grid(rows, env, run_of, dims=(1, 2))
        per_qd = grid_qd.conditional(given=0)
        x2 = grid_qd.axes[0].centers() * DEG
        bottom.bar(x2, 100 * per_qd, width=(x2[1] - x2[0]) * 0.9, color=color, label=name)
        numbers[f"{name}: speeds per configuration"] = f"mean {100 * np.nanmean(per_q):.0f} %, worst {100 * np.nanmin(per_q):.0f} %"
        numbers[f"{name}: accelerations per speed"] = f"mean {100 * np.nanmean(per_qd):.0f} %, worst {100 * np.nanmin(per_qd):.0f} %"
    top.set_ylabel("% of reachable qd")
    top.set_xlabel("q from bottom (deg)")
    top.set_title("Speeds covered in each configuration", loc="left")
    bottom.set_ylabel("% of reachable qdd")
    bottom.set_xlabel("qd (deg/s)")
    bottom.set_title("Accelerations covered at each speed", loc="left")
    top.set_ylim(0, 105)
    bottom.set_ylim(0, 105)
    top.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    return Card("conditional-coverage", "joint1", "Conditional coverage: % of reachable speeds per angle, accelerations per speed",
                "Zach's question as a number: 'are we going many velocities in different configurations?'",
                style.to_data_uri(fig), _source(ctx),
                "Conditional entropy / stratified coverage; the marginal-given-bin view of TFDV slicing", numbers,
                "1-DOF joint", wide=True)


def speed_fill(ctx) -> Card:
    env = ctx["env"]
    full = pend.axes(env, 36, 100, 50)
    reach = pend.reachable(env, full).any(axis=2)
    q_centers = full[0].centers()
    qd_centers = full[1].centers()
    allowed_up = np.array([qd_centers[r].max() if r.any() else 0 for r in reach])
    allowed_down = np.array([qd_centers[r].min() if r.any() else 0 for r in reach])
    fig, ax = style.figure(5.6, 3.0)
    x = (q_centers - env["bottom"]) * DEG
    ax.fill_between(x, allowed_down * DEG, allowed_up * DEG, color="#e3ecf7", step="mid", linewidth=0, label="reachable")
    numbers = {}
    for which, color, name in (("every", ACCENT, "all"), ("campaign", "#9db8d9", "rounds 0-9 only")):
        rows, _ = ctx[which]
        index = np.clip(np.digitize(rows[:, 0], full[0].edges()) - 1, 0, len(q_centers) - 1)
        up = np.array([rows[index == i, 1].max() if (index == i).any() else 0 for i in range(len(q_centers))])
        down = np.array([rows[index == i, 1].min() if (index == i).any() else 0 for i in range(len(q_centers))])
        up, down = np.minimum(up, allowed_up), np.maximum(down, allowed_down)
        ax.fill_between(x, down * DEG, up * DEG, color=color, step="mid", linewidth=0, label=name, alpha=0.95)
        fill = np.mean(np.clip((up - down) / np.maximum(allowed_up - allowed_down, 1e-9), 0, 1))
        numbers[f"{name}: speed envelope filled"] = f"{100 * fill:.0f} %"
    ax.plot(x, allowed_up * DEG, color=LIMIT, linewidth=1, drawstyle="steps-mid")
    ax.plot(x, allowed_down * DEG, color=LIMIT, linewidth=1, drawstyle="steps-mid")
    ax.set_xlabel("q from bottom (deg)")
    ax.set_ylabel("qd (deg/s)")
    ax.legend(loc="lower left", ncols=3)
    return Card("speed-fill", "joint1", "Speed envelope fill: fastest reached vs fastest allowed, per angle",
                "Are we going fast in every configuration, or slow at the edges?", style.to_data_uri(fig), _source(ctx),
                "Velocity-limit curves of time-optimal path parameterization (TOPP-RA's admissible velocity region)",
                numbers, "1-DOF joint")


def phase_spaghetti(ctx) -> Card:
    env = ctx["env"]
    fig, axes = style.figure(8.6, 3.0, ncols=2)
    for ax, which, name, alpha in ((axes[0], "campaign", "rounds 0-9", 0.025), (axes[1], "envelope", "Nov 18 full-turn", 0.12)):
        for run in ctx["runs"][which]:
            if run.get("qd") is None:
                continue
            ax.plot((run["q"] - env["bottom"]) * DEG, run["qd"] * DEG, color=style.ACCENT_DARK, alpha=alpha, linewidth=0.5)
        ax.axvline((env["low"] - env["bottom"]) * DEG, color=LIMIT, linewidth=1)
        ax.axvline((env["high"] - env["bottom"]) * DEG, color=LIMIT, linewidth=1)
        ax.axhline(env["qd"] * DEG, color=LIMIT, linewidth=1)
        ax.axhline(-env["qd"] * DEG, color=LIMIT, linewidth=1)
        ax.set_title(name, loc="left")
        ax.set_xlabel("q from bottom (deg)")
        ax.set_ylabel("qd (deg/s)")
        ax.set_xlim((env["low"] - env["bottom"]) * DEG - 10, (env["high"] - env["bottom"]) * DEG + 10)
        ax.set_ylim(-env["qd"] * DEG * 1.25, env["qd"] * DEG * 1.25)
    fig.tight_layout()
    return Card("phase-spaghetti", "joint1", "Phase-portrait spaghetti: every run's (q, qd) path at low opacity",
                "Strava for phase space: which paths are worn in, which are untouched?", style.to_data_uri(fig),
                _source(ctx), "Strava global heatmap; alpha-blended overplotting (Wegman 'saturation brushing')", {},
                "1-DOF joint", wide=True)


def polar_phase(ctx) -> Card:
    env = ctx["env"]
    rows, _ = ctx["every"]
    angle = rows[:, 0] - env["bottom"]
    speed = np.abs(rows[:, 1])
    fig = style.plt.figure(figsize=(4.0, 3.6))
    ax = fig.add_subplot(projection="polar")
    angle_edges = np.linspace(-np.pi, np.pi, 73)
    speed_edges = np.linspace(0, env["qd"], 11)
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    counts, _, _ = np.histogram2d(wrapped, speed, [angle_edges, speed_edges])
    vmax = np.percentile(counts[counts > 0], 98)
    mesh = ax.pcolormesh(angle_edges, speed_edges * DEG, np.ma.masked_where(counts.T <= 0, counts.T),
                         cmap=style.EMPTY_WHITE, vmin=1, vmax=vmax)
    ax.set_theta_zero_location("S")      # the bottom of the swing points down, as on the rig
    ax.set_theta_direction(-1)
    for limit in (env["low"], env["high"]):
        ax.plot([limit - env["bottom"]] * 2, [0, env["qd"] * DEG], color=LIMIT, linewidth=1.2)
    ax.set_yticks([90, 180, 270])
    ax.tick_params(labelsize=6.5, colors=MUTED)
    ax.grid(color=HAIR, linewidth=0.5)
    ax.set_title("angle around, |speed| outward", fontsize=8.5, pad=12)
    return Card("polar-phase", "joint1", "Polar coverage for a continuous joint: angle around the dial, |qd| outward",
                "For a joint that turns all the way round: which part of the circle is fast, which is slow?",
                style.to_data_uri(fig), _source(ctx), "Phase portraits on a cylinder; wind roses", {}, "1-DOF joint")


def ridgeline_rounds(ctx) -> Card:
    env = ctx["env"]
    runs = pend.can11_runs()
    groups = []
    for name in [f"round{i}" for i in range(10)]:
        chosen = [run for run in runs["campaign"] if run["round"] == name]
        if chosen:
            groups.append((name.replace("round", "round "), np.concatenate([run["q"] for run in chosen])))
    groups.append(("Nov 18 full turn", np.concatenate([run["q"] for run in runs["envelope"]])))
    fig, ax = style.figure(5.6, 4.0)
    edges = np.linspace(env["low"] - env["bottom"], env["high"] - env["bottom"], 120)
    spacing = 0.55
    for index, (name, q) in enumerate(groups):
        counts, _ = np.histogram(q - env["bottom"], edges)
        height = counts / counts.max()
        base = -index * spacing
        colour = style.plt.cm.Oranges(0.35 + 0.6 * index / len(groups))
        ax.fill_between(edges[:-1] * DEG, base, base + height, step="post", color=colour, zorder=len(groups) - index,
                        linewidth=0)
        ax.plot(edges[:-1] * DEG, base + height, drawstyle="steps-post", color="white", linewidth=0.6,
                zorder=len(groups) - index)
        ax.text(edges[-1] * DEG + 6, base + 0.05, name, fontsize=7, color=MUTED, va="bottom")
    for limit in (env["low"], env["high"]):
        ax.axvline((limit - env["bottom"]) * DEG, color=LIMIT, linewidth=1, zorder=100)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("q from bottom (deg)")
    ax.set_title("q distribution per round, oldest on top", loc="left")
    return Card("ridgeline-rounds", "joint1", "Ridgeline of q per round (TensorBoard 'histograms', offset mode)",
                "How did the dataset's spread change round by round? Did a new round reach new angles?",
                style.to_data_uri(fig), _source(ctx), "TensorBoard Histograms dashboard (offset mode); joy plots", {},
                "1-DOF joint")


def distribution_bands(ctx) -> Card:
    env = ctx["env"]
    runs = ctx["runs"]["campaign"] + ctx["runs"]["envelope"]
    percentiles = np.array([0, 7, 16, 31, 50, 69, 84, 93, 100])
    bands = np.array([np.percentile(run["qd"], percentiles) for run in runs if run.get("qd") is not None]) * DEG
    fig, ax = style.figure(6.0, 2.8)
    x = np.arange(len(bands))
    for low, high, alpha in ((0, 8, 0.18), (1, 7, 0.3), (2, 6, 0.45), (3, 5, 0.6)):
        ax.fill_between(x, bands[:, low], bands[:, high], color=ACCENT, alpha=alpha, linewidth=0)
    ax.plot(x, bands[:, 4], color=style.ACCENT_DARK, linewidth=0.6)
    for limit in (env["qd"], -env["qd"]):
        ax.axhline(limit * DEG, color=LIMIT, linewidth=1)
    ax.axvline(len(ctx["runs"]["campaign"]) - 0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(len(ctx["runs"]["campaign"]) + 2, env["qd"] * DEG * 0.85, "full-turn runs →", fontsize=7, color=MUTED)
    ax.set_xlabel("run, in recording order")
    ax.set_ylabel("qd (deg/s)")
    ax.set_title("qd percentile bands per run (min, 7, 16, 31, 50, 69, 84, 93, max)", loc="left")
    return Card("distribution-bands", "joint1", "Distribution bands per run (TensorBoard 'distributions')",
                "Which runs actually went fast, and did the spread stay put across the campaign?", style.to_data_uri(fig),
                _source(ctx), "TensorBoard Distributions dashboard (the same percentiles it draws)", {}, "1-DOF joint",
                wide=True)


def qdd_slices(ctx) -> Card:
    env = ctx["env"]
    rows, _ = ctx["every"]
    full = pend.axes(env, 24, 14, 5)
    reach = pend.reachable(env, full)
    counts, _ = np.histogramdd(rows[:, :3], bins=[axis.edges() for axis in full])
    fig, axes = style.figure(10.0, 2.3, ncols=5, sharey=True)
    vmax = np.percentile(counts[counts > 0], 98)
    for index, ax in enumerate(axes):
        style.heatmap(ax, counts[:, :, index], _q_edges(env, full[0]), full[1].edges() * DEG, reach[:, :, index],
                      vmax=vmax, colorbar_label=None)
        low, high = full[2].edges()[index:index + 2] * DEG
        visited = ((counts[:, :, index] > 0) & reach[:, :, index]).sum() / max(reach[:, :, index].sum(), 1)
        ax.set_title(f"qdd {low:.0f}…{high:.0f}: {100 * visited:.0f} %", loc="left", fontsize=8)
        ax.set_xlabel("q (deg)")
    axes[0].set_ylabel("qd (deg/s)")
    fig.tight_layout()
    return Card("qdd-slices", "joint1", "The 3-D grid unfolded: q × qd for each acceleration slab",
                "Which (angle, speed) cells did we visit while braking hard, cruising, accelerating hard?",
                style.to_data_uri(fig), _source(ctx), "Small multiples (Tufte); slicing a 3-D occupancy grid", {},
                "1-DOF joint", wide=True)


def plan_vs_real(ctx) -> Card:
    env = ctx["env"]
    runs = ctx["runs"]["envelope"]
    real_rows, real_run, _ = pend.samples(runs)
    plan_rows, plan_run, _ = pend.samples(runs, target=True)
    real = pend.grid(real_rows, env, real_run)
    plan = pend.grid(plan_rows, env, plan_run)
    both = (real.counts > 0) & (plan.counts > 0)
    plan_only = (plan.counts > 0) & ~(real.counts > 0)
    real_only = (real.counts > 0) & ~(plan.counts > 0)
    category = np.zeros(real.counts.shape)
    category[both], category[plan_only], category[real_only] = 1, 2, 3
    from matplotlib.colors import ListedColormap
    fig, ax = style.figure(5.0, 3.1)
    x_edges = _q_edges(env, real.axes[0])
    y_edges = real.axes[1].edges() * DEG
    ax.pcolormesh(x_edges, y_edges, category.T, cmap=ListedColormap(["#ffffff", "#9db8d9", "#e8590c", "#2b9348"]),
                  vmin=-0.5, vmax=3.5, shading="flat")
    blocked = np.ma.masked_where(real.reachable.T, np.ones_like(real.reachable.T, float))
    ax.pcolormesh(x_edges, y_edges, blocked, cmap=ListedColormap([UNREACHABLE]), shading="flat")
    ax.set_xlabel("q from bottom (deg)")
    ax.set_ylabel("qd (deg/s)")
    handles = [style.plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#9db8d9", "#e8590c", "#2b9348")]
    ax.legend(handles, ["plan and real", "plan only (lost)", "real only (overshoot)"], loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncols=3)
    return Card("plan-vs-real", "joint1", "Planned vs achieved coverage: cells the plan visited that the joint didn't",
                "Did the robot actually go where the plan sent it? (Coverage of the target is not coverage of the data.)",
                style.to_data_uri(fig), _source(ctx, "envelope") + "; plan = q_target / qd_target on the same ticks",
                "Train/serve skew in TFDV: the same statistics on two views of one dataset",
                {"plan cells": f"{int((plan.counts > 0).sum())}", "lost": f"{int(plan_only.sum())}",
                 "overshoot": f"{int(real_only.sum())}"}, "1-DOF joint")


def cards() -> list[Card]:
    ctx = _context()
    builders = [limit_bars, reachable_envelope, acceleration_envelope, torque_speed, q_qd_linear, runs_per_cell, four_scales,
                heat_vs_mono, qd_qdd, conditional_curves, speed_fill, phase_spaghetti, polar_phase, ridgeline_rounds,
                distribution_bands, qdd_slices, plan_vs_real]
    return [build(ctx) for build in builders]


CONTEXT = _context
