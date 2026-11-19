"""Section 5: the dataset as a whole, the way the ML community looks at one (class balance, growth, splits, value).

All on the real 1-DOF rack campaign: CAN 11's plays against its envelope, and the six stations as a fleet.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

import data
import pendulum as pend
import style
from cards import Card
from coverage import Grid, excitation, gap_distances, largest_holes, rarefaction
from figures_joint import CONTEXT, _source
from style import ACCENT, CATEGORY, INK, LIMIT, MUTED

DEG = pend.DEG


# WHY a coarser 3-D grid here (18 x 10 x 12 = 20° x 58°/s x 19 rad/s²) than on the 2-D planes: in three dimensions a fine
# grid has more cells than 423 plays can ever fill, so every % reads ~0; a visited % only means something with its grid.
from core_types import GRID3  # noqa: E402


def _grid3(rows, run_of, env):
    full = pend.axes(env, **GRID3)
    return Grid.of(rows[:, :3], full, reachable=pend.reachable(env, full), run_ids=run_of)


def rarefaction_card(ctx) -> Card:
    env = ctx["env"]
    fig, ax = style.figure(5.6, 3.2)
    numbers = {}
    for which, color, name in (("campaign", "#9db8d9", "rounds 0-9"), ("envelope", ACCENT, "full-turn runs"),
                               ("every", style.ACCENT_DARK, "all")):
        rows, run_of = ctx[which]
        grid = _grid3(rows, run_of, env)
        k, mean, band = rarefaction(grid)
        ax.fill_between(k, band[0], band[1], color=color, alpha=0.2, linewidth=0)
        ax.plot(k, mean, color=color, linewidth=1.6, label=name)
        estimate = 100 * grid.chao2_reachable_estimate() / grid.reachable_cells
        ax.axhline(min(estimate, 100), color=color, linewidth=0.8, linestyle=(0, (2, 2)))
        numbers[name] = f"{mean[-1]:.0f} % after {k[-1]} runs; Chao2 says this kind of run tops out near {min(estimate, 100):.0f} %"
    ax.axhline(100, color=LIMIT, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("runs recorded (random order, log)")
    ax.set_ylabel("% of reachable (q, qd, qdd) cells")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")
    ax.set_title("dashed: where more of the same kind of run would top out", loc="left", fontsize=8)
    return Card("rarefaction", "dataset", "Coverage accumulation (rarefaction) curve with a Chao2 ceiling",
                "Will recording more of the same buy coverage, or do we need a different kind of run?",
                style.to_data_uri(fig), _source(ctx) + "; 18 × 10 × 12 (q, qd, qdd) cells",
                "Ecology's species-accumulation curves and the Chao2 unseen-species estimator; ML data scaling curves",
                numbers, "dataset")


def imbalance(ctx) -> Card:
    env = ctx["env"]
    rows, run_of = ctx["every"]
    grid = _grid3(rows, run_of, env)
    counts = np.sort(grid.counts[grid.reachable].ravel())
    lorenz = np.concatenate([[0], np.cumsum(counts) / counts.sum()])
    cells = np.linspace(0, 1, len(lorenz))
    gini = 1 - 2 * np.trapezoid(lorenz, cells)
    fleet = [run for run in data.pendulum_runs()]
    kinds = Counter(run["kind"].replace("_multi", "") for run in fleet)
    fig, (left, right) = style.figure(8.4, 3.0, ncols=2, gridspec_kw={"width_ratios": [1.1, 1]})
    names, values = zip(*kinds.most_common(8))
    left.barh(range(len(names)), values, color=ACCENT)
    left.set_yticks(range(len(names)), names)
    left.invert_yaxis()
    left.set_xlabel("plays (all six stations)")
    left.set_title("runs per kind: the classes", loc="left")
    right.plot(cells, lorenz, color=ACCENT, linewidth=1.6)
    right.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=":")
    right.fill_between(cells, lorenz, cells, color=ACCENT, alpha=0.12)
    right.set_xlabel("share of reachable cells (emptiest first)")
    right.set_ylabel("share of samples")
    right.set_title(f"CAN 11 samples across cells: Gini {gini:.2f}", loc="left")
    fig.tight_layout()
    return Card("imbalance", "dataset", "Class imbalance: runs per kind, and a Lorenz curve of samples over cells",
                "Is the data piled into a few kinds and a few cells? (Gini 0 = even, 1 = all in one cell.)",
                style.to_data_uri(fig), f"All six stations, {len(fleet)} board-plays; Lorenz on CAN 11's 3-D grid",
                "Class-imbalance ratio and Gini (Buda et al. 2018); Lorenz curves",
                {"Gini over cells": f"{gini:.2f}", "imbalance ratio (max / median cell)": f"{grid.imbalance_ratio():.0f}×",
                 "effective cells (exp entropy)": f"{grid.effective_pct():.0f} % of reachable"}, "dataset", wide=True)


def novelty(ctx) -> Card:
    """Each run's unique contribution: the reachable cells only it visited. Zero = the dataset would not miss it."""

    env = ctx["env"]
    rows, run_of = ctx["every"]
    grid = _grid3(rows, run_of, env)
    unique = np.array([int((grid.runs_per_cell.ravel()[cells] == 1).sum()) for cells in grid.run_cells])
    visited = np.array([len(cells) for cells in grid.run_cells])
    runs = ctx["runs"]["campaign"] + ctx["runs"]["envelope"]
    duration = np.array([run["t"][-1] - run["t"][0] for run in runs])[: len(unique)]
    rounds = [run["round"] for run in runs][: len(unique)]
    order = sorted(set(rounds), key=lambda name: (name.startswith("envelope"), name))
    fig, ax = style.figure(5.6, 3.2)
    for index, name in enumerate(order):
        mask = np.array([r == name for r in rounds])
        color = ACCENT if name.startswith("envelope") else style.plt.cm.Greys(0.35 + 0.5 * index / len(order))
        ax.scatter(duration[mask], unique[mask] + np.random.default_rng(index).uniform(-0.25, 0.25, mask.sum()), s=9,
                   color=color, alpha=0.8, linewidth=0, label="full-turn runs" if name == order[-1] else None)
    ax.set_xlabel("run duration (s)")
    ax.set_ylabel("cells only this run visited")
    ax.set_xscale("log")
    ax.legend(loc="upper left")
    redundant = int((unique == 0).sum())
    return Card("novelty-per-run", "dataset", "Per-run novelty: the cells only that run visited (data valuation)",
                "Which runs carry the coverage, and which could be dropped without losing a cell?",
                style.to_data_uri(fig), _source(ctx) + "; grey = rounds 0-9 (darker = later), blue = full-turn",
                "Data valuation (KNN-Shapley in Cleanlab), FiftyOne uniqueness, coreset selection",
                {"runs adding no unique cell": f"{redundant} of {len(unique)}",
                 "median cells per run": f"{np.median(visited):.0f}"}, "dataset")


def held_out_support(ctx) -> Card:
    """Round 0's CAN 13 ladder: rungs 0.2-0.9 trained, rung 1.0 held out. Is the held-out set inside the train coverage?"""

    runs = [run for run in data.pendulum_runs() if run["round"] == "round0" and run["station"] == "can13"]
    train = [run for run in runs if "/can13_r1.0" not in run["file"]]
    held = [run for run in runs if "/can13_r1.0_held_out" in run["file"]]
    train_rows, train_run, _ = pend.samples(train)
    held_rows, _, _ = pend.samples(held)
    q_edges = np.linspace(min(train_rows[:, 0].min(), held_rows[:, 0].min()), max(train_rows[:, 0].max(), held_rows[:, 0].max()), 31)
    qd_limit = max(np.abs(train_rows[:, 1]).max(), np.abs(held_rows[:, 1]).max()) * 1.05
    qd_edges = np.linspace(-qd_limit, qd_limit, 25)
    train_counts, _, _ = np.histogram2d(train_rows[:, 0], train_rows[:, 1], [q_edges, qd_edges])
    q_index = np.clip(np.digitize(held_rows[:, 0], q_edges) - 1, 0, len(q_edges) - 2)
    qd_index = np.clip(np.digitize(held_rows[:, 1], qd_edges) - 1, 0, len(qd_edges) - 2)
    inside = train_counts[q_index, qd_index] > 0
    fig, ax = style.figure(5.0, 3.2)
    style.heatmap(ax, train_counts, q_edges * DEG, qd_edges * DEG, None, colorbar_label="train samples", figure=fig)
    ax.scatter(held_rows[inside, 0] * DEG, held_rows[inside, 1] * DEG, s=2, color="#2b9348", alpha=0.5, linewidth=0,
               label="held-out, inside train cells")
    ax.scatter(held_rows[~inside, 0] * DEG, held_rows[~inside, 1] * DEG, s=2, color=LIMIT, alpha=0.7, linewidth=0,
               label="held-out, outside: extrapolation")
    ax.set_xlabel("q (deg)")
    ax.set_ylabel("qd (deg/s)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, markerscale=4)
    return Card("held-out-support", "dataset", "Held-out inside the training coverage? (train/test support overlap)",
                "When a model is scored on held-out runs, is it interpolating or extrapolating?", style.to_data_uri(fig),
                f"CAN 13, round 0: {len(train)} train plays (rungs 0.2-0.9) vs {len(held)} held-out plays (rung 1.0)",
                "TFDV train/serving skew; FiftyOne leaky splits; kNN out-of-distribution scores (Sun et al. 2022)",
                {"held-out samples outside train cells": f"{100 * (~inside).mean():.0f} %"}, "dataset")


def excitation_card(ctx) -> Card:
    """Which model terms does the data pin down? Fit the rig model to the MEASURED torque per round and report each
    parameter's relative standard deviation (Gautier, Janot & Vandanjon 2008): large %σ = that term is not excited."""

    env = ctx["env"]
    runs = pend.can11_runs()
    groups = [(f"r{i}", [run for run in runs["campaign"] if run["round"] == f"round{i}"]) for i in (0, 3, 6, 9)]
    groups += [("full turn", runs["envelope"]), ("all", runs["campaign"] + runs["envelope"])]
    terms = ["inertia J", "gravity sin", "gravity cos", "Coulomb", "viscous"]
    table, conditions = [], []
    for name, chosen in groups:
        rows, _, tau = pend.samples(chosen)
        keep = np.isfinite(tau)
        q, qd, qdd = rows[keep, 0] - env["bottom"], rows[keep, 1], rows[keep, 2]
        # tau = J qdd + G sin q + G' cos q + Fc tanh(qd / 0.05) + b qd: the rig model, linear in its five parameters
        regressor = np.column_stack([qdd, np.sin(q), np.cos(q), np.tanh(qd / 0.05), qd])
        theta, residual, *_ = np.linalg.lstsq(regressor, tau[keep], rcond=None)
        noise = float(np.sum((regressor @ theta - tau[keep]) ** 2) / (len(tau[keep]) - len(theta)))
        covariance = noise * np.linalg.inv(regressor.T @ regressor)
        table.append(100 * np.sqrt(np.diag(covariance)) / np.abs(theta))
        conditions.append(excitation(regressor)["condition"])
    table = np.array(table)
    fig, ax = style.figure(6.2, 3.0)
    width = 0.8 / len(groups)
    for index, (name, _) in enumerate(groups):
        color = ACCENT if name in ("full turn", "all") else style.plt.cm.Greys(0.3 + 0.12 * index)
        ax.bar(np.arange(len(terms)) + (index - len(groups) / 2 + 0.5) * width, table[index], width, color=color, label=name)
    ax.set_yscale("log")
    ax.axhline(10, color=LIMIT, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(len(terms) - 0.5, 11, "10 %: identified", ha="right", fontsize=7, color=LIMIT)
    ax.set_xticks(range(len(terms)), terms)
    ax.set_ylabel("relative std dev %σ (log)")
    ax.legend(ncols=3, loc="upper left", fontsize=7)
    numbers = {f"{name}: worst term": f"{terms[int(np.argmax(row))]} {row.max():.1f} %" for (name, _), row in zip(groups, table)
               if name in ("r9", "full turn", "all")}
    numbers["condition number (unit columns)"] = f"{min(conditions):.1f} – {max(conditions):.1f}"
    return Card("excitation", "dataset", "Excitation: how well does the data pin each model parameter? (%σ per term)",
                "The analytic 'many speeds in many configurations': which physics does this data actually identify?",
                style.to_data_uri(fig), _source(ctx) + "; fit to the board's measured torque",
                "Relative standard deviation of identified parameters (Gautier, Janot & Vandanjon 2008); persistent "
                "excitation, Swevers et al. 1997", numbers, "dataset")


def dispersion_card(ctx) -> Card:
    env = ctx["env"]
    full = pend.axes(env)

    def inside(points):
        q, qd, qdd = points.T
        torque_ok = np.abs(pend.model_torque(env, q, qd, qdd)) <= env["torque_limit"]
        room = np.where(qd > 0, env["high"] - q, q - env["low"])
        return torque_ok & (qd ** 2 / (2 * env["qdd_plan"]) <= room)

    fig, ax = style.figure(5.4, 3.1)
    numbers = {}
    for which, color, name in (("campaign", "#9db8d9", "rounds 0-9"), ("envelope", ACCENT, "full-turn runs"),
                               ("every", style.ACCENT_DARK, "all")):
        rows, _ = ctx[which]
        gaps = np.sort(gap_distances(rows[:, :3], full, inside, probes=8000, max_points=12000))
        ax.plot(gaps, np.linspace(0, 100, len(gaps)), color=color, linewidth=1.6, label=name)
        numbers[name] = f"90 % of reachable within {np.percentile(gaps, 90):.2f}; largest hole {gaps[-1]:.2f}"
    ax.set_xlabel("distance to the nearest sample (limit spans = 1)")
    ax.set_ylabel("% of the reachable space")
    ax.legend(loc="lower right")
    return Card("dispersion", "dataset", "Gap distances: how far is any reachable state from the nearest recorded one?",
                "Bin-free coverage: '90 % of what the joint can do is within r of something we recorded.'",
                style.to_data_uri(fig), _source(ctx) + "; (q, qd, qdd) each scaled to its limit span",
                "Dispersion (LaValle, motion planning); k-center coverage radius (Sener & Savarese 2018); Naeem 2020 coverage",
                numbers, "dataset")


def holes_card(ctx) -> Card:
    env = ctx["env"]
    rows, run_of = ctx["every"]
    full = pend.axes(env)

    def inside(points):
        q, qd, qdd = points.T
        torque_ok = np.abs(pend.model_torque(env, q, qd, qdd)) <= env["torque_limit"]
        room = np.where(qd > 0, env["high"] - q, q - env["low"])
        return torque_ok & (qd ** 2 / (2 * env["qdd_plan"]) <= room)

    holes = largest_holes(rows[:, :3], full, inside, count=6, probes=12000)
    grid = pend.grid(rows, env, run_of)
    fig, ax = style.figure(5.6, 3.1)
    q_edges = (grid.axes[0].edges() - env["bottom"]) * DEG
    style.heatmap(ax, grid.counts, q_edges, grid.axes[1].edges() * DEG, grid.reachable, colorbar_label=None)
    numbers = {}
    from matplotlib.patches import Ellipse
    for rank, (centre, radius) in enumerate(holes, 1):
        q, qd, qdd = centre
        x, y = (q - env["bottom"]) * DEG, qd * DEG
        ax.add_patch(Ellipse((x, y), 2 * radius * (full[0].high - full[0].low) * DEG, 2 * radius * (full[1].high - full[1].low) * DEG,
                             facecolor="none", edgecolor=LIMIT, linewidth=1.1))
        ax.text(x, y, f"{rank}", fontsize=7.5, color=LIMIT, ha="center", va="center", fontweight="bold")
        numbers[f"hole {rank}"] = f"q {x:.0f}°, qd {y:.0f}°/s, qdd {qdd * DEG:.0f}°/s², radius {radius:.2f}"
    ax.set_xlabel("q from bottom (deg)")
    ax.set_ylabel("qd (deg/s)")
    ax.set_title("the six biggest empty balls in (q, qd, qdd), drawn on the q × qd plane", loc="left", fontsize=8)
    return Card("holes", "dataset", "Hole finder: the biggest empty reachable regions, ranked: what to record next",
                "Where exactly should the next run go? (Each hole lives in 3-D; its qdd is in the list.)",
                style.to_data_uri(fig), _source(ctx) + "; radius in limit spans",
                "k-center / farthest-first traversal as a collection plan (Gonzalez 1985; Sener & Savarese 2018)",
                numbers, "dataset")


def embedding_atlas(ctx) -> Card:
    """One point per 2 s window of every play on all six stations, by what the window did (PCA of its statistics)."""

    features, stations = [], []
    for run in data.pendulum_runs():
        if run.get("qd") is None or len(run["q"]) < 250:
            continue
        rows, _, _ = pend.samples([run])
        for start in range(0, len(rows) - 200, 200):
            window = rows[start:start + 200]
            q, qd, qdd = window[:, 0], window[:, 1], window[:, 2]
            features.append([np.ptp(q), np.std(qd), np.abs(qd).max(), np.std(qdd), np.abs(qdd).max(),
                             np.mean(np.abs(qd) < 0.05), np.mean(np.sign(qd[1:]) != np.sign(qd[:-1]))])
            stations.append(run["station"])
    features = np.log1p(np.abs(np.array(features)))
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    _, _, components = np.linalg.svd(features, full_matrices=False)
    projected = features @ components[:2].T
    fig, ax = style.figure(5.4, 3.6)
    order = sorted(set(stations), key=lambda name: int(name[3:]))
    for index, station in enumerate(order):
        mask = np.array([s == station for s in stations])
        ax.scatter(projected[mask, 0], projected[mask, 1], s=3, alpha=0.35, color=CATEGORY[index], linewidth=0, label=station)
    ax.set_xlabel("PC 1 (amplitude and speed)")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper right", markerscale=4, ncols=2)
    return Card("embedding-atlas", "dataset", "Behaviour atlas: every 2 s window of every play, embedded",
                "Which kinds of motion does the dataset hold, which station did them, and where are the empty regions?",
                style.to_data_uri(fig),
                f"All six stations, {len(stations)} windows of 2 s; 7 window statistics, log-scaled, PCA",
                "TensorBoard Embedding Projector; FiftyOne embeddings; Activation Atlas (Carter et al. 2019)",
                {"windows": f"{len(stations)}"}, "dataset")


def fleet_balance(ctx) -> Card:
    """Samples and plays per station and round: the stations as classes."""

    runs = data.pendulum_runs()
    stations = sorted({run["station"] for run in runs}, key=lambda name: int(name[3:]))
    rounds = [f"round{i}" for i in range(10)] + ["envelope"]
    matrix = np.zeros((len(stations), len(rounds)))
    for run in runs:
        column = rounds.index("envelope" if run["round"].startswith("envelope") else run["round"])
        matrix[stations.index(run["station"]), column] += (run["t"][-1] - run["t"][0]) / 60.0
    fig, ax = style.figure(6.4, 2.6)
    style.heatmap(ax, matrix.T, np.arange(len(rounds) + 1), np.arange(len(stations) + 1), None,
                  colorbar_label="minutes recorded", figure=fig)
    for i in range(len(rounds)):
        for j in range(len(stations)):
            if matrix[j, i] > 0:
                ax.text(i + 0.5, j + 0.5, f"{matrix[j, i]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if matrix[j, i] > matrix.max() * 0.5 else INK)
    ax.set_xticks(np.arange(len(rounds)) + 0.5, [r.replace("round", "r") for r in rounds], fontsize=7)
    ax.set_yticks(np.arange(len(stations)) + 0.5, stations, fontsize=7)
    return Card("fleet-balance", "dataset", "Minutes recorded per station per round (the co-occurrence matrix)",
                "Is one station or round carrying the dataset? Which combinations were never recorded?",
                style.to_data_uri(fig), f"All six stations, {len(runs)} board-plays",
                "Know Your Data's relations matrix; Open X-Embodiment's per-embodiment composition bars",
                {"total": f"{matrix.sum() / 60:.1f} h"}, "dataset", wide=True)


def cards() -> list[Card]:
    ctx = CONTEXT()
    builders = [rarefaction_card, holes_card, novelty, imbalance, fleet_balance, held_out_support, excitation_card,
                dispersion_card, embedding_atlas]
    return [build(ctx) for build in builders]
