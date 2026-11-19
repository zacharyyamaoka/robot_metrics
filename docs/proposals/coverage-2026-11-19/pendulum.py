"""CAN 11's pendulum as a coverage problem: its limits, its reachable (q, qd, qdd) cells and its recorded samples.

The envelope is the one the Nov 18 full-turn challenge proved (envelope_1118/challenge2/config.json): the probe-proven
window less 0.1 rad, 5 rad/s, 20 rad/s^2, an 11 N·m torque limit, and the rig model (inertia, gravity, Coulomb) the
controller's feed-forward carries. A cell is reachable when the model's torque at its centre fits the limit and the
joint can brake from that speed before the window ends (sim2real.schedule.joint.reachable_cells' rule, linear bins).
"""

from __future__ import annotations

import numpy as np

import data
from coverage import Axis, Grid

DEG = 180.0 / np.pi
SMOOTH_SAMPLES = 7          # 70 ms at the rack's 100 Hz: enough to differentiate qd twice without inventing jerk


def envelope() -> dict:
    config = data.envelope_config("challenge2")
    model = config["model"]
    env = {"low": config["room_rad"][0], "high": config["room_rad"][1], "bottom": config["bottom_rad"],
           "qd": config["max_velocity_rad_s"], "qdd_plan": config["max_acceleration_rad_s2"],
           "torque_limit": config["torque_limit_nm"], "inertia": model["model_inertia_kg_m2"],
           "gravity": model["model_gravity_nm"], "coulomb": model["model_coulomb_nm"],
           "viscous": config["controller"]["ff_viscous_nm_per_rad_s"]}
    # WHY the qdd axis spans the torque limit, not the planner's 20 rad/s^2: coverage is relative to what the joint CAN
    # do (Zach, 2026-11-19); the steps in rounds 0-9 really did reach ~100 rad/s^2. The planner's limit is drawn dashed.
    env["qdd"] = (env["torque_limit"] + env["gravity"]) / env["inertia"]
    env["qdd_at_bottom"] = env["torque_limit"] / env["inertia"]
    return env


def model_torque(env: dict, q, qd, qdd):
    """The rig model's demand: J qdd + G sin(q - bottom) + Fc tanh(qd / 0.05) + b qd."""

    return (env["inertia"] * qdd + env["gravity"] * np.sin(q - env["bottom"])
            + env["coulomb"] * np.tanh(qd / 0.05) + env["viscous"] * qd)


def axes(env: dict, q_bins: int = 36, qd_bins: int = 20, qdd_bins: int = 24) -> tuple[Axis, Axis, Axis]:
    return (Axis("q", env["low"], env["high"], q_bins, "rad", DEG, "deg"),
            Axis("qd", -env["qd"], env["qd"], qd_bins, "rad/s", DEG, "deg/s"),
            Axis("qdd", -env["qdd"], env["qdd"], qdd_bins, "rad/s²", DEG, "deg/s²"))


def reachable(env: dict, grid_axes) -> np.ndarray:
    q, qd, qdd = np.meshgrid(*(axis.centers() for axis in grid_axes), indexing="ij")
    torque_ok = np.abs(model_torque(env, q, qd, qdd)) <= env["torque_limit"]
    room = np.where(qd > 0, env["high"] - q, q - env["low"])
    braking_ok = qd ** 2 / (2 * env["qdd_plan"]) <= room + 1e-12     # stop at the planner's deceleration
    return torque_ok & braking_ok


def derivatives(t: np.ndarray, q: np.ndarray, qd: np.ndarray | None):
    """qd (measured when present), qdd and qddd by differentiating a moving average: the board reports q and qd only."""

    kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES
    t = np.asarray(t, float)
    if qd is None:
        qd = np.gradient(np.convolve(q, kernel, mode="same"), t)
    qdd = np.gradient(np.convolve(qd, kernel, mode="same"), t)
    qddd = np.gradient(np.convolve(qdd, kernel, mode="same"), t)
    edge = SMOOTH_SAMPLES
    keep = np.zeros(len(t), bool)
    keep[edge:-edge] = True                    # the moving average is biased at the ends
    return qd[keep], qdd[keep], qddd[keep], keep


def samples(runs: list[dict], target: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, 4) array of q, qd, qdd, qddd, the run index of each sample, and tau (NaN where not measured)."""

    rows, run_of, torque = [], [], []
    for index, run in enumerate(runs):
        q = run["q_target"] if target else run["q"]
        qd = run["qd_target"] if target else run.get("qd")
        if q is None or len(q) < 4 * SMOOTH_SAMPLES:
            continue
        qd_, qdd, qddd, keep = derivatives(run["t"], q, qd)
        rows.append(np.column_stack([q[keep], qd_, qdd, qddd]))
        run_of.append(np.full(keep.sum(), index))
        tau = run.get("tau")
        torque.append(tau[keep] if tau is not None and not target else np.full(keep.sum(), np.nan))
    return np.concatenate(rows), np.concatenate(run_of), np.concatenate(torque)


def can11_runs() -> dict[str, list[dict]]:
    """CAN 11's plays split the way the story goes: the rounds 0-9 campaign, and the Nov 18 full-turn envelope runs."""

    runs = [run for run in data.pendulum_runs() if run["station"] == "can11"]
    return {"campaign": [run for run in runs if run["round"].startswith("round")],
            "envelope": [run for run in runs if run["round"].startswith("envelope/challenge")]}


def grid(sample_rows: np.ndarray, env: dict, run_of=None, dims=(0, 1), **bins) -> Grid:
    full = axes(env, **bins)
    chosen = tuple(full[d] for d in dims)
    reach = reachable(env, full).any(axis=tuple(d for d in range(3) if d not in dims))
    return Grid.of(sample_rows[:, list(dims)], chosen, reachable=reach, run_ids=run_of)
