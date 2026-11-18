"""The 1-DOF rack's round 8 as runs of robot_msgs streams: 252 real plays (6 stations x 6 rungs x 5 repeats + 2 held
out), and the twin's answer to each.

    real_runs()            -> tuple[Run, ...]   streams "plan" (the ticks), "state" (the replies), "target" (aligned)
    sim_runs(twins)        -> tuple[Run, ...]   the twin's answers; ``twins`` names which twin answered each station
    stations()             -> {station: {"tip_length_m", "omega_max_rad_s"}}

A port of the metrics-API proposals' fixture (robot_models f0daa34, docs/proposals/metrics-api-2026-11-17/fixture.py),
the round's own rules:
- **The episode is the tape.** The runner's prelude (its move to A from wherever the joint rested) is dropped: rows on
  a plan tick at or after ``prelude_s``.
- **The target is the plan at each reply's own instant**, made by the explicit alignment step
  (``robot_metrics.align(plan, replies, rule="hermite", tolerance_s=one tick)``), not by the reader.
- **Paired runs** (one file, two boards, ``<field>_can<id>`` columns) are one play per board, each read as its own
  single-board run: its rows are the ones that board answered, its tape starts at ``tape_start_s``.
- **Rung and kind** come from the round's plan index (``plans/index.json``), matched by station and run name.
- **A sim answer** is the memo file whose station, twin, controller hash and plan step match the play.

WHY header stamps of 0 with the wall clock in ``header.meta``: robot_msgs aligns on absolute times (stamp + t), and at
today's epoch float64 resolves 0.24 us, which moves an interpolated target by up to ~1e-6 rad on a fast rung. The
streams carry run-relative time so that comparing a target and a reply on the same instants is exact.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from robot_msgs.std.header import Header
from robot_msgs.trajectory import JointTrajectory

from ..alignment import align
from ..run import Run

ROUND = Path("/archive/datasets/bam_rig/onedof_2026-11-15/round8")
SIM_MEMO = Path(os.environ.get("BAM_PENDULUM_SIM_MEMO") or Path.home() / ".cache/bam_rack_twin/pendulum_sim_memo")
DEPLOYMENT = "one-dof-test-rack"
PART_TYPE = "integrated_servo"
TICK_S = 0.01          # the runner's plan tick unless a run's meta says another
NO_TICK = -1           # a row that is no plan tick's command
MODE_POSITION = 10     # moteus position mode
STEP_IDENTITY_IGNORES = ("label", "schedule")   # repeats differ in label and stamp, never in their input
SAMPLES_PER_BOARD = ("ff_coulomb_nm", "ff_gravity_nm", "ff_gravity_bottom_rad", "ff_drive_torque_nm", "kp_scale_cmd",
                     "kd_scale_cmd", "trajectory_limits", "ff_inertia_kg_m2", "ff_viscous_nm_per_rad_s", "velocity_mode")


def available() -> bool:
    return (ROUND / "plans" / "index.json").is_file() and (ROUND / "real").is_dir()


def _meta_of(npz: Path) -> dict[str, Any]:
    path = npz.with_suffix(".json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _board_columns(npz: Path, board: int | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """A run file's columns in the single-board layout, and its tape start."""

    meta = _meta_of(npz)
    with np.load(npz) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    tick_s = _finite(meta.get("tick_s")) or TICK_S
    if board is None:
        return data, {"prelude_s": float(meta.get("prelude_s") or 0.0), "tick_s": tick_s}
    suffix = f"_can{board}"
    own = {name[: -len(suffix)]: values for name, values in data.items() if name.endswith(suffix)}
    rows = np.isfinite(np.asarray(own["mode"], float))
    columns = {name: np.asarray(data[name])[rows] for name in ("t", "wall", "tick") if name in data}
    columns.update({name: values[rows] for name, values in own.items()})
    tape_start = _finite(meta.get("tape_start_s"))
    if tape_start is None:
        tape_start = (math.floor((_finite(meta.get("approach_s")) or 0.0) / tick_s + 1e-9) + 1) * tick_s
    return columns, {"prelude_s": tape_start, "tick_s": tick_s}


def _header(frame: str, **meta: Any) -> Header:
    return Header(stamp=0.0, frame_id=frame, meta={key: str(value) for key, value in meta.items() if value is not None})


def _real_streams(npz: Path, board: int | None, station: str) -> tuple[JointTrajectory, JointTrajectory]:
    """The plan on its ticks and the replies (position-mode rows of the tape), as robot_msgs streams."""

    data, info = _board_columns(npz, board)
    t = np.asarray(data["t"], float)
    tick = np.asarray(data["tick"]).astype(np.int64)
    tick_s = info["tick_s"]
    q_target = np.asarray(data["q_target"], float)
    qd_target = np.asarray(data["qd_target"], float)
    tape = (tick != NO_TICK) & (tick * tick_s >= info["prelude_s"] - 1e-9)
    mode = np.asarray(data.get("mode", np.full(t.size, MODE_POSITION)), float)
    # WHY position-mode rows only: the round scores those; round 8 has none outside it on the tape
    keep = tape & (mode == MODE_POSITION) & np.isfinite(np.asarray(data["q"], float))
    wall0 = float(data["wall"][0]) if "wall" in data and np.size(data["wall"]) else None
    header = _header("output", wall0=wall0, source=str(npz), board=board)
    names = (station,)
    channels = {"fault": np.asarray(data.get("fault", np.zeros(t.size)), float)[keep]}
    if "ff_model_nm" in data:
        channels["ff_nm"] = np.asarray(data["ff_model_nm"], float)[keep]
    replies = JointTrajectory(t=t[keep], joint_names=names, positions=np.asarray(data["q"], float)[keep][:, None],
                              velocities=np.asarray(data["qd"], float)[keep][:, None],
                              effort=np.asarray(data["tau"], float)[keep][:, None], channels=channels, header=header)
    order = np.flatnonzero(keep & (tick != NO_TICK))
    plan_t, first = np.unique(tick[order] * tick_s, return_index=True)
    plan = JointTrajectory(t=plan_t, joint_names=names, positions=q_target[order][first][:, None],
                           velocities=qd_target[order][first][:, None], header=header)
    return plan, replies


def _identity(step: Mapping[str, Any]) -> str:
    return json.dumps({key: value for key, value in step.items() if key not in STEP_IDENTITY_IGNORES}, sort_keys=True,
                      default=str)


def _per_board(value: Any, can_id: int) -> Any:
    if isinstance(value, Mapping):
        return value[str(can_id)] if str(can_id) in value else value.get(can_id)
    return value


def _board_step(step: Mapping[str, Any], can_id: int) -> dict[str, Any]:
    """One board's part of a paired plan step, as the session built it (its identity finds the twin's answer)."""

    samples = _per_board(step["samples_q"], can_id)
    samples = [float(value) for value in (samples.split(",") if isinstance(samples, str) else samples)]
    part: dict[str, Any] = {"kind": "samples", "can_id": int(can_id), "cap_nm": _per_board(step["cap_nm"], can_id),
                            "origin_rad": samples[0], "samples_q": [value - samples[0] for value in samples]}
    for key in SAMPLES_PER_BOARD:
        if key in step and _per_board(step[key], can_id) is not None:
            part[key] = _per_board(step[key], can_id)
    if _per_board(step.get("ff_band_rad_s"), can_id) is not None:
        part["ff_band_rad_s"] = _per_board(step["ff_band_rad_s"], can_id)
    if float(_per_board(step.get("ff_hold_nm"), can_id) or 0.0):
        part.update({key: _per_board(step[key], can_id) for key in ("ff_hold_nm", "ff_hold_band_rad", "ff_hold_after_s")
                     if key in step})
    if float(_per_board(step.get("kp_scale_hold"), can_id) or 1.0) != 1.0:
        part["kp_scale_hold"] = _per_board(step["kp_scale_hold"], can_id)
    rate = _finite(step.get("rate_hz"))
    if rate is not None and abs(rate * TICK_S - 1.0) > 1e-9:
        part["rate_hz"] = rate
    for key in STEP_IDENTITY_IGNORES:
        if key in step:
            part[key] = step[key]
    return part


@dataclass(frozen=True)
class _Play:
    station: str
    npz: Path
    board: int | None
    step: Mapping[str, Any]
    status: str
    fraction: float
    kind: str
    id: str


@functools.cache
def _index() -> dict[str, Any]:
    return json.loads((ROUND / "plans" / "index.json").read_text(encoding="utf-8"))


@functools.cache
def _plays() -> tuple[_Play, ...]:
    """Every play of round 8 in session order: one per board of each session step, its rung and kind from the index."""

    by_run = {(entry["station"], entry["run_name"]): entry for entry in _index()["runs"]}
    plays = []
    for session_dir in sorted(p for p in (ROUND / "real").iterdir() if (p / "session.json").is_file()):
        session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        plan_steps = (session.get("plan") or {}).get("steps") or []
        for entry in session.get("steps") or []:
            boards = entry.get("can_ids") if entry.get("can_id") is None else None
            if entry.get("kind") == "pause" or (entry.get("can_id") is None and not boards):
                continue
            step = plan_steps[int(entry["index"]) - 1]
            npz = session_dir / next(f for f in entry["files"] if f.endswith(".npz"))
            for can_id in boards or [int(entry["can_id"])]:
                station = f"can{can_id}"
                status = str((entry.get("boards") or {}).get(str(can_id)) or entry.get("status")) if boards else str(
                    entry.get("status"))
                found = by_run.get((station, npz.stem))
                if found is None:  # a step the plan index never named (a park, a recovery): scored nowhere
                    continue
                own = _board_step(step, can_id) if boards else step
                play_id = str(npz) if not boards else f"{npz}#can{can_id}"
                plays.append(_Play(station, npz, int(can_id) if boards else None, own, status,
                                   float(found["fraction"]), str(found["kind"]), play_id))
    return tuple(plays)


def stations() -> dict[str, dict[str, float]]:
    """Each station's facts from the round's plan index: tip length (m) and top speed (rad/s)."""

    return {name: {"tip_length_m": float(facts["tip_length_m"]), "omega_max_rad_s": float(facts["omega_max_rad_s"])}
            for name, facts in _index()["stations"].items()}


def _labels(play: _Play, **extra: Any) -> dict[str, Any]:
    return {"deployment": DEPLOYMENT, "station": play.station, "part_type": PART_TYPE, "rung": play.fraction,
            "kind": play.kind, **extra}


@functools.cache
def real_runs() -> tuple[Run, ...]:
    """Round 8's real plays, in session order. ``target`` is the plan at each reply's instant (hermite, one tick)."""

    runs = []
    for play in _plays():
        plan, replies = _real_streams(play.npz, play.board, play.station)
        aligned = align(plan, replies, rule="hermite", tolerance_s=TICK_S)
        runs.append(Run(id=play.id, streams={"plan": plan, "state": aligned.measured, "target": aligned.target},
                        labels=_labels(play), provenance="real",
                        outcome="complete" if play.status == "complete" else "faulted", alignment=aligned.alignment,
                        links={"source": str(play.npz)}))
    return tuple(runs)


@functools.cache
def _memo_index() -> dict[tuple[str, str, str, str], Path]:
    """(station, twin, controller sha256, step identity) -> the memo's npz; the newest file wins a duplicate key."""

    found: dict[tuple[str, str, str, str], tuple[str, Path]] = {}
    for path in SIM_MEMO.glob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        key = (entry.get("station"), entry.get("twin"), entry.get("controller"), entry.get("step"))
        if None in key:
            continue
        written = str(entry.get("written_utc") or "")
        if key not in found or written > found[key][0]:
            found[key] = (written, path.with_suffix(".npz"))
    return {key: value[1] for key, value in found.items()}


def _controller_sha() -> str:
    controller = json.loads((ROUND / "controller.json").read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(controller, sort_keys=True, default=str).encode()).hexdigest()


def sim_runs(twins: Mapping[str, str]) -> tuple[Run, ...]:
    """The twin's answer to each real play (a play without one is left out). ``twins`` names the twin version that
    answered each station, e.g. {"can11": "can11-r18a", ...}: the round's scoring metadata, passed in by the caller."""

    return _sim_runs(tuple(sorted(twins.items())))


@functools.cache
def _sim_runs(twins: tuple[tuple[str, str], ...]) -> tuple[Run, ...]:
    by_station, controller, memo = dict(twins), _controller_sha(), _memo_index()
    runs = []
    for play in _plays():
        path = memo.get((play.station, by_station.get(play.station), controller, _identity(play.step)))
        if path is None or not path.is_file():
            continue
        with np.load(path) as archive:
            answer = {name: np.asarray(archive[name]) for name in archive.files}
        t = np.asarray(answer["t"], float)
        q_target = np.asarray(answer["q_target"], float)
        qd_target = np.asarray(answer["qd_target"], float) if "qd_target" in answer else np.gradient(q_target, t)
        prelude = float(answer["prelude_s"]) if "prelude_s" in answer else 0.0
        tape = t >= t[0] + prelude - 1e-9
        mode = np.asarray(answer["mode"], float) if "mode" in answer else np.full(t.size, MODE_POSITION)
        keep = tape & (mode == MODE_POSITION) & np.isfinite(np.asarray(answer["q"], float))
        header = _header("output", source=str(path), twin=by_station[play.station])
        names = (play.station,)
        plan = JointTrajectory(t=t[keep], joint_names=names, positions=q_target[keep][:, None],
                               velocities=qd_target[keep][:, None], header=header)
        replies = JointTrajectory(t=t[keep], joint_names=names, positions=np.asarray(answer["q"], float)[keep][:, None],
                                  velocities=np.asarray(answer["qd"], float)[keep][:, None],
                                  effort=np.asarray(answer["tau"], float)[keep][:, None],
                                  channels={"fault": np.asarray(answer.get("fault", np.zeros(t.size)), float)[keep]},
                                  header=header)
        # WHY same_clock with tolerance 0: a sim answers on its exact ticks, so its target is its logged plan row
        aligned = align(plan, replies, rule="same_clock", tolerance_s=0.0)
        runs.append(Run(id=play.id, streams={"plan": plan, "state": aligned.measured, "target": aligned.target},
                        labels=_labels(play, twin=by_station[play.station]), provenance="sim",
                        alignment=aligned.alignment, links={"source": str(path)}))
    return tuple(runs)
