"""The stop kernel: where the stops are, one sample at a time, with the open dwell carried across chunks as a value.

    state = track(StopsState(), t, q, target, target_speed, joint="can11")   # a chunk (live) or a whole run (offline)
    state = finish(state, include_last_dwell=True)                           # the run boundary closes the last dwell
    state.closed                                                             # tuple[Stop, ...]
    stops(target, measured)                                                  # the same fold over a whole recording
    settle_time(stop, band)                                                  # any band, from the rows the stop carries

The rule is round 8's (bam_rack_twin pendulum_score.stops, via the metrics-API proposals' shared kernel ``stock.py``
at robot_models f0daa34, ported line for line with its state lifted out of the tracker object): a stop is a dwell of
at least 0.25 s at zero commanded speed on one constant target that a move led into (a dwell the episode starts in is
no stop); its error is the mean of measured - target over the dwell's last 50 ms (T4's settled tail); its settle time
runs from the dwell's start (the move's end) to the first row after which every row stays inside ``band``.

WHY here: robot_msgs.compare has no stop rule yet. Proposed move (PEP 0001 §Next): ``robot_msgs.compare.stops``
beside ``goal``, so sim2real.metrics.actuator and RobotDx's T4 call the same code.

WHY one per-sample rule for live and offline (Zach, 2026-11-17: "Able to compute live and offline ... it doesn't matter
if the sensor stream is online or offline"): ``stops()`` is literally ``track`` over the recording, so a stop split
across two chunks is found exactly as in one piece.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from robot_msgs.trajectory import JointTrajectory

STILL_RAD_S = 1e-9      # sim2real.metrics.actuator.stops: a row at rest has |commanded speed| below this
DWELL_MIN_S = 0.25      # stop_accuracy.DWELL_MIN_S (T4): a stop lasts at least this long on one constant target
SETTLE_WINDOW_S = 0.05  # stop_accuracy.SETTLE_WINDOW_S (T4): a stop's error is the mean over its last 50 ms
SAME_TARGET_RAD = 1e-9  # a dwell is one constant target


@dataclass(frozen=True, eq=False)
class Stop:
    """One settled stop. ``error`` is signed, measured - target, in the signal's unit (rad for a joint). ``move_*``
    hold the move into it, from the last still row before it to the dwell's end; the dwell starts at ``move_end``."""

    t_s: float             # the dwell's start, from the run's first sample
    target: float
    error: float
    move_t: np.ndarray
    move_q: np.ndarray
    move_target: np.ndarray
    move_end: int
    joint: str = "joint"

    @property
    def dwell_s(self) -> float:
        return float(self.move_t[-1] - self.move_t[self.move_end])

    def move(self) -> tuple[JointTrajectory, JointTrajectory]:
        """The move as robot_msgs.compare.goal wants it: the reference up to the move's end (its last point is the
        stop's target), the measured run through the whole dwell (goal's settle phase)."""

        names = (self.joint,)
        reference = JointTrajectory(t=self.move_t[: self.move_end + 1], joint_names=names,
                                    positions=self.move_target[: self.move_end + 1, None])
        measured = JointTrajectory(t=self.move_t, joint_names=names, positions=self.move_q[:, None])
        return reference, measured


@dataclass(frozen=True, eq=False)
class OpenDwell:
    """What the tracker carries between chunks: the rows since the last still row before this move, and the open
    dwell inside them. Bounded by one move plus one dwell, whatever the run's length."""

    rows: int                   # samples seen in this run
    t0: float                   # the run's first sample time
    since: np.ndarray           # (n, 3): t, q, target
    dwell_start: int | None     # index into ``since`` of the open dwell's first row
    dwell_target: float
    dwell_first_row: int
    joint: str = "joint"


@dataclass(frozen=True, eq=False)
class StopsState:
    closed: tuple[Stop, ...] = ()
    open: OpenDwell | None = None


def _close(since: list, dwell_start: int, dwell_first_row: int, t0: float, joint: str) -> Stop | None:
    rows = np.asarray(since, float)
    dwell = rows[dwell_start:]
    if dwell_first_row > 0 and dwell[-1, 0] - dwell[0, 0] >= DWELL_MIN_S:
        tail = dwell[dwell[:, 0] >= dwell[-1, 0] - SETTLE_WINDOW_S]
        error = float(np.mean(tail[:, 1] - tail[:, 2]))
        return Stop(float(dwell[0, 0] - t0), float(dwell[0, 2]), error, rows[:, 0].copy(), rows[:, 1].copy(),
                    rows[:, 2].copy(), dwell_start, joint)
    return None


def track(state: StopsState, t, q, target, target_speed, *, joint: str = "joint") -> StopsState:
    """The stop rule over one chunk of samples; returns a new state (the old one is untouched)."""

    closed = list(state.closed)
    if state.open is None:
        rows, t0, since, dwell_start, dwell_target, dwell_first_row = 0, None, [], None, 0.0, 0
    else:
        o = state.open
        rows, t0, since = o.rows, o.t0, o.since.tolist()
        dwell_start, dwell_target, dwell_first_row = o.dwell_start, o.dwell_target, o.dwell_first_row
    for t_row, q_row, target_row, speed_row in zip(np.asarray(t, float).tolist(), np.asarray(q, float).tolist(),
                                                   np.asarray(target, float).tolist(),
                                                   np.asarray(target_speed, float).tolist()):
        if t0 is None:
            t0 = t_row
        still = abs(speed_row) < STILL_RAD_S
        if dwell_start is not None and not (still and abs(target_row - dwell_target) < SAME_TARGET_RAD):
            stop = _close(since, dwell_start, dwell_first_row, t0, joint)
            if stop is not None:
                closed.append(stop)
            since = [list(since[-1])]          # the dwell's last row is the "last still row" the next move starts from
            dwell_start = None
        since.append([t_row, q_row, target_row])
        if still and dwell_start is None:
            dwell_start, dwell_target, dwell_first_row = len(since) - 1, target_row, rows
        rows += 1
    if t0 is None:
        return state
    return StopsState(tuple(closed), OpenDwell(rows, t0, np.asarray(since, float).reshape(-1, 3), dwell_start,
                                               dwell_target, dwell_first_row, joint))


def finish(state: StopsState, *, include_last_dwell: bool = True) -> StopsState:
    """The run boundary: close the open dwell. WHY a flag, default True: the pendulum's rule counts the episode's last
    dwell (a ladder play is A -> B -> A and its return to A is a commanded stop); sim2real.metrics.actuator.stops and
    RobotDx's T4 drop it (a tape's last dwell is parking). One rule holds both instead of two functions drifting apart."""

    o = state.open
    if o is None:
        return state
    if o.dwell_start is None or not include_last_dwell:
        return StopsState(state.closed, None)
    stop = _close(o.since.tolist(), o.dwell_start, o.dwell_first_row, o.t0, o.joint)
    return StopsState(state.closed + ((stop,) if stop is not None else ()), None)


def stops(target: JointTrajectory, measured: JointTrajectory, *, joint: int = 0,
          include_last_dwell: bool = True) -> tuple[Stop, ...]:
    """Every stop of a whole recording: ``track`` folded over it once, then ``finish``."""

    state = track(StopsState(), measured.t, measured.positions[:, joint], target.positions[:, joint],
                  target.velocities[:, joint], joint=target.joint_names[joint])
    return finish(state, include_last_dwell=include_last_dwell).closed


def settle_time(stop: Stop, band: float, scale: float = 1.0) -> float:
    """A stop's settle time against any band, from the dwell rows it carries: from the move's end to the first row
    after which every row stays inside ``band``; 0 inside from the start; inf never. WHY a function of the stop and not
    of the tracker: the band is the caller's config (a tip's 1 mm, a joint's share), so one pass serves every band."""

    t = stop.move_t[stop.move_end:]
    error = scale * np.abs(stop.move_q[stop.move_end:] - stop.move_target[stop.move_end:])
    outside = np.flatnonzero(error > band)
    if not outside.size:
        return 0.0
    if outside[-1] == t.size - 1:
        return math.inf
    return float(t[outside[-1] + 1] - t[0])
