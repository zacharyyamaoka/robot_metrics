"""The stock metrics: math only, each a frozen config whose state is a value. The meaning comes from the data.

    ErrorStat(statistic="rms", rows="moving")     target - measured over the rows it reads     state (sum e², n, max |e|, sum |e|)
    StopStat(of="error", statistic="median")      a quantity of every stop, pooled             state StopsState (the stop kernel)
    TimeOffset()                                  how late the state runs behind the command   per-run offsets      live=False
    CodeShare(codes=(97, 98, 99, 102))            share of rows holding one of the codes       (hits, rows)
    Repeatability(k=3)          Input Repeat      ISO 9283 RP over same-input repeats          per-repeat stop errors
    PerformanceGain(metric=..)  Input HeldOutPair a metric with nominal minus with fitted      (nominal, fitted)

WHY math only (Zach, 2026-11-17: "most metrics has the mathematical meaning but the semantic meaning ... from the actual
data you end up passing into them"): one ErrorStat serves servo tracking, tip tracking, the five-bar's tool error and
the feed-forward residual, by the streams its alias hands it. Units belong to the alias.

WHY robot_msgs.compare for the error (the brief: reuse it, never re-implement it): ``compare(target, measured).error``
is the error signal on the measured samples; the metric only folds it into a mergeable state. Overshoot is
``robot_msgs.compare.goal`` on each stop's move, lag ``robot_msgs.compare.estimate_time_offset``.

WHY no limits in any config (the prior-art note §1 on RobotDx: its Limits ride inside each config, so a new target
changes the metric's identity): targets live on ``Objective`` only.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
from robot_msgs.compare import compare, estimate_time_offset, goal
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory, concatenate

from . import stops as stop_kernel
from .metric import AlignedPair, CommandAndState, Metric, check_aligned

ROBOTDX = "robotdx.diagnostics.stock (io-diagnostics main a7fc658)"
STILL_RAD_S = stop_kernel.STILL_RAD_S


def moving(target: JointTrajectory, joint: int = 0) -> np.ndarray:
    """Rows whose commanded speed is not zero (the moving part of a move): tracking RMS is read on these."""

    if target.velocities is None:
        raise ValueError("rows='moving' needs the target's velocities (the commanded speed)")
    return np.abs(np.asarray(target.velocities[:, joint], float)) >= STILL_RAD_S


def error_signal(target: Any, measured: Any, field: str = "positions", joint: int = 0, rows: str = "all") -> np.ndarray:
    """target - measured on the measured samples, from robot_msgs.compare: one joint's field, or a pose's position
    error magnitude (compare's own "position" metric); finite samples only."""

    comparison = compare(target, measured)
    error = comparison.error
    if len(error) != len(measured):
        raise ValueError(f"compare dropped {comparison.dropped} samples: align the pair first")
    if isinstance(error, PoseTrajectory):
        if field != "positions":
            raise ValueError("a pose's ErrorStat reads its position error (field='positions')")
        values = np.linalg.norm(error.positions, axis=1)
    else:
        column = getattr(error, field)
        if column is None:
            raise ValueError(f"no {field!r} on both streams to compare")
        values = np.asarray(column[:, joint], float)
    if rows == "moving":
        values = values[moving(target, joint)]
    elif rows != "all":
        raise ValueError(f"rows is 'all' or 'moving', not {rows!r}")
    return values[np.isfinite(values)]


@dataclass(frozen=True)
class ErrorStat(Metric):
    statistic: str = "rms"          # rms | max_abs | mean_abs
    rows: str = "all"               # all | moving (commanded speed not zero)
    field: str = "positions"        # positions | velocities | effort (a pose: its position error)
    joint: int = 0
    name = "error statistic"
    description = "One statistic of target - measured over the rows it reads: rms, max_abs or mean_abs."
    references = ("robot_msgs.compare.ErrorStats", ROBOTDX + " tracking_error", "evo APE (Grupp 2017)")

    def __post_init__(self) -> None:
        if self.statistic not in ("rms", "max_abs", "mean_abs"):
            raise ValueError(f"statistic is rms, max_abs or mean_abs, not {self.statistic!r}")

    @property
    def state_key(self) -> tuple:
        return (type(self), self.rows, self.field, self.joint)   # rms, max_abs, mean_abs share one state

    def empty(self) -> tuple[float, int, float, float]:
        return (0.0, 0, 0.0, 0.0)

    def update(self, state: tuple, target: Any, measured: Any) -> tuple:
        if not len(measured):
            return state
        check_aligned(self, target, measured)
        error = error_signal(target, measured, self.field, self.joint, self.rows)
        if not error.size:
            return state
        return (state[0] + float(np.sum(error * error)), state[1] + int(error.size),
                max(state[2], float(np.max(np.abs(error)))), state[3] + float(np.sum(np.abs(error))))

    def merge(self, a: tuple, b: tuple) -> tuple:
        return (a[0] + b[0], a[1] + b[1], max(a[2], b[2]), a[3] + b[3])

    def compute(self, state: tuple) -> float | None:
        if not state[1]:
            return None
        if self.statistic == "rms":
            return math.sqrt(state[0] / state[1])
        return state[2] if self.statistic == "max_abs" else state[3] / state[1]


STOP_QUANTITIES = ("error", "settle", "settled", "overshoot", "count")
STOP_STATISTICS = ("median", "max", "p95", "p90", "mean", "count", "all")


@dataclass(frozen=True)
class StopStat(Metric):
    """A quantity of every stop, pooled: |error|, settle time against ``band``, settled within ``band`` (bool),
    overshoot past the target along the move (robot_msgs.compare.goal), or the count. ``band`` is in the signal's unit
    (rad for a joint); every StopStat on one signal shares one pass of the stop kernel."""

    of: str = "error"
    statistic: str = "median"
    band: float = math.inf
    include_last_dwell: bool = True
    joint: int = 0
    name = "stop statistic"
    description = ("A quantity of every stop (a dwell >= 0.25 s at zero commanded speed on one target that a move led "
                   "into; its error the mean of the dwell's last 50 ms), pooled over the stops it is given.")
    references = ("bam_rack_twin pendulum_score.stops / RobotDx T4 stop_accuracy", ROBOTDX + " stop_error",
                  "robot_msgs.compare.goal (overshoot)", "ISO 9283:1998 §7.2 (pose accuracy at a stop)")
    hint = "distribution"

    def __post_init__(self) -> None:
        if self.of not in STOP_QUANTITIES or self.statistic not in STOP_STATISTICS:
            raise ValueError(f"of in {STOP_QUANTITIES}, statistic in {STOP_STATISTICS}")

    @property
    def value_type(self) -> str:  # type: ignore[override]
        return "bool" if self.statistic == "all" else "number"

    @property
    def state_key(self) -> tuple:
        return (type(self), self.joint, self.include_last_dwell)     # one stop pass serves every reading

    def empty(self) -> stop_kernel.StopsState:
        return stop_kernel.StopsState()

    def update(self, state: stop_kernel.StopsState, target: Any, measured: Any) -> stop_kernel.StopsState:
        if not len(measured):
            return state
        check_aligned(self, target, measured)
        j = self.joint
        return stop_kernel.track(state, measured.t, measured.positions[:, j], target.positions[:, j],
                                 target.velocities[:, j], joint=target.joint_names[j])

    def close(self, state: stop_kernel.StopsState) -> stop_kernel.StopsState:
        return stop_kernel.finish(state, include_last_dwell=self.include_last_dwell)

    def merge(self, a: stop_kernel.StopsState, b: stop_kernel.StopsState) -> stop_kernel.StopsState:
        return stop_kernel.StopsState(self.close(a).closed + self.close(b).closed, None)

    def stops(self, state: stop_kernel.StopsState) -> tuple[stop_kernel.Stop, ...]:
        """The stops so far; an open dwell counts as if it ended now (it is closed on a copy, the state is a value)."""

        return self.close(state).closed

    def items(self, state: stop_kernel.StopsState) -> list[float]:
        found = self.stops(state)
        if self.of == "error":
            return [abs(stop.error) for stop in found]
        if self.of == "settle":
            return [stop_kernel.settle_time(stop, self.band) for stop in found]
        if self.of == "settled":
            return [float(math.isfinite(stop_kernel.settle_time(stop, self.band))) for stop in found]
        if self.of == "overshoot":
            return [goal(*stop.move(), band=math.inf)[f"positions/{stop.joint}"].overshoot for stop in found]
        return [1.0 for _ in found]

    def compute(self, state: stop_kernel.StopsState) -> Any:
        values = self.items(state)
        if self.statistic == "count":
            return float(len(values))
        if self.statistic == "all":
            return all(value == 1.0 for value in values)
        if not values:
            return 0.0 if self.of == "overshoot" else None
        if self.statistic == "median":
            result = statistics.median(values)
        elif self.statistic == "max":
            result = max(values)
        elif self.statistic == "mean":
            result = float(np.mean(values))
        else:
            result = float(np.percentile(values, 95.0 if self.statistic == "p95" else 90.0))
        # WHY None and never a clipped number: a statistic that lands on a never-settled stop (inf) has no value; the
        # scorecard writes None there ("None or inf when that median never settled"), and None is never a pass.
        return float(result) if math.isfinite(result) else None


@dataclass(frozen=True)
class TimeOffset(Metric):
    """How late the state runs behind the command (robot_msgs.compare.estimate_time_offset), per run; across runs the
    median. It needs the whole run (a cross-correlation), so it declares live=False and folds chunks until close."""

    max_offset_s: float = 0.25
    name = "time offset"
    description = "How late the response runs behind the command (estimate_time_offset), one offset per run."
    references = ("robot_msgs.compare.estimate_time_offset", "evo associate_trajectories offset_2")
    Input = CommandAndState
    live = False
    align_tolerance_s = math.inf   # WHY: it is the step that finds the clock offset, so it cannot demand aligned input

    def empty(self) -> tuple:
        return ((), (), ())

    def update(self, state: tuple, command: Any, measured: Any) -> tuple:
        offsets, commands, states = state
        return (offsets, commands + ((command,) if len(command) else ()), states + ((measured,) if len(measured) else ()))

    def close(self, state: tuple) -> tuple:
        offsets, commands, states = state
        if not commands or not states:
            return (offsets, (), ())
        command = commands[0] if len(commands) == 1 else concatenate(list(commands))
        measured = states[0] if len(states) == 1 else concatenate(list(states))
        offset = estimate_time_offset(command, measured, max_offset_s=self.max_offset_s).offset_s
        return (offsets + (float(offset),), (), ())

    def merge(self, a: tuple, b: tuple) -> tuple:
        return (self.close(a)[0] + self.close(b)[0], (), ())

    def compute(self, state: tuple) -> float | None:
        offsets = self.close(state)[0]
        return float(statistics.median(offsets)) if offsets else None


@dataclass(frozen=True)
class CodeShare(Metric):
    codes: tuple[int, ...] = (97, 98, 99, 102)
    channel: str = "fault"
    name = "code share"
    description = "The share of the state's rows whose code channel holds one of the codes."
    references = ("moteus register 0x00f fault / limiter codes", ROBOTDX + " fault_codes")

    def empty(self) -> tuple[int, int]:
        return (0, 0)

    def update(self, state: tuple, target: Any, measured: Any) -> tuple:
        if not len(measured):
            return state
        codes = np.asarray(measured.channels[self.channel]).astype(int)
        return (state[0] + int(np.isin(codes, self.codes).sum()), state[1] + int(codes.size))

    def merge(self, a: tuple, b: tuple) -> tuple:
        return (a[0] + b[0], a[1] + b[1])

    def compute(self, state: tuple) -> float | None:
        return state[0] / state[1] if state[1] else None


@dataclass(frozen=True)
class Repeat:
    """One real repeat of one command: the input of Repeatability. ``command`` names the input it repeats."""

    command: Any
    target: Any
    measured: Any


@dataclass(frozen=True)
class Repeatability(Metric):
    """ISO 9283 pose repeatability RP at each stop over same-input repeats, mean(l) + k S_l with l the distance of each
    repeat's stop error from the repeats' mean; the worst stop index. A repeat of another command is refused."""

    k: float = 3.0
    include_last_dwell: bool = True
    name = "repeatability"
    description = "ISO 9283 RP at each stop over same-input repeats: mean(l) + k S_l, the worst stop index."
    references = ("ISO 9283:1998 §7.2.2 pose repeatability", ROBOTDX + " pose_accuracy",
                  "robot_msgs.compare.aggregate")
    Input = Repeat
    live = False

    def empty(self) -> tuple:
        return (None, ())

    def update(self, state: tuple, repeat: Repeat) -> tuple:  # type: ignore[override]
        if state[0] is not None and repeat.command != state[0]:
            raise ValueError(f"{repeat.command!r} is not a repeat of {state[0]!r}")
        errors = tuple(stop.error for stop in stop_kernel.stops(repeat.target, repeat.measured,
                                                                include_last_dwell=self.include_last_dwell))
        return (repeat.command, state[1] + (errors,))

    def merge(self, a: tuple, b: tuple) -> tuple:
        if a[0] is not None and b[0] is not None and a[0] != b[0]:
            raise ValueError("repeats of two commands do not merge")
        return (a[0] if a[0] is not None else b[0], a[1] + b[1])

    def compute(self, state: tuple) -> float | None:
        errors = state[1]
        if len(errors) < 2 or len({len(e) for e in errors}) != 1:
            return None
        matrix = np.asarray(errors)
        distance = np.abs(matrix - matrix.mean(axis=0))
        return float(np.max(distance.mean(axis=0) + self.k * distance.std(axis=0, ddof=1)))


@dataclass(frozen=True)
class HeldOutPair:
    """The same held-out runs played twice: with the nominal values and with the fitted ones."""

    nominal: tuple
    fitted: tuple


@dataclass(frozen=True)
class PerformanceGain(Metric):
    """A metric folded over held-out runs with nominal values and with fitted ones; nominal minus fitted.

    ``metric`` takes a Metric or an Alias; an alias is reduced to its metric and signal, so the gain's identity is the
    alias's identity (its name stays a label). WHY (Zach, 3000-ft view, 2026-11-17): "Fitting a model is only useful
    if it gives us better performance on the things that actually matter" - the gain is measured with the performance
    metric itself, on runs the fit never saw."""

    metric: Any = dataclass_field(default_factory=ErrorStat)
    signal: tuple[str, str] = ("target", "state")
    name = "performance gain"
    description = "A metric on held-out runs with nominal and with fitted values; nominal minus fitted."
    references = ("hold-out validation (Hastie, Tibshirani, Friedman, ESL §7.2)", "robot_models select_best")
    Input = HeldOutPair
    live = False

    def __post_init__(self) -> None:
        if not isinstance(self.metric, Metric) and hasattr(self.metric, "metric") and hasattr(self.metric, "signal"):
            object.__setattr__(self, "signal", tuple(self.metric.signal))
            object.__setattr__(self, "metric", self.metric.metric)
        if not isinstance(self.metric, Metric):
            raise TypeError(f"PerformanceGain(metric=...) takes a Metric or an Alias, not {type(self.metric).__name__}")

    def empty(self) -> tuple:
        return (None, None)

    def _side(self, runs: tuple) -> Any:
        state = self.metric.empty()
        for run in runs:     # each run closed before the next: a run boundary
            state = self.metric.close(self.metric.update(state, run.streams[self.signal[0]],
                                                         run.streams[self.signal[1]]))
        return self.metric.compute(state)

    def update(self, state: tuple, pair: HeldOutPair) -> tuple:  # type: ignore[override]
        return (self._side(pair.nominal), self._side(pair.fitted))

    def merge(self, a: tuple, b: tuple) -> tuple:
        raise TypeError("a gain is one comparison; merge its sides' runs instead")

    def sides(self, state: tuple) -> tuple[Any, Any]:
        """(nominal, fitted): the metric's value on each side."""

        return state

    def compute(self, state: tuple) -> float | None:
        return None if state[0] is None or state[1] is None else state[0] - state[1]
