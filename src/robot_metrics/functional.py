"""The plain function beside each class (torchmetrics.functional): one call, one input, an SI value.

WHY each function is its class run once and not a second formula: torchmetrics keeps two implementations (a functional
core and the class around it); here a function cannot drift from its class.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import stops as stop_kernel
from .stock import (CodeShare, ErrorPercentile, ErrorStat, HeldOutPair, PerformanceGain, Repeat, Repeatability, StopStat,
                    TimeOffset)


def error_stat(target: Any, measured: Any, **config: Any) -> float | None:
    return ErrorStat(**config)(target, measured)


def error_percentile(target: Any, measured: Any, **config: Any) -> float | None:
    return ErrorPercentile(**config)(target, measured)


def stop_stat(target: Any, measured: Any, **config: Any) -> Any:
    return StopStat(**config)(target, measured)


def stops(target: Any, measured: Any, *, joint: int = 0, include_last_dwell: bool = True,
          dwell_min_s: float = stop_kernel.DWELL_MIN_S, settle_window_s: float = stop_kernel.SETTLE_WINDOW_S,
          count_first_dwell: bool = False) -> tuple[stop_kernel.Stop, ...]:
    return stop_kernel.stops(target, measured, joint=joint, include_last_dwell=include_last_dwell,
                             dwell_min_s=dwell_min_s, settle_window_s=settle_window_s,
                             count_first_dwell=count_first_dwell)


def stop_summary(signed: Iterable[float], percentile: float = 90.0) -> dict[str, Any]:
    """T4's numbers over signed stop errors in the caller's unit (``stops.summary``): not a class run once, because it
    reads stops already found (their errors scaled to the caller's unit), not a recording."""

    return stop_kernel.summary(list(signed), percentile)


def time_offset(command: Any, measured: Any, **config: Any) -> float | None:
    return TimeOffset(**config)(command, measured)


def code_share(target: Any, measured: Any, **config: Any) -> float | None:
    return CodeShare(**config)(target, measured)


def repeatability(repeats: Iterable[Repeat], **config: Any) -> float | None:
    metric = Repeatability(**config)
    return metric.compute(metric.fold(repeats))


def performance_gain(pair: HeldOutPair, **config: Any) -> float | None:
    return PerformanceGain(**config)(pair)
