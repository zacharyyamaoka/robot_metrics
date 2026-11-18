"""The plain function beside each class (torchmetrics.functional): one call, one input, an SI value.

WHY each function is its class run once and not a second formula: torchmetrics keeps two implementations (a functional
core and the class around it); here a function cannot drift from its class.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from . import stops as stop_kernel
from .stock import CodeShare, ErrorStat, HeldOutPair, PerformanceGain, Repeat, Repeatability, StopStat, TimeOffset


def error_stat(target: Any, measured: Any, **config: Any) -> float | None:
    return ErrorStat(**config)(target, measured)


def stop_stat(target: Any, measured: Any, **config: Any) -> Any:
    return StopStat(**config)(target, measured)


def stops(target: Any, measured: Any, *, joint: int = 0, include_last_dwell: bool = True) -> tuple[stop_kernel.Stop, ...]:
    return stop_kernel.stops(target, measured, joint=joint, include_last_dwell=include_last_dwell)


def time_offset(command: Any, measured: Any, **config: Any) -> float | None:
    return TimeOffset(**config)(command, measured)


def code_share(target: Any, measured: Any, **config: Any) -> float | None:
    return CodeShare(**config)(target, measured)


def repeatability(repeats: Iterable[Repeat], **config: Any) -> float | None:
    metric = Repeatability(**config)
    return metric.compute(metric.fold(repeats))


def performance_gain(pair: HeldOutPair, **config: Any) -> float | None:
    return PerformanceGain(**config)(pair)
