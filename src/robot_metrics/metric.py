"""The core: a metric is a frozen config whose state is a value; live is a fold over chunks, offline the same fold once.

    rms = ErrorStat(statistic="rms", rows="moving")     # a frozen dataclass: identity = class + fields, hashable
    state = rms.empty()
    for target, measured in chunks:                     # live
        state = rms.update(state, target, measured)     # returns a NEW state; nothing hidden is mutated
    rms.compute(rms.close(state))                       # close = the run boundary (an open dwell ends here)
    rms(target, measured)                               # offline: one update of the whole recording, from empty()
    rms.merge(a, b)                                     # two runs, two rungs, two stations: one state
    stream = rms.stream()                               # torchmetrics' face: update / compute / reset / plot

WHY the state as a value (clu, torcheval; the prior-art note §1): merging runs needs no extra code, a state can be
stored and resumed, and live differs from offline only in where the fold's chunks come from. "Live == offline" holds
within float reassociation, never bit for bit by promise: a sum folded in another order may round differently.

WHY a frozen dataclass (Zach, 2026-11-17, "# Ok homing in on this..."): a base metric is math only, configured in its
constructor; its identity is (class, frozen config), so two aliases of one identity compute once. What a metric IS
(name, description, references, value type, input, live, version) sits on the class; its config on the instance; its
display unit on the alias, never here (SI inside).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

DEGREES = 180.0 / math.pi


@dataclass(frozen=True)
class AlignedPair:
    """The common input: a target and a measured stream on the same instants (see ``robot_metrics.align``)."""

    target: Any
    measured: Any


@dataclass(frozen=True)
class CommandAndState:
    """The command on its own ticks and the state as it came back: not aligned (the metric finds the offset itself)."""

    command: Any
    state: Any


@dataclass(frozen=True)
class Metric:
    """The base. Subclasses define ``empty`` / ``update`` / ``merge`` / ``compute`` and, when a state spans a run
    boundary, ``close``. Metrics whose ``state_key`` is equal share one state (their update / merge / close are one
    code path and only ``compute`` differs): torchmetrics' compute groups."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    references: ClassVar[tuple[str, ...]] = ()
    value_type: ClassVar[str] = "number"     # "number" | "bool" (Zach: "a bool type or like a value type")
    unit: ClassVar[str] = "SI"               # the signal's own SI unit; the display unit belongs to the alias
    hint: ClassVar[str] = "kpi_trend"        # the data-ui view that shows it (RobotDx core.HINTS)
    Input: ClassVar[type] = AlignedPair      # what update() takes
    live: ClassVar[bool] = True              # can it produce its value while the run is still arriving
    version: ClassVar[str] = "1"             # stored with every value (the prior-art note: recompute is then safe)
    align_tolerance_s: ClassVar[float] = 1e-9   # an AlignedPair's two clocks may differ by at most this

    # ---- the value-state contract --------------------------------------------------------------------------------
    def empty(self) -> Any:
        raise NotImplementedError

    def update(self, state: Any, *inputs: Any) -> Any:
        raise NotImplementedError

    def merge(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def compute(self, state: Any) -> Any:
        raise NotImplementedError

    def close(self, state: Any) -> Any:
        """The run boundary (a dwell never spans two runs). Most states have nothing to close."""

        return state

    def items(self, state: Any) -> list[float] | None:
        """The occurrences a gate counts ("100 % of stops <= 1 mm"); None when the occurrence is the run itself."""

        return None

    @property
    def state_key(self) -> Any:
        """Metrics with equal keys share one state. By default a metric shares only with its own identity."""

        return self

    # ---- offline and functional ----------------------------------------------------------------------------------
    def __call__(self, *inputs: Any) -> Any:
        """Offline: one update of the whole input from empty(), closed, computed."""

        return self.compute(self.close(self.update(self.empty(), *inputs)))

    def fold(self, inputs: Iterable[Any]) -> Any:
        """The state after every input in turn (a tuple is unpacked as update's arguments), closed."""

        state = self.empty()
        for item in inputs:
            state = self.update(state, *item) if isinstance(item, tuple) else self.update(state, item)
        return self.close(state)

    # ---- torchmetrics' face -----------------------------------------------------------------------------------------
    def stream(self) -> Stream:
        return Stream(self)

    # ---- identity, card, view ---------------------------------------------------------------------------------------
    @property
    def identity(self) -> tuple[str, tuple]:
        """What makes two metrics one computation: the base class and its frozen config (an alias is only a label)."""

        return type(self).__name__, tuple((f.name, _identity_of(getattr(self, f.name))) for f in dataclasses.fields(self))

    def config(self) -> str:
        return ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in dataclasses.fields(self))

    def card(self) -> dict[str, Any]:
        return {"base": type(self).__name__, "config": self.config(), "description": self.description,
                "references": list(self.references), "value_type": self.value_type, "input": self.Input.__name__,
                "live": self.live, "version": self.version, "hint": self.hint}

    def view(self, state: Any, *, scale: float = 1.0, unit: str = "", title: str = "") -> Any:
        """The points behind the value: every occurrence when it has some, else the one value."""

        from .views import View

        items = self.items(state)
        if items is None:
            value = self.compute(state)
            items = [] if value is None else [float(value)]
        values = [scale * float(item) for item in items]
        return View(title=title or self.name, unit=unit or self.unit, hint=self.hint, x=list(range(len(values))),
                    y=values, x_label="occurrence")

    def plot(self, state: Any, ax: Any = None, *, path: Any = None, scale: float = 1.0, unit: str = "",
             title: str = "") -> tuple[Any, Any]:
        """torchmetrics' ``Metric.plot``: (fig, ax) on matplotlib (the optional 'viz' extra)."""

        from .views import plot

        return plot(self.view(state, scale=scale, unit=unit, title=title), ax=ax, path=path)


def _identity_of(value: Any) -> Any:
    return value.identity if isinstance(value, Metric) else value


def check_aligned(metric: Metric, target: Any, measured: Any) -> None:
    """Refuse a pair that is not on one clock: alignment is an explicit step before the metric, never inside it."""

    if len(target) != len(measured) or (len(target) and not np.allclose(
            np.asarray(target.t, float), np.asarray(measured.t, float), rtol=0.0, atol=metric.align_tolerance_s)):
        raise ValueError(f"{type(metric).__name__}: target and measured are not on the same instants (tolerance "
                         f"{metric.align_tolerance_s} s); align them first with robot_metrics.align")


class Stream:
    """torchmetrics' stateful face over a value-state metric: update / compute / reset / plot, and the state to store."""

    def __init__(self, metric: Metric) -> None:
        self.metric = metric
        self.state = metric.empty()

    def update(self, *inputs: Any) -> None:
        self.state = self.metric.update(self.state, *inputs)

    def end_run(self) -> None:
        self.state = self.metric.close(self.state)

    def compute(self) -> Any:
        return self.metric.compute(self.metric.close(self.state))

    def reset(self) -> None:
        self.state = self.metric.empty()

    def plot(self, ax: Any = None, **kwargs: Any) -> tuple[Any, Any]:
        return self.metric.plot(self.metric.close(self.state), ax=ax, **kwargs)

    def __repr__(self) -> str:
        return f"Stream({self.metric!r})"
