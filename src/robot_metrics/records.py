"""Records: what gets stored or handed on. Frozen dataclasses, SI unless the field says otherwise.

- ``MetricRecord``: one stored value with the identity and version that produced it (the prior-art note: a version on
  every stored value is what makes "recompute from the recording" safe when a metric changes).
- ``CalibrationOutcome`` / ``Interval``: what one calibration produced (amendment 4). WHY here and not in sim2real:
  sim2real is the outer layer; a record it owned would reverse the import arrow. Proposed move (PEP 0001): a
  robot_msgs message, beside ``Metrics``, once its fields have settled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricRecord:
    alias: str
    value: float | bool | None     # in ``unit`` (the alias's display unit)
    unit: str
    base: str
    config: str
    version: str
    run: str
    provenance: str
    labels: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Interval:
    value: float
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class CalibrationOutcome:
    """Per parameter: the nominal value, the recovered one (with an interval when the fit gives one), the truth when it
    is known (a sealed sim only, never a rig), and the relative tolerance a recovery is held to."""

    part_type: str
    model: str
    nominal: Mapping[str, float]
    recovered: Mapping[str, Interval]
    truth: Mapping[str, float] = field(default_factory=dict)
    tolerance: Mapping[str, float] = field(default_factory=dict)
    fitted_on: str = ""
    held_out: str = ""

    def values(self, which: str) -> dict[str, float]:
        """``"nominal"``: the starting values; ``"fitted"``: those with every recovered parameter put in."""

        if which == "nominal":
            return dict(self.nominal)
        if which != "fitted":
            raise ValueError(f"which is 'nominal' or 'fitted', not {which!r}")
        return {**self.nominal, **{name: interval.value for name, interval in self.recovered.items()}}

    def relative_error(self) -> dict[str, float]:
        """|recovered - truth| / |truth| for every parameter whose truth is known."""

        return {name: abs(self.recovered[name].value - truth) / abs(truth)
                for name, truth in self.truth.items() if name in self.recovered and truth}

    def within_tolerance(self) -> dict[str, bool]:
        return {name: error <= self.tolerance[name] for name, error in self.relative_error().items()
                if name in self.tolerance}
