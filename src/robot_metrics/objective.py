"""Gates outside compute(): OpenSLO-style occurrence objectives, judged per stratum into a small verdict record.

    Objective("tip stop error", op="<=", value=1.0, unit="mm", share=1.0, on="items")   # "100 % of stops <= 1 mm"
    Objective("tip stop error", op="<=", value=1.0, unit="mm", on="value")              # the median stop <= 1 mm
    Gate("tip stop (every stop)", (objective, ...)).judge(stratum)                      # passes when every one passes

WHY OpenSLO's occurrence form (op, value, target share; the prior-art note §1): "99 % of searches under 500 ms" reads
here as "100 % of stops under 1 mm", which says what a median hides (CAN 11's tail). WHY outside the metric (Zach,
2026-11-17: "100% agreed keep the target outside of compute"): the same state is judged against any target, and a new
target never changes the metric's identity, so nothing is recomputed. Each objective names its owner (Zach / agent /
proposed). The value is in the alias's display unit and the unit is checked, never assumed.

robot_metrics returns its own ``Verdict``; it does not import RobotDx. RobotDx converts one in a line (PEP 0001 §5):
``rdx.result({"objective": v.objective, "owner": v.owner}, value=v.observed, unit=v.unit, metric=v.measure,
limits=rdx.Limits.at_least(v.share) if v.on == "items" else ..., verdict=v.verdict, title=v.objective,
message=v.message, hint=v.hint)``.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .summary import Stratum

PASS, FAIL, NOT_RUN = "pass", "fail", "not_run"
OPS = {"<=": operator.le, "lte": operator.le, "<": operator.lt, "lt": operator.lt,
       ">=": operator.ge, "gte": operator.ge, ">": operator.gt, "gt": operator.gt}
SYMBOL = {"lte": "<=", "lt": "<", "gte": ">=", "gt": ">"}


@dataclass(frozen=True)
class Verdict:
    """One objective judged on one stratum. ``observed`` is the share of occurrences that passed (on="items") or the
    value itself (on="value"); no value is ``not_run``, never a pass (RobotDx: "can't tell is never green")."""

    objective: str
    verdict: str
    observed: float | None
    passed: int
    total: int
    owner: str
    stratum: tuple
    on: str
    share: float
    unit: str
    measure: str          # "share" | "value"
    message: str
    hint: str


@dataclass(frozen=True)
class Objective:
    alias: str
    op: str
    value: float
    share: float = 1.0
    on: str = "items"           # items: every occurrence (each stop) | value: the stratum's value of the alias
    unit: str | None = None     # checked against the alias's display unit when given
    owner: str = "proposed"

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"op is one of {sorted(OPS)}, not {self.op!r}")
        if self.on not in ("items", "value"):
            raise ValueError(f"on is 'items' or 'value', not {self.on!r}")
        if not 0.0 <= self.share <= 1.0:
            raise ValueError(f"share is a fraction in [0, 1], not {self.share!r}")

    @property
    def title(self) -> str:
        bound = f"{SYMBOL.get(self.op, self.op)} {self.value:g}{' ' + self.unit if self.unit else ''}"
        return f"{self.share:.0%} of {self.alias} {bound}" if self.on == "items" else f"{self.alias} {bound}"

    def judge(self, stratum: Stratum) -> Verdict:
        if self.alias not in stratum.values:
            raise KeyError(f"no alias {self.alias!r} in this stratum: {sorted(stratum.values)}")
        unit = stratum.units.get(self.alias, "")
        if self.unit is not None and self.unit != unit:
            raise ValueError(f"{self.title}: the objective is in {self.unit!r} but the alias's unit is {unit!r}")
        # WHY: the target is read here and never by a metric (Zach 2026-11-17: "keep the target outside of compute"),
        # so judging the same state against a new target recomputes nothing and changes no identity
        passes = OPS[self.op]
        if self.on == "value":
            value = stratum.values[self.alias]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return self._verdict(stratum, NOT_RUN, None, 0, 0, unit, "value", f"{self.alias}: no value")
            ok = bool(passes(float(value), self.value))
            return self._verdict(stratum, PASS if ok else FAIL, float(value), int(ok), 1, unit, "value",
                                 f"{self.alias} {float(value):.4g} {unit} ({self.owner})".replace("  ", " "))
        items = stratum.occurrences.get(self.alias)
        if items is None:
            raise KeyError(f"{self.alias!r} has no occurrences to count (a bool alias is judged on='value')")
        if not items:
            return self._verdict(stratum, NOT_RUN, None, 0, 0, "1", "share", f"{self.alias}: no occurrences")
        ok = sum(bool(passes(float(item), self.value)) for item in items)
        observed = ok / len(items)
        return self._verdict(stratum, PASS if observed >= self.share else FAIL, observed, ok, len(items), "1", "share",
                             f"{ok} of {len(items)} pass, worst {max(items):.4g} {unit} ({self.owner})")

    def _verdict(self, stratum: Stratum, verdict: str, observed: Any, passed: int, total: int, unit: str, measure: str,
                 message: str) -> Verdict:
        return Verdict(self.title, verdict, observed, passed, total, self.owner, stratum.key, self.on, self.share, unit,
                       measure, message, "distribution" if self.on == "items" else "threshold_band")


@dataclass(frozen=True)
class GateVerdict:
    name: str
    verdict: str
    verdicts: tuple[Verdict, ...]


@dataclass(frozen=True)
class Gate:
    """Objectives that must all pass on one stratum: fail if any fails, not_run if any could not be judged."""

    name: str
    objectives: tuple[Objective, ...]

    def judge(self, stratum: Stratum) -> GateVerdict:
        verdicts = tuple(objective.judge(stratum) for objective in self.objectives)
        kinds = {verdict.verdict for verdict in verdicts}
        overall = FAIL if FAIL in kinds else NOT_RUN if NOT_RUN in kinds or not verdicts else PASS
        return GateVerdict(self.name, overall, verdicts)


def worst(verdicts: Sequence[str]) -> str:
    """RobotDx's rank: fail over not_run over pass."""

    kinds = set(verdicts)
    return FAIL if FAIL in kinds else NOT_RUN if NOT_RUN in kinds or not kinds else PASS
