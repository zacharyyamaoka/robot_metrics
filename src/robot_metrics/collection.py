"""Aliases, a part's metrics, and the collection that computes each identity once.

    Alias("tip stop error", StopStat(of="error", statistic="median"), unit="mm", scale=1e3)   # a label over a metric
    part = PartMetrics("arm", "pendulum_arm", aliases, children=(servo,), context={"lever_m": 0.339})
    collection = part.collection()          # or Collection(aliases)
    state = collection.fold(run)            # offline; live: state = collection.update(state, chunk) ... close(state)
    collection.values(state)                # {alias: display value}; si=True for SI
    collection.computations                 # distinct states: what one update costs

WHY aliases (Zach, 2026-11-17: "we can always create like alias for the metrics"): a semantic name is a configured
instance with a display name, a display unit and the signal it reads. Its identity is the metric's (base + frozen
config), so two aliases of one identity are one state, and metrics whose ``state_key`` matches share one update
(torchmetrics' compute groups: every StopStat on a signal reads one pass of the stop kernel).

WHY ``of_child`` / ``scale_by`` (v5's rule): an assembly reads its children's results through context instead of
recomputing them. The pendulum arm declares no math of its own: its tip stop is the servo's stop state read through
the lever (``scale_by="lever_m"``), so the arm adds no computation to its servo.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .metric import Metric
from .records import MetricRecord

ACROSS = ("pool", "median", "max")


@dataclass(frozen=True)
class Alias:
    """A label over a configured metric.

    ``signal`` names the run's two streams it reads (target, measured). ``unit`` is the display unit and ``scale`` the
    SI -> display factor (WHY at display only: SI inside, degrees and mm on screen, Zach 2026-11-11). ``across`` says
    how runs combine: ``pool`` merges their states (every stop of a rung in one median), ``median`` / ``max`` take the
    runs' own values. ``of_child`` names the child part whose state this reads; ``scale_by`` a context key of the
    declaring part multiplied into ``scale`` (the lever)."""

    name: str
    metric: Metric
    signal: tuple[str, str] = ("target", "state")
    unit: str = ""
    scale: float = 1.0
    across: str = "pool"
    of_child: str = ""
    scale_by: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.across not in ACROSS:
            raise ValueError(f"{self.name}: across is one of {ACROSS}, not {self.across!r}")

    @classmethod
    def through(cls, child: PartMetrics, alias: str, *, name: str, unit: str, scale: float = 1.0, scale_by: str = "",
                note: str = "") -> Alias:
        """The child's alias shown at the assembly: the same metric and signal (so the same state), a new label."""

        source = child.alias(alias)
        return cls(name, source.metric, source.signal, unit, scale, source.across, of_child=child.name,
                   scale_by=scale_by, note=note or f"{child.name}'s {alias!r} through {scale_by or 'a fixed factor'}")

    def card(self) -> dict[str, Any]:
        """Zach's Metric fields (name, description, references, function, unit, target, value_type) and the identity."""

        card = self.metric.card()
        return {"name": self.name, "unit": self.unit, "signal": "/".join(self.signal), "target": None,
                "function": f"robot_metrics.{card['base']}", "across": self.across,
                "of_child": self.of_child, **card}


@dataclass(frozen=True)
class PartMetrics:
    """A part type's metrics: the seed of ``RobotPart.metrics``, beside ``.models``. ``children`` are the parts an
    assembly is made of; ``context`` the assembly's own numbers its aliases read through (the lever)."""

    name: str
    part_type: str
    aliases: tuple[Alias, ...]
    children: tuple[PartMetrics, ...] = ()
    context: Mapping[str, float] = field(default_factory=dict)

    def alias(self, name: str) -> Alias:
        found = [alias for alias in self.aliases if alias.name == name]
        if not found:
            raise KeyError(f"{self.part_type} has no alias {name!r}: {[a.name for a in self.aliases]}")
        return found[0]

    def child(self, name: str) -> PartMetrics:
        found = [child for child in self.children if child.name == name]
        if not found:
            raise KeyError(f"{self.part_type} has no child {name!r}")
        return found[0]

    def flattened(self, prefix: str = "") -> list[Alias]:
        """Every alias of the tree with its display factor resolved: children first, named ``<child>/<alias>``."""

        out = []
        for child in self.children:
            out += child.flattened(f"{prefix}{child.name}/")
        for alias in self.aliases:
            if alias.of_child:
                self.child(alias.of_child)
            scale = alias.scale
            if alias.scale_by:
                if alias.scale_by not in self.context:
                    raise KeyError(f"{alias.name}: scaled by {alias.scale_by!r}, which {self.part_type}'s context lacks")
                scale *= float(self.context[alias.scale_by])
            out.append(replace(alias, name=prefix + alias.name, scale=scale, scale_by=""))
        return out

    def collection(self) -> Collection:
        return Collection(self)

    def table(self) -> str:
        collection = self.collection()
        width = max(len(name) for name in collection.aliases)
        lines = [f"{self.part_type}.metrics: {len(collection.aliases)} aliases over {collection.computations} states"]
        for name, alias in collection.aliases.items():
            unit = alias.unit or "-"
            lines.append(f"  {name:<{width}}  [{unit:>3}]  {alias.metric!r}{'' if alias.metric.live else '  (end of run)'}")
        return "\n".join(lines)


class Collection:
    """A part's aliases over value states: one state per (signal, state key); each alias reads its own value off it."""

    def __init__(self, aliases: PartMetrics | Sequence[Alias]) -> None:
        resolved = aliases.flattened() if isinstance(aliases, PartMetrics) else list(aliases)
        self.part = aliases if isinstance(aliases, PartMetrics) else None
        self.aliases: dict[str, Alias] = {}
        self.groups: dict[tuple, Metric] = {}       # state key -> the metric whose update feeds it
        self.key_of: dict[str, tuple] = {}
        for alias in resolved:
            if alias.name in self.aliases:
                raise ValueError(f"two aliases named {alias.name!r}")
            if alias.scale_by:
                raise ValueError(f"{alias.name}: scale_by needs a PartMetrics with that context")
            key = (tuple(alias.signal), alias.metric.state_key)
            self.groups.setdefault(key, alias.metric)
            self.aliases[alias.name] = alias
            self.key_of[alias.name] = key

    @property
    def computations(self) -> int:
        return len(self.groups)

    # ---- the value-state fold ------------------------------------------------------------------------------------
    def empty(self) -> dict:
        return {key: metric.empty() for key, metric in self.groups.items()}

    def update(self, state: Mapping, run: Any) -> dict:
        """One chunk (live) or the whole run (offline); a new state, the old one untouched."""

        out = dict(state)
        for key, metric in self.groups.items():
            target_name, measured_name = key[0]
            out[key] = metric.update(state[key], run.streams[target_name], run.streams[measured_name])
        return out

    def close(self, state: Mapping) -> dict:
        return {key: self.groups[key].close(value) for key, value in state.items()}

    def merge(self, a: Mapping, b: Mapping) -> dict:
        return {key: self.groups[key].merge(a[key], b[key]) for key in a}

    def fold(self, run: Any) -> dict:
        return self.close(self.update(self.empty(), run))

    def fold_many(self, runs: Iterable[Any]) -> dict:
        """Every run closed on its own, then merged: a pooled state (a rung, a station)."""

        merged = None
        for run in runs:
            state = self.fold(run)
            merged = state if merged is None else self.merge(merged, state)
        return merged if merged is not None else self.empty()

    # ---- reading values --------------------------------------------------------------------------------------------
    def value(self, state: Mapping, name: str, *, si: bool = False) -> Any:
        alias = self.aliases[name]
        value = alias.metric.compute(state[self.key_of[name]])
        if value is None or isinstance(value, bool) or si:
            return value
        # WHY: SI inside, the display unit only here (Zach 2026-11-11: angles shown in degrees, code and data SI)
        return float(value) * alias.scale

    def values(self, state: Mapping, *, si: bool = False) -> dict[str, Any]:
        return {name: self.value(state, name, si=si) for name in self.aliases}

    def items(self, state: Mapping, name: str) -> list[float] | None:
        """The alias's occurrences in display units (every stop for a stop reading); None when the occurrence is the
        run itself."""

        alias = self.aliases[name]
        items = alias.metric.items(state[self.key_of[name]])
        return None if items is None else [float(item) * alias.scale for item in items]

    def records(self, state: Mapping, run: Any) -> tuple[MetricRecord, ...]:
        """The values as stored records: alias, display value and unit, the identity and version that made them."""

        values = self.values(state)
        return tuple(MetricRecord(name, values[name], alias.unit, type(alias.metric).__name__, alias.metric.config(),
                                  alias.metric.version, run.id, run.provenance, dict(run.labels))
                     for name, alias in self.aliases.items())

    def stream(self) -> CollectionStream:
        return CollectionStream(self)

    # ---- views ---------------------------------------------------------------------------------------------------------
    def view(self, name: str, states: Mapping | Mapping[str, Mapping], *, title: str = "", target: float | None = None,
             target_label: str = "") -> Any:
        """The alias's occurrences (or values) as one View; ``states`` is one state or {group label: state}."""

        from .views import View

        alias = self.aliases[name]
        groups = states if states and all(isinstance(key, str) for key in states) else {"": states}
        x, y, group = [], [], []
        for label, state in groups.items():
            items = self.items(state, name)
            if items is None:
                value = self.value(state, name)
                items = [] if value is None else [float(value)]
            for item in items:
                x.append(len(x))
                y.append(item)
                group.append(label)
        return View(title=title or name, unit=alias.unit, hint=alias.metric.hint, x=x, y=y,
                    group=group if any(group) else [], target=target, target_label=target_label,
                    x_label="stop" if alias.metric.hint == "distribution" else "", y_label=alias.unit)

    def plot(self, name: str, states: Mapping, *, ax: Any = None, path: Any = None, **view: Any) -> tuple[Any, Any]:
        from .views import plot

        return plot(self.view(name, states, **view), ax=ax, path=path)

    def __repr__(self) -> str:
        return f"Collection({len(self.aliases)} aliases over {self.computations} states)"


class CollectionStream:
    """torchmetrics' MetricCollection face: update a chunk, end the run, compute every alias, reset."""

    def __init__(self, collection: Collection) -> None:
        self.collection = collection
        self.state = collection.empty()

    def update(self, run: Any) -> None:
        self.state = self.collection.update(self.state, run)

    def end_run(self) -> None:
        self.state = self.collection.close(self.state)

    def compute(self) -> dict[str, Any]:
        return self.collection.values(self.collection.close(self.state))

    def reset(self) -> None:
        self.state = self.collection.empty()


def across_runs(alias: Alias, values: Sequence[Any]) -> Any:
    """An alias's per-run values combined by its ``across`` rule (median / max); None when no run has a value."""

    present = [value for value in values if value is not None]
    if not present:
        return None
    return statistics.median(present) if alias.across == "median" else max(present)
