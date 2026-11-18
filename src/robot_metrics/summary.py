"""Across runs: strata, their tails, Wilson shares for bools, and the highest rung that passes.

    strata = summarize(collection, runs, by=("station", "rung"))   # {(station, rung): Stratum}
    stratum.values["tip stop error"]          # each alias by its own across rule (pooled stops, median of runs, ...)
    stratum.tails["tip stop error"]           # n, median, p95, worst of its occurrences: what a gate reads
    stratum.shares["all stops settled"]       # a bool over runs: successes of trials, Wilson 95 %
    highest_passing({(station, rung): passed, ...})                # {station: the highest rung with every one below it passing}

WHY a tail per stratum and never an IQM for a gate (the prior-art note §1 and falsification 7): rliable's IQM trims a
quarter off each end, and CAN 11's over-1 mm stops sit in that trimmed quarter, so how far over they are could never
move it. A gate reads a tail (the worst stop, p95) per stratum with n beside it; an interval estimate is for comparing
two controllers, not for a gate. Strata are never pooled across stations: their levers differ.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .collection import Collection, across_runs

BUILT_IN = ("runs", "faulted runs")


@dataclass(frozen=True)
class Tail:
    """A stratum's occurrences of one alias (display units): how many, the middle, and the tail."""

    n: int
    median: float | None
    p95: float | None
    worst: float | None

    @classmethod
    def of(cls, values: Sequence[float]) -> Tail:
        if not values:
            return cls(0, None, None, None)
        array = np.asarray(values, float)
        return cls(int(array.size), _quantile(array, 50.0), _quantile(array, 95.0), float(np.max(array)))


def _quantile(array: np.ndarray, percent: float) -> float:
    """A quantile that keeps a never-settled stop (inf) truthful: interpolating between two infs is inf, not NaN."""

    with np.errstate(invalid="ignore"):
        value = float(np.percentile(array, percent))
    return math.inf if math.isnan(value) else value


@dataclass(frozen=True)
class Share:
    """A bool alias over a stratum's runs: successes of trials, with the Wilson 95 % interval."""

    successes: int
    trials: int
    low: float
    high: float

    @property
    def share(self) -> float | None:
        return self.successes / self.trials if self.trials else None


@dataclass(frozen=True)
class Stratum:
    key: tuple
    by: tuple[str, ...]
    run_ids: tuple[str, ...]
    faulted: int
    merged: Mapping                        # the runs' states merged (pooled)
    states: tuple[Mapping, ...]            # each run's own closed state
    values: Mapping[str, Any]              # display units: each alias by its across rule, plus "runs", "faulted runs"
    units: Mapping[str, str]
    occurrences: Mapping[str, tuple[float, ...]]   # every stop for a stop reading, every run's value otherwise
    tails: Mapping[str, Tail]
    shares: Mapping[str, Share]

    def __repr__(self) -> str:
        labels = ", ".join(f"{name}={value}" for name, value in zip(self.by, self.key))
        return f"Stratum({labels}: {len(self.run_ids)} runs, {self.faulted} faulted)"


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a share (a bool metric over runs), 95 % by default; (0, 1) with no trials."""

    if trials == 0:
        return (0.0, 1.0)
    share = successes / trials
    centre = (share + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z / (1 + z * z / trials) * math.sqrt(share * (1 - share) / trials + z * z / (4 * trials * trials))
    return (centre - half, centre + half)


def summarize(collection: Collection, runs: Sequence[Any], by: Sequence[str]) -> dict[tuple, Stratum]:
    """Group runs by their labels, fold each run, merge each stratum, read every alias by its across rule."""

    grouped: dict[tuple, list] = {}
    for run in runs:
        grouped.setdefault(tuple(run.labels[name] for name in by), []).append(run)
    out = {}
    for key, members in grouped.items():
        states = tuple(collection.fold(run) for run in members)
        merged = states[0]
        for state in states[1:]:
            merged = collection.merge(merged, state)
        per_run = {name: [collection.value(state, name) for state in states] for name in collection.aliases}
        values: dict[str, Any] = {}
        occurrences: dict[str, tuple[float, ...]] = {}
        tails: dict[str, Tail] = {}
        shares: dict[str, Share] = {}
        for name, alias in collection.aliases.items():
            values[name] = (collection.value(merged, name) if alias.across == "pool"
                            else across_runs(alias, per_run[name]))
            if alias.metric.value_type == "bool":
                trials = [value for value in per_run[name] if value is not None]
                successes = sum(bool(value) for value in trials)
                shares[name] = Share(successes, len(trials), *wilson(successes, len(trials)))
                continue
            items = collection.items(merged, name)
            found = tuple(items) if items is not None else tuple(v for v in per_run[name] if v is not None)
            occurrences[name] = found
            tails[name] = Tail.of(found)
        faulted = sum(run.outcome != "complete" for run in members)
        values.update({"runs": len(members), "faulted runs": faulted})
        units = {name: alias.unit for name, alias in collection.aliases.items()} | {"runs": "", "faulted runs": ""}
        out[key] = Stratum(key, tuple(by), tuple(run.id for run in members), faulted, merged, states, values, units,
                           occurrences, tails, shares)
    return out


def highest_passing(passed: Mapping[tuple, bool]) -> dict[Any, Any]:
    """Per first key (a station), the highest second key (a rung) such that it and every lower one pass; 0.0 when
    the lowest fails. WHY "every rung up to it": a ladder is only as high as its first failing step."""

    by_station: dict[Any, list] = {}
    for (station, rung), ok in passed.items():
        by_station.setdefault(station, []).append((rung, ok))
    out = {}
    for station, rungs in sorted(by_station.items()):
        reached = 0.0
        for rung, ok in sorted(rungs):
            if not ok:
                break
            reached = rung
        out[station] = reached
    return out
