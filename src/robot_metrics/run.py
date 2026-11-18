"""One run, the atomic unit (Zach, 3000-ft view, 2026-11-17: "runs are the atomic unit"): named robot_msgs streams plus
the labels a roll-up groups by, its provenance, its outcome, how its pair was aligned, and links out.

    run = Run(id, streams={"plan": plan, "state": replies, "target": plan_at_each_reply}, labels={"station": "can11", ...})
    run.chunks(7)            # the run as a live stream would deliver it: every stream cut at the same instants
    run.split_at([t1, t2])   # the same, cut at chosen instants (a chunk boundary in the middle of a dwell)

WHY these fields (the run-data MOC, Nov 17: "the missing layer is the wrapper around one run"): a stable id, named
streams, labels, provenance, an outcome and links. Provenance (real | sim | replay) travels with the values though no
metric reads it: nuPlan's open- and closed-loop scores are negatively correlated, so they are never averaged blind.
Rerun stays a pointer in ``links`` (Zach, Oct 26: the dataset is the dataset; viewers are presentation layers).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .alignment import Alignment


@dataclass(frozen=True)
class Run:
    id: str
    streams: Mapping[str, Any]                   # "plan", "state", "target", "tool/target", "tool/state": robot_msgs
    labels: Mapping[str, Any] = field(default_factory=dict)   # deployment, station, part_type, rung, kind
    provenance: str = "real"                     # real | sim | replay
    outcome: str = "complete"                    # complete | faulted
    alignment: Alignment | None = None           # how "target" was put on the state's instants
    links: Mapping[str, str] = field(default_factory=dict)    # "source" file, "rrd", "wandb"

    @property
    def faulted(self) -> bool:
        return self.outcome != "complete"

    def split_at(self, times: Sequence[float]) -> tuple[Run, ...]:
        """Chunks that end at each of ``times`` (and one after the last): a sample at t belongs to the chunk whose
        window (previous edge, edge] holds it. Every stream is cut at the same instants."""

        edges = sorted({float(edge) for edge in times}) + [math.inf]
        out, low = [], -math.inf
        for high in edges:
            out.append(replace(self, streams={name: stream[np.flatnonzero((np.asarray(stream.t) > low)
                                                                          & (np.asarray(stream.t) <= high))]
                                              for name, stream in self.streams.items()}))
            low = high
        return tuple(out)

    def chunks(self, rows: int, *, by: str = "state") -> tuple[Run, ...]:
        """The run as a live stream would deliver it: ``rows`` samples of ``by`` at a time."""

        return self.split_at(np.asarray(self.streams[by].t)[rows - 1::rows].tolist())

    def __repr__(self) -> str:
        streams = ", ".join(f"{name} {len(stream)}" for name, stream in self.streams.items())
        labels = " ".join(f"{key}={value}" for key, value in self.labels.items() if key in ("station", "rung", "kind"))
        return f"Run({self.provenance} {labels} · {streams})"
