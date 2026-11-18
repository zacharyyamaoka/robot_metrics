"""Standard views: one ``View`` record, three sinks (matplotlib here; W&B and Rerun in ``robot_metrics.adapters``).

    view = collection.view("tip stop error", {"0.2": state, ...}, target=1.0, target_label="Zach's 1 mm")
    fig, ax = plot(view, path="stops.png")         # torchmetrics' Metric.plot contract: (fig, ax)

``hint`` is the data-ui view contract (RobotDx core.HINTS): the same string tells data-ui which stock view shows the
number, so a metric carries it instead of each sink guessing. WHY one record for every sink (Zach, 2026-11-17:
"standard visuzliation ... matplotlib but also like w&B and rerun"): one result feeds every viewer. matplotlib is
imported inside ``plot`` only: the runtime core stays numpy + robot_msgs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HINTS = ("gate_matrix", "threshold_band", "kpi_trend", "rollup_tree", "evidence", "distribution", "level_history",
         "path_overlay")


@dataclass(frozen=True)
class View:
    """What a result looks like as data: points (x, y) in groups, the display unit, the target line if any."""

    title: str
    unit: str
    hint: str
    x: Sequence[Any]
    y: Sequence[float]
    group: Sequence[str] = ()
    target: float | None = None
    target_label: str = ""
    x_label: str = ""
    y_label: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hint not in HINTS:
            raise ValueError(f"hint {self.hint!r} is not a data-ui view: {HINTS}")


def plot(view: View, ax: Any = None, path: str | Path | None = None) -> tuple[Any, Any]:
    """(fig, ax) on matplotlib (the 'viz' extra); draws into ``ax`` when given, saves to ``path`` when given."""

    import matplotlib

    if ax is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8.6, 3.6), dpi=130)
    groups = list(dict.fromkeys(view.group)) if view.group else [""]
    for name in groups:
        rows = [i for i in range(len(view.y)) if not view.group or view.group[i] == name]
        xs, ys = [view.x[i] for i in rows], [view.y[i] for i in rows]
        if view.hint == "distribution":
            ax.plot(xs, ys, "o", ms=3.5, alpha=0.85, label=name or None)
        else:
            ax.plot(xs, ys, "o-", ms=4, lw=1.4, label=name or None)
    if view.target is not None:
        ax.axhline(view.target, color="#c0392b", lw=1.2, ls="--", label=view.target_label or f"target {view.target:g}")
    ax.set_title(view.title, fontsize=10)
    ax.set_xlabel(view.x_label)
    ax.set_ylabel(view.y_label or view.unit)
    ax.grid(alpha=0.3)
    if len(groups) > 1 or view.target is not None:
        # WHY outside the axes: a legend inside can hide the very point a gate fails on (the worst stop)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig, ax
