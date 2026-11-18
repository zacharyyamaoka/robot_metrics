"""Rerun: a View as log calls (one scalar series per group, the target as its own series), and an .rrd writer.

    calls(view, "metrics/can11/tip_stop_mm")              # (entity, archetype, args): built, inspectable, testable
    write_rrd(view, "metrics/can11/tip_stop_mm", path)    # logs them to a local .rrd when rerun-sdk is importable

A local .rrd only: a link to it goes through the always-on launcher (bam-replay-link on 127.0.0.1:8771), never a
server someone has to start (Zach, 3000-ft view, 2026-11-17).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..views import View


def calls(view: View, entity: str) -> list[tuple[str, str, dict[str, Any]]]:
    out = []
    for i in range(len(view.y)):
        group = view.group[i] if view.group else "value"
        out.append((f"{entity}/{group}", "Scalars", {"index": int(i), "scalars": [float(view.y[i])]}))
    if view.target is not None:
        out.append((f"{entity}/target", "Scalars", {"index": 0, "scalars": [float(view.target)]}))
    return out


def write_rrd(view: View, entity: str, path: str | Path, application: str = "robot_metrics") -> Path | None:
    """The calls logged to ``path``; None when rerun-sdk is not installed (the 'rerun' extra)."""

    try:
        import rerun as rr
    except ImportError:
        return None
    recording = rr.RecordingStream(application)
    recording.save(str(path))
    for name, archetype, args in calls(view, entity):
        recording.set_time("index", sequence=args["index"])
        recording.log(name, getattr(rr, archetype)(args["scalars"]))
    recording.flush()
    return Path(path)
