"""W&B: a View's summary scalars on a run; its rows stay on this computer (one W&B project per deployment, Zach 2026-11-17).

    payload(view, "can11/tip_stop_mm")                    # built, never sent: the rows + the summary, for any caller to inspect
    log(view, "can11/tip_stop_mm", run)                    # sends the summary scalars to an open wandb run (the 'wandb' extra)
    log(view, "can11/tip_stop_mm", run, rows_path=path)   # + the rows as jsonl at `path`, and that path on the run

Summary statistics only, since 2026-11-21 (Zach: "to help limit W and B storage I think we need to just be using it for
summary statistics; for the media-rich things we should be using rerun!"). ``log`` used to send ``payload``'s rows as a
``wandb.Table``; it now sends ``n``, ``max`` and ``target`` as scalars, and the rows go to a local jsonl file (one object
per row) whose path is logged as a string. The picture of the rows is ``adapters.rerun.write_rrd`` (a local .rrd).
``payload`` is unchanged: it still builds the rows, it just never ships them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..views import View


def payload(view: View, prefix: str) -> dict[str, Any]:
    columns = ["x", view.y_label or view.unit] + (["group"] if view.group else [])
    data = [[view.x[i], float(view.y[i])] + ([view.group[i]] if view.group else []) for i in range(len(view.y))]
    summary: dict[str, Any] = {f"{prefix}/n": len(view.y), f"{prefix}/max": max(view.y) if view.y else None}
    if view.target is not None:
        summary[f"{prefix}/target"] = view.target
    return {"table": {"columns": columns, "data": data}, "summary": summary, "title": view.title, "hint": view.hint}


def write_rows(built: dict[str, Any], path: str | Path) -> Path:
    """``payload``'s rows as jsonl at ``path``: one ``{column: value}`` object per line, written to a .partial then renamed
    so a reader never sees half a file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = built["table"]["columns"]
    partial = path.with_name(path.name + ".partial")
    partial.write_text("".join(json.dumps(dict(zip(columns, row)), default=str) + "\n" for row in built["table"]["data"]),
                       encoding="utf-8")
    partial.replace(path)
    return path


def log(view: View, prefix: str, run: Any = None, rows_path: str | Path | None = None) -> dict[str, Any]:
    """Send the summary scalars to ``run`` (default: wandb.run); with ``rows_path``, write the rows there as jsonl and
    send that path as the string ``<prefix>/rows_path``. Returns exactly what was sent."""

    import wandb

    built = payload(view, prefix)
    target = run if run is not None else wandb.run
    if target is None:
        raise RuntimeError("no open wandb run: wandb.init(project=<the deployment>) first")
    # WHY scalars and a path, never a wandb.Table (Zach, 2026-11-21: "to help limit W and B storage I think we need to just
    # be using it for summary statistics; for the media-rich things we should be using rerun!"): a Table uploads a
    # .table.json per log call; the rows stay local and W&B gets where they are.
    sent: dict[str, Any] = {key: value for key, value in built["summary"].items() if value is not None}
    if rows_path is not None:
        sent[f"{prefix}/rows_path"] = str(write_rows(built, rows_path).resolve())
    target.log(sent)
    return sent
