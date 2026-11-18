"""W&B: a View as a wandb.Table payload plus summary keys (one W&B project per deployment, Zach 2026-11-17).

    payload(view, "can11/tip_stop_mm")     # built, never sent: a dict any caller can inspect or test
    log(view, "can11/tip_stop_mm", run)    # sends it to an open wandb run (the 'wandb' extra)
"""

from __future__ import annotations

from typing import Any

from ..views import View


def payload(view: View, prefix: str) -> dict[str, Any]:
    columns = ["x", view.y_label or view.unit] + (["group"] if view.group else [])
    data = [[view.x[i], float(view.y[i])] + ([view.group[i]] if view.group else []) for i in range(len(view.y))]
    summary: dict[str, Any] = {f"{prefix}/n": len(view.y), f"{prefix}/max": max(view.y) if view.y else None}
    if view.target is not None:
        summary[f"{prefix}/target"] = view.target
    return {"table": {"columns": columns, "data": data}, "summary": summary, "title": view.title, "hint": view.hint}


def log(view: View, prefix: str, run: Any = None) -> dict[str, Any]:
    """Send the payload to ``run`` (default: wandb.run). Returns what was sent."""

    import wandb

    built = payload(view, prefix)
    target = run if run is not None else wandb.run
    if target is None:
        raise RuntimeError("no open wandb run: wandb.init(project=<the deployment>) first")
    table = wandb.Table(columns=built["table"]["columns"], data=built["table"]["data"])
    target.log({f"{prefix}/table": table, **{key: value for key, value in built["summary"].items() if value is not None}})
    return built
