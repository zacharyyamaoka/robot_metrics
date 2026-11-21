"""adapters.wandb.log sends W&B summary scalars only; the rows go to a local jsonl (Zach, 2026-11-21: "to help limit W
and B storage I think we need to just be using it for summary statistics; for the media-rich things we should be using
rerun!").

Runs ``log`` against a fake ``wandb`` module: every W&B type that uploads a file (Table, Image, Video, Html, Object3D,
Plotly, Audio, Histogram, Artifact, plot.*, plot_table) raises, and the fake run records what it is handed. Nothing here
reaches W&B.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from robot_metrics.adapters import wandb as wandb_adapter
from robot_metrics.views import View

UPLOADING = ("Table", "Image", "Video", "Html", "Object3D", "Plotly", "Audio", "Histogram", "Artifact", "plot_table")


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.summary: dict = {}

    def log(self, data: dict, *args, **kwargs) -> None:
        self.logged.append(dict(data))


def _refuse(name: str):
    def refuse(*args, **kwargs):
        raise AssertionError(f"wandb.{name} would upload more than summary statistics")
    return refuse


@pytest.fixture
def fake_wandb(monkeypatch):
    module = types.ModuleType("wandb")
    for name in UPLOADING:
        setattr(module, name, _refuse(name))
    module.plot = types.SimpleNamespace(**{name: _refuse(f"plot.{name}") for name in (
        "line", "line_series", "bar", "histogram", "scatter", "confusion_matrix", "pr_curve", "roc_curve")})
    module.run = None
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def _view() -> View:
    return View(title="tip stops", unit="mm", hint="distribution", x=[0, 1, 2, 3], y=[0.4, 1.3, 0.2, 0.9],
                group=["rung 0.2", "rung 0.2", "rung 1", "rung 1"], target=1.0)


def _only_scalars(sent: dict, allowed_strings: set[str] = frozenset()) -> None:
    for key, value in sent.items():
        if key in allowed_strings:
            assert isinstance(value, str), (key, value)
        else:
            assert isinstance(value, (int, float)), f"{key} = {value!r} is not a summary scalar"


def test_log_sends_only_the_summary_scalars(fake_wandb):
    run = FakeRun()
    sent = wandb_adapter.log(_view(), "can11/tip_stop_mm", run)
    assert run.logged == [sent] and run.summary == {}
    assert sent == {"can11/tip_stop_mm/n": 4, "can11/tip_stop_mm/max": 1.3, "can11/tip_stop_mm/target": 1.0}
    _only_scalars(sent)


def test_log_writes_the_rows_locally_and_sends_their_path(fake_wandb, tmp_path):
    run = FakeRun()
    fake_wandb.run = run                                  # the default target: wandb.run
    rows_path = tmp_path / "views" / "tip_stop_mm.jsonl"
    sent = wandb_adapter.log(_view(), "can11/tip_stop_mm", rows_path=rows_path)
    assert run.logged == [sent]
    _only_scalars(sent, {"can11/tip_stop_mm/rows_path"})
    assert sent["can11/tip_stop_mm/rows_path"] == str(rows_path.resolve())
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    built = wandb_adapter.payload(_view(), "can11/tip_stop_mm")
    assert rows == [dict(zip(built["table"]["columns"], row)) for row in built["table"]["data"]]
    assert not rows_path.with_name(rows_path.name + ".partial").exists()


def test_log_without_an_open_run_still_refuses(fake_wandb):
    with pytest.raises(RuntimeError, match="no open wandb run"):
        wandb_adapter.log(_view(), "can11/tip_stop_mm")
