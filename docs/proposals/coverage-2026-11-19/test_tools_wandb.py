"""The gallery's W&B column sends summary statistics only (Zach, 2026-11-21: "to help limit W and B storage I think we
need to just be using it for summary statistics; for the media-rich things we should be using rerun!").

Runs `tools_wandb.log_runs()` on the real cached gallery data against a fake `wandb` module: every W&B type that uploads
a file (Table, Image, Video, Html, Object3D, Plotly, Audio, Histogram, Artifact, plot.*, plot_table) and the API raise,
and each fake run records what it is handed. Then builds the report's panels against a fake `wandb_workspaces` whose
media and custom-chart panels raise, and checks every panel reads a key the runs logged. Nothing here reaches W&B.

    cd /home/bam/robot_metrics/docs/proposals/coverage-2026-11-19 && taskset -c 2-5,16,17 /home/bam/robot_metrics/.venv/bin/python -m pytest -q test_tools_wandb.py
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import types

import pytest

from core_types import CACHE, TYPES

UPLOADING = ("Table", "Image", "Video", "Html", "Object3D", "Plotly", "Audio", "Histogram", "Artifact", "plot_table",
             "Api", "save", "log_artifact")
PLOTS = ("line", "line_series", "bar", "histogram", "scatter", "confusion_matrix", "pr_curve", "roc_curve")

pytestmark = pytest.mark.skipif(
    not all((CACHE / f"core_{core.id.replace('-', '_')}.pkl").is_file() for core in TYPES),
    reason=f"the gallery's cached payloads are not in {CACHE} (core_types.py builds them from /archive)")


def _refuse(name: str):
    def refuse(*args, **kwargs):
        raise AssertionError(f"wandb.{name} sends W&B more than summary statistics (or reaches W&B)")
    return refuse


class Config(dict):
    def update(self, values=(), **more):  # wandb's run.config.update
        super().update(values, **more)


class FakeRun:
    def __init__(self, **init):
        self.init, self.id, self.url = init, init["id"], f"https://wandb.ai/fake/{init['id']}"
        self.config = Config(init.get("config") or {})
        self.history: list[dict] = []
        self.summary: dict = {}
        self.finished = False

    def log(self, data: dict, step=None, commit=None) -> None:
        self.history.append(dict(data))
        self.summary.update(data)

    def define_metric(self, *args, **kwargs) -> None:
        pass

    def finish(self) -> None:
        self.finished = True


@pytest.fixture
def column(monkeypatch, tmp_path):
    """tools_wandb imported against the fake wandb; returns (module, runs started)."""
    runs: list[FakeRun] = []
    fake = types.ModuleType("wandb")
    for name in UPLOADING:
        setattr(fake, name, _refuse(name))
    fake.plot = types.SimpleNamespace(**{name: _refuse(f"plot.{name}") for name in PLOTS})
    fake.init = lambda **init: runs.append(FakeRun(**init)) or runs[-1]
    monkeypatch.setitem(sys.modules, "wandb", fake)
    sys.modules.pop("tools_wandb", None)
    module = importlib.import_module("tools_wandb")
    monkeypatch.setattr(module, "WANDB_DIR", tmp_path / "wandb")
    yield module, runs
    sys.modules.pop("tools_wandb", None)


def _is_scalar(value) -> bool:
    return isinstance(value, (int, float)) and (isinstance(value, int) or math.isfinite(value))


def test_every_type_logs_only_summary_scalars(column):
    tools_wandb, runs = column
    logged = tools_wandb.log_runs()
    assert list(logged) == [core.id for core in TYPES] and len(runs) == len(TYPES)
    recorded_runs = len(tools_wandb.by_id("histogram-over-runs").data()["qd_per_run"])
    for core, run in zip(TYPES, runs):
        assert run.finished and run.init["job_type"] == core.id
        for row in run.history:
            bad = {key: type(value).__name__ for key, value in row.items() if not _is_scalar(value)}
            assert not bad, f"{core.id} logged non-scalars: {bad}"
        bad = {key: type(value).__name__ for key, value in run.summary.items() if not _is_scalar(value)}
        assert not bad, f"{core.id} summary holds non-scalars: {bad}"
        # small config, and the pointer to the Rerun column's picture of this type
        assert len(json.dumps(dict(run.config))) < 4000, f"{core.id} config is not small"
        assert run.config["rerun_rrd"].endswith(f"tools/rerun/{core.id}.rrd")
        assert run.config["rerun_writer"].endswith(f"tools_rerun.py::{core.id.replace('-', '_')}")
        # verify()'s expectations match what the column now logs
        for key, rows in tools_wandb.EXPECTED[core.id].items():
            assert key in run.summary, f"{core.id}: EXPECTED names {key}, which it never logs"
            if rows is not None:
                found = sum(1 for row in run.history if key in row)
                assert found == (recorded_runs if rows == "runs" else rows), (core.id, key, found)


class _Panel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def keys(self) -> list[str]:
        k = self.kwargs
        return list(k.get("y") or []) + list(k.get("metrics") or []) + ([k["metric"]] if k.get("metric") else [])


def test_the_report_reads_only_logged_scalars(column):
    tools_wandb, runs = column
    tools_wandb.log_runs()
    wr = types.SimpleNamespace(Layout=lambda **kwargs: kwargs, LinePlot=_Panel, BarPlot=_Panel, ScalarChart=_Panel,
                               MarkdownPanel=_Panel, MediaBrowser=_refuse("MediaBrowser"),
                               CustomChart=types.SimpleNamespace(from_table=_refuse("CustomChart.from_table")))
    for core, run in zip(TYPES, runs):
        panels = tools_wandb._report_panels(wr, core.id)
        assert panels, core.id
        for panel in panels:
            missing = [key for key in panel.keys() if key not in run.summary]
            assert not missing, f"{core.id}'s report reads {missing}, which its run never logs"
        pointers = [panel.kwargs["markdown"] for panel in panels if "markdown" in panel.kwargs]
        assert all(f"{core.id}.rrd" in text for text in pointers), core.id
