"""The catalog (the seed of RobotPart.metrics) carries Zach's Metric fields, and every result has a standard view.

Zach's `Metric` (Daily Note Nov 17, amendment 1): a name, a description, references, a function, a unit, a target kept
outside compute(), a bool or number value. G7: units belong to the alias, never to the base class's name.
"""

from __future__ import annotations

import re

import pytest

import synthetic
from robot_metrics import Run, catalog
from robot_metrics.adapters import rerun as rerun_adapter
from robot_metrics.adapters import wandb as wandb_adapter

ZACH_FIELDS = {"name", "description", "references", "function", "unit", "target", "value_type"}
UNIT_WORDS = {"deg", "degrees", "mm", "rad", "mrad", "ms", "millimetres", "radians", "s", "seconds", "nm"}


def all_cards():
    parts = (catalog.integrated_servo(), catalog.pendulum_arm(0.339), catalog.rigid_chain())
    return [card for part in parts for card in (alias.card() for alias in part.collection().aliases.values())]


def test_every_alias_card_has_zachs_fields():
    cards = all_cards()
    assert all(ZACH_FIELDS <= set(card) for card in cards)
    assert all(card["target"] is None for card in cards)            # the target lives on an Objective
    assert all(card["description"] and card["references"] for card in cards)
    assert {"number", "bool"} <= {card["value_type"] for card in cards}
    assert all(card["live"] in (True, False) and card["version"] for card in cards)


def test_no_base_metric_names_a_unit():
    for card in all_cards():
        words = {word.lower() for word in re.findall(r"[A-Z][a-z]*|[a-z]+", card["base"])}
        assert not words & UNIT_WORDS, card["base"]


def test_one_base_serves_several_meanings():
    bases = [card["base"] for card in all_cards()]
    assert bases.count("ErrorStat") >= 4 and bases.count("StopStat") >= 4


def test_both_part_types_declare_their_metrics():
    assert {alias.name for alias in catalog.integrated_servo().aliases} >= {
        "tracking RMS", "tracking peak", "stop error", "stop error p95", "stop error worst", "overshoot", "lag",
        "limiter share", "stops"}
    assert {alias.name for alias in catalog.rigid_chain().collection().aliases.values()} >= {
        "j2/tracking RMS", "c1/tracking RMS", "tool tracking RMS", "tool worst"}
    gates = catalog.pendulum_gates()
    assert set(gates) == {"tracks", "tip stop (median)", "tip stop (every stop)"}
    assert all(objective.owner for gate in gates.values() for objective in gate.objectives)


def _stops_view():
    collection = catalog.pendulum_arm(0.339).collection()
    states = {f"seed {seed}": collection.fold(synthetic.run(seed)) for seed in range(3)}
    return collection, collection.view("tip stop error", states, target=1.0, target_label="Zach's 1 mm")


def test_a_view_carries_the_display_unit_and_the_data_ui_hint():
    collection, view = _stops_view()
    assert view.unit == "mm" and view.hint == "distribution" and view.target == 1.0
    assert len(view.y) == 6 and set(view.group) == {"seed 0", "seed 1", "seed 2"}


def test_plot_returns_a_matplotlib_figure_and_axes(tmp_path):
    pytest.importorskip("matplotlib")
    collection, view = _stops_view()
    fig, ax = collection.plot("tip stop error", {"one run": collection.fold(synthetic.run())}, path=tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 1000 and ax.get_ylabel() == "mm"
    metric = collection.aliases["tip stop error"].metric           # the base metric plots its own SI items
    fig, ax = metric.plot(collection.fold(synthetic.run())[collection.key_of["tip stop error"]])
    assert fig is ax.figure


def test_the_same_view_feeds_wandb_and_rerun(tmp_path):
    _, view = _stops_view()
    payload = wandb_adapter.payload(view, "synthetic/tip_stop_mm")
    assert payload["table"]["columns"][:2] == ["x", "mm"] and len(payload["table"]["data"]) == 6
    assert payload["summary"]["synthetic/tip_stop_mm/target"] == 1.0 and payload["hint"] == "distribution"
    calls = rerun_adapter.calls(view, "metrics/synthetic/tip_stop_mm")
    assert len(calls) == 7 and calls[-1][0].endswith("/target")
    written = rerun_adapter.write_rrd(view, "metrics/synthetic/tip_stop_mm", tmp_path / "stops.rrd")
    assert written is None or written.stat().st_size > 0


def test_run_chunks_cover_every_sample_once():
    run = synthetic.run()
    chunks = run.chunks(7)
    for name, stream in run.streams.items():
        assert sum(len(chunk.streams[name]) for chunk in chunks) == len(stream)
    assert isinstance(chunks[0], Run) and chunks[0].id == run.id
