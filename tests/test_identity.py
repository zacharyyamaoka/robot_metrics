"""Test 4 · identity: a metric is its class + frozen config; changing an Objective's target never changes it.

WHY (Zach, 2026-11-17: "100% agreed keep the target outside of compute"; the prior-art note: RobotDx's Limits ride
inside each config, so a new target changes the metric's identity): here the target lives on the Objective only.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

import synthetic
from robot_metrics import Alias, ErrorStat, Objective, StopStat, catalog, summarize

UPDATES: list[str] = []


@dataclass(frozen=True)
class CountedStopStat(StopStat):
    def update(self, state, target, measured):
        UPDATES.append(repr(self))
        return super().update(state, target, measured)


def test_identity_is_class_plus_frozen_config():
    a, b = ErrorStat(statistic="rms", rows="moving"), ErrorStat(statistic="rms", rows="moving")
    assert a == b and hash(a) == hash(b) and a.identity == b.identity
    assert a != ErrorStat(statistic="rms", rows="all")
    assert a.identity == ("ErrorStat", (("statistic", "rms"), ("rows", "moving"), ("field", "positions"), ("joint", 0)))
    with pytest.raises(FrozenInstanceError):
        a.statistic = "max_abs"


def test_an_alias_name_is_only_a_label():
    metric = StopStat(of="error", statistic="median")
    one = Alias("tip stop error", metric, unit="mm", scale=339.0)
    two = Alias("T4 stop error", metric, unit="deg", scale=57.29577951308232)
    assert one.metric.identity == two.metric.identity
    assert one.card()["base"] == two.card()["base"] == "StopStat"


def test_changing_an_objective_target_never_changes_the_metric_identity():
    collection = catalog.pendulum_arm(0.339).collection()
    before = {name: alias.metric.identity for name, alias in collection.aliases.items()}
    keys_before = dict(collection.key_of)
    strict = Objective("tip stop error", op="<=", value=1e-6, unit="mm", share=1.0, on="items", owner="proposed")
    loose = Objective("tip stop error", op="<=", value=1.0, unit="mm", share=1.0, on="items", owner="Zach 2026-11-11")
    stratum = next(iter(summarize(collection, [synthetic.run(seed) for seed in range(3)], by=("station",)).values()))
    strict_verdict, loose_verdict = strict.judge(stratum), loose.judge(stratum)
    assert {name: alias.metric.identity for name, alias in collection.aliases.items()} == before
    assert collection.key_of == keys_before
    assert strict != loose and strict_verdict.total == loose_verdict.total > 0
    assert (strict_verdict.verdict, loose_verdict.verdict) == ("fail", "pass")


def test_a_new_target_judges_the_same_state_without_recomputing():
    UPDATES.clear()
    collection = catalog.pendulum_arm(0.339).collection()
    counted = type(collection)((Alias("tip stop error", CountedStopStat(of="error", statistic="median"), unit="mm",
                                      scale=339.0),))
    stratum = next(iter(summarize(counted, [synthetic.run()], by=("station",)).values()))
    computed = len(UPDATES)
    for limit in (0.01, 0.1, 1.0, 10.0):
        Objective("tip stop error", op="<=", value=limit, unit="mm").judge(stratum)
    assert len(UPDATES) == computed == 1


def test_an_objective_in_the_wrong_unit_is_refused():
    collection = catalog.pendulum_arm(0.339).collection()
    stratum = next(iter(summarize(collection, [synthetic.run()], by=("station",)).values()))
    with pytest.raises(ValueError, match="unit"):
        Objective("tip stop error", op="<=", value=1.0, unit="deg").judge(stratum)


def test_every_stored_value_carries_identity_and_version():
    collection = catalog.pendulum_arm(0.339).collection()
    run = synthetic.run()
    records = collection.records(collection.fold(run), run)
    assert {record.alias for record in records} == set(collection.aliases)
    record = next(r for r in records if r.alias == "tip stop error")
    assert (record.base, record.unit, record.run, record.provenance) == ("StopStat", "mm", run.id, "sim")
    assert record.version and "of='error'" in record.config
