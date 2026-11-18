"""Test 3 · computed once: two aliases of one identity share one state, and an assembly reads its child's computation.

WHY (Zach, 2026-11-17, "# Ok homing in on this..."): an alias is a label; a metric's identity is its base + frozen
config, so two names for one identity are one computation. v5's rule on top: the pendulum arm's tip numbers are the
servo's stop and tracking states read through the lever (context), never a second pass over the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import synthetic
from robot_metrics import DEGREES, Alias, Collection, ErrorStat, PartMetrics, catalog

UPDATES: list[str] = []


@dataclass(frozen=True)
class CountedErrorStat(ErrorStat):
    """ErrorStat that logs each update it runs: the dedup counter."""

    def update(self, state, target, measured):
        UPDATES.append(repr(self))
        return super().update(state, target, measured)


def test_two_aliases_of_one_identity_compute_once():
    UPDATES.clear()
    collection = Collection((
        Alias("tracking RMS", CountedErrorStat(statistic="rms", rows="moving"), unit="deg", scale=DEGREES),
        Alias("tip tracking RMS", CountedErrorStat(statistic="rms", rows="moving"), unit="mm", scale=300.0),
    ))
    assert collection.computations == 1
    run = synthetic.run()
    values = collection.values(collection.fold(run))
    assert len(UPDATES) == 1                            # one update for two aliases
    assert math.isclose(values["tip tracking RMS"], values["tracking RMS"] / DEGREES * 300.0, rel_tol=1e-15)


def test_different_configs_compute_separately():
    UPDATES.clear()
    collection = Collection((
        Alias("moving RMS", CountedErrorStat(statistic="rms", rows="moving")),
        Alias("all-rows RMS", CountedErrorStat(statistic="rms", rows="all")),
    ))
    assert collection.computations == 2
    collection.fold(synthetic.run())
    assert len(UPDATES) == 2


def test_live_chunks_cost_one_update_per_identity_per_chunk():
    UPDATES.clear()
    collection = Collection((
        Alias("a", CountedErrorStat(statistic="rms", rows="moving")),
        Alias("b", CountedErrorStat(statistic="rms", rows="moving")),
        Alias("c", CountedErrorStat(statistic="rms", rows="moving")),
    ))
    chunks = synthetic.run().chunks(50)
    state = collection.empty()
    for chunk in chunks:
        state = collection.update(state, chunk)
    assert len(UPDATES) == len(chunks)


def test_the_arm_adds_no_computation_to_its_servo():
    servo = catalog.integrated_servo()
    arm = catalog.pendulum_arm(tip_length_m=0.339)
    assert arm.child("servo") == servo
    assert arm.collection().computations == servo.collection().computations
    assert len(arm.collection().aliases) > len(servo.collection().aliases)


def test_the_tip_stop_is_the_servo_stop_through_the_lever():
    lever = 0.339
    collection = catalog.pendulum_arm(tip_length_m=lever).collection()
    values = collection.values(collection.fold(synthetic.run()))
    servo_rad = values["servo/stop error"] / DEGREES
    assert math.isclose(values["tip stop error"], servo_rad * lever * 1e3, rel_tol=1e-12)
    assert math.isclose(values["tip stop worst"], values["servo/stop error worst"] / DEGREES * lever * 1e3, rel_tol=1e-12)
    assert math.isclose(values["tip tracking RMS"], values["servo/tracking RMS"] / DEGREES * lever * 1e3, rel_tol=1e-12)


def test_an_alias_through_a_missing_context_key_is_refused():
    servo = catalog.integrated_servo()
    broken = PartMetrics("arm", "pendulum_arm", (
        Alias.through(servo, "stop error", name="tip stop error", unit="mm", scale=1e3, scale_by="lever_m"),
    ), children=(servo,), context={})
    try:
        broken.collection()
    except KeyError as error:
        assert "lever_m" in str(error)
    else:
        raise AssertionError("an alias scaled by a context key the part lacks must be refused")
