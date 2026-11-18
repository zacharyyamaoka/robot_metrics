"""Test 6 · inputs other than one run recording (amendment 4): Repeatability over repeats, PerformanceGain over a
held-out pair, both equal to the proposal's own oracles; each class declares its input type.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from robot_msgs.std.header import Header
from robot_msgs.trajectory import JointTrajectory

import oracle
import synthetic
from robot_metrics import (Alias, CalibrationOutcome, CodeShare, ErrorStat, HeldOutPair, Interval, PerformanceGain,
                           Repeat, Repeatability, Run, StopStat, TimeOffset, functional)

FF_RESIDUAL = Alias("feed-forward residual RMS", ErrorStat(statistic="rms", field="effort"), unit="N·m")


def held_out(side: list[dict]) -> tuple[Run, ...]:
    runs = []
    for row in side:
        t = np.asarray(row["t"])
        names, zeros = ("servo",), np.zeros((t.size, 1))
        header = Header(stamp=0.0, frame_id="output", meta={"source": row["id"]})
        speed = np.asarray(row["speed"])[:, None]
        target = JointTrajectory(t=t, joint_names=names, positions=zeros, velocities=speed,
                                 effort=np.asarray(row["feedforward"])[:, None], header=header)
        state = JointTrajectory(t=t, joint_names=names, positions=zeros, velocities=speed,
                                effort=np.asarray(row["torque"])[:, None], header=header)
        runs.append(Run(id=row["id"], streams={"target": target, "state": state},
                        labels={"plateau": row["plateau"]}, provenance="sim"))
    return tuple(runs)


def test_each_class_declares_its_input():
    inputs = {cls.__name__: cls.Input for cls in (ErrorStat, StopStat, TimeOffset, CodeShare, Repeatability,
                                                   PerformanceGain)}
    assert inputs["Repeatability"] is Repeat and inputs["PerformanceGain"] is HeldOutPair
    assert len({value.__name__ for value in inputs.values()}) >= 3


@pytest.mark.archive
def test_repeatability_equals_the_iso_9283_oracle(real_runs):
    group = [run for run in real_runs if run.labels["station"] == "can11" and run.labels["rung"] == 0.9
             and run.labels["kind"] == "repeat"]
    assert len(group) == 5
    metric = Repeatability(k=3.0)
    state = metric.empty()
    for run in group:      # one real repeat at a time
        state = metric.update(state, Repeat(("can11", 0.9), run.streams["target"], run.streams["state"]))
    value = metric.compute(state)
    want = oracle.repeatability_oracle([(r.streams["state"].t, r.streams["state"].positions[:, 0],
                                         r.streams["target"].positions[:, 0], r.streams["target"].velocities[:, 0])
                                        for r in group], 3.0)
    assert oracle.close(value, want, oracle.REPEATABILITY), (value, want)
    assert round(value * 1e3 * 0.339, 3) == 0.387          # the proposal's result: 0.387 mm at CAN 11's tip
    assert functional.repeatability([Repeat(("can11", 0.9), r.streams["target"], r.streams["state"]) for r in group],
                                    k=3.0) == value


def test_repeatability_refuses_a_different_command():
    metric = Repeatability()
    run = synthetic.run()
    state = metric.update(metric.empty(), Repeat("A to B", run.streams["target"], run.streams["state"]))
    with pytest.raises(ValueError, match="repeat"):
        metric.update(state, Repeat("B to A", run.streams["target"], run.streams["state"]))
    assert metric.compute(state) is None                    # one repeat has no spread


def test_performance_gain_equals_the_held_out_oracle():
    data = oracle.sealed_ladder()
    nominal, fitted = held_out(data["runs"]["nominal"]), held_out(data["runs"]["fitted"])
    gain = PerformanceGain(metric=FF_RESIDUAL)
    state = gain.update(gain.empty(), HeldOutPair(nominal=nominal, fitted=fitted))
    sides = gain.sides(state)
    want_nominal = oracle.residual_rms_oracle(data["runs"]["nominal"])
    want_fitted = oracle.residual_rms_oracle(data["runs"]["fitted"])
    for got, want, recorded in ((sides[0], want_nominal, data["oracle"]["nominal"]),
                                (sides[1], want_fitted, data["oracle"]["fitted"])):
        assert oracle.close(got, want, oracle.GAIN) and oracle.close(got, recorded, oracle.GAIN), (got, want, recorded)
    assert oracle.close(gain.compute(state), want_nominal - want_fitted, oracle.GAIN)
    assert round(sides[0], 4) == 0.2220 and round(sides[1], 4) == 0.0042   # the proposal's 0.2220 -> 0.0042 N·m


def test_performance_gain_reuses_the_alias_by_identity_not_by_name():
    renamed = Alias("residual torque", FF_RESIDUAL.metric, unit="N·m")
    assert PerformanceGain(metric=FF_RESIDUAL) == PerformanceGain(metric=renamed)
    assert PerformanceGain(metric=FF_RESIDUAL).metric == FF_RESIDUAL.metric
    assert PerformanceGain(metric=FF_RESIDUAL) != PerformanceGain(metric=ErrorStat(statistic="max_abs", field="effort"))


def test_calibration_outcome_is_a_frozen_record():
    record = oracle.sealed_ladder()["outcome"]
    outcome = CalibrationOutcome(part_type=record["part_type"], model=record["model"], nominal=record["nominal"],
                                 recovered={name: Interval(value) for name, value in record["recovered"].items()},
                                 truth=record["truth"], tolerance=record["tolerance"], fitted_on=record["fitted_on"],
                                 held_out=record["held_out"])
    fitted = outcome.values("fitted")
    assert fitted["coulomb_friction"] == record["recovered"]["coulomb_friction"]
    assert fitted["gear_ratio"] == record["nominal"]["gear_ratio"]
    assert outcome.values("nominal") == record["nominal"]
    assert all(outcome.within_tolerance().values())
    assert math.isclose(outcome.relative_error()["coulomb_friction"], abs(0.4194633040218535 - 0.42) / 0.42)
