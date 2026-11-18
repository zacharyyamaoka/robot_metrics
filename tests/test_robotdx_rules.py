"""Test 9 · what RobotDx's stock diagnostics need to compute through robot_metrics, bit for bit with what they computed.

RobotDx (io-diagnostics, robotdx/diagnostics/stock) moves its tracking and stop math onto these metrics (PEP 0001 §8
step 1). Its published numbers were computed by sim2real.metrics.actuator (main 6c6e964), so each addition here is held
to that code's answer exactly, by an oracle copied into this file (robot_metrics never imports sim2real):

- ``ErrorPercentile``: the p95 of |target - measured| (RobotDx's ``statistic="p95"``), robot_msgs' ``p95_abs``.
- StopStat's T4 rule options: RobotDx's T4 counts a dwell its tape starts in (after the prelude the tape may open on a
  held target), takes ``dwell_min_s`` / ``settle_window_s`` from its config, and drops the tape's last dwell (parking).
  Defaults keep the pendulum's rule unchanged.
- ``StopStat(statistic="p<q>")``: any percentile, so a RobotDx config's ``percentile`` is part of the metric's identity.
- ``functional.stop_summary``: T4's numbers over signed stop errors in the caller's unit (sim2real's stop_summary).

Written before the implementation (red first).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from robot_msgs.compare import error_stats
from robot_msgs.std.header import Header
from robot_msgs.trajectory import JointTrajectory

import synthetic
from robot_metrics import ErrorPercentile, ErrorStat, StopStat, functional, stops


# ---- the oracle: sim2real.metrics.actuator at main 6c6e964, copied unchanged ------------------------------------------
def t4_stops_oracle(t, q, q_target, qd_target, *, dwell_min_s, settle_window_s, scale=1e3, origin=None, open_end=False):
    origin = t[0] if origin is None else origin
    still = np.abs(qd_target) < 1e-9
    errors, start, count = [], 0, len(t)
    while start < count:
        if not still[start]:
            start += 1
            continue
        end = start
        while end + 1 < count and still[end + 1] and abs(q_target[end + 1] - q_target[start]) < 1e-9:
            end += 1
        if t[end] - t[start] >= dwell_min_s and (end + 1 < count or open_end):
            window = (t >= t[end] - settle_window_s) & (t <= t[end])
            errors.append((float(scale * np.mean(q[window] - q_target[window])), float(t[start] - origin), float(t[end] - origin)))
        start = end + 1
    return errors


def stop_summary_oracle(signed, percentile=90.0):
    magnitudes = np.abs(np.array(signed))
    return {"value": float(np.percentile(magnitudes, percentile)), "n": len(signed), "median": float(np.median(magnitudes)),
            "max": float(magnitudes.max()), "mean_signed": float(np.mean(signed)), "worst": int(np.argmax(magnitudes))}


# ---- fixtures ---------------------------------------------------------------------------------------------------------
def streams(t, q, q_target, qd_target):
    header, names = Header(stamp=0.0, frame_id="output"), ("servo",)
    target = JointTrajectory(t=t, joint_names=names, positions=q_target[:, None], velocities=qd_target[:, None], header=header)
    measured = JointTrajectory(t=t, joint_names=names, positions=q[:, None], header=header)
    return target, measured


def tape(seed: int = 0, *, start_held: bool = True):
    """A tape that OPENS on a held target (0.6 s), moves, dwells 1.2 s, moves, dwells 0.15 s (short), moves, parks 0.8 s."""

    rate = 100.0
    pieces = [(0.6, "rest", 0.0)] if start_held else []
    pieces += [(1.0, "move", (0.0, 0.5)), (1.2, "rest", 0.5), (0.5, "move", (0.5, 0.3)), (0.15, "rest", 0.3),
               (0.7, "move", (0.3, 0.0)), (0.8, "rest", 0.0)]
    t_all, q_all, qd_all, begin = [], [], [], 0.0
    for duration, kind, where in pieces:
        t = begin + np.arange(int(round(duration * rate))) / rate
        if kind == "rest":
            q, qd = np.full(t.size, where), np.zeros(t.size)
        else:
            s = (t - begin) / duration
            low, high = where
            q = low + (high - low) * 0.5 * (1.0 - np.cos(np.pi * s))
            qd = (high - low) * 0.5 * np.pi / duration * np.sin(np.pi * s)
            qd[0] = 1e-6 * np.sign(high - low)
        t_all.append(t), q_all.append(q), qd_all.append(qd)
        begin += duration
    t, q_target, qd_target = np.concatenate(t_all), np.concatenate(q_all), np.concatenate(qd_all)
    t = t + 3.25                       # a tape cut out of a longer run: its clock does not start at 0
    rng = np.random.default_rng(seed)
    q = q_target + 3e-4 + rng.normal(0.0, 5e-5, t.size)
    return t, q, q_target, qd_target


T4 = dict(count_first_dwell=True, include_last_dwell=False)


# ---- ErrorPercentile ----------------------------------------------------------------------------------------------------
def test_error_percentile_is_robot_msgs_p95_abs_bit_for_bit():
    t, q, q_target, qd_target = tape()
    target, measured = streams(t, q, q_target, qd_target)
    got = ErrorPercentile(percentile=95.0)(target, measured)
    assert got == error_stats(q - q_target).p95_abs == float(np.percentile(np.abs(q - q_target), 95.0))
    assert ErrorPercentile(percentile=50.0)(target, measured) == float(np.percentile(np.abs(q - q_target), 50.0))


def test_error_percentile_folds_like_one_recording_and_keeps_its_identity():
    t, q, q_target, qd_target = tape()
    target, measured = streams(t, q, q_target, qd_target)
    metric = ErrorPercentile()
    state = metric.empty()
    for rows in (slice(0, 37), slice(37, 200), slice(200, None)):
        state = metric.update(state, target[rows], measured[rows])
    assert metric.compute(metric.close(state)) == metric(target, measured)
    halves = metric.merge(metric.update(metric.empty(), target[:150], measured[:150]),
                          metric.update(metric.empty(), target[150:], measured[150:]))
    assert metric.compute(halves) == metric(target, measured)
    assert metric.identity == ("ErrorPercentile", (("percentile", 95.0), ("rows", "all"), ("field", "positions"), ("joint", 0)))
    assert metric.compute(metric.empty()) is None
    moving = ErrorPercentile(rows="moving")(target, measured)
    assert moving == float(np.percentile(np.abs((q - q_target)[np.abs(qd_target) >= 1e-9]), 95.0))
    with pytest.raises(ValueError):
        ErrorPercentile(percentile=101.0)


def test_error_stat_is_unchanged_beside_it():
    # WHY here: RobotDx's rms, max and final read ErrorStat itself; its identity and state are untouched
    t, q, q_target, qd_target = tape()
    target, measured = streams(t, q, q_target, qd_target)
    stats = error_stats(q - q_target)
    assert ErrorStat(statistic="rms")(target, measured) == stats.rms
    assert ErrorStat(statistic="max_abs")(target, measured) == stats.max_abs
    assert ErrorStat(statistic="mean_abs")(target, measured) == float(np.mean(np.abs(q - q_target)))


# ---- StopStat: RobotDx's T4 rule ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("open_end", [False, True])
@pytest.mark.parametrize("rule", [dict(dwell_min_s=0.25, settle_window_s=0.05), dict(dwell_min_s=0.1, settle_window_s=0.02),
                                  dict(dwell_min_s=0.5, settle_window_s=0.1)])
def test_the_t4_rule_finds_the_t4_stops_bit_for_bit(rule, open_end):
    for seed in range(3):
        t, q, q_target, qd_target = tape(seed)
        target, measured = streams(t, q, q_target, qd_target)
        metric = StopStat(of="error", statistic="p90", count_first_dwell=True, include_last_dwell=open_end, **rule)
        found = metric.stops(metric.update(metric.empty(), target, measured))
        expected = t4_stops_oracle(t, q, q_target, qd_target, open_end=open_end, origin=1.0, **rule)
        got = [(1e3 * stop.error, float(stop.move_t[stop.move_end] - 1.0), float(stop.move_t[-1] - 1.0)) for stop in found]
        assert got == expected
        # the opening dwell and B always; the 0.15 s dwell only under a shorter dwell_min_s; parking only in a live window
        assert len(found) == 2 + (rule["dwell_min_s"] <= 0.14) + open_end
        assert found[0].t_s == 0.0


def test_the_t4_rule_holds_live():
    t, q, q_target, qd_target = tape()
    target, measured = streams(t, q, q_target, qd_target)
    metric = StopStat(of="error", statistic="p90", **T4)
    offline = metric.stops(metric.update(metric.empty(), target, measured))
    state = metric.empty()
    for edge in range(0, len(t), 13):          # chunk edges inside every dwell, the opening one too
        state = metric.update(state, target[edge:edge + 13], measured[edge:edge + 13])
    live = metric.stops(state)
    assert [(s.error, s.t_s, s.dwell_s) for s in live] == [(s.error, s.t_s, s.dwell_s) for s in offline]


def test_the_default_rule_is_still_the_pendulums():
    t, q, q_target, qd_target = tape()
    target, measured = streams(t, q, q_target, qd_target)
    default, t4 = StopStat(of="error", statistic="median"), StopStat(of="error", statistic="median", **T4)
    # the pendulum's rule: a dwell the episode starts in is no stop, its last dwell counts; T4 the other way round
    default_starts = [round(s.t_s, 6) for s in default.stops(default.update(default.empty(), target, measured))]
    t4_starts = [round(s.t_s, 6) for s in t4.stops(t4.update(t4.empty(), target, measured))]
    assert default_starts == [1.6, 4.15] and t4_starts == [0.0, 1.6]
    run = synthetic.run()
    assert StopStat(of="count", statistic="count")(run.streams["target"], run.streams["state"]) == 2.0
    assert len(stops.stops(run.streams["target"], run.streams["state"])) == 2
    assert len(functional.stops(run.streams["target"], run.streams["state"], count_first_dwell=True)) == 3


def test_the_rule_is_part_of_the_state_key_and_the_identity():
    a, b = StopStat(of="error", statistic="median"), StopStat(of="error", statistic="p90")
    assert a.state_key == b.state_key                                  # one stop pass serves both readings
    assert StopStat(of="error", statistic="median", **T4).state_key != a.state_key
    assert StopStat(dwell_min_s=0.5).state_key != StopStat().state_key
    assert StopStat(settle_window_s=0.1).identity != StopStat().identity


# ---- StopStat percentiles -----------------------------------------------------------------------------------------------
def test_any_percentile_is_a_statistic():
    run = synthetic.run()
    target, measured = run.streams["target"], run.streams["state"]
    errors = [abs(stop.error) for stop in stops.stops(target, measured)]
    for statistic, q in (("p90", 90.0), ("p95", 95.0), ("p99", 99.0), ("p50", 50.0), ("p97.5", 97.5)):
        assert StopStat(of="error", statistic=statistic)(target, measured) == float(np.percentile(errors, q))
    for bad in ("p", "p101", "px", "p-1"):
        with pytest.raises(ValueError):
            StopStat(statistic=bad)


# ---- T4's summary -------------------------------------------------------------------------------------------------------
def test_stop_summary_is_t4s_numbers_bit_for_bit():
    rng = np.random.default_rng(7)
    for n in (1, 2, 5, 36, 101):
        signed = [float(v) for v in rng.normal(0.4, 1.3, n)]
        for percentile in (90.0, 95.0, 50.0, 99.0):
            assert functional.stop_summary(signed, percentile) == stop_summary_oracle(signed, percentile)
    assert stops.summary([-2.0, 1.0], 90.0)["worst"] == 0
    assert math.isnan(functional.stop_summary([float("nan"), 1.0])["value"])
