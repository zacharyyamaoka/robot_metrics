"""Test 2 · live == offline: the same collection fed a run in chunks (split mid-dwell) gives the offline numbers.

Tolerance, stated here: |live - offline| <= LIVE_REL x max(1, |offline|) in display units, LIVE_REL = 1e-12. WHY that
number: the contract's G4 allows 1e-9; the prior-art note's default is 1e-12 relative; a sum folded chunk by chunk
rounds differently from one sum (the note measured 3.5e-18 rad), and nothing else may differ. Stops, their errors and
their settle times must agree exactly, because the stop kernel is one per-sample rule run over the same rows.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import synthetic
from robot_metrics import Aligner, ErrorStat, Stream, align, catalog, stops
from robot_metrics.datasets import onedof_round8

LIVE_REL = 1e-12


def assert_same(live: dict, offline: dict) -> None:
    assert live.keys() == offline.keys()
    for name, value in offline.items():
        got = live[name]
        if value is None or isinstance(value, bool):
            assert got == value, (name, got, value)
        elif math.isinf(value):
            assert got == value, (name, got, value)
        else:
            assert abs(got - value) <= LIVE_REL * max(1.0, abs(value)), (name, got, value)


def dwell_midpoints(run) -> list[float]:
    """The middle instant of every stop's dwell: a chunk boundary there splits an open dwell across two chunks."""

    found = stops.stops(run.streams["target"], run.streams["state"])
    t0 = float(run.streams["state"].t[0])
    return [t0 + stop.t_s + 0.5 * stop.dwell_s for stop in found]


def live(collection, chunks) -> dict:
    state = collection.empty()
    for chunk in chunks:
        state = collection.update(state, chunk)
    return collection.values(collection.close(state))


def test_synthetic_run_live_equals_offline_split_mid_dwell():
    run = synthetic.run()
    collection = catalog.pendulum_arm(0.3).collection()
    middles = dwell_midpoints(run)
    assert len(middles) == 2                      # B and the return to A (the starting dwell is no stop)
    offline = collection.values(collection.fold(run))
    assert_same(live(collection, run.split_at(middles)), offline)
    assert_same(live(collection, run.chunks(1)), offline)      # one sample at a time
    assert_same(live(collection, run.chunks(7)), offline)


@pytest.mark.archive
def test_every_real_play_live_equals_offline(real_runs, collections):
    split_inside_a_dwell = 0
    for run in real_runs:
        collection = collections[run.labels["station"]]
        middles = dwell_midpoints(run)
        edges = sorted(set(middles) | set(run.streams["state"].t[6::7].tolist()))
        split_inside_a_dwell += len(middles)
        offline = collection.values(collection.fold(run))
        assert_same(live(collection, run.split_at(edges)), offline)
    assert split_inside_a_dwell > 500                # every stop of round 8 was split mid-dwell at least once


@pytest.mark.archive
def test_one_sample_chunks_on_a_real_play(real_runs, collections):
    run = next(r for r in real_runs if r.labels["station"] == "can11" and r.labels["rung"] == 0.9)
    collection = collections["can11"]
    assert_same(live(collection, run.chunks(1)), collection.values(collection.fold(run)))


def test_stream_is_the_torchmetrics_face():
    run = synthetic.run()
    metric = ErrorStat(statistic="rms", rows="moving")
    stream = Stream(metric)
    for chunk in run.chunks(5):
        stream.update(chunk.streams["target"], chunk.streams["state"])
    offline = metric(run.streams["target"], run.streams["state"])
    assert abs(stream.compute() - offline) <= LIVE_REL * offline
    stream.reset()
    assert stream.compute() is None
    assert metric.stream().compute() is None


@pytest.mark.archive
def test_the_live_aligner_pairs_raw_streams_like_the_offline_step(real_runs):
    """v3's explicit alignment step, live: plan knots and replies arrive one message at a time, in arrival order; each
    reply is held until the knot after it exists. The released pairs equal the offline hermite alignment exactly."""

    for run in [r for r in real_runs if r.labels["station"] == "can11"][:12]:
        plan, state = run.streams["plan"], run.streams["state"]
        aligner = Aligner(rule="hermite", tolerance_s=onedof_round8.TICK_S)
        released = []
        events = sorted([(float(t), 0, row) for row, t in enumerate(plan.t)] +
                        [(float(t), 1, row) for row, t in enumerate(state.t)])
        for _, kind, row in events:
            out = aligner.push(reference=plan[row:row + 1]) if kind == 0 else aligner.push(measured=state[row:row + 1])
            if out is not None:
                released.append(out)
        tail = aligner.finish()
        if tail is not None:
            released.append(tail)
        positions = np.concatenate([piece.target.positions[:, 0] for piece in released])
        velocities = np.concatenate([piece.target.velocities[:, 0] for piece in released])
        offline = align(plan, state, rule="hermite", tolerance_s=onedof_round8.TICK_S)
        assert np.array_equal(positions, offline.target.positions[:, 0])
        assert np.array_equal(velocities, offline.target.velocities[:, 0])
        assert np.array_equal(positions, run.streams["target"].positions[:, 0])
        assert aligner.max_held <= 2      # the latency paid: a reply waits for at most the next knot


def test_alignment_refuses_to_extrapolate_past_its_tolerance():
    run = synthetic.run()
    plan = run.streams["plan"]
    late = run.streams["state"]
    shifted = type(late)(t=late.t + 0.5, joint_names=late.joint_names, positions=late.positions,
                         header=late.header)
    out = align(plan, shifted, rule="hermite", tolerance_s=0.01)
    assert out.alignment.dropped == int(np.count_nonzero(shifted.t > plan.t[-1] + 0.01))
    assert len(out.measured) + out.alignment.dropped == len(shifted)
