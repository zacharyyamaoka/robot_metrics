"""Test 1 · parity with round 8: robot_metrics reproduces the scorecard at the frozen check's tolerances.

Over the 252 real plays of /archive/datasets/bam_rig/onedof_2026-11-15/round8/real/*/runs/*.npz against
scorecards/latest_real.json (read-only), the twin's answers to the same plays, and the five-bar ladder's summary.json.
The oracle and tolerances are check_scenario.py's, ported unchanged (tests/oracle.py). Never loosen a tolerance: a
failure here is a FAIL with its gap.
"""

from __future__ import annotations

import pytest

import oracle
import scoring

pytestmark = pytest.mark.archive


def test_every_real_play_matches_the_scorecard(real_runs, collections, scorecard):
    assert len(real_runs) == 252
    plays = scoring.play_rows(real_runs, collections)
    problems, worst = [], {}
    oracle.compare_rows("real_plays", plays, scorecard["plays"], oracle.PLAY_TOLERANCES, problems, worst)
    assert problems == [], problems[:5]
    assert set(plays) == set(scorecard["plays"])


def test_every_real_rung_matches_the_scorecard(real_strata, scorecard):
    rungs = scoring.rung_rows(real_strata)
    problems, worst = [], {}
    oracle.compare_rows("real_rungs", rungs, scorecard["rungs"], oracle.RUNG_TOLERANCES, problems, worst)
    assert problems == [], problems[:5]
    assert len(rungs) == 36


def test_alignment_keeps_every_reply(real_runs):
    # the hermite step (the plan at each reply's own instant) must pair every reply the scorecard scored
    assert all(run.alignment is not None and run.alignment.dropped == 0 for run in real_runs)
    assert {run.alignment.rule for run in real_runs} == {"hermite"}


def test_the_twin_side_matches_the_scorecard(sim_runs, collections, scorecard):
    plays = scoring.play_rows(sim_runs, collections)
    problems, worst = [], {}
    oracle.compare_rows("sim_plays", {k: {"tracking_rms_deg": v["tracking_rms_deg"]} for k, v in plays.items()},
                        scorecard["sim_plays"], oracle.SIM_PLAY_TOLERANCES, problems, worst)
    rungs = scoring.rung_rows(scoring.strata(sim_runs, collections))
    oracle.compare_rows("sim_rungs", rungs, scorecard["sim_rungs"], oracle.SIM_RUNG_TOLERANCES, problems, worst)
    assert problems == [], problems[:5]


def test_highest_rung_each_station_tracks(real_strata, scorecard):
    judged = scoring.verdicts(real_strata)
    got = {station: float(rung) for station, rung in scoring.highest_rungs(judged).items()}
    assert got == {station: float(rung) for station, rung in scorecard["highest"].items()}


def test_the_rigid_chain_matches_the_ladder_summary(chain):
    want = oracle.chain_oracle()
    got = scoring.chain_rows(chain["real"] + chain["sim"])
    assert set(want) <= set(got)
    for key, expected in want.items():
        for name, value in expected.items():
            deviation = abs(float(got[key][name]) - value) / max(abs(value), 1e-12)
            assert deviation <= oracle.CHAIN_REL, (key, name, got[key][name], value)


def test_overshoot_matches_its_independent_oracle(real_runs, collections):
    plays = scoring.play_rows(real_runs, collections)
    for run in real_runs:
        state, target = run.streams["state"], run.streams["target"]
        want = oracle.overshoot_oracle(state.t, state.positions[:, 0], target.positions[:, 0], target.velocities[:, 0])
        assert oracle.close(plays[run.id]["overshoot_deg"], want, oracle.OVERSHOOT_DEG), (run.id, want)


def test_lag_matches_estimate_time_offset(real_runs, collections):
    from robot_msgs.compare import estimate_time_offset

    plays = scoring.play_rows(real_runs, collections)
    for index, run in enumerate(real_runs):
        if index % 25 == 0:   # check_scenario's sampling, unchanged
            want = estimate_time_offset(run.streams["plan"], run.streams["state"]).offset_s * 1e3
            assert oracle.close(plays[run.id]["lag_ms"], want, oracle.LAG_MS), (run.id, want)
