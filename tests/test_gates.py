"""Test 5 · gate counts equal the contract's: tracking 31, median tip stop 36, every stop 33 of 36 rung cells.

The contract (robot_models docs/proposals/metrics-api-2026-11-17/README.md, "Result") states the three counts; the
scorecard oracle recomputes them from round8.json. Gates are OpenSLO-style occurrence objectives judged outside
compute(), per stratum (station x rung), on a tail statistic, never an IQM.
"""

from __future__ import annotations

import math

import pytest

import oracle
import scoring
from robot_metrics import Objective, wilson

CONTRACT = {"tracking": [31, 36], "tool_stop_median": [36, 36], "tool_stop_worst": [33, 36]}


@pytest.mark.archive
def test_gate_counts_equal_the_contract_and_the_scorecard(real_strata, scorecard):
    counts = scoring.gate_counts(scoring.verdicts(real_strata))
    assert counts == CONTRACT
    assert counts == scorecard["gates"]


@pytest.mark.archive
def test_the_every_stop_gate_reads_the_tail_the_median_hides(real_strata):
    judged = scoring.verdicts(real_strata)
    hidden = [key for key, cell in judged.items()
              if cell["tip stop (median)"] == "pass" and cell["tip stop (every stop)"] == "fail"]
    assert len(hidden) == 3
    for key in hidden:
        tail = real_strata[key].tails["tip stop error"]
        assert tail.worst > 1.0 >= tail.median and tail.n == len(real_strata[key].occurrences["tip stop error"])


@pytest.mark.archive
def test_an_occurrence_objective_counts_items_not_runs(real_strata):
    key = next(k for k in real_strata if k[0] == "can11" and k[1] == 1.0)
    stratum = real_strata[key]
    every = Objective("tip stop error", op="<=", value=1.0, unit="mm", share=1.0, on="items").judge(stratum)
    most = Objective("tip stop error", op="<=", value=1.0, unit="mm", share=0.5, on="items").judge(stratum)
    stops = stratum.occurrences["tip stop error"]
    assert every.total == most.total == len(stops) > len(stratum.run_ids)
    assert every.passed == sum(stop <= 1.0 for stop in stops)
    assert math.isclose(every.observed, every.passed / every.total)


def test_wilson_interval_matches_the_textbook_formula():
    successes, trials, z = 38, 42, 1.96
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z / (1 + z * z / trials) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    low, high = wilson(successes, trials)
    assert math.isclose(low, centre - half, rel_tol=1e-12) and math.isclose(high, centre + half, rel_tol=1e-12)
    assert wilson(0, 0) == (0.0, 1.0)
    assert (round(low, 2), round(high, 2)) == (0.78, 0.96)


@pytest.mark.archive
def test_bool_metrics_summarize_as_a_wilson_share(real_runs, collections):
    from robot_metrics import summarize

    can11 = [run for run in real_runs if run.labels["station"] == "can11"]
    station = summarize(collections["can11"], can11, by=("station",))[("can11",)]
    share = station.shares["all stops settled"]
    assert (share.successes, share.trials) == (38, 42)     # the v4 proposal's transcript: 38 of 42, 95 % 0.78-0.96
    assert (share.low, share.high) == wilson(38, 42)
    assert "tip stop error" not in station.shares          # a number gets a tail, a bool a share
