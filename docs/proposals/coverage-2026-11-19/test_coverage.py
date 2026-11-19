"""The coverage math on cases whose answer is known by construction, plus parity with robot_msgs.coverage.

    cd /home/bam/robot_metrics/docs/proposals/coverage-2026-11-19 && taskset -c 2-5,16,17 /home/bam/robot_metrics/.venv/bin/python -m pytest -q test_coverage.py
"""

from __future__ import annotations

import numpy as np
import pytest

from coverage import Axis, Grid, excitation, gap_distances, largest_holes, limit_use, rarefaction, sphere_cell

AXES = (Axis("q", -1.0, 1.0, 10), Axis("qd", -1.0, 1.0, 10))


def uniform_with_hole(seed: int = 0, n: int = 20000) -> np.ndarray:
    """Uniform on the square except the 4 x 4-cell corner q > 0.2, qd > 0.2 (16 of 100 cells)."""

    samples = np.random.default_rng(seed).uniform(-1, 1, (n, 2))
    return samples[~((samples[:, 0] > 0.2) & (samples[:, 1] > 0.2))]


def test_visited_and_conditional_see_the_planted_hole():
    grid = Grid.of(uniform_with_hole(), AXES)
    assert grid.visited_pct() == pytest.approx(84.0)
    per_q = grid.conditional(given=0)
    assert np.allclose(per_q[:6], 1.0) and np.allclose(per_q[6:], 0.6)
    (centre, size), = grid.holes(top=1)
    assert size == 16 and centre == pytest.approx((0.6, 0.6))


def test_unreachable_cells_leave_the_denominator():
    reachable = np.ones((10, 10), bool)
    reachable[6:, 6:] = False                     # the hole is unreachable, so the data covers everything it can
    grid = Grid.of(uniform_with_hole(), AXES, reachable=reachable)
    assert grid.visited_pct() == pytest.approx(100.0)


def test_sufficient_and_evenness_separate_spread_from_piled():
    rng = np.random.default_rng(1)
    even = rng.uniform(-1, 1, (10000, 2))
    piled = np.vstack([even[:100], np.zeros((9900, 2)) + 0.05])
    assert Grid.of(even, AXES).effective_pct() > 95
    assert Grid.of(piled, AXES).effective_pct() < 10
    assert Grid.of(piled, AXES).sufficient_pct(5) < Grid.of(even, AXES).sufficient_pct(5)


def test_visited_matches_robot_msgs_coverage_rule():
    from robot_msgs.coverage import binned
    samples = uniform_with_hole(n=3000)
    edges = [list(axis.edges()) for axis in AXES]
    theirs = binned([tuple(row) for row in samples], edges)
    assert Grid.of(samples, AXES).visited_pct() == pytest.approx(theirs.pct, abs=0.01)


def test_rarefaction_is_monotone_and_chao2_bounds_the_kind_of_run():
    samples = uniform_with_hole()
    runs = np.random.default_rng(2).integers(0, 30, len(samples))
    grid = Grid.of(samples, AXES, run_ids=runs)
    k, mean, band = rarefaction(grid)
    assert np.all(np.diff(mean) >= -1e-9) and mean[-1] == pytest.approx(84.0)
    assert grid.chao2_reachable_estimate() == pytest.approx(84.0, abs=1.0)   # every visitable cell is already seen


def test_largest_holes_finds_the_planted_hole_first():
    (centre, radius), *_ = largest_holes(uniform_with_hole(), AXES, count=3, probes=6000)
    assert centre[0] > 0.2 and centre[1] > 0.2
    assert 0.25 < radius < 0.45                    # the empty corner is 0.4 x 0.4 of the normalized square


def test_gap_distances_are_small_when_covered():
    dense = np.random.default_rng(3).uniform(-1, 1, (20000, 2))
    assert np.percentile(gap_distances(dense, AXES, probes=4000), 90) < 0.02


def test_excitation_flags_collinear_columns():
    x = np.random.default_rng(4).normal(size=(2000, 2))
    good = excitation(np.column_stack([x[:, 0], x[:, 1]]))
    collinear = excitation(np.column_stack([x[:, 0], x[:, 0] + 1e-6 * x[:, 1]]))
    assert good["condition"] < 2 and collinear["condition"] > 1e4


def test_limit_use_and_sphere_cells():
    assert limit_use(np.linspace(-0.5, 0.5, 101), -1.0, 1.0)["span_pct"] == pytest.approx(50.0)
    directions = np.random.default_rng(5).normal(size=(200000, 3))
    counts = np.bincount(sphere_cell(directions, 6, 12), minlength=72)
    assert counts.min() / counts.max() > 0.9       # equal-area: uniform directions fill every cell equally
