"""Shared fixtures. Real data is read-only and optional: every test that needs /archive skips cleanly without it."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import oracle  # noqa: E402


def _skip_without(path: pathlib.Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} is absent ({path}); this test reads real data from /archive")


@pytest.fixture(scope="session")
def round8():
    from robot_metrics.datasets import onedof_round8

    _skip_without(onedof_round8.ROUND / "plans" / "index.json", "round 8")
    _skip_without(oracle.SCORECARD, "round 8's scorecard")
    return onedof_round8


@pytest.fixture(scope="session")
def stations(round8):
    return round8.stations()


@pytest.fixture(scope="session")
def real_runs(round8):
    return round8.real_runs()


@pytest.fixture(scope="session")
def collections(stations):
    import scoring

    return scoring.arm_collections(stations)


@pytest.fixture(scope="session")
def scorecard(round8):
    return oracle.scorecard_oracle()


@pytest.fixture(scope="session")
def real_strata(real_runs, collections):
    import scoring

    return scoring.strata(real_runs, collections)


@pytest.fixture(scope="session")
def sim_runs(round8):
    _skip_without(round8.SIM_MEMO, "the twin's sim memo")
    return round8.sim_runs(oracle.twins_used())


@pytest.fixture(scope="session")
def chain():
    from robot_metrics.datasets import fivebar_ladder

    _skip_without(fivebar_ladder.LADDER, "the five-bar speed ladder")
    _skip_without(oracle.CHAIN_SUMMARY, "the five-bar ladder's summary.json")
    return {source: fivebar_ladder.runs(source) for source in ("real", "sim")}
