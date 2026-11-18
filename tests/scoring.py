"""Round 8 scored with robot_metrics' public API only, in the frozen check's shape (check_scenario.run()'s keys).

Shared by the parity, gate and live tests and by the report builder (docs/build_report.py), so the report's parity
table is the tests' own computation, not a second one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from robot_metrics import Collection, Run, Stratum, catalog, highest_passing, summarize

# check_scenario's gate names -> the catalog's gates
GATE_NAMES = {"tracking": "tracks", "tool_stop_median": "tip stop (median)", "tool_stop_worst": "tip stop (every stop)"}


def arm_collections(stations: Mapping[str, Mapping[str, float]]) -> dict[str, Collection]:
    """One collection per station: the 1-DOF pendulum arm with that station's lever (its tip length)."""

    return {station: catalog.pendulum_arm(facts["tip_length_m"]).collection() for station, facts in stations.items()}


def play_rows(runs: Iterable[Run], collections: Mapping[str, Collection]) -> dict[str, dict]:
    out = {}
    for run in runs:
        collection = collections[run.labels["station"]]
        values = collection.values(collection.fold(run))
        out[run.id] = {"tracking_rms_deg": values["servo/tracking RMS"],
                       "tracking_peak_deg": values["servo/tracking peak"],
                       "stops": int(values["servo/stops"]), "overshoot_deg": values["servo/overshoot"],
                       "lag_ms": values["servo/lag"]}
    return out


def strata(runs: Sequence[Run], collections: Mapping[str, Collection]) -> dict[tuple, Stratum]:
    out: dict[tuple, Stratum] = {}
    for station, collection in collections.items():
        out.update(summarize(collection, [run for run in runs if run.labels["station"] == station],
                             by=("station", "rung")))
    return out


def rung_rows(cells: Mapping[tuple, Stratum]) -> dict[str, dict]:
    out = {}
    for (station, rung), stratum in cells.items():
        values = stratum.values
        share = values["servo/limiter share"]
        out[f"{station}@{float(rung):g}"] = {
            "tracking_rms_deg": values["servo/tracking RMS"], "tracking_peak_deg": values["servo/tracking peak"],
            "stop_tool_mm": values["tip stop error"], "stop_tool_max_mm": values["tip stop worst"],
            "settle_s": values["tip settle"], "limiter_share": None if share is None else share / 100.0}
    return out


def verdicts(cells: Mapping[tuple, Stratum]) -> dict[tuple, dict[str, str]]:
    gates = catalog.pendulum_gates()
    return {key: {name: gate.judge(stratum).verdict for name, gate in gates.items()} for key, stratum in cells.items()}


def gate_counts(judged: Mapping[tuple, Mapping[str, str]]) -> dict[str, list[int]]:
    return {name: [sum(cell[gate] == "pass" for cell in judged.values()), len(judged)]
            for name, gate in GATE_NAMES.items()}


def highest_rungs(judged: Mapping[tuple, Mapping[str, str]]) -> dict[str, float]:
    return highest_passing({key: cell["tracks"] == "pass" and cell["tip stop (median)"] == "pass"
                            for key, cell in judged.items()})


def chain_rows(runs: Iterable[Run]) -> dict[str, dict]:
    collection = catalog.rigid_chain().collection()
    out = {}
    for run in runs:
        values = collection.values(collection.fold(run), si=True)
        out[run.id] = {"positions/j2": values["j2/tracking RMS"], "positions/c1": values["c1/tracking RMS"],
                       "tool_position_rms_m": values["tool tracking RMS"], "tool_position_max_m": values["tool worst"]}
    return out
