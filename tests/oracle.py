"""The parity oracle, ported from the metrics-API proposals' frozen check (robot_models f0daa34,
docs/proposals/metrics-api-2026-11-17/check_scenario.py). Test-side only: the package never reads these files.

What is ported unchanged:
- the tolerances (frozen with the contract on 2026-11-17 at 20:05 MST, amendment 1 folded in before any variant);
- ``_scorecard_oracle`` (per play, per rung, sim side, highest rung, gate counts from latest_real.json);
- ``_chain_oracle`` (the five-bar ladder's own summary.json, every number robot_msgs.compare's);
- ``_overshoot_oracle`` and ``_repeatability_oracle`` (independent of the stop kernel: their own dwell segmentation);
- ``calibration.oracle_residual_rms`` (amendment 4: RMS of measured - feed-forward torque over every held-out sample);
- ``_close`` and ``_compare_rows`` (first problem per label, worst |deviation| per field).

What changed, and only this: the oracles read robot_msgs streams (``t, q, target, target_speed`` arrays) instead of the
proposal fixture's ``Run`` attributes, because robot_metrics' runs name their streams ("state", "target").
"""

from __future__ import annotations

import json
import math
import pathlib
from collections.abc import Sequence

import numpy as np

SCORECARD = pathlib.Path("/archive/datasets/bam_rig/onedof_2026-11-15/scorecards/latest_real.json")
CHAIN_SUMMARY = pathlib.Path("/archive/datasets/bam_rig/rack_2026-11-10/fivebar_real/ladder_real_124755/ladder/summary.json")
SEALED_LADDER = pathlib.Path(__file__).resolve().parent / "data" / "sealed_ladder_held_out.json"

# WHY these tolerances (check_scenario.py, verbatim): each variant runs the same arithmetic on the same files as the
# scorer, so anything above float reassociation is a different ruler. The scorecard prints 0.01 deg and gates at 0.5 deg /
# 1 mm; the nearest wrong ruler (scoring each reply against its plan TICK instead of the plan at its own instant) moves
# play RMS by ~1e-3 deg, three orders above TRACKING_DEG. Never loosen these.
TRACKING_DEG = 1e-6
STOP_MM = 1e-6
SETTLE_S = 1e-9
SHARE = 1e-12
OVERSHOOT_DEG = 1e-6
LAG_MS = 1e-6
CHAIN_REL = 1e-9
LIVE = 1e-9
GAIN = 1e-9            # check_scenario: performance_gain nominal / fitted / gain within 1e-9
REPEATABILITY = 1e-12  # check_scenario: repeatability value within 1e-12


def _key(station: str, fraction: float) -> str:
    return f"{station}@{float(fraction):g}"


def scorecard_oracle(path: pathlib.Path = SCORECARD) -> dict:
    """check_scenario._scorecard_oracle, unchanged."""

    card = json.loads(path.read_text(encoding="utf-8"))
    limits = card["thresholds"]
    plays, sim_plays, rungs, sim_rungs, gates, highest = {}, {}, {}, {}, {"tracking": 0, "tool_stop_median": 0,
                                                                            "tool_stop_worst": 0}, {}
    total = 0
    for station, block in card["stations"].items():
        for rung in block["rungs"]:
            key = _key(station, rung["fraction"])
            real, sim = rung["real"], rung["sim"]
            rungs[key] = {name: real.get(name) for name in ("tracking_rms_deg", "tracking_peak_deg", "stop_tool_mm",
                                                             "stop_tool_max_mm", "settle_s", "limiter_share")}
            sim_rungs[key] = {name: sim.get(name) for name in ("tracking_rms_deg", "tracking_peak_deg", "stop_tool_mm",
                                                                "stop_tool_max_mm")}
            for play in rung["plays"]:
                play_id = play["run"] + (f"#{play['board']}" if play.get("board") else "")
                plays[play_id] = {"tracking_rms_deg": play["tracking_rms_deg"],
                                  "tracking_peak_deg": play["tracking_peak_deg"], "stops": play["stops"]}
                sim_plays[play_id] = {"tracking_rms_deg": play.get("sim_tracking_rms_deg")}
            total += 1
            gates["tracking"] += bool(real["tracking_rms_deg"] <= limits["tracking_rms_deg"]
                                      and real["tracking_peak_deg"] <= limits["tracking_peak_deg"]
                                      and (real.get("limiter_share") or 0.0) <= limits["limiter_share_max"]
                                      and not real.get("faults"))
            gates["tool_stop_median"] += bool(real["stop_tool_mm"] <= limits["stop_tool_mm"]
                                              and real.get("settle_s") is not None
                                              and real["settle_s"] <= limits["settle_s"])
            gates["tool_stop_worst"] += bool(real["stop_tool_max_mm"] <= limits["stop_tool_mm"])
        highest[station] = card["goal"][f"max_trackable_fraction_{station}"]
    return {"plays": plays, "sim_plays": sim_plays, "rungs": rungs, "sim_rungs": sim_rungs, "highest": highest,
            "gates": {name: [count, total] for name, count in gates.items()}}


def twins_used(path: pathlib.Path = SCORECARD) -> dict[str, str]:
    """Which twin answered each station: the scoring's own label, read here so the package never opens the scorecard."""

    return dict(json.loads(path.read_text(encoding="utf-8"))["twins_used"])


def chain_oracle(path: pathlib.Path = CHAIN_SUMMARY) -> dict:
    """check_scenario._chain_oracle, unchanged."""

    summary = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for step in summary["steps"]:
        if step.get("kind") != "rung":
            continue
        for source, block in (("real", step.get("metrics")), ("sim", step.get("sim_metrics"))):
            if block:
                stats = block["stats"]
                out[f"{source}:{step['name']}"] = {"positions/j2": stats["joints/positions/j2"]["rms"],
                                                   "positions/c1": stats["joints/positions/c1"]["rms"],
                                                   "tool_position_rms_m": stats["tool/position"]["rms"],
                                                   "tool_position_max_m": stats["tool/position"]["max_abs"]}
    return out


def overshoot_oracle(t: np.ndarray, q: np.ndarray, target: np.ndarray, target_speed: np.ndarray) -> float:
    """check_scenario._overshoot_oracle, unchanged: dwells from runs of zero commanded speed on one target, the overshoot
    past each stop's target along its move, read on the dwell's rows (goal()'s settle phase). Degrees."""

    still = np.abs(target_speed) < 1e-9
    worst, last_still, row = 0.0, 0, 0
    while row < len(t):
        if not still[row]:
            row += 1
            continue
        end = row
        while end + 1 < len(t) and still[end + 1] and abs(target[end + 1] - target[row]) < 1e-9:
            end += 1
        if row > 0 and t[end] - t[row] >= 0.25:
            direction = np.sign(target[row] - target[last_still])
            worst = max(worst, float(np.max((q[row:end + 1] - target[row]) * direction)))
        last_still, row = end, end + 1
    return math.degrees(max(worst, 0.0))


def repeatability_oracle(group: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], k: float) -> float | None:
    """check_scenario._repeatability_oracle, unchanged: ISO 9283 RP at each stop index over same-input repeats,
    mean(l) + k * std(l, ddof=1), l = |e - mean(e)| of the signed stop errors (mean of the dwell's last 50 ms); the worst
    stop index, radians. Each repeat is (t, q, target, target_speed)."""

    per_repeat = []
    for t, q, target, speed in group:
        still = np.abs(speed) < 1e-9
        errors, row = [], 0
        while row < len(t):
            if not still[row]:
                row += 1
                continue
            end = row
            while end + 1 < len(t) and still[end + 1] and abs(target[end + 1] - target[row]) < 1e-9:
                end += 1
            if row > 0 and t[end] - t[row] >= 0.25:
                tail = np.arange(row, end + 1)[t[row:end + 1] >= t[end] - 0.05]
                errors.append(float(np.mean(q[tail] - target[tail])))
            row = end + 1
        per_repeat.append(errors)
    if len({len(errors) for errors in per_repeat}) != 1 or len(per_repeat) < 2:
        return None
    matrix = np.asarray(per_repeat)
    distance = np.abs(matrix - matrix.mean(axis=0))
    return float(np.max(distance.mean(axis=0) + k * distance.std(axis=0, ddof=1)))


def sealed_ladder() -> dict:
    """The frozen held-out runs (tools/freeze_sealed_ladder.py) with the proposal's own oracle numbers."""

    return json.loads(SEALED_LADDER.read_text(encoding="utf-8"))


def residual_rms_oracle(side: Sequence[dict]) -> float:
    """calibration.oracle_residual_rms, unchanged: RMS of (measured - feed-forward torque) over every held-out sample."""

    residual = np.concatenate([np.asarray(run["torque"]) - np.asarray(run["feedforward"]) for run in side])
    return float(math.sqrt(np.mean(residual ** 2)))


def close(got, want, tol: float) -> bool:
    """check_scenario._close, unchanged: None or non-finite on the oracle side wants None or non-finite back."""

    if want is None or (isinstance(want, float) and not math.isfinite(want)):
        return got is None or (isinstance(got, float) and not math.isfinite(got))
    return got is not None and math.isfinite(float(got)) and abs(float(got) - float(want)) <= tol


def compare_rows(label: str, got: dict, want: dict, tolerances: dict, problems: list, worst: dict) -> None:
    """check_scenario._compare_rows, unchanged: the first problem per label, every row's |deviation| into ``worst``."""

    missing = sorted(set(want) - set(got))
    if missing:
        problems.append(f"{label}: {len(missing)} of {len(want)} missing, e.g. {missing[0]}")
    for key, expected in want.items():
        if key not in got:
            continue
        for name, tol in tolerances.items():
            if name not in expected:
                continue
            value, oracle = got[key].get(name), expected[name]
            if oracle is not None and value is not None and math.isfinite(float(oracle)) and math.isfinite(float(value)):
                worst[f"{label}.{name}"] = max(worst.get(f"{label}.{name}", 0.0), abs(float(value) - float(oracle)))
            if not close(value, oracle, tol):
                problems.append(f"G1 {label} {key} {name}: {value!r} vs scorecard {oracle!r} (tolerance {tol:g})")
                return


PLAY_TOLERANCES = {"tracking_rms_deg": TRACKING_DEG, "tracking_peak_deg": TRACKING_DEG, "stops": 0}
RUNG_TOLERANCES = {"tracking_rms_deg": TRACKING_DEG, "tracking_peak_deg": TRACKING_DEG, "stop_tool_mm": STOP_MM,
                   "stop_tool_max_mm": STOP_MM, "settle_s": SETTLE_S, "limiter_share": SHARE}
SIM_PLAY_TOLERANCES = {"tracking_rms_deg": TRACKING_DEG}
SIM_RUNG_TOLERANCES = {"tracking_rms_deg": TRACKING_DEG, "tracking_peak_deg": TRACKING_DEG, "stop_tool_mm": STOP_MM,
                       "stop_tool_max_mm": STOP_MM}
