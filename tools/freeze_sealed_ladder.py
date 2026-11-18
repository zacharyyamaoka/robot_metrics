"""Freeze amendment 4's held-out runs and their oracle into tests/data, so robot_metrics' tests need no robot_models.

    /home/bam/robot_models/.venv/bin/python /home/bam/robot_metrics/tools/freeze_sealed_ladder.py

Run once, out of band, with robot_models' own venv: it imports the metrics-API proposal's ``calibration.py``
(robot_models main f0daa34), which plays the sealed friction ladder's held-out plateaus with the nominal (M1.datasheet)
and the fitted (M3.stribeck) values. It writes the two recordings per plateau (feed-forward torque and the torque the rig
needed), the CalibrationOutcome, and the proposal's own oracle numbers (``calibration.oracle_residual_rms``).

WHY frozen and not imported: robot_metrics must not import robot_models (robot_models will import robot_metrics for
``.metrics``; a test-time import back would be a cycle). The frozen file is the oracle's input; the oracle's arithmetic
is ported verbatim into tests/oracle.py and re-checked against the numbers recorded here.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

PROPOSALS = pathlib.Path("/home/bam/robot_models/docs/proposals/metrics-api-2026-11-17")
OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "data" / "sealed_ladder_held_out.json"
sys.path.insert(0, str(PROPOSALS))

import calibration  # noqa: E402


def main() -> None:
    commit = subprocess.run(["git", "-C", str(PROPOSALS), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    sides = {}
    for which in ("nominal", "fitted"):
        sides[which] = [{"id": run.id, "plateau": int(run.labels["plateau"]), "t": run.target.t.tolist(),
                         "speed": run.target.velocities[:, 0].tolist(), "feedforward": run.target.effort[:, 0].tolist(),
                         "torque": run.measured.effort[:, 0].tolist()} for run in calibration.held_out_runs(which)]
    outcome = calibration.outcome()
    payload = {
        "source": f"robot_models {commit}: docs/proposals/metrics-api-2026-11-17/calibration.py held_out_runs()",
        "units": {"t": "s", "speed": "rad/s", "feedforward": "N·m", "torque": "N·m"},
        "oracle": {"nominal": calibration.oracle_residual_rms("nominal"),
                   "fitted": calibration.oracle_residual_rms("fitted")},
        "outcome": {"part_type": outcome.part_type, "model": outcome.model, "nominal": dict(outcome.nominal),
                    "recovered": {name: interval.value for name, interval in outcome.recovered.items()},
                    "truth": dict(outcome.truth), "tolerance": dict(outcome.tolerance),
                    "fitted_on": outcome.fitted_on, "held_out": outcome.held_out},
        "runs": sides,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes): oracle {payload['oracle']}")


if __name__ == "__main__":
    main()
