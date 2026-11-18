"""The bam_fb five-bar's real speed ladder of 2026-11-10 (and its sim twin) as runs: a rigid chain's example.

    runs("real" | "sim")   -> tuple[Run, ...]   streams "target" / "state" (the two motors, rad) and
                                                "tool/target" / "tool/state" (the L2 tip, PoseTrajectory)

Each rung file holds the tape as bam_fivebar.speed_ladder cut it (``tape_t``): the command and the encoders on one
clock, and the tool by FK of each. The rung's name, speed and feed-forward mode are read from its file name; the reader
never opens the ladder's scoring file.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from robot_msgs.std.header import Header
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory

from ..run import Run

LADDER = Path("/archive/datasets/bam_rig/rack_2026-11-10/fivebar_real/ladder_real_124755/ladder")
FOLDERS = {"real": "rungs", "sim": "sim"}
DEPLOYMENT = "bam_fb five-bar (rack 2026-11-10)"
JOINTS = ("j2", "c1")


def available() -> bool:
    return LADDER.is_dir()


def _tool(t: np.ndarray, xy: np.ndarray) -> PoseTrajectory:
    """The L2 tip in the five-bar's plane as poses (z = 0, identity orientation): what compare() takes for a tool."""

    matrices = np.repeat(np.eye(4)[None], len(xy), axis=0)
    matrices[:, 0, 3], matrices[:, 1, 3] = xy[:, 0], xy[:, 1]
    return PoseTrajectory.from_matrices(t, matrices, child_frame_id="L2 tip")


def _rung(path: Path) -> dict:
    """``rungs/vertical/rung_0.10mps_gravity_ff.npz`` -> name "vertical.0.10mps.gravity_ff", speed 0.10 m/s."""

    speed, ff_mode = path.stem.removeprefix("rung_").split("mps_", 1)
    gravity_mode = path.parent.name
    return {"name": f"{gravity_mode}.{speed}mps.{ff_mode}", "speed_m_s": float(speed), "ff_mode": ff_mode,
            "gravity_mode": gravity_mode}


@functools.cache
def runs(source: str = "real") -> tuple[Run, ...]:
    if source not in FOLDERS:
        raise ValueError(f"source is 'real' or 'sim', not {source!r}")
    out = []
    for path in sorted((LADDER / FOLDERS[source]).glob("*/rung_*.npz")):
        if path.stem.endswith("_approach"):
            continue
        rung = _rung(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        t = np.asarray(arrays["t"], float)
        t = t - t[0]
        count = min(len(arrays["tape_t"]), len(t)) if "tape_t" in arrays else len(t)
        header = Header(stamp=0.0, frame_id="five-bar plane", meta={"source": str(path)})
        target = JointTrajectory(t=t[:count], joint_names=JOINTS,
                                 positions=np.asarray(arrays["target_motors"], float)[:count], header=header)
        state = JointTrajectory(t=t[:count], joint_names=JOINTS,
                                positions=np.asarray(arrays["measured_motors"], float)[:count], header=header)
        out.append(Run(id=f"{source}:{rung['name']}",
                       streams={"target": target, "state": state,
                                "tool/target": _tool(t[:count], np.asarray(arrays["target_tool"], float)[:count]),
                                "tool/state": _tool(t[:count], np.asarray(arrays["measured_tool"], float)[:count])},
                       labels={"deployment": DEPLOYMENT, "station": "bam_fb five-bar", "part_type": "rigid_body_chain",
                               "rung": rung["name"], "speed_m_s": rung["speed_m_s"], "ff_mode": rung["ff_mode"],
                               "gravity_mode": rung["gravity_mode"]},
                       provenance=source, links={"source": str(path)}))
    return tuple(out)
