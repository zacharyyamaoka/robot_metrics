"""The real multi-joint recordings section 2 (many joints) and 3 (Cartesian) read, cut to what coverage needs and cached.

- `ladder_rungs()`: every five-bar speed-ladder rung that ran on the rack (Nov 10-11): the two motors (J2, C1, Craig rad, the
  reachable mask's frame), their rates, the board torque and the L2 tip in the plane, real and its sim twin where one ran.
- `wrist_runs()`: the UR wrist rig's multi-joint plays (CAN 14/15/16, 100 Hz board frame), the ones that move all three at once.
- `repicks()`: the full arm's real repick loop (Nov 13-18): measured J2, C1, J1, W1-W3 and the claw gap on the 50 Hz timeline, the
  L2 tip by the five-bar's FK, and per cycle the planned grasp and how the close went.
- `fivebar_limits()`, `wrist_limits()`: the committed limits and reachable masks (2026-11-11) every card measures against.

/archive is a slow HDD: the first call of each reads once and writes a pickle under `COVERAGE_GALLERY_CACHE`
(default ~/.cache/robot_metrics_coverage). Read-only on /archive and on bam_ws; nothing is written next to the data.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("COVERAGE_GALLERY_CACHE") or Path.home() / ".cache/robot_metrics_coverage")
RACK = Path("/archive/datasets/bam_rig/rack_2026-11-10")
FIVEBAR_REAL = RACK / "fivebar_real"
WRIST_RUNS = RACK / "ur_wrist/coverage/real"
FIVEBAR_REACH = Path("/home/bam/bam_ws/src/dev/bam_fivebar/bam_fivebar/data/reachable_2026-11-11")
WRIST_REACH = Path("/home/bam/bam_ws/src/dev/bam_ur_wrist_coverage/bam_ur_wrist_coverage/data/reachable_2026-11-11")
WRIST_BUNDLE = Path("/home/bam/.cache/cad2robot-build/bundles/3f7ef84/fixtures/cad2robot/ur_wrist_test_rig")
WRIST_HOLDER_BOARD = "#4-7"   # the holder's board the reachable file's workspace and facing are judged on

DEG = 180.0 / np.pi


def _cached(name: str, build, refresh: bool = False):
    path = CACHE / name
    if path.is_file() and not refresh:
        return pickle.loads(path.read_bytes())
    value = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    return value


# --- five-bar: limits, FK, ladders --------------------------------------------------------------------------------------

# bam_fivebar.bench.BenchFiveBar's geometry (m, rad) on the real assembly's branches (closure -1, secondary -1, the CAD lever).
# WHY a copy and not an import: bam_fivebar.kinematics.forward_pose needs the five_bar package; this circle intersection reproduces
# every ladder rung's recorded measured_tool to 1e-13 m (checked on ladder_real_124755), so it IS that FK for these runs.
_L_SHOULDER, _L1, _L2, _LC1, _LC2, _LE, _ELBOW_PHASE = 0.1375 - 0.0505, 0.240, 0.299658, 0.120, 0.260, 0.120, np.pi / 4


def fivebar_fk(j2, c1) -> tuple[np.ndarray, np.ndarray]:
    """Motors (J2, C1, Craig rad) -> the L2 tip (x, y in the plane, m) and L2's heading j2 + j3 (rad). NaN where it cannot close."""

    j2, c1 = np.asarray(j2, float), np.asarray(c1, float)
    ex, ey = _L_SHOULDER + _L1 * np.cos(j2), _L1 * np.sin(j2)
    dx, dy = _LC1 * np.cos(c1) - ex, _LC1 * np.sin(c1) - ey
    d = np.hypot(dx, dy)
    a = (_LE ** 2 - _LC2 ** 2 + d ** 2) / (2 * d)
    h2 = _LE ** 2 - a ** 2
    h = np.sqrt(np.where(h2 >= 0, h2, np.nan))
    px, py = ex + a * dx / d - h * (-dy) / d, ey + a * dy / d - h * dx / d
    heading = np.arctan2(py - ey, px - ex) - np.pi + _ELBOW_PHASE
    return np.column_stack([ex + _L2 * np.cos(heading), ey + _L2 * np.sin(heading)]), heading


def fivebar_limits() -> dict:
    """J2 / C1 joint descriptions, the 1-degree joint-plane masks and the 2 cm workspace pixels (bam_fivebar.reachable/1)."""

    document = json.loads(FIVEBAR_REACH.with_suffix(".json").read_text())
    with np.load(FIVEBAR_REACH.with_suffix(".npz")) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    joints = {}
    for can, name in (("12", "J2"), ("13", "C1")):
        d = document["joint_descriptions"][can]
        joints[name] = {"can": int(can), "low": d["origin_rad"] + d["range_rad"][0], "high": d["origin_rad"] + d["range_rad"][1],
                        "qd": d["speed_rad_s"], "qdd": d["accel_rad_s2"], "torque": d["torque_limit_nm"]}
    return {"joints": joints, "axis_j2_deg": arrays["axis_j2_deg"], "axis_c1_deg": arrays["axis_c1_deg"],
            "allowed": arrays["allowed"], "closes": arrays["closes"], "reason": arrays["reason"],
            "pixels_allowed": arrays["workspace_pixels_allowed"], "pixels_closes": arrays["workspace_pixels_closes"],
            "heading_allowed": arrays["heading_cells_allowed"], "pixel_m": document["workspace"]["pixel_m"],
            "keep_out_j2_deg": 20.0, "document": document}


def wrap_c1(c1, low_rad: float) -> np.ndarray:
    """The crank can sit whole turns from the mask's one turn (bam_fivebar.reachable.real_coverage's rule)."""

    return low_rad + np.mod(np.asarray(c1, float) - low_rad, 2 * np.pi)


def _rung_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        a = {name: np.asarray(archive[name]) for name in ("t", "measured_motors", "measured_motor_rate", "target_motors",
                                                           "measured_tool", "target_tool", "board_torque") if name in archive.files}
    keep = np.isfinite(a["measured_motors"]).all(axis=1) & np.isfinite(a["measured_tool"]).all(axis=1)
    return {name: value[keep].astype(np.float32) if value.ndim else value for name, value in a.items()}


def _build_ladders() -> list[dict]:
    out = []
    for summary in sorted(FIVEBAR_REAL.glob("ladder_real_*/ladder/summary.json")):
        ladder = summary.parent
        for step in json.loads(summary.read_text())["steps"]:
            if step.get("kind") != "rung" or step.get("result") != "ran" or not step.get("real"):
                continue
            entry = {"ladder": ladder.parent.name, "name": step["name"], "speed_m_s": float(step["speed_m_s"]),
                     "ff_mode": step.get("ff_mode"), "real": _rung_arrays(ladder / step["real"])}
            sim = ladder / step["sim"] if step.get("sim") else None
            entry["sim"] = _rung_arrays(sim) if sim is not None and sim.is_file() else None
            out.append(entry)
    return out


def ladder_rungs(refresh: bool = False) -> list[dict]:
    return _cached("arm_fivebar_ladders.pkl", _build_ladders, refresh)


# --- UR wrist rig -------------------------------------------------------------------------------------------------------

WRIST_CANS = (14, 15, 16)


def wrist_limits() -> dict:
    """The three joint descriptions (board frame), the CAD frame offsets and the committed 29^3 joint mask (bam_ur_wrist_coverage.reachable/1)."""

    document = json.loads(WRIST_REACH.with_suffix(".json").read_text())
    with np.load(WRIST_REACH.with_suffix(".npz")) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    joints = {}
    for can in WRIST_CANS:
        d = document["joint_descriptions"][str(can)]
        joints[can] = {"can": can, "low": d["window_rad"][0], "high": d["window_rad"][1], "origin": d["origin_rad"],
                       "qd": d["speed_rad_s"], "qdd": d["accel_rad_s2"], "torque": d["torque_limit_nm"],
                       "inertia": d["inertia_kg_m2"], "gravity": d["gravity_nm"], "bottom": d["gravity_bottom_rad"],
                       "coulomb": d["coulomb_nm"]}
    # WHY the reachable file's frames (CAN 14: the real lane's run_f homing; 15/16: bundle adjustment 8afe1d14) and not rig.OFFSET_RAD:
    # rig.OFFSET_RAD is the TWIN's board frame (multi3.py says so); the real plays are mapped with these (multi3.OFFSET_RAD)
    offsets = {int(can): float(deg) / DEG for can, deg in document["frames"]["cad_equals_board_plus_deg"].items()}
    return {"joints": joints, "offset_rad": offsets, "axis_cad_deg": arrays["axis_cad_deg"], "allowed": arrays["allowed"],
            "clear": arrays["clear"], "voxels_allowed": arrays["workspace_voxels_allowed"],
            "facing_allowed": arrays["facing_cells_allowed"], "voxel_m": document["workspace"]["voxel_m"], "document": document}


def _build_wrist() -> list[dict]:
    out = []
    for path in sorted(glob.glob(str(WRIST_RUNS / "**" / "runs" / "*_samples_multi.npz"), recursive=True)):
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            columns = {name: np.asarray(archive[name]) for name in archive.files}
        if not all(f"q_can{can}" in columns for can in WRIST_CANS):
            continue
        t = columns["t"].astype(float)
        stack = lambda prefix: np.column_stack([columns[f"{prefix}_can{can}"].astype(float) for can in WRIST_CANS])  # noqa: E731
        q, qd = stack("q"), stack("qd")
        keep = np.isfinite(t) & np.isfinite(q).all(axis=1) & np.isfinite(qd).all(axis=1)
        entry = {"file": str(path), "campaign": path.relative_to(WRIST_RUNS).parts[0],
                 "session": path.relative_to(WRIST_RUNS).parts[1], "t": t[keep], "q": q[keep].astype(np.float32),
                 "qd": qd[keep].astype(np.float32)}
        for prefix in ("tau", "q_target"):
            if all(f"{prefix}_can{can}" in columns for can in WRIST_CANS):
                entry[prefix] = stack(prefix)[keep].astype(np.float32)
        out.append(entry)
    return out


def wrist_runs(refresh: bool = False) -> list[dict]:
    return _cached("arm_wrist_multi.pkl", _build_wrist, refresh)


def wrist_holder(q_cad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The holder board #4-7's centre (base_link, m) and normal for each CAD pose, by MuJoCo on the rig's bundle (multi.holder_samples)."""

    import mujoco

    robot = json.loads((WRIST_BUNDLE / "robot.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(WRIST_BUNDLE / robot["mjcf"]))
    data = mujoco.MjData(model)
    marker = next(m for m in json.loads((WRIST_BUNDLE / "aruco_markers.json").read_text())["markers"]
                  if f"#{m['ids'][0]}-{m['ids'][-1]}" == WRIST_HOLDER_BOARD)
    body = model.body(marker.get("mjcf_body") or marker["link"]).id
    link_board = np.asarray(marker["T_link_marker"], float)
    names = ("revolute_11", "revolute_14", "revolute_17")      # CAN 14, 15, 16
    address = [model.jnt_qposadr[model.joint(name).id] for name in names]
    points, normals = np.empty((len(q_cad), 3)), np.empty((len(q_cad), 3))
    for index, q in enumerate(np.asarray(q_cad, float)):
        data.qpos[address] = q
        mujoco.mj_kinematics(model, data)
        pose = np.eye(4)
        pose[:3, :3] = data.xmat[body].reshape(3, 3)
        pose[:3, 3] = data.xpos[body]
        board = pose @ link_board
        points[index], normals[index] = board[:3, 3], board[:3, 2]
    return points, normals


WRIST_HOLDER_STRIDE = 4      # every 4th 100 Hz row (25 Hz): the holder moves < 1 cm between kept rows


def _build_wrist_holder() -> dict:
    limits = wrist_limits()
    runs = wrist_runs()
    poses, run_of = [], []
    for index, run in enumerate(runs):
        board = np.asarray(run["q"], float)[::WRIST_HOLDER_STRIDE]
        poses.append(board + np.array([limits["offset_rad"][can] for can in WRIST_CANS]))
        run_of.append(np.full(len(board), index))
    q_cad = np.concatenate(poses)
    points, normals = wrist_holder(q_cad)
    return {"q_cad": q_cad.astype(np.float32), "points": points.astype(np.float32), "normals": normals.astype(np.float32),
            "run": np.concatenate(run_of)}


def wrist_holder_samples(refresh: bool = False) -> dict:
    """Every multi-joint play's CAD pose (25 Hz) with the holder board's centre and normal: the wrist's Cartesian coverage."""

    return _cached("arm_wrist_holder.pkl", _build_wrist_holder, refresh)


# --- full arm repicks ---------------------------------------------------------------------------------------------------

_MOVE = re.compile(r"^m(\d+)_(.+)$")


def _repick(folder: Path) -> dict | None:
    try:
        summary = json.loads((folder / "summary.json").read_text())
        timeline = np.load(folder / "timeline.npz")
    except (OSError, ValueError):
        return None
    moves, rows, move_of = [], [], []
    for key in sorted(timeline.files):
        block = np.asarray(timeline[key], float)
        if block.ndim != 2 or len(block) == 0:
            continue
        match = _MOVE.match(key)
        moves.append(match.group(2) if match else key)
        rows.append(block)
        move_of.append(np.full(len(block), len(moves) - 1, np.int16))
    if not rows:
        return None
    block = np.concatenate(rows)
    move_of = np.concatenate(move_of)
    t = block[:, 0]
    out = {"run": folder.name, "area": None, "result": summary.get("result"), "attempts": summary.get("attempts") or 0,
           "successes": summary.get("successes") or 0, "moves": moves, "move": move_of, "t": t.astype(np.float64),
           "j2": block[:, 1].astype(np.float32), "c1": block[:, 2].astype(np.float32)}
    try:
        j1 = np.load(folder / "j1.npz")
        out["j1"] = np.interp(t, j1["t"], j1["joint_rad"]).astype(np.float32)
    except (OSError, ValueError, KeyError):
        out["j1"] = block[:, 3].astype(np.float32)
    try:
        loop = json.loads((folder / "pick_loop.json").read_text())
        area = loop.get("area") or {}
        out["area"] = {key: area.get(key) for key in ("name", "x_m", "z_m", "surface_y_m")} if area else None
        maps = loop["maps"]["wrist"]
        wrist = np.load(folder / "wrist_timeline.npz")
        columns = {str(name): index for index, name in enumerate(wrist["columns"])}
        trace = np.asarray(wrist["trace"], float)
        w = []
        for name, joint_map in zip(("W1", "W2", "W3"), maps):
            board = np.interp(t, trace[:, 0], trace[:, columns[f"{name}_board_rad"]])
            w.append(joint_map["sign"] * (board - joint_map["zero_board_rad"]) + joint_map["joint_at_zero_rad"])
        out["w"] = np.column_stack(w).astype(np.float32)
    except (OSError, ValueError, KeyError, TypeError):
        out["w"] = None
    try:
        claw = np.load(folder / "claw_timeline.npz")
        columns = {str(name): index for index, name in enumerate(claw["columns"])}
        trace = np.asarray(claw["trace"], float)
        gap = trace[:, columns["gap_mm"]]
        ok = np.isfinite(gap)
        out["gap_mm"] = (np.interp(t, trace[ok, 0], gap[ok]) if ok.sum() > 2 else np.full(len(t), np.nan)).astype(np.float32)
    except (OSError, ValueError, KeyError):
        out["gap_mm"] = None
    attempts = {}
    if (folder / "attempts.jsonl").is_file():
        for line in (folder / "attempts.jsonl").read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if "cycle" in record:
                close = record.get("close") or {}
                attempts[int(record["cycle"])] = {"outcome": record.get("outcome"), "bucket": record.get("bucket"),
                                                  "closed": bool(close), "grasped": close.get("grasped"),
                                                  "stall_gap_mm": close.get("stall_gap_mm")}
    plans = []
    for plan_json in sorted((folder / "plans").glob("c*.json")):
        try:
            document = json.loads(plan_json.read_text())
            arrays = np.load(plan_json.with_suffix(".npz"))
            grasp = document.get("grasp") or {}
            plans.append({"cycle": int(document.get("cycle") or 0), "legs": [leg["name"] for leg in document.get("legs", [])],
                          "tcp": np.asarray(arrays["tcp"], np.float32), "leg": np.asarray(arrays["leg"], np.int16),
                          "grasp_tcp_base": grasp.get("tcp_base_m"), "approach_base": grasp.get("approach_base"),
                          "psi_rad": grasp.get("psi_rad"), "item": document.get("item_id"),
                          "limits": document.get("limits"), "plane_frame": document.get("plane_frame")})
        except (OSError, ValueError, KeyError):
            continue
    out["attempt"] = attempts
    out["plans"] = plans
    return out


def _build_repicks() -> list[dict]:
    folders = sorted(path for path in FIVEBAR_REAL.glob("pick_loop_full_arm_repick_real_*") if path.is_dir())
    # WHY one process: the real arm's motion gate refuses on a loaded host (coordinator, 2026-11-19); the HDD is the bottleneck anyway
    runs = [_repick(folder) for folder in folders]
    return [run for run in runs if run is not None]


def repicks(refresh: bool = False) -> list[dict]:
    return _cached("arm_repicks.pkl", _build_repicks, refresh)


if __name__ == "__main__":
    import time

    for name, build in (("ladders", ladder_rungs), ("wrist", wrist_runs), ("repicks", repicks), ("wrist holder", wrist_holder_samples)):
        start = time.time()
        value = build(refresh=True)
        size = len(value["q_cad"]) if isinstance(value, dict) else len(value)
        print(f"{name}: {size} in {time.time() - start:.0f} s")
