"""The grasp section's real data, cut down to what coverage needs and cached outside the repo.

- `graspnet_object(id)`: one GraspNet-1Billion object's analytic 6-DOF labels (real scanned YCB objects). The label files
  are 450-850 MB of *stored* (uncompressed) npy members, so they are memory-mapped straight out of the zip and walked in
  point chunks; only the valid grasps (friction coefficient in (0, 0.4], no gripper/object collision, graspnetAPI's
  `loadGrasp` defaults) are kept, 8 bytes each.
- `graspclutter_objects()`: per-object counts for all 200 GraspClutter6D objects (compressed npz; ~0.8 s each).
- `repicks()`: every planned grasp of the real full-arm repick runs (rack 2026-11-10 campaign), joined to its attempt's
  outcome and the claw's measured gap.
- `sim_grasps()`: bam_sim grasp_success episodes (SIMULATION; outcomes are labelled by the sim).

/archive is a slow HDD: the first call of each reads once and writes a cache (`COVERAGE_GALLERY_CACHE`, default
~/.cache/robot_metrics_coverage). Read-only on /archive; nothing here writes next to the data.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("COVERAGE_GALLERY_CACHE") or Path.home() / ".cache/robot_metrics_coverage")
GRASPNET = Path("/archive/datasets/graspnet")
GRASPCLUTTER = Path("/archive/datasets/graspclutter6d")
REPICKS = Path("/archive/datasets/bam_rig/rack_2026-11-10/fivebar_real")
SIM = Path("/archive/datasets/bam_sim/grasp_success")

# GraspNet-1Billion's first 32 objects are YCB; ids matched to YCB by their label points' extents (e.g. 002: 66 x 67 x 100
# mm = the tomato soup can, 007: 116 x 92 x 79 mm = the mug, 005: 108 x 178 x 36 mm = the banana lying flat).
GRASPNET_OBJECTS = {0: "cracker box", 2: "soup can", 5: "banana", 7: "mug"}
FRICTION_MAX = 0.4                # graspnetAPI loadGrasp(fric_coef_thresh=0.4): lower friction needed = better grasp
NUM_VIEWS, NUM_ANGLES, NUM_DEPTHS = 300, 12, 4
DEPTHS_M = np.array([0.01, 0.02, 0.03, 0.04])
WIDTH_EDGES_MM = np.arange(0.0, 152.0, 2.0)


# --- GraspNet-1Billion ----------------------------------------------------------------------------------------------

def generate_views(count: int = NUM_VIEWS, phi: float = (np.sqrt(5) - 1) / 2) -> np.ndarray:
    """graspnetAPI.utils.utils.generate_views, verbatim: the Fibonacci-lattice viewpoints the 300 label views index."""

    index = np.arange(count, dtype=np.float32)
    z = (2 * index + 1) / count - 1
    x = np.sqrt(1 - z ** 2) * np.cos(2 * index * np.pi * phi)
    y = np.sqrt(1 - z ** 2) * np.sin(2 * index * np.pi * phi)
    return np.stack([x, y, z], axis=1)


def grasp_rotation(views: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """graspnetAPI's label -> rotation: batch_viewpoint_params_to_matrix(-views, angles).

    Columns: x = approach (from the viewpoint towards the point), y = the jaw-opening (width) axis, z = the finger
    height axis. The gripper's palm sits 2 cm behind the grasp point along -x (plot_gripper_pro_max's depth_base)."""

    towards = -np.asarray(views, float)
    axis_x = towards / np.linalg.norm(towards, axis=1, keepdims=True)
    axis_y = np.stack([-axis_x[:, 1], axis_x[:, 0], np.zeros(len(axis_x))], axis=1)
    flat = np.linalg.norm(axis_y, axis=1) == 0
    axis_y[flat] = (0.0, 1.0, 0.0)
    axis_y /= np.linalg.norm(axis_y, axis=1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    cos, sin = np.cos(angles), np.sin(angles)
    in_plane = np.zeros((len(angles), 3, 3))
    in_plane[:, 0, 0] = 1.0
    in_plane[:, 1, 1], in_plane[:, 1, 2], in_plane[:, 2, 1], in_plane[:, 2, 2] = cos, -sin, sin, cos
    return np.stack([axis_x, axis_y, axis_z], axis=-1) @ in_plane


def _stored_member(path: Path, name: str) -> np.memmap:
    """Memory-map one STORED (uncompressed) .npy member of an .npz, so a slice reads only its own bytes from disk."""

    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"{path}:{name} is compressed; memory-mapping needs a stored member")
    with open(path, "rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        name_length, extra_length = struct.unpack("<HH", header[26:30])
        handle.seek(info.header_offset + 30 + name_length + extra_length)
        version = np.lib.format.read_magic(handle)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(handle)
        offset = handle.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def _extract_graspnet(object_id: int) -> dict:
    path = GRASPNET / "grasp_label" / f"{object_id:03d}_labels.npz"
    points = np.asarray(_stored_member(path, "points.npy"), np.float32)
    scores = _stored_member(path, "scores.npy")
    collision = _stored_member(path, "collision.npy")
    offsets = _stored_member(path, "offsets.npy")
    count = len(points)
    valid_per_view = np.zeros((count, NUM_VIEWS), np.uint8)
    contact_per_view = np.zeros((count, NUM_VIEWS), np.uint8)      # any antipodal contact (friction <= 1.0)
    friction_counts = np.zeros(11, np.int64)                          # 0.1 .. 1.0 among collision-free, index 1..10; 0 = none
    parts = {key: [] for key in ("point", "view", "angle", "depth", "width_mm", "friction")}
    totals = {"labels": 0, "contact": 0, "collision": 0, "valid": 0}
    for start in range(0, count, 128):
        block = slice(start, min(start + 128, count))
        score = np.asarray(scores[block])
        hit = np.asarray(collision[block])
        offset = np.asarray(offsets[block])
        # WHY score > 0 already means collision-free: the labels store -1 for every gripper/object collision (checked:
        # no label has score > 0 and collision), so `collision` is kept only to count what it removed
        contact = (score > 0) & ~hit
        valid = contact & (score <= FRICTION_MAX)
        valid_per_view[block] = valid.sum(axis=(2, 3))
        contact_per_view[block] = contact.sum(axis=(2, 3))
        friction_counts += np.bincount(np.rint(score[contact] * 10).astype(int), minlength=11)[:11]
        totals["labels"] += score.size
        totals["contact"] += int(contact.sum())
        totals["collision"] += int(hit.sum())
        totals["valid"] += int(valid.sum())
        point, view, angle, depth = np.nonzero(valid)
        parts["point"].append((point + start).astype(np.uint16))
        parts["view"].append(view.astype(np.uint16))
        parts["angle"].append(angle.astype(np.uint8))
        parts["depth"].append(depth.astype(np.uint8))
        parts["width_mm"].append((offset[..., 2][valid] * 1000).astype(np.float16))
        parts["friction"].append(np.rint(score[valid] * 10).astype(np.uint8))
    out = {key: np.concatenate(value) for key, value in parts.items()}
    out.update(points=points, valid_per_view=valid_per_view, contact_per_view=contact_per_view,
               friction_counts=friction_counts, **{f"total_{k}": np.int64(v) for k, v in totals.items()})
    return out


def graspnet_object(object_id: int, refresh: bool = False) -> dict:
    cache = CACHE / f"graspnet_{object_id:03d}.npz"
    if cache.is_file() and not refresh:
        with np.load(cache) as archive:
            data = {name: archive[name] for name in archive.files}
    else:
        data = _extract_graspnet(object_id)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **data)
    data = {key: (value.item() if value.ndim == 0 else value) for key, value in data.items()}
    data["id"], data["name"] = object_id, GRASPNET_OBJECTS.get(object_id, f"object {object_id:03d}")
    return data


# --- GraspClutter6D ---------------------------------------------------------------------------------------------------

def _graspclutter_one(path: str) -> dict:
    with np.load(path) as archive:
        points = archive["points"]
        score = archive["scores"]
        hit = archive["collision"]
        contact = (score > 0) & ~hit
        valid = contact & (score <= FRICTION_MAX)
        friction_counts = np.bincount(np.rint(score[contact] * 10).astype(int), minlength=11)[:11]
        widths = archive["offsets"][..., 2][valid] * 1000.0
    pair = valid.any(axis=(2, 3))                       # (point, approach view) has at least one valid roll/depth
    directions_per_point = pair.sum(axis=1) / NUM_VIEWS
    return {"object": int(Path(path).stem.split("_")[1]), "points": int(len(points)), "labels": int(score.size),
            "valid": int(valid.sum()), "contact": int(contact.sum()), "collision": int(hit.sum()),
            "contact_in_collision": int(((score > 0) & hit).sum()),
            "friction_counts": friction_counts, "width_counts": np.histogram(widths, WIDTH_EDGES_MM)[0],
            "points_graspable": int(pair.any(axis=1).sum()), "views_graspable": int(pair.any(axis=0).sum()),
            "pair_share": float(pair.mean()), "directions_per_point_median": float(np.median(directions_per_point)),
            "points_with_10pct_directions": float(np.mean(directions_per_point >= 0.1)),
            "extent_mm": (points.max(axis=0) - points.min(axis=0)) * 1000.0}


def graspclutter_objects(refresh: bool = False, workers: int = 1) -> list[dict]:
    cache = CACHE / "graspclutter6d_objects.pkl"
    if cache.is_file() and not refresh:
        return pickle.loads(cache.read_bytes())
    files = sorted(glob.glob(str(GRASPCLUTTER / "grasp_label" / "obj_*_labels.npz")))
    # WHY one process by default (orchestrator, 2026-11-19): the real arm's motion gate refuses on a loaded host; a
    # rebuild takes ~4.5 min serial. workers=2 is the machine's ceiling when nothing is running on the rig.
    if workers <= 1:
        rows = [_graspclutter_one(path) for path in files]
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_graspclutter_one, files, chunksize=4)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    return rows


# --- our real repicks -------------------------------------------------------------------------------------------------

def _attempts(run: Path) -> dict[str, dict]:
    """cycle plan file name -> that attempt's record (outcome, claw opening and stall)."""

    found = {}
    path = run / "attempts.jsonl"
    if not path.is_file():
        return found
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "cycle" not in record:
            continue
        plan = (record.get("plan_file") or {}).get("path")
        name = Path(plan).name if plan else f"c{int(record['cycle']):03d}.json"
        opening = ((record.get("first_grasp") or {}).get("opening") or {})
        close = record.get("close") or {}
        item = ((record.get("look") or {}).get("item") or {})
        found[name] = {"outcome": record.get("outcome"), "bucket": record.get("bucket"),
                       "stage_failed": record.get("stage_failed"), "open_gap_mm": opening.get("gap_mm"),
                       "wanted_gap_mm": opening.get("wanted_gap_mm"), "item_width_mm": opening.get("item_width_mm")
                       or item.get("width_mm"), "stall_gap_mm": close.get("stall_gap_mm"), "grasped": close.get("grasped"),
                       "claw_span_mm": item.get("claw_span_mm"), "max_open_mm": opening.get("max_open_mm"),
                       "open_theta_deg": opening.get("theta_deg")}
    return found


def _read_run(run: Path) -> dict:
    attempts = _attempts(run)
    plans = []
    for plan_path in sorted((run / "plans").glob("c*.json")):
        try:
            plan = json.loads(plan_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        grasp = plan.get("grasp") or {}
        if "tcp_base_m" not in grasp:
            continue
        row = {"run": run.name, "cycle": plan_path.stem, "item": plan.get("item_id"),
               "tcp_base_m": grasp["tcp_base_m"], "tcp_plane_m": grasp.get("tcp_plane_m"),
               "approach_base": grasp.get("approach_base"), "opening_dir_base": grasp.get("opening_dir_base"),
               "psi_rad": grasp.get("psi_rad"), "theta_open_rad": grasp.get("theta_open_rad"),
               "theta_item_rad": grasp.get("theta_item_rad"), "j1_rad": grasp.get("j1_rad"), "w_rad": grasp.get("w_rad"),
               "tree": grasp.get("tree"), "fit_margin_mm": grasp.get("fit_margin_mm"), "limits": plan.get("limits"),
               "plane_frame": plan.get("plane_frame"), "stamp_utc": plan.get("stamp_utc")}
        row.update(attempts.get(plan_path.name, {"outcome": None}))
        plans.append(row)
    gaps = np.zeros(0)
    claw = run / "claw_timeline.npz"
    if claw.is_file():
        try:
            with np.load(claw) as archive:
                columns = [str(c) for c in archive["columns"]]
                if "gap_mm" in columns:
                    values = archive["trace"][:, columns.index("gap_mm")]
                    gaps = values[np.isfinite(values)].astype(np.float32)
        except Exception:
            pass
    area = None
    try:
        loop = json.loads((run / "pick_loop.json").read_text())
        if isinstance(loop.get("area"), dict) and "x_m" in loop["area"]:
            area = {"x_m": loop["area"]["x_m"], "z_m": loop["area"]["z_m"], "name": loop["area"].get("name", "")}
    except (OSError, json.JSONDecodeError):
        pass
    return {"run": run.name, "plans": plans, "gap_mm": gaps, "area": area}


def repicks(refresh: bool = False) -> list[dict]:
    cache = CACHE / "repicks_full_arm.pkl"
    if cache.is_file() and not refresh:
        return pickle.loads(cache.read_bytes())
    runs = [_read_run(Path(path)) for path in sorted(glob.glob(str(REPICKS / "pick_loop_full_arm_repick_real_*")))
            if Path(path).is_dir()]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(runs, protocol=pickle.HIGHEST_PROTOCOL))
    return runs


# --- bam_sim grasp_success (SIMULATION) -------------------------------------------------------------------------------

def sim_grasps(refresh: bool = False) -> list[dict]:
    cache = CACHE / "sim_grasp_success.pkl"
    if cache.is_file() and not refresh:
        return pickle.loads(cache.read_bytes())
    rows = []
    for run in ("run1", "run2", "run3_heldout"):
        for path in sorted((SIM / run).glob("s*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("status") != "ok" or not record.get("pick_xyz") or not record.get("outcome"):
                continue
            rows.append({"run": run, "name": record["name"], "variant": record.get("variant"),
                         "object": record.get("object_type"), "pick_xyz": record["pick_xyz"],
                         "belt_z": record.get("belt_z"), "close_width_m": record.get("close_width_m"),
                         "object_size_m": record.get("object_size_m"), "opening_axis": record.get("opening_axis"),
                         "mass_kg": record.get("mass_kg"), "passed": bool(record["outcome"].get("passed")),
                         "codes": record["outcome"].get("codes", []), "episode": record.get("episode_id")})
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    return rows


if __name__ == "__main__":
    import time
    for object_id in GRASPNET_OBJECTS:
        started = time.time()
        data = graspnet_object(object_id, refresh=True)
        print(f"graspnet {object_id:03d}: {len(data['points'])} points, {data['total_valid']} valid grasps, "
              f"{time.time() - started:.0f} s")
    started = time.time()
    print(len(repicks(refresh=True)), "repick runs", f"{time.time() - started:.0f} s")
    print(len(sim_grasps(refresh=True)), "sim episodes")
    started = time.time()
    print(len(graspclutter_objects(refresh=True)), "GraspClutter6D objects", f"{time.time() - started:.0f} s")
