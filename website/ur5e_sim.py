"""A simulated UR5e for the gallery: MuJoCo Menagerie's UR5e under its own position servos, as robot_metrics runs.

    sim = simulate(speed=1.0, controller="pd")               # plain arrays: t, joints, flange poses
    run = pick_and_place_run(speed=1.0, controller="pd")     # a Run: target / state (6 joints), tool/target / tool/state
    run = pick_and_place_run(speed=2.0, controller="pd+ff")  # the same path, twice as fast, with model feed-forward

SIMULATED, and labelled so wherever it is shown: no real UR5e run exists. The plant is Menagerie's
``universal_robots_ur5e/ur5e.xml`` (BSD-3-Clause, not vendored here) as shipped: each joint a PD position servo
(kp 2000 / kd 400 on the big joints, 500 / 100 on the wrist; the damping acts on the joint's speed, not on its speed
error), no gravity compensation. ``controller="pd+ff"`` adds MuJoCo's own gravity compensation (``body_gravcomp``)
and a velocity feed-forward (the damping turned onto the speed error), so the two controllers bracket a servo tune. The path is a joint-space pick-and-place through seven
waypoints, quintic between them, with a 0.6 s dwell at each (a stop the stop kernel can find).

The tool is the flange (Menagerie's ``attachment_site``): its pose by forward kinematics of the commanded joints and
of the simulated ones, so ``tool/target`` vs ``tool/state`` is the tool error the joints' errors make.

WHY here and not in the package: robot_metrics reads runs and never simulates; this is the gallery's data source.
Set ``MENAGERIE`` to a mujoco_menagerie checkout (default ``~/.cache/mujoco_menagerie``), or install
``robot_descriptions``, which fetches it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from robot_msgs.std.header import Header
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory

from robot_metrics import Run

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
HOME = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)       # Menagerie's "home" keyframe
ABOVE_PICK = (-0.9, -1.3, 1.8, -2.1, -1.35, 0.3)
PICK = (-0.9, -1.1, 1.95, -2.45, -1.35, 0.3)
ABOVE_PLACE = (-2.2, -1.3, 1.8, -2.1, -1.8, -0.5)
PLACE = (-2.2, -1.1, 1.95, -2.45, -1.8, -0.5)
WAYPOINTS = (HOME, ABOVE_PICK, PICK, ABOVE_PICK, ABOVE_PLACE, PLACE, ABOVE_PLACE, HOME)
DWELL_S = 0.6               # each stop: longer than the stop kernel's 0.25 s minimum dwell
NOMINAL_SPEED_RAD_S = 0.8   # the largest joint's mean speed on a segment at speed=1.0
RECORD_HZ = 100.0           # the run's sample rate (the sim steps at the model's own 2 ms)
CONTROLLERS = ("pd", "pd+ff")


def model_path() -> Path:
    root = os.environ.get("MENAGERIE")
    candidates = [Path(root)] if root else []
    candidates.append(Path.home() / ".cache" / "mujoco_menagerie")
    for candidate in candidates:
        path = candidate / "universal_robots_ur5e" / "ur5e.xml"
        if path.is_file():
            return path
    try:
        from robot_descriptions import ur5e_mj_description
    except ImportError as error:
        raise FileNotFoundError("no Menagerie UR5e: set MENAGERIE or pip install robot_descriptions") from error
    return Path(ur5e_mj_description.MJCF_PATH)


def path_segments(speed: float) -> list[tuple[str, tuple, tuple, float]]:
    """The pick-and-place as (kind, start, end, duration) segments: rest-to-rest quintic moves, a dwell after each."""

    segments = [("dwell", WAYPOINTS[0], WAYPOINTS[0], DWELL_S)]
    for start, end in zip(WAYPOINTS[:-1], WAYPOINTS[1:]):
        largest_move = float(np.max(np.abs(np.subtract(end, start))))
        duration = max(0.4, largest_move / (NOMINAL_SPEED_RAD_S * speed))
        segments.append(("move", start, end, duration))
        segments.append(("dwell", end, end, DWELL_S))
    return segments


def _sample(segments: list, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, qd, qdd = (np.zeros((t.size, len(JOINTS))) for _ in range(3))
    edge = 0.0
    for kind, start, end, duration in segments:
        rows = (t >= edge) & (t < edge + duration)
        start, end = np.asarray(start), np.asarray(end)
        if kind == "dwell":
            q[rows] = end
        else:
            tau = (t[rows] - edge)[:, None] / duration
            blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            blend_rate = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
            blend_accel = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
            q[rows] = start + blend * (end - start)
            qd[rows] = blend_rate * (end - start)
            qdd[rows] = blend_accel * (end - start)
        edge += duration
    q[t >= edge] = np.asarray(segments[-1][2])
    return q, qd, qdd


def _flange_poses(model, data, positions: np.ndarray, site: int) -> np.ndarray:
    import mujoco

    matrices = np.repeat(np.eye(4)[None], len(positions), axis=0)
    for row, joint_positions in enumerate(positions):
        data.qpos[:] = joint_positions
        mujoco.mj_kinematics(model, data)
        matrices[row, :3, :3] = data.site_xmat[site].reshape(3, 3)
        matrices[row, :3, 3] = data.site_xpos[site]
    return matrices


@dataclass(frozen=True)
class Simulated:
    """One simulated pick-and-place as plain arrays (SI): what a recorder would hand over, before any robot_msgs."""

    t: np.ndarray                 # (N,) s
    target_q: np.ndarray          # (N, 6) rad: the commanded joints
    target_qd: np.ndarray         # (N, 6) rad/s
    state_q: np.ndarray           # (N, 6) rad: the simulated joints
    state_qd: np.ndarray          # (N, 6) rad/s
    target_flange: np.ndarray     # (N, 4, 4): the flange by FK of the commanded joints
    state_flange: np.ndarray      # (N, 4, 4): the flange by FK of the simulated joints
    controller: str
    speed: float


def simulate(speed: float = 1.0, controller: str = "pd") -> Simulated:
    import mujoco

    if controller not in CONTROLLERS:
        raise ValueError(f"controller is one of {CONTROLLERS}, not {controller!r}")
    spec = mujoco.MjSpec.from_file(str(model_path()))
    if controller == "pd+ff":
        # WHY at compile time: MuJoCo counts the gravity-compensated bodies when it compiles; set later, it is ignored
        for body in spec.bodies:
            body.gravcomp = 1.0
    model = spec.compile()
    data = mujoco.MjData(model)
    step_s = float(model.opt.timestep)
    segments = path_segments(speed)
    duration = sum(segment[3] for segment in segments)
    sim_t = np.arange(0.0, duration, step_s)
    q_target, qd_target, _ = _sample(segments, sim_t)
    kp = np.asarray(model.actuator_gainprm[:, 0], float)         # Menagerie: force = kp (ctrl - q) - kd qd
    kd = -np.asarray(model.actuator_biasprm[:, 2], float)
    # WHY the lead: damping on speed (not on speed error) is what lags a move; gravity is what sags a stop
    lead = kd / kp if controller == "pd+ff" else np.zeros(len(JOINTS))   # ctrl = q + (kd/kp) qd: kd on speed error

    data.qpos[:] = q_target[0]
    data.ctrl[:] = q_target[0]
    mujoco.mj_forward(model, data)
    every = max(1, round(1.0 / (RECORD_HZ * step_s)))
    measured_q, measured_qd = [], []
    for row in range(sim_t.size):
        data.ctrl[:] = q_target[row] + lead * qd_target[row]
        if row % every == 0:
            measured_q.append(data.qpos.copy())
            measured_qd.append(data.qvel.copy())
        mujoco.mj_step(model, data)

    picked = np.arange(0, sim_t.size, every)
    target_q, state_q = q_target[picked], np.asarray(measured_q)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    scratch = mujoco.MjData(model)
    return Simulated(t=sim_t[picked], target_q=target_q, target_qd=qd_target[picked], state_q=state_q,
                     state_qd=np.asarray(measured_qd), target_flange=_flange_poses(model, scratch, target_q, site),
                     state_flange=_flange_poses(model, scratch, state_q, site), controller=controller,
                     speed=float(speed))


def pick_and_place_run(speed: float = 1.0, controller: str = "pd") -> Run:
    """``simulate`` as a robot_metrics Run: the streams the gallery's first cell builds by hand, and the labels."""

    sim = simulate(speed, controller)
    header = Header(stamp=0.0, frame_id="base", meta={"model": "mujoco_menagerie universal_robots_ur5e",
                                                       "controller": controller, "provenance": "simulated"})
    target = JointTrajectory(t=sim.t, joint_names=JOINTS, positions=sim.target_q, velocities=sim.target_qd,
                             header=header)
    state = JointTrajectory(t=sim.t, joint_names=JOINTS, positions=sim.state_q, velocities=sim.state_qd, header=header)
    tool_target = PoseTrajectory.from_matrices(sim.t, sim.target_flange, child_frame_id="flange")
    tool_state = PoseTrajectory.from_matrices(sim.t, sim.state_flange, child_frame_id="flange")
    return Run(id=f"ur5e-sim:{controller}:x{speed:g}",
               streams={"target": target, "state": state, "tool/target": tool_target, "tool/state": tool_state},
               labels={"deployment": "ur5e (simulated)", "station": "ur5e-sim", "part_type": "rigid_body_chain",
                       "controller": controller, "speed": float(speed)},
               provenance="sim")
