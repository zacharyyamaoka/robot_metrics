"""A small synthetic 1-DOF run for the tests that must run without /archive: A -> B -> A with three dwells.

The plan rests at A for 0.6 s, moves to B over 1 s (a cosine), rests 1.2 s, moves back over 1 s, rests 0.8 s; 100 Hz.
The response is the plan 15 ms late with a 0.4 mrad overshoot that decays in each dwell, plus seeded noise. Limiter
codes are planted on a few rows. Plan, target and state share one clock (the sim's ``same_clock`` case).
"""

from __future__ import annotations

import numpy as np
from robot_msgs.std.header import Header
from robot_msgs.trajectory import JointTrajectory

from robot_metrics import Run

RATE_HZ = 100.0


def plan(a: float = 0.0, b: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pieces = [(0.6, "rest", a), (1.0, "move", (a, b)), (1.2, "rest", b), (1.0, "move", (b, a)), (0.8, "rest", a)]
    t_all, q_all, qd_all, start = [], [], [], 0.0
    for duration, kind, where in pieces:
        t = start + np.arange(int(round(duration * RATE_HZ))) / RATE_HZ
        if kind == "rest":
            q, qd = np.full(t.size, where), np.zeros(t.size)
        else:
            s = (t - start) / duration
            low, high = where
            q = low + (high - low) * 0.5 * (1.0 - np.cos(np.pi * s))
            qd = (high - low) * 0.5 * np.pi / duration * np.sin(np.pi * s)
            qd[0] = 1e-6 * np.sign(high - low)    # the move's first tick already commands motion
        t_all.append(t), q_all.append(q), qd_all.append(qd)
        start += duration
    return np.concatenate(t_all), np.concatenate(q_all), np.concatenate(qd_all)


def run(seed: int = 0, *, lag_s: float = 0.015, overshoot_rad: float = 4e-4, rung: float = 0.5,
        station: str = "synthetic") -> Run:
    t, q_target, qd_target = plan()
    rng = np.random.default_rng(seed)
    lagged = np.interp(t - lag_s, t, q_target)
    settling = np.zeros_like(t)
    still = np.abs(qd_target) < 1e-9
    since_move = 0.0
    for row in range(1, t.size):
        since_move = since_move + (t[row] - t[row - 1]) if still[row] else 0.0
        if still[row] and q_target[row] != q_target[0]:
            settling[row] = overshoot_rad * np.exp(-since_move / 0.08)
    q = lagged + settling + rng.normal(0.0, 2e-5, t.size)
    fault = np.zeros(t.size)
    fault[[150, 151, 152]] = 98            # three limiter rows
    header = Header(stamp=0.0, frame_id="output", meta={"source": f"synthetic seed {seed}"})
    names = (station,)
    target = JointTrajectory(t=t, joint_names=names, positions=q_target[:, None], velocities=qd_target[:, None],
                             header=header)
    state = JointTrajectory(t=t, joint_names=names, positions=q[:, None], velocities=np.gradient(q, t)[:, None],
                            effort=np.zeros((t.size, 1)), channels={"fault": fault}, header=header)
    return Run(id=f"synthetic:{seed}", streams={"plan": target, "target": target, "state": state},
               labels={"deployment": "synthetic", "station": station, "rung": rung, "kind": "repeat"},
               provenance="sim")
