"""Time alignment: an explicit step before any metric, with a tolerance, never inside one (v3's aligner).

    out = align(plan, replies, rule="hermite", tolerance_s=0.01)   # offline: the plan at each reply's own instant
    out.target, out.measured, out.alignment.dropped                 # an AlignedPair, and how it was made
    aligner = Aligner(rule="hermite", tolerance_s=0.01)             # live: the same answer, message by message
    aligner.push(reference=knot) / aligner.push(measured=reply)     # -> Aligned | None (what it could release)
    aligner.finish()                                                # the replies after the plan's last knot

Rules:
- ``hermite``: the reference is a plan on its ticks (positions and commanded speeds). Each reply is compared with the
  cubic Hermite through the knots either side of its own instant (round 8's scorecard rule, ``align: measured``), and
  its commanded speed is its own knot's. A reply before the first knot or after the last is paired by the edge knot's
  speed only within ``tolerance_s``; further out it is dropped, never extrapolated.
- ``same_clock``: the reference was sampled on the reply's clock (a sim, a ladder's tape). Each reply takes the
  reference sample nearest in time, within ``tolerance_s``; a reply with none is dropped.

WHY a separate step (the prior-art note §1, falsification 3: evo ``max_diff``, Autoware ``sync_tolerance_ms``): every
robotics tool lines streams up outside the metric, with a tolerance, and records what it dropped. Every metric of a
pair needs the same answer, so it is computed once, here. Live, a hermite reply waits for the knot after it: a stated
one-tick latency, never an extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from robot_msgs.trajectory import JointTrajectory, concatenate

RULES = ("hermite", "same_clock")


@dataclass(frozen=True)
class Alignment:
    """How a pair was lined up, kept beside the numbers (robot_msgs.compare.Comparison's time_offset_s / dropped /
    coverage idea): the rule, its tolerance, the replies dropped, the share of the reference's span the replies cover."""

    rule: str
    tolerance_s: float
    dropped: int
    coverage: float


@dataclass(frozen=True)
class Aligned:
    target: JointTrajectory
    measured: JointTrajectory
    alignment: Alignment


def _hermite(t: np.ndarray, knots_t: np.ndarray, knots_q: np.ndarray, knots_qd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The plan at each t (pendulum_score.measured_target's rule) and its own knot's commanded speed."""

    count = knots_t.size
    own = np.clip(np.searchsorted(knots_t, t, side="right") - 1, 0, count - 1)
    if count == 1:
        return knots_q[0] + knots_qd[0] * (t - knots_t[0]), knots_qd[own]
    j = np.clip(np.searchsorted(knots_t, t, side="right") - 1, 0, count - 2)
    dt = knots_t[j + 1] - knots_t[j]
    s = (t - knots_t[j]) / dt
    p0, p1, m0, m1 = knots_q[j], knots_q[j + 1], knots_qd[j], knots_qd[j + 1]
    inside = (s >= 0.0) & (s <= 1.0)
    h01 = s * s * (3.0 - 2.0 * s)
    hermite = p0 + h01 * (p1 - p0) + dt * (s * (1.0 - s) ** 2 * m0 + s * s * (s - 1.0) * m1)
    hermite = np.clip(hermite, np.minimum(p0, p1), np.maximum(p0, p1))
    before = knots_q[0] + knots_qd[0] * (t - knots_t[0])
    after = knots_q[-1] + knots_qd[-1] * (t - knots_t[-1])
    return np.where(inside, hermite, np.where(t < knots_t[0], before, after)), knots_qd[own]


def _coverage(t: np.ndarray, reference_t: np.ndarray) -> float:
    span = float(reference_t[-1] - reference_t[0]) if reference_t.size else 0.0
    if span <= 0.0 or not t.size:
        return 1.0 if t.size else 0.0
    covered = min(float(t[-1]), float(reference_t[-1])) - max(float(t[0]), float(reference_t[0]))
    return float(np.clip(covered / span, 0.0, 1.0))


def _single_joint(reference: JointTrajectory) -> None:
    if reference.positions.shape[1] != 1:
        raise ValueError("hermite alignment pairs one joint's plan with its replies; select() the joint first")
    if reference.velocities is None:
        raise ValueError("hermite alignment needs the plan's commanded speeds (velocities)")


def _hermite_pair(reference: JointTrajectory, replies: JointTrajectory) -> JointTrajectory:
    t = np.asarray(replies.t, float)
    q, qd = _hermite(t, np.asarray(reference.t, float), np.asarray(reference.positions[:, 0], float),
                     np.asarray(reference.velocities[:, 0], float))
    return JointTrajectory(t=t, joint_names=replies.joint_names, positions=q[:, None], velocities=qd[:, None],
                           header=replies.header)


def align(reference: JointTrajectory, measured: JointTrajectory, *, rule: str = "hermite",
          tolerance_s: float = 0.01) -> Aligned:
    """``measured`` with the reference at each of its own instants: an AlignedPair plus how it was made."""

    if rule not in RULES:
        raise ValueError(f"rule is one of {RULES}, not {rule!r}")
    t = np.asarray(measured.t, float)
    reference_t = np.asarray(reference.t, float)
    if rule == "hermite":
        _single_joint(reference)
        keep = (t >= reference_t[0] - tolerance_s) & (t <= reference_t[-1] + tolerance_s)
        inside = measured[np.flatnonzero(keep)]
        target = _hermite_pair(reference, inside)
    else:
        nearest = np.clip(np.searchsorted(reference_t, t), 1, max(reference_t.size - 1, 1))
        left = np.clip(nearest - 1, 0, reference_t.size - 1)
        right = np.clip(nearest, 0, reference_t.size - 1)
        pick = np.where(np.abs(reference_t[left] - t) <= np.abs(reference_t[right] - t), left, right)
        keep = np.abs(reference_t[pick] - t) <= tolerance_s
        inside = measured[np.flatnonzero(keep)]
        rows = reference[pick[keep]]
        target = type(rows)(**{**_fields(rows), "t": np.asarray(inside.t, float), "header": inside.header})
    dropped = int(np.count_nonzero(~keep))
    return Aligned(target, inside, Alignment(rule, float(tolerance_s), dropped, _coverage(np.asarray(inside.t),
                                                                                           reference_t)))


def _fields(trajectory: JointTrajectory) -> dict:
    return {"joint_names": trajectory.joint_names, "positions": trajectory.positions,
            "velocities": trajectory.velocities, "accelerations": trajectory.accelerations,
            "effort": trajectory.effort, "channels": dict(trajectory.channels)}


class Aligner:
    """The hermite (or same-clock) step, live: knots and replies arrive in any chunking, in arrival order; a reply is
    released once the knot after it exists, so every released pair is exactly what ``align`` gives offline.

    A data adapter, not a metric: it holds the replies it cannot pair yet. ``max_held`` is the latency actually paid,
    in replies."""

    def __init__(self, rule: str = "hermite", tolerance_s: float = 0.01) -> None:
        if rule not in RULES:
            raise ValueError(f"rule is one of {RULES}, not {rule!r}")
        self.rule, self.tolerance_s = rule, float(tolerance_s)
        self.knots: list[JointTrajectory] = []
        self.held: list[JointTrajectory] = []
        self.dropped = 0
        self.max_held = 0

    def _plan(self) -> JointTrajectory | None:
        if not self.knots:
            return None
        if len(self.knots) > 1:
            self.knots = [concatenate(self.knots)]
        return self.knots[0]

    def push(self, *, reference: JointTrajectory | None = None, measured: JointTrajectory | None = None) -> Aligned | None:
        if reference is not None and len(reference):
            self.knots.append(reference)
        if measured is not None and len(measured):
            self.held.append(measured)
            self.max_held = max(self.max_held, sum(len(piece) for piece in self.held))
        plan = self._plan()
        if plan is None or not self.held:
            return None
        waiting = self.held[0] if len(self.held) == 1 else concatenate(self.held)
        last = float(plan.t[-1])
        ready = np.asarray(waiting.t, float) < last if self.rule == "hermite" else (
            np.asarray(waiting.t, float) <= last + self.tolerance_s)
        if not ready.any():
            self.held = [waiting]
            return None
        self.held = [waiting[np.flatnonzero(~ready)]] if not ready.all() else []
        return self._release(plan, waiting[np.flatnonzero(ready)])

    def finish(self) -> Aligned | None:
        plan = self._plan()
        if not self.held:
            return None
        waiting = self.held[0] if len(self.held) == 1 else concatenate(self.held)
        self.held = []
        if plan is None:
            self.dropped += len(waiting)
            return None
        return self._release(plan, waiting)

    def _release(self, plan: JointTrajectory, replies: JointTrajectory) -> Aligned:
        out = align(plan, replies, rule=self.rule, tolerance_s=self.tolerance_s)
        self.dropped += out.alignment.dropped
        return out
