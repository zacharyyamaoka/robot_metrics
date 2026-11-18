"""The part catalog: each part type's metrics as aliases over stock metrics. The seed of ``RobotPart.metrics``.

    catalog.integrated_servo()            # the servo's performance rows (target vs state), PartMetrics
    catalog.pendulum_arm(tip_length_m)    # an assembly: the servo plus a lever; its tip rows read the servo's states
    catalog.rigid_chain(("j2", "c1"))     # a rigid chain (the bam_fb five-bar): joint RMS per joint, the tool's error
    catalog.pendulum_gates()              # round 8's three verdicts as OpenSLO-style objectives, each with its owner

WHY here for now (the brief, 2026-11-17): ``RobotPart.metrics`` sits beside ``RobotPart.models`` (Zach: "RobotPart.models
-> ModelFamily, RobotPart.metrics -> metrics"). This module is that list's first home; wiring it into robotparts and
robot_models (``IntegratedServo.metrics``) is the next step (PEP 0001 §Next), and then this module moves out.

WHY one stock metric per row, never a function per part (Zach, robot-metrics-design): the meaning comes from the
streams an alias reads and the unit it shows, so the servo's stop error and the arm's tip stop error are one StopStat.
"""

from __future__ import annotations

from collections.abc import Sequence

from .collection import Alias, PartMetrics
from .metric import DEGREES
from .objective import Gate, Objective
from .stock import CodeShare, ErrorStat, StopStat, TimeOffset

LIMITER_CODES = (97, 98, 99, 102)    # moteus: the rows the limiter or a fault shaped (the scorecard's limiter share)
TIP_BAND_M = 1e-3                    # WHY 1 mm: Zach 2026-11-11, "we certainly need final tool error sub 1mm"
COMMAND = ("plan", "state")          # the plan on its own ticks and the replies: the lag reads these, unaligned


def integrated_servo() -> PartMetrics:
    """An integrated servo judged at its output: target (the plan at each reply's instant) vs state."""

    return PartMetrics("servo", "integrated_servo", (
        Alias("tracking RMS", ErrorStat(statistic="rms", rows="moving"), unit="deg", scale=DEGREES, across="median",
              note="RMS of target - measured over the rows the plan moves; a rung reads the median of its runs"),
        Alias("tracking peak", ErrorStat(statistic="max_abs"), unit="deg", scale=DEGREES, across="max",
              note="the worst |error| of the run, every row"),
        Alias("stop error", StopStat(of="error", statistic="median"), unit="deg", scale=DEGREES,
              note="the median |error| of every stop, pooled over the runs"),
        Alias("stop error p95", StopStat(of="error", statistic="p95"), unit="deg", scale=DEGREES),
        Alias("stop error worst", StopStat(of="error", statistic="max"), unit="deg", scale=DEGREES),
        Alias("overshoot", StopStat(of="overshoot", statistic="max"), unit="deg", scale=DEGREES,
              note="the largest robot_msgs.compare.goal overshoot over the stops (0 with none)"),
        Alias("lag", TimeOffset(), signal=COMMAND, unit="ms", scale=1e3, across="median",
              note="estimate_time_offset(plan, state); negative on the rack: a reply is stamped after its tick"),
        Alias("limiter share", CodeShare(codes=LIMITER_CODES), unit="%", scale=100.0,
              note="rows the limiter or a fault shaped, pooled over the runs' rows"),
        Alias("stops", StopStat(of="count", statistic="count"), note="how many stops the runs made"),
    ))


def pendulum_arm(tip_length_m: float) -> PartMetrics:
    """The 1-DOF pendulum arm: an assembly of the servo and a lever of ``tip_length_m``. Every tip row is the servo's
    state read through the lever (v5's rule): the arm declares no new pass over the data."""

    servo = integrated_servo()
    band = TIP_BAND_M / tip_length_m      # 1 mm at the tip, in the joint's radians (the base math's unit)
    tip = "tip = joint x lever"
    return PartMetrics("arm", "pendulum_arm", (
        Alias.through(servo, "tracking RMS", name="tip tracking RMS", unit="mm", scale=1e3, scale_by="lever_m", note=tip),
        Alias.through(servo, "stop error", name="tip stop error", unit="mm", scale=1e3, scale_by="lever_m", note=tip),
        Alias.through(servo, "stop error p95", name="tip stop p95", unit="mm", scale=1e3, scale_by="lever_m", note=tip),
        Alias.through(servo, "stop error worst", name="tip stop worst", unit="mm", scale=1e3, scale_by="lever_m",
                      note=tip),
        Alias("tip settle", StopStat(of="settle", statistic="median", band=band), unit="s", of_child="servo",
              note="the median stop's settle time into 1 mm at the tip (None when that median never settled)"),
        Alias("all stops settled", StopStat(of="settled", statistic="all", band=band), of_child="servo",
              note="every stop of the run settled into 1 mm at the tip (a bool: summarized as a Wilson share)"),
    ), children=(servo,), context={"lever_m": float(tip_length_m)})


def rigid_chain(joints: Sequence[str] = ("j2", "c1")) -> PartMetrics:
    """A rigid chain through robot_msgs.compare (traj_compare's core): each joint's RMS, the tool's position error.
    The tool streams are PoseTrajectories (FK of the command and of the encoders), compared by compare()."""

    tool = ("tool/target", "tool/state")
    children = tuple(PartMetrics(joint, "integrated_servo", (
        Alias("tracking RMS", ErrorStat(statistic="rms", joint=index), unit="deg", scale=DEGREES),)) for index, joint
        in enumerate(joints))
    return PartMetrics("chain", "rigid_body_chain", (
        Alias("tool tracking RMS", ErrorStat(statistic="rms"), signal=tool, unit="mm", scale=1e3),
        Alias("tool worst", ErrorStat(statistic="max_abs"), signal=tool, unit="mm", scale=1e3, across="max"),
    ), children=children)


AGENT = "agent (pendulum_goal)"
ZACH = "Zach 2026-11-11"


def pendulum_gates() -> dict[str, Gate]:
    """Round 8's three verdicts per (station, rung) cell. The thresholds are the scorecard's: tracking is the agent's,
    the tip's 1 mm is Zach's; "every stop" is proposed (Zach's 1 mm read on the worst stop, not the median)."""

    return {
        "tracks": Gate("tracks", (
            Objective("servo/tracking RMS", "<=", 0.5, unit="deg", on="value", owner=AGENT),
            Objective("servo/tracking peak", "<=", 1.5, unit="deg", on="value", owner=AGENT),
            Objective("servo/limiter share", "<=", 2.0, unit="%", on="value", owner=AGENT),
            Objective("faulted runs", "<=", 0, on="value", owner=AGENT),
        )),
        "tip stop (median)": Gate("tip stop (median)", (
            Objective("tip stop error", "<=", 1.0, unit="mm", on="value", owner=ZACH + " (the scorecard's median)"),
            Objective("tip settle", "<=", 0.3, unit="s", on="value", owner=AGENT),
        )),
        "tip stop (every stop)": Gate("tip stop (every stop)", (
            Objective("tip stop error", "<=", 1.0, unit="mm", share=1.0, on="items",
                      owner="proposed: " + ZACH + " on every stop"),
        )),
    }
