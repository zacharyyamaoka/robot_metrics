"""robot_metrics: performance metrics for robot parts. torchmetrics' shape with numpy inside, over robot_msgs streams.

    from robot_metrics import catalog, summarize
    from robot_metrics.datasets import onedof_round8
    arm = catalog.pendulum_arm(tip_length_m=0.339).collection()      # RobotPart.metrics' seed: aliases over stock metrics
    run = onedof_round8.real_runs()[32]                               # one run: named robot_msgs streams + labels
    arm.values(arm.fold(run))                                         # offline; live: arm.update(state, chunk) ... close

The pieces (PEP 0001 has the why of each):
- ``Metric``: a frozen config whose state is a value (empty / update / merge / compute / close); ``Stream`` is
  torchmetrics' update / compute / reset / plot face over it; ``functional`` the plain function beside each class.
- stock metrics, math only: ``ErrorStat``, ``ErrorPercentile``, ``StopStat``, ``TimeOffset``, ``CodeShare``, ``Repeatability``,
  ``PerformanceGain``; the stop kernel in ``stops``.
- ``align`` / ``Aligner``: time alignment as an explicit step with a tolerance.
- ``Alias``, ``PartMetrics``, ``Collection``: a part's metrics, each identity computed once, an assembly reading its
  children's states through context.
- ``summarize`` / ``Objective`` / ``Gate``: strata, tails and Wilson shares; OpenSLO-style gates outside compute().

Layering: robot_msgs <- robot_metrics <- (RobotDx, sim2real, robotparts / robot_models' ``.metrics``). The core imports
numpy, robot_msgs and the stdlib only; matplotlib, wandb and rerun-sdk only inside their adapters.
"""

from . import catalog, functional, stops
from .alignment import Aligned, Aligner, Alignment, align
from .collection import Alias, Collection, CollectionStream, PartMetrics
from .metric import DEGREES, AlignedPair, CommandAndState, Metric, Stream
from .objective import Gate, GateVerdict, Objective, Verdict
from .records import CalibrationOutcome, Interval, MetricRecord
from .run import Run
from .stock import (CodeShare, ErrorPercentile, ErrorStat, HeldOutPair, PerformanceGain, Repeat, Repeatability,
                    StopStat, TimeOffset)
from .summary import Share, Stratum, Tail, highest_passing, summarize, wilson
from .views import View

__version__ = "0.0.1"

__all__ = [
    "DEGREES", "Alias", "Aligned", "AlignedPair", "Aligner", "Alignment", "CalibrationOutcome", "CodeShare",
    "Collection", "CollectionStream", "CommandAndState", "ErrorPercentile", "ErrorStat", "Gate", "GateVerdict", "HeldOutPair", "Interval",
    "Metric", "MetricRecord", "Objective", "PartMetrics", "PerformanceGain", "Repeat", "Repeatability", "Run", "Share",
    "StopStat", "Stratum", "Stream", "Tail", "TimeOffset", "Verdict", "View", "align", "catalog", "functional",
    "highest_passing", "stops", "summarize", "wilson",
]
