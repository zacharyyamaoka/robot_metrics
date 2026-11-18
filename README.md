# robot_metrics

Performance metrics for robot parts: torchmetrics' shape with numpy inside, over robot_msgs streams. A metric is a
frozen config whose state is a value, so one object scores a live stream chunk by chunk or a whole recording, and runs
merge into rungs, stations and a deployment. Targets stay outside `compute()`, on OpenSLO-style objectives.

**v0, local only.** It reproduces round 8's scorecard at float round-off: 252 real plays, the twin's answers, the
five-bar ladder. Why it is shaped this way: [docs/peps/0001-robot-metrics-v0.md](docs/peps/0001-robot-metrics-v0.md).

## Five lines on a real run

```python
from robot_metrics import catalog
from robot_metrics.datasets import onedof_round8
run = onedof_round8.real_runs()[32]                              # CAN 11, rung 0.9, one repeat
arm = catalog.pendulum_arm(tip_length_m=0.339).collection()      # the arm's metrics: 15 aliases over 5 states
arm.values(arm.fold(run))   # {'servo/tracking RMS': 0.1459, ..., 'tip stop error': 0.2309, 'servo/lag': -0.7412, ...}
```

Values come back in each alias's display unit (deg, mm, ms, %); the math runs in SI. Live is the same fold:
`state = arm.update(state, chunk)` per chunk, then `arm.values(arm.close(state))`. Across runs:
`summarize(arm, runs, by=("station", "rung"))`, then `catalog.pendulum_gates()[name].judge(stratum)`.

## What is in it

| module | what |
|---|---|
| `metric` | `Metric` (empty / update / merge / compute / close), `Stream` (torchmetrics' update / compute / reset / plot) |
| `stock`, `functional` | `ErrorStat`, `ErrorPercentile`, `StopStat`, `TimeOffset`, `CodeShare`, `Repeatability`, `PerformanceGain`, and a function beside each |
| `stops` | the stop kernel: dwells found one sample at a time, safe across chunk boundaries |
| `alignment` | `align` / `Aligner`: the plan at each reply's instant, with a tolerance, offline or live |
| `collection`, `catalog` | `Alias`, `PartMetrics` (the seed of `RobotPart.metrics`), `Collection` (each identity once) |
| `summary`, `objective` | strata, tails, Wilson shares; `Objective` / `Gate` / `Verdict` |
| `views`, `adapters` | `View` to `plot()` (matplotlib), a W&B payload, a Rerun `.rrd` |
| `datasets` | readers for round 8 and the five-bar ladder on /archive (read-only) |

Runtime dependencies: numpy and robot_msgs. matplotlib, wandb and rerun-sdk are extras (`viz`, `wandb`, `rerun`).

## Run the tests

```bash
cd ~/robot_metrics && taskset -c 2-5,16,17 uv run --extra test pytest -q
```

The tests that read /archive skip cleanly without it. Report: `reports/robot-metrics-v0-2026-11-17.html`.
