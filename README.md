# robot_metrics

[![test](https://github.com/zacharyyamaoka/robot_metrics/actions/workflows/test.yml/badge.svg)](https://github.com/zacharyyamaoka/robot_metrics/actions/workflows/test.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Performance metrics for robot parts: torchmetrics' shape with numpy inside, over robot_msgs streams. A metric is a
frozen config whose state is a value, so one object scores a live stream chunk by chunk or a whole recording, and runs
merge into rungs, stations and a deployment. Targets stay outside `compute()`, on OpenSLO-style objectives.

**Landing page: <https://zacharyyamaoka.github.io/robot_metrics/>**. It has the API at a glance, a catalog of every
stock metric with its stock `plot()`, and a gallery on three arms: a 1-DOF pendulum (real), a five-bar arm (real and
its sim twin) and a UR5e (simulated).

v0.1.0. It reproduces a real 1-DOF test rack's scorecard at float round-off: 252 real plays, the twin's answers and a
five-bar ladder. Why it is shaped this way: [docs/peps/0001-robot-metrics-v0.md](docs/peps/0001-robot-metrics-v0.md).

## Install

```bash
pip install "robot-metrics[viz] @ git+https://github.com/zacharyyamaoka/robot_metrics@v0.1.0"
```

Runtime dependencies: numpy and [robot_msgs](https://github.com/zacharyyamaoka/robot_msgs) (`robot-msgs` on PyPI).
matplotlib, wandb and rerun-sdk are extras (`viz`, `wandb`, `rerun`).

## What goes in, what comes out

```python
import numpy as np
from robot_msgs.trajectory import JointTrajectory
from robot_metrics import DEGREES, Alias, Collection, ErrorStat, Objective, Run, StopStat, summarize

t = np.arange(0.0, 3.0, 0.01)                                # a 2 s move, then a 1 s dwell
q_target = np.clip(t / 2.0, 0.0, 1.0)[:, None]
qd_target = np.where(t < 2.0, 0.5, 0.0)[:, None]
q_measured = q_target - 0.002 * np.exp(-t)[:, None]         # your recording, or a live feed

target = JointTrajectory(t=t, joint_names=("j1",), positions=q_target, velocities=qd_target)
measured = JointTrajectory(t=t, joint_names=("j1",), positions=q_measured)
run = Run("run-1", streams={"target": target, "state": measured}, labels={"rung": 0.5})

metrics = Collection([
    Alias("tracking RMS", ErrorStat(statistic="rms"), unit="deg", scale=DEGREES),
    Alias("stop error", StopStat(of="error", statistic="max"), unit="deg", scale=DEGREES),
])
metrics.values(metrics.fold(run))          # offline: the whole recording, one fold
# {'tracking RMS': 0.04695760451301328, 'stop error': 0.005879514223425522}

state = metrics.empty()                    # live: the same fold, chunk by chunk
for chunk in run.chunks(25):
    state = metrics.update(state, chunk)
metrics.values(metrics.close(state))
# {'tracking RMS': 0.04695760451301328, 'stop error': 0.005879514223425522}

strata = summarize(metrics, [run], by=("rung",))      # many runs: values, tails, Wilson shares
Objective("stop error", "<=", 0.1, unit="deg", on="value").judge(strata[(0.5,)]).verdict
# 'pass'
```

Values come back in each alias's display unit (deg, mm, ms, %); the math runs in SI. The landing page runs the same
API on real runs and shows every output.

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
| `datasets` | readers for the recordings the v0 was proven on (not public; `ROBOT_METRICS_ROUND8`, `ROBOT_METRICS_FIVEBAR_LADDER`) |

## Run the tests

```bash
uv run --extra test pytest -q
```

53 of the 69 tests need nothing else; CI runs them on Python 3.11 to 3.13. The parity tests read the private
recordings and skip cleanly without them.

## The landing page

`website/build_site.py` builds it from real runs: every code block on it is a cell it ran, and every image a `plot()`
it drew. It needs the recordings and MuJoCo Menagerie's UR5e, so it is built locally and published as a build
artifact on the `gh-pages` branch (`website/publish_pages.py`); main's history holds no images or data.

```bash
uv run --extra site python website/build_site.py && python3 website/publish_pages.py
```

## License

MIT, see [LICENSE](LICENSE). [CHANGELOG](CHANGELOG.md).
