# Changelog

## 0.1.0 — 2026-11-18

The first public release: the v0 core, made to stand on its own.

- **Public package.** MIT licence; `robot-msgs` comes from PyPI (`>=0.3.0`) instead of a local checkout, so a fresh
  clone installs and tests as is. Version 0.1.0.
- **CI.** GitHub Actions runs the tests that need no recordings on Python 3.11, 3.12 and 3.13 (53 of 69; the round-8
  and five-bar parity tests skip without the recordings).
- **Landing page.** `website/build_site.py` builds the GitHub Pages site from real runs: the API at a glance, a catalog
  of every stock metric with its `plot()`, the aliases per part type, and a gallery (the 1-DOF pendulum on real
  round-8 runs, the five-bar arm's real and sim speed ladder, a simulated UR5e).
- **Dataset readers** take their folder from `ROBOT_METRICS_ROUND8` / `ROBOT_METRICS_FIVEBAR_LADDER`.
- **README** example is self-contained and tested (`tests/test_readme_example.py`): it runs as written and shows its
  real output.

## 0.0.1 — 2026-11-17

The v0 contract (`docs/peps/0001-robot-metrics-v0.md`): a metric is a frozen config whose state is a value
(empty / update / merge / close / compute), so live is a fold over chunks and offline the same fold once.

- Stock metrics: `ErrorStat`, `ErrorPercentile`, `StopStat`, `TimeOffset`, `CodeShare`, `Repeatability`,
  `PerformanceGain`, and a plain function beside each in `functional`.
- `Alias`, `PartMetrics`, `Collection`: a part's metrics, each identity computed once; `catalog` seeds
  `RobotPart.metrics` (the integrated servo, the 1-DOF pendulum arm, a rigid chain).
- `summarize` (strata, tails, Wilson shares) and OpenSLO-style `Objective` / `Gate` outside `compute()`.
- One `View`, three sinks: matplotlib `plot()`, a W&B payload, a Rerun `.rrd`.
- Reproduces round 8's scorecard at float round-off: 252 real plays, the twin's answers, the five-bar ladder.
