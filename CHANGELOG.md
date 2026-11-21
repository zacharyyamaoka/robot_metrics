# Changelog

## Unreleased

- **W&B gets summary statistics only** (Zach, 2026-11-21: "to help limit W and B storage I think we need to just be
  using it for summary statistics; for the media-rich things we should be using rerun!"). `adapters.wandb.log` no
  longer sends a `wandb.Table`: it sends the summary scalars (`n`, `max`, `target`) and returns exactly what it sent;
  the new `rows_path=` writes the rows as jsonl on this computer and sends that path as a string. `payload()` is
  unchanged. The coverage gallery's W&B column (`docs/proposals/coverage-2026-11-19/tools_wandb.py`) logs each
  view's counts, shares, ranges and percentiles, and points at the Rerun column's .rrd for the picture.

Otherwise docs and checks only:

- **Every Python block shown to a reader runs as shown.** The landing page's glance, each catalog card, each gallery
  arm, the v0 report's glance and usage session, and the README: each imports what it uses, says where its data comes
  from, and runs alone in a fresh `python -I` printing the outputs the page shows (`tests/test_shown_examples.py`,
  which extracts the code from the built page). The catalog cards no longer lean on an unshown setup, and no example
  uses a name the build put in its namespace.
- **Plain words.** The landing page, the README and the report's comments say what `fold` / `update` / `close` do
  ("run every sample through the metrics once", "a running tally"), group runs instead of "strata", and show
  readable outputs instead of `Collection(15 aliases over 5 states)`.
- **The report is public-safe.** Its builder scrubs this machine's paths and refuses to write one that leaks.

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
