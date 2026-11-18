# robot_metrics — working notes for agents

Performance metrics for robot parts: torchmetrics' shape with numpy inside, over robot_msgs streams. The decision
record is `docs/peps/0001-robot-metrics-v0.md`; read it before changing the core.

## Rules that are Zach's (keep the `WHY:` comments that carry them)

- A base metric is math only; the meaning comes from the streams an alias hands it. Base + frozen config, a subclass
  only when config cannot say it. Never a metric function per part type: configure a stock metric and alias it.
- Identity = class + frozen config. An alias is a label; two aliases of one identity compute once.
- The target stays outside `compute()`: thresholds live on `Objective` only, never in a metric's config.
- SI inside; degrees and mm only at display (the alias's `unit` and `scale`).
- `RobotPart.metrics` sits beside `.models`: `catalog.py` is that list's seed until robotparts / robot_models wire it.
- `plot()` goes to matplotlib; W&B and Rerun are adapters over the same `View`.

## Layering

- Runtime deps: numpy + robot_msgs (`robot-msgs` from PyPI, no local source override) + the stdlib. matplotlib, wandb and rerun-sdk
  are extras imported only inside `views.plot` and `adapters/`. `tests/test_import_isolation.py` enforces it.
- Never import io-diagnostics (RobotDx), sim2real, robot_models or robotparts: they will import this package.
- Missing something robot_msgs should have? Write it here and list it as a proposed move in a PEP; never edit
  robot_msgs from this repo.

## Tests

```bash
cd ~/robot_metrics && taskset -c 2-5,16,17 uv run --extra test pytest -q
```

- The acceptance tests were committed before the implementation (`7f8763c`). Don't edit a green test except to add.
- Every Python block a reader sees (README, landing page, report) runs as shown: `tests/test_shown_examples.py` re-runs
  the code extracted from the built pages, alone, in a fresh `python -I`. After editing `website/gallery.py` or
  `docs/usage_session.py`, rebuild the site / report. Never put a name in a shown script's namespace.
- `tests/oracle.py` is the frozen check's oracle and tolerances (robot_models f0daa34, `check_scenario.py`), ported
  unchanged. Only tests read the oracle files (round 8's scorecard, the ladder's summary); the package never does.
- Real data on /archive is read-only; archive tests skip cleanly without it. `tests/data/` holds the sealed ladder
  frozen by `tools/freeze_sealed_ladder.py` (run with robot_models' venv, out of band).

## Machine

- At most 2 worker processes, pinned `taskset -c 2-5,16,17`; never CPUs 22-23 (the fdcanusb IRQ).
- Never touch /dev/ttyACM*, CAN, cameras or moteus. No whole-disk find/du.
- Public: github.com/zacharyyamaoka/robot_metrics (MIT). Push verified main; tag releases `vX.Y.Z` with a
  CHANGELOG entry. Never commit recordings, images or data to main: they live on /archive and the site's gh-pages.
- Landing page: `website/build_site.py` (cells in `website/gallery.py`, run for real) then
  `website/publish_pages.py` (gh-pages). Live: https://zacharyyamaoka.github.io/robot_metrics/
- Reports: `/home/bam/robot_metrics/reports/<name>-<date>.html`, built by `docs/build_report.py`.
