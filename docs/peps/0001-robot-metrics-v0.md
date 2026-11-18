# PEP 0001: robot_metrics v0 — the recommended splice of the five metrics-API proposals

- **Status:** accepted as the v0 contract, 2026-11-17. The AI's recommended combination, taken as the default under
  Zach's standing rule for reversible work ("your recommendation is the default, take it"). Undo: the repo is local
  and private; `git revert` any commit, or delete `/home/bam/robot_metrics`.
- **Ask:** build `robot_metrics` v0 from the combination recommended at the end of the five-proposal babble round
  (robot_models main `f0daa34`, `docs/proposals/metrics-api-2026-11-17/README.md`, "AI pick: v4, 84 ... Splice
  recommended").
- **Follows:** Zach's design (memory `robot-metrics-design`, Daily Note Nov 17 "# Ok torch metrics" and "# Ok homing
  in on this..."); the 3000-ft view (memory `deployment-view-3000ft`: performance first, runs are the atomic unit);
  the prior-art note `Metric Library APIs — Prior Art (2026-11-17)` §1 and §5; the run-data MOC (Nov 17); the house
  layering (memory `project_robot_msgs`).

## 1. The answer in eight lines

1. **A metric is a frozen dataclass; its identity is its class + fields** (v4). Two aliases of one identity are one
   state, and metrics whose `state_key` matches share one update (every `StopStat` on a signal reads one stop pass).
2. **Its state is an immutable value**: `empty`, `update(state, chunk) -> state`, `merge`, `close` (the run boundary),
   `compute`. Live is a fold over chunks, offline the same fold once, and runs merge into rungs and stations (v4).
3. **The face is torchmetrics'** (v1): `metric.stream()` / `Stream` gives `update` / `compute` / `reset` / `plot`;
   `functional` holds a plain function beside each class; `Collection` shares state across aliases.
4. **Each metric declares `Input`, `live` and `version`**; every stored `MetricRecord` carries identity and version.
5. **Alignment is an explicit step with a tolerance** (v3): `align(plan, replies, rule="hermite", tolerance_s=one
   tick)` offline and `Aligner` live give bit-identical targets; a reply past the tolerance is dropped and counted.
6. **An assembly reads its children's states through context** (v5): the pendulum arm's tip rows are the servo's
   states times the lever (`Alias.through(..., scale_by="lever_m")`); the arm adds no computation to its servo.
7. **Gates are OpenSLO occurrence objectives outside compute()** (v4): `Objective(alias, op, value, share, on,
   unit, owner)`; `summarize(..., by=("station", "rung"))` gives a tail per stratum (n, median, p95, worst), never an
   IQM, and a Wilson interval for bool metrics. The verdict is robot_metrics' own small `Verdict` record.
8. **Inputs other than a run recording** (amendment 4): `Repeatability(k)` over `Repeat`s, `PerformanceGain(metric=
   <alias>)` over a `HeldOutPair`; `CalibrationOutcome` is a robot_metrics record.

## 2. The package and its layering

```
robot_msgs  <-  robot_metrics  <-  RobotDx (io-diagnostics), sim2real, robotparts / robot_models (.metrics)
```

- Runtime: numpy + robot_msgs + the stdlib. matplotlib (`views.plot`), wandb (`adapters/wandb.py`) and rerun-sdk
  (`adapters/rerun.py`) are extras imported inside the function that needs them.
- It imports none of io-diagnostics, sim2real, robot_models or robotparts: each of them will import it. Test-time
  data from robot_models (the sealed ladder) was frozen out of band (`tools/freeze_sealed_ladder.py`) rather than
  imported, so there is no test-time cycle either.
- `tests/test_import_isolation.py` enforces both (an AST check held red by planted violations, an import check, and
  the brief's `env -u VIRTUAL_ENV uv run --isolated` run, whose env holds exactly numpy, dataclass-values, robot-msgs
  and robot-metrics).

## 3. The combination, element by element

| element | from | where | why |
|---|---|---|---|
| frozen config, identity = class + fields | v4 | `metric.Metric`, `Metric.identity` | Zach: base + config; equal aliases compute once |
| state as a value; live = fold, offline = same fold, runs merge | v4 (clu, torcheval) | `Metric.empty/update/merge/close/compute` | merging and resuming need no extra code |
| `live`, `version`, `Input` per metric | prior-art §1, amendment 4 | class attributes | an at-end metric (lag) says so; stored values stay readable |
| `update / compute / reset / plot` face | v1 (torchmetrics) | `metric.Stream`, `Metric.plot` | Zach's pick of torchmetrics' shape |
| a function beside each class | v1 | `functional` | each function IS its class run once: no drift |
| collection sharing state across aliases | v1 + v4 | `collection.Collection` (`state_key` groups) | compute groups |
| alignment step with a tolerance | v3 | `alignment.align`, `alignment.Aligner` | the field aligns outside the metric (evo `max_diff`) |
| assembly reads children through context | v5 | `PartMetrics(children, context)`, `Alias.through(scale_by=)` | the arm shows, the servo computes |
| occurrence objectives, tails per stratum, Wilson | v4 (OpenSLO, rliable's lesson) | `objective`, `summary` | "100 % of stops <= 1 mm" says what a median hides |
| other inputs | amendment 4 | `stock.Repeatability`, `stock.PerformanceGain` | repeats and held-out pairs are not one recording |
| one View, three sinks | all five (`_viz.py`) | `views`, `adapters` | one result feeds every viewer |

The stop kernel (`stops.py`) is the proposals' shared `stock.py` rule, ported line for line with its state lifted
out of the tracker object: the open dwell travels in the state, so a stop split across chunks is found exactly.

## 4. Run data

`Run(id, streams, labels, provenance, outcome, alignment, links)`: named robot_msgs streams (`plan`, `state`,
`target`, `tool/target`, `tool/state`), the labels a roll-up groups by, provenance (real / sim / replay; carried though
no metric reads it), the outcome, how the target was aligned, and links out (the source file; the `.rrd` once the
recorder registers one with the 8771 launcher). This is the run-data MOC's recommended wrapper. The two readers in
`datasets/` (round 8, the five-bar ladder) are ports of the proposals' fixture and read nothing the oracle reads.

## 5. Gates, and how RobotDx converts a verdict

```python
gate = Objective("tip stop error", op="<=", value=1.0, unit="mm", share=1.0, on="items", owner="Zach 2026-11-11")
v = gate.judge(stratum)          # Verdict(objective, verdict, observed, passed, total, owner, ..., hint)
# in RobotDx (it imports robot_metrics; never the other way):
rdx.result({"objective": v.objective, "owner": v.owner}, value=v.observed, unit=v.unit, metric=v.measure,
           limits=rdx.Limits.at_least(v.share) if v.on == "items" else rdx.Limits.at_most(gate.value),
           verdict=v.verdict, title=v.objective, message=v.message, hint=v.hint)
```

The verdict strings are RobotDx's (`pass` / `fail` / `not_run`; no value is never a pass), so the conversion is a
field copy. A `Gate` passes when all its objectives pass.

## 6. Proof (`reports/robot-metrics-v0-2026-11-17.html` has the outputs)

- **Parity with round 8**, check_scenario.py's oracle and tolerances ported unchanged: 252 real plays, 36 rungs, the
  twin's 252 answers, the highest rung per station, the five-bar ladder. Worst deviation 0 on tracking (bit-equal),
  2.8e-17 mm on the rung stop median (5.6e-17 sim side), 0 relative on the chain. Two wrong rulers go red under the
  same oracle (the plan linearly interpolated: 9.4e-4 deg; each reply against its own tick: 0.105 deg).
- **Live == offline** within 1e-12 relative on every real play, with a chunk boundary in the middle of each of the
  504 stops' dwells, and one sample at a time; fed one message at a time on 12 CAN 11 plays, the live aligner equals
  the offline alignment bit for bit.
- **Gate counts** equal the contract's: tracking 31, median tip stop 36, every stop 33 of 36.
- **Repeatability** (k=3) on CAN 11's five 0.9 repeats equals the ISO 9283 oracle (0.387 mm at the tip);
  **PerformanceGain** equals the held-out oracle (0.2220 -> 0.0042 N·m).
- The tests were committed red before the implementation (`7f8763c`).

## 7. Proposed moves into robot_msgs (not made: this repo never edits robot_msgs)

1. `stops.py` -> `robot_msgs.compare.stops` beside `goal` (the stop rule is generic signal math).
2. The hermite rule -> an option of `robot_msgs.compare.align` (`compare.align` interpolates linearly, which is the
   9.4e-4 deg wrong ruler on the rack's plan).
3. `CalibrationOutcome` / `Interval` -> a robot_msgs message beside `std.Metrics`, once a second caller uses it.

## 8. What is next

1. **RobotDx migrates its stock onto these classes**: `robotdx.diagnostics.stock` keeps `Limits`, `judge`, `Result`
   and records, takes its formulas from robot_metrics, and moves the `limits` out of every config.
2. **sim2real.metrics.actuator** (`tracking_statistic`, `stops`, `stop_summary`) is replaced by robot_metrics; its
   fidelity helpers (`floor`, `fidelity_ratios`) stay until a fidelity family is designed (out of scope here).
3. **robotparts and robot_models gain `.metrics`**: `IntegratedServo.metrics = catalog.integrated_servo()` beside
   `.models`; this catalog module then moves out.
4. The recorder's run reader replaces `datasets/`; each run's `.rrd` registered with `data_recorder.runlink` lands in
   `Run.links`.
5. Windows in seconds (river's `TimeRolling`) and `live` levels (exact / windowed / at end) when a live monitor needs
   them; a one-sided tolerance bound per stratum once a stratum holds 59 stops (the prior-art note's open question 4).

## 9. Alternatives rejected (each built and checked in the babble round; all five passed parity)

- **v1 as built** (torchmetrics ported literally, `add_state`): state hides in attributes; merging needs every reduce
  rule right and `compute()` mid-run deep-copies the open tracker. Kept: its face.
- **v2** (metrics in `robot_models` on the ModelFamily, cat buffers): puts metrics inside the models package (Zach:
  "as we are in robot models all there really is is models") and its state grows with the run (18 ms compute).
- **v3 as built** (math in `robot_msgs.metrics`, IQM + bootstrap across runs): grows a framework inside robot_msgs,
  and the IQM trims CAN 11's failing stops. Kept: its aligner.
- **v4 as built**: loaded RobotDx's `Result` from io-diagnostics at import (a reverse arrow) and had no torchmetrics
  face. Kept: its core, objectives and summaries; the verdict is now robot_metrics' own.
- **v5 as built** (formulas in robotparts' part tree): RobotDx and sim2real would reach up above robot_models for
  their math. Kept: its context rule.
- **Adopting torchmetrics itself**: 3.9 s and 450 MB to import, 2.7 GB of venv, and it rejects numpy arrays.

## 10. Decided here (reversible, each with its undo)

- **Module `alignment`, function `align`** (a module named `align` was shadowed by the exported function). Undo:
  rename back and export the function under another name.
- **`state_key` compute groups** go one step past identity (rms / max_abs / mean_abs of one field share a state).
  Undo: return `self` from `ErrorStat.state_key`.
- **A statistic that lands on a never-settled stop returns None**, never a clipped number (the scorecard's convention
  for the median, extended to max / p95). Undo: `StopStat.compute`.
- **Datasets readers live in the package** for now, so a usage session is five lines. Undo: move `datasets/` to
  `tests/`.
- **The plot legend sits outside the axes**: inside, it hid CAN 11's worst rung-1.0 stop (1.317 mm). Undo:
  `views.plot`.

## 11. Not verified

- `adapters.wandb.log` (the send) is not exercised: wandb is not installed and nothing is uploaded; the payload is.
- The Rerun `.rrd` is written and non-empty; it was not opened in a viewer.
- Only the pendulum and the five-bar are run; the other part types are not declared yet.
