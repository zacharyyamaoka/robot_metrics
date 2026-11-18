"""The v0 usage session on CAN 11's round-8 runs: the cells the report shows, run for real.

    cd ~/robot_metrics && taskset -c 2-5,16,17 uv run --extra test --extra viz python docs/usage_session.py

Prints the transcript; ``run(media)`` returns the Session (the report builder calls it) and writes the views to
``media``: the plot() images and a Rerun .rrd (when rerun-sdk is installed).
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from session import Session  # noqa: E402

CELLS = [
    ("a part's metrics", "The pendulum arm is an assembly: the servo's aliases plus tip rows that read the servo's states "
     "through the lever. 15 aliases, 5 states.", '''
from robot_metrics import DEGREES, Alias, Collection, ErrorStat, Objective, Repeat, Repeatability, View, catalog, summarize
from robot_metrics.datasets import onedof_round8
tips = {station: facts["tip_length_m"] for station, facts in onedof_round8.stations().items()}
arm = catalog.pendulum_arm(tip_length_m=tips["can11"])
print(arm.table())
'''),
    ("one CAN 11 run, offline", "A run is named robot_msgs streams plus labels; the target was put on the replies' "
     "instants by the explicit alignment step (hermite, one tick) when the run was read.", '''
run = onedof_round8.real_runs()[32]
print(run, "\\n", run.alignment)
collection = arm.collection()
offline = collection.values(collection.fold(run))
for name, value in offline.items():
    print(f"  {name:<20} {value:9.4f} {collection.aliases[name].unit}" if isinstance(value, float) else f"  {name:<20} {value}")
'''),
    ("the same run, live", "Fed 7 samples at a time (a stop's dwell spans several chunks), the fold gives the offline "
     "numbers. One live update costs well under a 10 ms tick.", '''
import time
chunks = run.chunks(7)
state = collection.empty()
for chunk in chunks:
    state = collection.update(state, chunk)
live = collection.values(collection.close(state))
print("largest |live - offline| over", len(offline), "aliases:", max(abs(live[n] - offline[n]) for n in offline if isinstance(offline[n], float)))
mid = collection.update(collection.empty(), chunks[9])
start = time.perf_counter()
for _ in range(300):
    collection.update(mid, chunks[10])
print(f"one live update: {len(collection.aliases)} aliases over {collection.computations} states, {(time.perf_counter() - start) / 300 * 1e3:.2f} ms per 7-sample chunk")
stream = ErrorStat(statistic="rms", rows="moving").stream()       # torchmetrics' face: update / compute / reset
for chunk in chunks:
    stream.update(chunk.streams["target"], chunk.streams["state"])
print(f"Stream: {stream.compute() * DEGREES:.6f} deg (offline {offline['servo/tracking RMS']:.6f})")
'''),
    ("CAN 11's rungs, gates applied", "Strata are station x rung. Each gate is OpenSLO-style objectives judged outside "
     "compute(); the every-stop gate reads the tail the median hides.", '''
can11 = [r for r in onedof_round8.real_runs() if r.labels["station"] == "can11"]
strata = summarize(collection, can11, by=("station", "rung"))
gates = catalog.pendulum_gates()
short = {"tracks": "tracks", "tip stop (median)": "median stop", "tip stop (every stop)": "every stop"}
for (station, rung), cell in sorted(strata.items()):
    v, tail = cell.values, cell.tails["tip stop error"]
    marks = ", ".join(f"{short[name]} {gate.judge(cell).verdict}" for name, gate in gates.items())
    print(f"{station}@{rung:<3g} RMS {v['servo/tracking RMS']:.3f}° | tip stop median {v['tip stop error']:.3f} p95 {tail.p95:.3f} worst {tail.worst:.3f} mm, n={tail.n} | {marks}")
'''),
    ("one target, two shares, a Wilson interval", "The same stop state judged against two occurrence targets, nothing "
     "recomputed; a bool metric summarizes as a share of runs with its Wilson interval.", '''
station = summarize(collection, can11, by=("station",))[("can11",)]
for share in (1.0, 0.9):
    verdict = Objective("tip stop error", op="<=", value=1.0, unit="mm", share=share, on="items", owner="proposed").judge(station)
    print(f"{verdict.objective}: {verdict.verdict} · {verdict.message}")
settled = station.shares["all stops settled"]
print(f"runs with every stop settled into 1 mm: {settled.successes} of {settled.trials}, Wilson 95 % [{settled.low:.2f}, {settled.high:.2f}]")
'''),
    ("views: plot(), W&B, Rerun", "One View feeds matplotlib, a W&B payload (built, not sent) and a local .rrd.", '''
from robot_metrics.adapters import rerun as rerun_adapter, wandb as wandb_adapter
by_rung = {f"rung {rung:g}": cell.merged for (_, rung), cell in sorted(strata.items())}
view = collection.view("tip stop error", by_rung, title="CAN 11 · every stop at the tip, by rung (round 8, real)", target=1.0, target_label="Zach's 1 mm")
fig, ax = collection.plot("tip stop error", by_rung, title=view.title, target=1.0, target_label="Zach's 1 mm", path=MEDIA / "can11-tip-stops.png")
print("plot ->", MEDIA / "can11-tip-stops.png", "·", len(view.y), "stops")
print("W&B summary:", wandb_adapter.payload(view, "can11/tip_stop_mm")["summary"])
print("Rerun ->", rerun_adapter.write_rrd(view, "metrics/can11/tip_stop_mm", MEDIA / "can11-tip-stops.rrd"))
'''),
    ("every station's tracking, one view", "The same stratum values across the deployment, as a kpi_trend View: the "
     "agent's 0.5° tracking gate, CAN 11 tracking the 1.0 rung.", '''
from robot_metrics.views import plot
xs, ys, groups = [], [], []
for station_name, tip in sorted(tips.items()):
    station_collection = catalog.pendulum_arm(tip).collection()
    runs = [r for r in onedof_round8.real_runs() if r.labels["station"] == station_name]
    for (_, rung), cell in sorted(summarize(station_collection, runs, by=("station", "rung")).items()):
        xs.append(rung); ys.append(cell.values["servo/tracking RMS"]); groups.append(station_name)
trend = View(title="Tracking RMS by rung, every station (median of each rung's 7 plays)", unit="deg", hint="kpi_trend", x=xs, y=ys, group=groups, target=0.5, target_label="tracking gate 0.5° (agent)", x_label="rung (share of top speed)")
fig, ax = plot(trend, path=MEDIA / "tracking-by-rung.png")
print("plot ->", MEDIA / "tracking-by-rung.png")
'''),
    ("other inputs: repeats", "Repeatability takes one real repeat at a time and refuses another command's.", '''
repeats = [Repeat(("can11", 0.9), r.streams["target"], r.streams["state"]) for r in can11 if r.labels["rung"] == 0.9 and r.labels["kind"] == "repeat"]
rp = Repeatability(k=3.0)
print(f"ISO 9283 RP (k=3) over {len(repeats)} real repeats: {rp.compute(rp.fold(repeats)) * 1e3 * tips['can11']:.3f} mm at CAN 11's tip")
'''),
    ("a new metric is one alias", "A new row is a configured stock metric; it adds one state, the others are untouched.",
     '''
faster = Collection(list(collection.aliases.values()) + [Alias("velocity error RMS", ErrorStat(statistic="rms", rows="moving", field="velocities"), unit="deg/s", scale=DEGREES, across="median")])
print(f"velocity error RMS {faster.values(faster.fold(run))['velocity error RMS']:.3f} deg/s · states {collection.computations} -> {faster.computations}")
record = next(r for r in collection.records(collection.fold(run), run) if r.alias == "tip stop error")
record
'''),
]


def run(media: pathlib.Path) -> Session:
    media.mkdir(parents=True, exist_ok=True)
    session = Session({"MEDIA": media})
    for label, caption, code in CELLS:
        session.cell(label, caption, code)
    return session


if __name__ == "__main__":
    print(run(HERE.parent / "reports" / "media" / "robot-metrics-v0-2026-11-17").transcript())
