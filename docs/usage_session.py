"""The v0 usage session on CAN 11's round-8 runs: the cells the report shows, run for real.

    cd ~/robot_metrics && taskset -c 2-5,16,17 uv run --extra test --extra viz --extra rerun python docs/usage_session.py

Prints the transcript; ``run(media)`` returns the Session (the report builder calls it) and writes the views to
``media``: the plot() images and a Rerun .rrd (when rerun-sdk is installed). ``GLANCE`` runs first and every cell after
it continues its namespace; ``MEDIA`` (the report's media folder) is the one name the harness puts there.

WHY written this way (Zach, /rate bad 2026-11-18, on the metrics-API gallery: "you are doing too much per line ... I am
not seeing the API design at a glance"): the API at a glance comes first; then one statement per line, each followed
by its own output; intermediates named; no comprehension inside an f-string; lines under ~90 characters; a blank line
between ideas.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from session import Session  # noqa: E402

GLANCE = ("API at a glance", "Configure a stock metric, alias it, take the pendulum arm's metrics from the catalog, "
          "fold a run live and offline, keep a gate outside compute, summarize by station and rung. Runnable as is.", '''
from robot_metrics import DEGREES, Alias, ErrorStat, Objective, catalog, summarize
from robot_metrics.datasets import onedof_round8

real_runs = onedof_round8.real_runs()
can11_runs = [run for run in real_runs if run.labels["station"] == "can11"]
run = can11_runs[32]

tracking_rms = ErrorStat(statistic="rms", rows="moving")
tracking = Alias("tracking RMS", tracking_rms, unit="deg", scale=DEGREES)

arm_metrics = catalog.pendulum_arm(tip_length_m=0.339)    # the arm's .metrics
collection = arm_metrics.collection()
collection

state = collection.empty()                                # live
for chunk in run.chunks(7):
    state = collection.update(state, chunk)
live = collection.values(collection.close(state))
live["tip stop error"]

offline = collection.values(collection.fold(run))         # offline
offline["tip stop error"]

every_stop = Objective("tip stop error", "<=", 1.0, unit="mm", share=1.0, on="items")
every_stop.title

strata = summarize(collection, can11_runs, by=("station", "rung"))
verdict = every_stop.judge(strata[("can11", 0.9)])
verdict.verdict, verdict.message
''')

CELLS = [
    ("a part's metrics", "The pendulum arm is an assembly: the servo's aliases plus tip rows that read the servo's "
     "states through the lever. 15 aliases, 5 states.", '''
stations = onedof_round8.stations()
stations["can11"]

arm_metrics.context

servo_metrics = arm_metrics.child("servo")
servo_metrics.part_type

print(arm_metrics.table())
'''),
    ("one CAN 11 run, offline", "A run is named robot_msgs streams plus labels. The target was put on the replies' "
     "instants by the explicit alignment step (hermite, one tick) when the run was read.", '''
run

run.labels

run.alignment

for name, value in offline.items():
    unit = collection.aliases[name].unit
    shown = f"{value:.4f}" if isinstance(value, float) else str(value)
    print(f"{name:<22} {shown:>8} {unit}")
'''),
    ("the same run, live", "Fed 7 samples at a time (a stop's dwell spans several chunks), the fold gives the "
     "offline numbers. One live update costs well under a 10 ms tick.", '''
import time

chunks = run.chunks(7)
len(chunks)

number_names = [name for name, value in offline.items() if isinstance(value, float)]
gaps = [abs(live[name] - offline[name]) for name in number_names]
len(gaps)
max(gaps)

collection.computations

mid_run_state = collection.update(collection.empty(), chunks[9])
next_chunk = chunks[10]
started = time.perf_counter()
for _ in range(300):
    collection.update(mid_run_state, next_chunk)
milliseconds = (time.perf_counter() - started) / 300 * 1e3
print(f"one live update: {milliseconds:.2f} ms per 7-sample chunk")

stream = tracking_rms.stream()    # torchmetrics' face: update / compute / reset
for chunk in chunks:
    stream.update(chunk.streams["target"], chunk.streams["state"])
stream.compute() * DEGREES
offline["servo/tracking RMS"]
'''),
    ("CAN 11's rungs, gates applied", "Strata are station x rung. Each gate is OpenSLO-style objectives judged "
     "outside compute(); the every-stop gate reads the tail the median hides.", '''
len(strata)

gates = catalog.pendulum_gates()
list(gates)

for (station, rung), stratum in sorted(strata.items()):
    tracking_deg = stratum.values["servo/tracking RMS"]
    median_mm = stratum.values["tip stop error"]
    tail = stratum.tails["tip stop error"]

    verdicts = [gate.judge(stratum).verdict for gate in gates.values()]
    verdict_text = " / ".join(verdicts)

    tracking_text = f"tracking {tracking_deg:.3f}°"
    stops_text = f"stops median {median_mm:.3f} p95 {tail.p95:.3f} worst {tail.worst:.3f}"
    row = f"rung {rung:<3g} | {tracking_text} | {stops_text} mm, n={tail.n}"
    print(f"{row} | {verdict_text}")
'''),
    ("one target, two shares, a Wilson interval", "The same stop state judged against two occurrence targets, nothing "
     "recomputed; a bool metric summarizes as a share of runs with its Wilson interval.", '''
station_strata = summarize(collection, can11_runs, by=("station",))
can11 = station_strata[("can11",)]
can11

for share in (1.0, 0.9):
    objective = Objective("tip stop error", "<=", 1.0, unit="mm", share=share, on="items")
    judged = objective.judge(can11)
    print(f"{judged.objective}: {judged.verdict} · {judged.message}")

settled = can11.shares["all stops settled"]
interval = f"Wilson 95 % [{settled.low:.2f}, {settled.high:.2f}]"
counts = f"{settled.successes} of {settled.trials}"
print(f"runs with every stop settled into 1 mm: {counts}, {interval}")
'''),
    ("views: plot(), W&B, Rerun", "One View feeds matplotlib, a W&B payload (built, not sent) and a local .rrd.", '''
from robot_metrics.adapters import rerun as rerun_adapter
from robot_metrics.adapters import wandb as wandb_adapter
from robot_metrics.views import plot

states_by_rung = {}
for (station, rung), stratum in sorted(strata.items()):
    states_by_rung[f"rung {rung:g}"] = stratum.merged
list(states_by_rung)

view = collection.view(
    "tip stop error",
    states_by_rung,
    title="CAN 11 · every stop at the tip, by rung (round 8, real)",
    target=1.0,
    target_label="Zach's 1 mm",
)
len(view.y)

figure, axes = plot(view, path=MEDIA / "can11-tip-stops.png")

payload = wandb_adapter.payload(view, "can11/tip_stop_mm")
payload["summary"]

rrd_file = MEDIA / "can11-tip-stops.rrd"
rrd_path = rerun_adapter.write_rrd(view, "metrics/can11/tip_stop_mm", rrd_file)
rrd_path.name
'''),
    ("every station's tracking, one view", "The same stratum values across the deployment, as a kpi_trend View: the "
     "agent's 0.5° tracking gate, CAN 11 tracking the 1.0 rung.", '''
from robot_metrics import View

rungs, tracking_values, station_names = [], [], []
for station_name, facts in sorted(stations.items()):
    station_collection = catalog.pendulum_arm(facts["tip_length_m"]).collection()
    station_runs = [run for run in real_runs if run.labels["station"] == station_name]
    by_rung = summarize(station_collection, station_runs, by=("station", "rung"))

    for (_, rung), stratum in sorted(by_rung.items()):
        rungs.append(rung)
        tracking_values.append(stratum.values["servo/tracking RMS"])
        station_names.append(station_name)
len(tracking_values)

trend = View(
    title="Tracking RMS by rung, every station (median of each rung's 7 plays)",
    unit="deg",
    hint="kpi_trend",
    x=rungs,
    y=tracking_values,
    group=station_names,
    target=0.5,
    target_label="tracking gate 0.5° (agent)",
    x_label="rung (share of top speed)",
)
figure, axes = plot(trend, path=MEDIA / "tracking-by-rung.png")
'''),
    ("other inputs: repeats", "Repeatability takes one real repeat at a time and refuses another command's.", '''
from robot_metrics import Repeat, Repeatability

rung_runs = [run for run in can11_runs if run.labels["rung"] == 0.9]
repeat_runs = [run for run in rung_runs if run.labels["kind"] == "repeat"]

repeats = []
for repeat_run in repeat_runs:
    target = repeat_run.streams["target"]
    measured = repeat_run.streams["state"]
    repeats.append(Repeat(("can11", 0.9), target, measured))
len(repeats)

repeatability = Repeatability(k=3.0)
rp_joint_rad = repeatability.compute(repeatability.fold(repeats))
rp_tip_mm = rp_joint_rad * stations["can11"]["tip_length_m"] * 1e3
print(f"ISO 9283 RP (k=3) at CAN 11's tip: {rp_tip_mm:.3f} mm")
'''),
    ("a new metric is one alias", "A new row is a configured stock metric; it adds one state, the others are "
     "untouched.", '''
from robot_metrics import Collection

velocity_rms = ErrorStat(statistic="rms", rows="moving", field="velocities")
velocity_alias = Alias(
    "velocity error RMS", velocity_rms, unit="deg/s", scale=DEGREES, across="median"
)

extended = Collection([*collection.aliases.values(), velocity_alias])
extended

extended_values = extended.values(extended.fold(run))
extended_values["velocity error RMS"]

records = collection.records(collection.fold(run), run)
tip_record = next(record for record in records if record.alias == "tip stop error")
tip_record
'''),
]


def run(media: pathlib.Path) -> Session:
    media.mkdir(parents=True, exist_ok=True)
    session = Session({"MEDIA": media})
    for label, caption, code in [GLANCE, *CELLS]:
        session.cell(label, caption, code)
    return session


if __name__ == "__main__":
    print(run(HERE.parent / "reports" / "media" / "robot-metrics-v0-2026-11-17").transcript())
