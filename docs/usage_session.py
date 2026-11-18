"""The v0 usage session on CAN 11's round-8 runs: the cells the report shows, run for real.

    cd ~/robot_metrics && taskset -c 2-5,16,17 uv run --extra test --extra viz --extra rerun python docs/usage_session.py

Prints the transcripts; ``run(media)`` returns two Sessions (the report builder calls it) and writes the views to
``media``: the plot() images and a Rerun .rrd (when rerun-sdk is installed). Two scripts, each on its own: ``GLANCE``,
and ``CELLS`` (the usage session, its steps in order). Nothing is put in their namespaces; each imports what it uses
and saves its files by bare name in the folder it runs in (``tests/test_shown_examples.py`` re-runs both, alone, in a
fresh ``python -I``, against the report).

WHY written this way (Zach, /rate bad 2026-11-18, on the metrics-API gallery: "you are doing too much per line ... I am
not seeing the API design at a glance"): the API at a glance comes first; then one statement per line, each followed
by its own output; intermediates named; no comprehension inside an f-string; lines under ~90 characters; a blank line
between ideas.
WHY each script stands alone and the comments are plain words (Zach, /rate bad 2026-11-18: "where did you import it
from", and "I don't know what you mean by fold etc."): a reader can copy either script and run it; each fold / update
/ close says what it does; a result that would echo as jargon is shown as something readable. The API is unchanged.
"""

from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from session import Session  # noqa: E402

GLANCE = ("API at a glance", "Configure a stock metric and name it, take the pendulum arm's metrics from the catalog, "
          "score a run live and from the whole recording, judge a pass/fail target kept apart from the math, and "
          "group many runs by station and rung. One script on its own; it reads round 8's recordings, which are "
          "not public.", """
from robot_metrics import DEGREES, Alias, ErrorStat, Objective, catalog, summarize
from robot_metrics.datasets import onedof_round8

real_runs = onedof_round8.real_runs()       # round 8's 252 real plays
can11_runs = [run for run in real_runs if run.labels["station"] == "can11"]
run = can11_runs[32]

tracking_rms = ErrorStat(statistic="rms", rows="moving")                    # a stock metric
tracking = Alias("tracking RMS", tracking_rms, unit="deg", scale=DEGREES)   # its name, unit

arm_metrics = catalog.pendulum_arm(tip_length_m=0.339)    # the arm's .metrics
collection = arm_metrics.collection()
list(collection.aliases)                                  # the 15 numbers it reports

tally = collection.empty()                                # live: an empty running tally
for chunk in run.chunks(7):                               # the recording, 7 samples at a time
    tally = collection.update(tally, chunk)
live = collection.values(collection.close(tally))         # the run ended: read the numbers
live["tip stop error"]

offline = collection.values(collection.fold(run))         # the whole recording in one call
offline["tip stop error"]

every_stop = Objective("tip stop error", "<=", 1.0, unit="mm", share=1.0, on="items")
every_stop.title

by_rung = summarize(collection, can11_runs, by=("station", "rung"))     # score, then group
verdict = every_stop.judge(by_rung[("can11", 0.9)])
verdict.verdict, verdict.message
""")

CELLS = [
    ("a part's metrics", "The pendulum arm is an assembly: the servo's 9 numbers plus tip numbers, which are the "
     "servo's scaled through the lever. 15 numbers from 5 computations.", """
from robot_metrics import catalog
from robot_metrics.datasets import onedof_round8

stations = onedof_round8.stations()         # round 8's six stations
stations["can11"]

arm_metrics = catalog.pendulum_arm(tip_length_m=0.339)
arm_metrics.context

servo_metrics = arm_metrics.child("servo")
servo_metrics.part_type

# each row: a name, its unit, the stock metric and settings behind it
# ("15 aliases over 5 states": 15 names, 5 computations shared between them)
print(arm_metrics.table())
"""),
    ("one CAN 11 run, scored whole", "A run is named robot_msgs streams plus labels. The target was put on the "
     "replies' instants by the explicit alignment step (hermite, one tick) when the run was read.", """
real_runs = onedof_round8.real_runs()       # round 8's 252 real plays
can11_runs = [run for run in real_runs if run.labels["station"] == "can11"]
run = can11_runs[32]
run

run.labels

run.alignment

collection = arm_metrics.collection()
offline = collection.values(collection.fold(run))     # every sample, once: {name: number}

for name, value in offline.items():
    unit = collection.aliases[name].unit
    shown = f"{value:.4f}" if isinstance(value, float) else str(value)
    print(f"{name:<22} {shown:>8} {unit}")
"""),
    ("the same run, live", "Fed 7 samples at a time (a stop spans several chunks), the running tally gives the "
     "whole-recording numbers. One live update costs well under a 10 ms tick.", """
import time

from robot_metrics import DEGREES, ErrorStat

chunks = run.chunks(7)
len(chunks)

tally = collection.empty()                  # an empty running tally
for chunk in chunks:
    tally = collection.update(tally, chunk)     # add 7 samples
live = collection.values(collection.close(tally))

number_names = [name for name, value in offline.items() if isinstance(value, float)]
gaps = [abs(live[name] - offline[name]) for name in number_names]
len(gaps)
max(gaps)

collection.computations                     # 15 numbers, 5 computations

mid_run_tally = collection.update(collection.empty(), chunks[9])
next_chunk = chunks[10]
started = time.perf_counter()
for _ in range(300):
    collection.update(mid_run_tally, next_chunk)
milliseconds = (time.perf_counter() - started) / 300 * 1e3
print(f"one live update: {milliseconds:.2f} ms per 7-sample chunk")

tracking_rms = ErrorStat(statistic="rms", rows="moving")
stream = tracking_rms.stream()    # torchmetrics' face: update / compute / reset
for chunk in chunks:
    stream.update(chunk.streams["target"], chunk.streams["state"])
stream.compute() * DEGREES
offline["servo/tracking RMS"]
"""),
    ("CAN 11's rungs, gates applied", "Groups are station x rung. Each gate is pass/fail targets judged apart from "
     "the math; the every-stop gate reads the worst stops the median hides.", """
from robot_metrics import summarize

by_rung = summarize(collection, can11_runs, by=("station", "rung"))     # score, then group
len(by_rung)

gates = catalog.pendulum_gates()
list(gates)

for (station, rung), group in sorted(by_rung.items()):
    tracking_deg = group.values["servo/tracking RMS"]
    median_mm = group.values["tip stop error"]
    tail = group.tails["tip stop error"]       # every stop in the group: median, p95, worst, n

    verdicts = [gate.judge(group).verdict for gate in gates.values()]
    verdict_text = " / ".join(verdicts)

    tracking_text = f"tracking {tracking_deg:.3f}°"
    stops_text = f"stops median {median_mm:.3f} p95 {tail.p95:.3f} worst {tail.worst:.3f}"
    row = f"rung {rung:<3g} | {tracking_text} | {stops_text} mm, n={tail.n}"
    print(f"{row} | {verdict_text}")
"""),
    ("one target, two shares, a Wilson interval", "The same stops judged against two targets, nothing recomputed; "
     "a yes/no metric summarizes as a share of runs with its Wilson interval.", """
from robot_metrics import Objective

by_station = summarize(collection, can11_runs, by=("station",))
can11 = by_station[("can11",)]
can11

for share in (1.0, 0.9):
    objective = Objective("tip stop error", "<=", 1.0, unit="mm", share=share, on="items")
    judged = objective.judge(can11)
    print(f"{judged.objective}: {judged.verdict} · {judged.message}")

settled = can11.shares["all stops settled"]
interval = f"Wilson 95 % [{settled.low:.2f}, {settled.high:.2f}]"
counts = f"{settled.successes} of {settled.trials}"
print(f"runs with every stop settled into 1 mm: {counts}, {interval}")
"""),
    ("pictures: plot(), W&B, Rerun", "One View feeds matplotlib, a W&B payload (built, not sent) and a local .rrd.",
     """
from robot_metrics.adapters import rerun as rerun_adapter
from robot_metrics.adapters import wandb as wandb_adapter
from robot_metrics.views import plot

tallies_by_rung = {}
for (station, rung), group in sorted(by_rung.items()):
    tallies_by_rung[f"rung {rung:g}"] = group.merged     # the rung's runs, added up
list(tallies_by_rung)

view = collection.view(
    "tip stop error",
    tallies_by_rung,
    title="CAN 11 · every stop at the tip, by rung (round 8, real)",
    target=1.0,
    target_label="Zach's 1 mm",
)
len(view.y)

figure, axes = plot(view, path="can11-tip-stops.png")     # saved in this folder

payload = wandb_adapter.payload(view, "can11/tip_stop_mm")
payload["summary"]

rrd_path = rerun_adapter.write_rrd(view, "metrics/can11/tip_stop_mm", "can11-tip-stops.rrd")
rrd_path.name
"""),
    ("every station's tracking, one picture", "The same group numbers across the deployment, as a kpi_trend View: "
     "the agent's 0.5° tracking gate, CAN 11 tracking the 1.0 rung.", """
from robot_metrics import View

rungs, tracking_values, station_names = [], [], []
for station_name, facts in sorted(stations.items()):
    station_collection = catalog.pendulum_arm(facts["tip_length_m"]).collection()
    station_runs = [run for run in real_runs if run.labels["station"] == station_name]
    station_groups = summarize(station_collection, station_runs, by=("station", "rung"))

    for (_, rung), group in sorted(station_groups.items()):
        rungs.append(rung)
        tracking_values.append(group.values["servo/tracking RMS"])
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
figure, axes = plot(trend, path="tracking-by-rung.png")
"""),
    ("other inputs: repeats", "Repeatability takes one real repeat at a time and refuses another command's.", """
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
rp_joint_rad = repeatability.compute(repeatability.fold(repeats))     # all five repeats
rp_tip_mm = rp_joint_rad * stations["can11"]["tip_length_m"] * 1e3
print(f"ISO 9283 RP (k=3) at CAN 11's tip: {rp_tip_mm:.3f} mm")
"""),
    ("a new metric is one name", "A new number is a stock metric with new settings and a name; it adds one "
     "computation, the others are untouched.", """
from robot_metrics import Alias, Collection

velocity_rms = ErrorStat(statistic="rms", rows="moving", field="velocities")
velocity_alias = Alias(
    "velocity error RMS", velocity_rms, unit="deg/s", scale=DEGREES, across="median"
)

extended = Collection([*collection.aliases.values(), velocity_alias])
extended.computations                       # one more than the arm's 5

extended_values = extended.values(extended.fold(run))
extended_values["velocity error RMS"]

records = collection.records(collection.fold(run), run)     # what you would store
tip_record = next(record for record in records if record.alias == "tip stop error")
tip_record.value, tip_record.unit, tip_record.base, tip_record.version
"""),
]


def run(media: pathlib.Path) -> tuple[Session, Session]:
    """(the glance, the usage session): two scripts, each in its own fresh namespace, run inside ``media`` so the
    images and the .rrd they save by bare file name land there."""

    media.mkdir(parents=True, exist_ok=True)
    previous = pathlib.Path.cwd()
    os.chdir(media)
    try:
        glance = Session({"__name__": "__main__"})
        glance.cell(*GLANCE)
        session = Session({"__name__": "__main__"})
        for label, caption, code in CELLS:
            session.cell(label, caption, code)
    finally:
        os.chdir(previous)
    return glance, session


if __name__ == "__main__":
    for script in run(HERE.parent / "reports" / "media" / "robot-metrics-v0-2026-11-17"):
        print(script.transcript(), end="\n\n")
