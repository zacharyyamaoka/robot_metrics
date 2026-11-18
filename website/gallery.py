"""The landing page's code: the API at a glance, the catalog cards and the gallery, as cells the builder runs for real.

Each entry is (label, caption, code); a catalog card is its code alone. ``build_site.py`` runs each script (the glance,
one catalog card, one gallery arm's cells in order) in its own fresh ``docs/session.py`` Session, from a folder laid
out like a clone's root, so the code on the page is exactly the code that ran and each statement's output is the
output it printed. Nothing is put in a namespace for them: images are saved by bare file name (the builder collects
them), ``tests/data/`` and ``website/ur5e_sim.py`` are read by their path in a clone.

WHY written this way (Zach, /rate bad 2026-11-18, on the metrics-API gallery: "you are doing too much per line ... I
am not seeing the API design at a glance"): the API at a glance comes first; then one statement per line, each
followed by its own output; intermediates named; no comprehension inside an f-string; lines under ~90 characters; a
blank line between ideas.
WHY every script stands alone (Zach, /rate bad 2026-11-18: "I cannot read this pseudo code because it's not valid ...
where did you import it from"): each imports what it uses and says where its data comes from; it runs top to bottom
in a fresh ``python -I`` (``tests/test_shown_examples.py`` checks the built page).
WHY plain words in the comments (Zach, 2026-11-18: "I don't know what you mean by fold etc."): each call to fold /
update / close says in plain words what it does; a result that would echo as jargon is shown as something readable.
The API itself is unchanged in 0.1.x.
"""

from __future__ import annotations

RECORDINGS = ("It reads round 8's recordings, which are not public (point <code>ROBOT_METRICS_ROUND8</code> at a "
              "copy); the README's example runs anywhere.")

GLANCE = ("API at a glance", "A real run of a 1-DOF pendulum: the arm's metrics from the catalog, the same numbers "
          "from the whole recording and from a live feed, then a pass/fail target judged on its own. Every output is "
          "from the run that built this page.", '''
from robot_metrics import Objective, catalog, summarize
from robot_metrics.datasets import onedof_round8

runs = onedof_round8.real_runs()      # 252 real plays of a 1-DOF pendulum rack
run = runs[32]                        # one play: its streams, samples in each
run

arm = catalog.pendulum_arm(tip_length_m=0.339).collection()
list(arm.aliases)                     # the 15 numbers the catalog defines

scored = arm.fold(run)                # run every sample through them, once
offline = arm.values(scored)          # read them out: {name: number}
offline["tip stop error"]             # mm: the tip's median stop error

tally = arm.empty()                   # live: an empty running tally
for chunk in run.chunks(7):           # the same recording, 7 samples at a time
    tally = arm.update(tally, chunk)  # add these 7 samples
live = arm.values(arm.close(tally))   # the run ended: read the numbers out
live == offline                       # the same numbers either way

# score every CAN 11 run, grouped by (station, rung)
can11_runs = [run for run in runs if run.labels["station"] == "can11"]
by_rung = summarize(arm, can11_runs, by=("station", "rung"))

# a pass/fail target, kept apart from the math: every stop within 1 mm
every_stop = Objective(
    "tip stop error", "<=", 1.0, unit="mm", share=1.0, on="items"
)
every_stop.judge(by_rung[("can11", 0.9)]).message
''')

PENDULUM = [
    ("six stations, one catalog entry", "Round 8 of the 1-DOF test rack: six pendulum stations, six speed rungs each, "
     "five repeats plus two held-out plays per rung. Each station is the same part type with its own lever.", '''
from robot_metrics import View, catalog, highest_passing, summarize
from robot_metrics.datasets import onedof_round8
from robot_metrics.views import plot

stations = onedof_round8.stations()     # round 8's six pendulum stations
stations["can11"]

runs = onedof_round8.real_runs()
len(runs)

gates = catalog.pendulum_gates()        # three pass/fail gates for each (station, rung)
list(gates)
'''),
    ("every station by rung, every gate", "summarize() scores every run, groups the runs by (station, rung) and "
     "combines each group's numbers the way each metric says (a median across runs, or every stop pooled). The gates "
     "judge each group; a station reaches the highest rung with every rung below it passing.", '''
groups_by_station = {}
for station, facts in sorted(stations.items()):
    arm = catalog.pendulum_arm(facts["tip_length_m"]).collection()
    station_runs = [run for run in runs if run.labels["station"] == station]
    groups_by_station[station] = summarize(arm, station_runs, by=("station", "rung"))
len(groups_by_station)

passed_by_gate = {name: {} for name in gates}
for groups in groups_by_station.values():
    for key, group in groups.items():
        for name, gate in gates.items():
            passed_by_gate[name][key] = gate.judge(group).verdict == "pass"

for name, passed in passed_by_gate.items():
    reached = highest_passing(passed)       # the top rung with every rung below it passing
    print(f"{name:<22} {reached}")
'''),
    ("CAN 11, rung by rung", "The long-lever station (339 mm to the tip). The every-stop gate reads the worst stops "
     "the median hides.", '''
can11 = groups_by_station["can11"]

for (station, rung), group in sorted(can11.items()):
    tracking_deg = group.values["servo/tracking RMS"]
    tail = group.tails["tip stop error"]        # every stop in the group: median, p95, worst, n
    tracking_text = f"tracking {tracking_deg:.3f}°"
    middle_text = f"stops median {tail.median:.3f} · p95 {tail.p95:.3f}"
    worst_text = f"worst {tail.worst:.3f} mm (n={tail.n})"
    print(f"rung {rung:<3g} | {tracking_text} | {middle_text} · {worst_text}")
'''),
    ("pictures", "One View per picture, drawn by the stock plot(): every stop at the tip by rung against Zach's 1 mm, "
     "and every station's tracking against the agent's 0.5° gate.", '''
can11_arm = catalog.pendulum_arm(stations["can11"]["tip_length_m"]).collection()

tallies_by_rung = {}
for (station, rung), group in sorted(can11.items()):
    tallies_by_rung[f"rung {rung:g}"] = group.merged     # the rung's runs, added up

stops_view = can11_arm.view(
    "tip stop error",
    tallies_by_rung,
    title="CAN 11 · every stop at the tip, by rung (round 8, real)",
    target=1.0,
    target_label="1 mm (Zach)",
)
figure, axes = plot(stops_view, path="pendulum-can11-tip-stops.png")    # saved in this folder

rungs, tracking_values, station_names = [], [], []
for station, groups in sorted(groups_by_station.items()):
    for (_, rung), group in sorted(groups.items()):
        rungs.append(rung)
        tracking_values.append(group.values["servo/tracking RMS"])
        station_names.append(station)

tracking_view = View(
    title="Tracking RMS by rung, every station (median of each rung's 7 plays)",
    unit="deg",
    hint="kpi_trend",
    x=rungs,
    y=tracking_values,
    group=station_names,
    target=0.5,
    target_label="0.5° (agent)",
    x_label="rung (share of the station's top speed)",
)
figure, axes = plot(tracking_view, path="pendulum-tracking-by-rung.png")
'''),
]

FIVEBAR = [
    ("a rigid chain from the catalog", "The bam_fb five-bar's speed ladder of 2026-11-10: the same five rungs played "
     "on the real arm and on its sim twin. Its metrics are a rigid chain: each motor's RMS and the tool's error.", '''
from robot_metrics import View, catalog, summarize
from robot_metrics.datasets import fivebar_ladder
from robot_metrics.views import plot

real_runs = fivebar_ladder.runs("real")     # the 2026-11-10 ladder's recordings
sim_runs = fivebar_ladder.runs("sim")
real_runs[0]

# each row: a name, its unit, the stock metric and settings behind it
# ("4 aliases over 3 states": 4 names, 3 computations; the tool's two rows share one)
chain = catalog.rigid_chain(("j2", "c1"))
print(chain.table())
'''),
    ("real vs sim, rung by rung", "One set of metrics over both sources, so the numbers compare.", '''
collection = chain.collection()
real = summarize(collection, real_runs, by=("rung",))      # score every run, grouped by rung
sim = summarize(collection, sim_runs, by=("rung",))

for (rung,), group in sorted(real.items()):
    real_mm = group.values["tool tracking RMS"]
    sim_mm = sim[(rung,)].values["tool tracking RMS"]
    worst_mm = group.values["tool worst"]
    tool_text = f"tool RMS real {real_mm:.2f} · sim {sim_mm:.2f} mm"
    print(f"{rung:<28} {tool_text} · worst real {worst_mm:.1f} mm")
'''),
    ("pictures", "Tool tracking RMS against speed, real and sim per feed-forward mode; and one rung's tool path in "
     "the plane, commanded and measured (every 20th sample).", '''
labels_by_rung = {}
for run in real_runs:
    labels_by_rung[run.labels["rung"]] = run.labels

speeds, tool_values, series = [], [], []
for source, groups in (("real", real), ("sim", sim)):
    for (rung,), group in sorted(groups.items()):
        rung_labels = labels_by_rung[rung]
        speeds.append(rung_labels["speed_m_s"])
        tool_values.append(group.values["tool tracking RMS"])
        series.append(f"{source} · {rung_labels['ff_mode']}")

speed_view = View(
    title="Five-bar tool tracking RMS by speed (2026-11-10 ladder)",
    unit="mm",
    hint="kpi_trend",
    x=speeds,
    y=tool_values,
    group=series,
    x_label="tool speed (m/s)",
)
figure, axes = plot(speed_view, path="fivebar-tool-rms-by-speed.png")    # saved in this folder

run = real_runs[2]
commanded_xy = run.streams["tool/target"].positions[::20, :2] * 1e3
measured_xy = run.streams["tool/state"].positions[::20, :2] * 1e3
samples = len(commanded_xy)

path_view = View(
    title=f"Tool path in the plane, {run.labels['rung']} (real)",
    unit="mm",
    hint="path_overlay",
    x=[*commanded_xy[:, 0], *measured_xy[:, 0]],
    y=[*commanded_xy[:, 1], *measured_xy[:, 1]],
    group=["commanded"] * samples + ["measured"] * samples,
    x_label="x (mm)",
)
figure, axes = plot(path_view)
axes.set_aspect("equal", adjustable="datalim")
figure.savefig("fivebar-tool-path.png")
'''),
]

UR5E = [
    ("what goes in: robot_msgs streams", "SIMULATED. MuJoCo Menagerie's UR5e follows a joint-space pick-and-place "
     "under its own PD position servos. The arrays become robot_msgs streams, and the streams a Run.", '''
import sys

sys.path.insert(0, "website")       # website/ur5e_sim.py, in a clone of this repo
import ur5e_sim                     # MuJoCo + Menagerie's UR5e (pip install mujoco)
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory
from robot_metrics import DEGREES, Alias, Collection, Objective, Run
from robot_metrics import StopStat, TimeOffset, View, catalog, summarize
from robot_metrics.views import plot

sim = ur5e_sim.simulate(speed=1.0, controller="pd")     # numpy arrays from MuJoCo
sim.t.shape
sim.target_q.shape

target = JointTrajectory(
    t=sim.t, joint_names=ur5e_sim.JOINTS, positions=sim.target_q, velocities=sim.target_qd
)
state = JointTrajectory(
    t=sim.t, joint_names=ur5e_sim.JOINTS, positions=sim.state_q, velocities=sim.state_qd
)
flange = "flange"
tool_target = PoseTrajectory.from_matrices(sim.t, sim.target_flange, child_frame_id=flange)
tool_state = PoseTrajectory.from_matrices(sim.t, sim.state_flange, child_frame_id=flange)

streams = {
    "target": target,
    "state": state,
    "tool/target": tool_target,
    "tool/state": tool_state,
}
labels = {"station": "ur5e-sim", "controller": "pd", "speed": 1.0}
run = Run("ur5e-sim:pd:x1", streams=streams, labels=labels, provenance="sim")
run
'''),
    ("a six-joint arm's metrics", "The catalog's rigid chain over six joints, plus each joint's worst stop and the "
     "lag: new rows are stock metrics with new settings, never new functions.", '''
chain = catalog.rigid_chain(ur5e_sim.JOINTS)

worst_stops = []
for index, joint in enumerate(ur5e_sim.JOINTS):
    worst_stop = StopStat(of="error", statistic="max", joint=index)
    worst_stops.append(Alias(f"{joint}/worst stop", worst_stop, unit="deg", scale=DEGREES))

lag = Alias("lag", TimeOffset(), unit="ms", scale=1e3, across="median")
ur5e = Collection([*chain.collection().aliases.values(), *worst_stops, lag])
list(ur5e.aliases)                  # the numbers this arm reports

scored = ur5e.fold(run)             # run every sample of the run through them, once
values = ur5e.values(scored)        # {name: number in its unit}
values["tool tracking RMS"]
values["shoulder_lift/worst stop"]
values["lag"]
'''),
    ("two controllers, three speeds", "The same path at half, full and double speed, with the PD servos as shipped "
     "and with model feed-forward added. summarize() by controller and speed; a 1 mm target on the worst tool error.",
     '''
sweep = []
for controller in ("pd", "pd+ff"):
    for speed in (0.5, 1.0, 2.0):
        sweep.append(ur5e_sim.pick_and_place_run(speed, controller))
len(sweep)

groups = summarize(ur5e, sweep, by=("controller", "speed"))     # one group per cell
tool_gate = Objective("tool worst", "<=", 1.0, unit="mm", on="value")

for (controller, speed), group in sorted(groups.items()):
    rms_mm = group.values["tool tracking RMS"]
    worst_mm = group.values["tool worst"]
    lag_ms = group.values["lag"]
    verdict = tool_gate.judge(group).verdict
    tool_text = f"tool RMS {rms_mm:6.2f} · worst {worst_mm:6.2f} mm"
    print(f"{controller:<5} x{speed:<3g} | {tool_text} | lag {lag_ms:5.1f} ms | {verdict}")
'''),
    ("pictures", "Tool tracking RMS by speed for both controllers (log scale), and every stop of the shoulder lift "
     "joint at full speed.", '''
speeds, tool_values, controllers = [], [], []
for (controller, speed), group in sorted(groups.items()):
    speeds.append(speed)
    tool_values.append(group.values["tool tracking RMS"])
    controllers.append(controller)

speed_view = View(
    title="UR5e (simulated) · tool tracking RMS by speed",
    unit="mm",
    hint="kpi_trend",
    x=speeds,
    y=tool_values,
    group=controllers,
    target=1.0,
    target_label="1 mm",
    x_label="speed (x the nominal path)",
)
figure, axes = plot(speed_view)
axes.set_yscale("log")
figure.savefig("ur5e-tool-rms-by-speed.png")        # saved in this folder

full_speed = {}
for controller in ("pd", "pd+ff"):
    full_speed[controller] = groups[(controller, 1.0)].merged     # its runs, added up

stops_view = ur5e.view(
    "shoulder_lift/worst stop",
    full_speed,
    title="UR5e (simulated) · every stop of the shoulder lift, full speed",
)
figure, axes = plot(stops_view, path="ur5e-shoulder-lift-stops.png")
'''),
]

# ---- the catalog's pictures: one script per stock metric, each on its own, on CAN 11's real round-8 runs ----------

CATALOG = {
    "ErrorStat": '''
import matplotlib.pyplot as plt
from robot_metrics import DEGREES, Alias, Collection, ErrorStat, View
from robot_metrics import summarize
from robot_metrics.datasets import onedof_round8
from robot_metrics.views import plot

# round 8's real plays, CAN 11's only
runs = onedof_round8.real_runs()
can11_runs = [run for run in runs if run.labels["station"] == "can11"]

statistics = ("rms", "max_abs", "mean_abs")
aliases = []
for statistic in statistics:
    metric = ErrorStat(statistic=statistic, rows="moving")
    alias = Alias(statistic, metric, unit="deg", scale=DEGREES)
    aliases.append(alias)
tracking = Collection(aliases)

# the three statistics share one computation
tracking.computations

# score every run, grouped by rung
by_rung = summarize(tracking, can11_runs, by=("rung",))
rungs, values, series = [], [], []
for (rung,), group in sorted(by_rung.items()):
    for statistic in statistics:
        rungs.append(rung)
        values.append(group.values[statistic])
        series.append(statistic)

view = View(
    title="CAN 11 · tracking error while moving, by rung",
    unit="deg",
    hint=ErrorStat.hint,
    x=rungs,
    y=values,
    group=series,
    x_label="rung",
)

# the picture, saved in this folder
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path="catalog-errorstat.png")
''',
    "ErrorPercentile": '''
import matplotlib.pyplot as plt
from robot_metrics import DEGREES, Alias, Collection, ErrorPercentile
from robot_metrics import View, summarize
from robot_metrics.datasets import onedof_round8
from robot_metrics.views import plot

# round 8's real plays, CAN 11's only
runs = onedof_round8.real_runs()
can11_runs = [run for run in runs if run.labels["station"] == "can11"]

aliases = []
for percentile in (50.0, 95.0, 99.0):
    metric = ErrorPercentile(percentile=percentile, rows="moving")
    name = f"p{percentile:g}"
    aliases.append(Alias(name, metric, unit="deg", scale=DEGREES))
percentiles = Collection(aliases)
list(percentiles.aliases)

# score every run, grouped by rung
by_rung = summarize(percentiles, can11_runs, by=("rung",))
rungs, values, series = [], [], []
for (rung,), group in sorted(by_rung.items()):
    for alias in aliases:
        rungs.append(rung)
        values.append(group.values[alias.name])
        series.append(alias.name)

view = View(
    title="CAN 11 · |error| percentiles while moving, by rung",
    unit="deg",
    hint="kpi_trend",
    x=rungs,
    y=values,
    group=series,
    x_label="rung",
)

# the picture, saved in this folder
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path="catalog-errorpercentile.png")
''',
    "StopStat": '''
import matplotlib.pyplot as plt
from robot_metrics import Alias, Collection, StopStat, summarize
from robot_metrics.datasets import onedof_round8
from robot_metrics.views import plot

# round 8's real plays, CAN 11's only, and CAN 11's lever
runs = onedof_round8.real_runs()
can11_runs = [run for run in runs if run.labels["station"] == "can11"]
lever_m = onedof_round8.stations()["can11"]["tip_length_m"]

# rad at the joint -> mm at the tip
tip_scale = 1e3 * lever_m
tip_stop = Alias(
    "tip stop error", StopStat(), unit="mm", scale=tip_scale
)
stops = Collection([tip_stop])

# score every run, grouped by rung; add up each rung's runs
by_rung = summarize(stops, can11_runs, by=("rung",))
tallies_by_rung = {}
for (rung,), group in sorted(by_rung.items()):
    tallies_by_rung[f"rung {rung:g}"] = group.merged

view = stops.view(
    "tip stop error",
    tallies_by_rung,
    title="CAN 11 · every stop at the tip, by rung",
    target=1.0,
    target_label="1 mm",
)
len(view.y)                     # stops in the picture

# the picture, saved in this folder
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path="catalog-stopstat.png")
''',
    "TimeOffset": '''
import matplotlib.pyplot as plt
from robot_metrics import Alias, Collection, TimeOffset, View
from robot_metrics import summarize
from robot_metrics.datasets import onedof_round8
from robot_metrics.views import plot

# round 8's real plays, CAN 11's only
runs = onedof_round8.real_runs()
can11_runs = [run for run in runs if run.labels["station"] == "can11"]

# the plan (the command, on its own ticks) against the replies,
# not lined up in time: finding the offset is the metric's job
lag = Alias(
    "lag",
    TimeOffset(),
    signal=("plan", "state"),
    unit="ms",
    scale=1e3,
    across="median",
)
timing = Collection([lag])

# one lag per run, the median per rung
by_rung = summarize(timing, can11_runs, by=("rung",))
rungs, values = [], []
for (rung,), group in sorted(by_rung.items()):
    rungs.append(rung)
    values.append(group.values["lag"])

# ms; below zero on this rack: each reply is stamped after its tick
values

view = View(
    title="CAN 11 · reply lag behind the plan, median per rung",
    unit="ms",
    hint=TimeOffset.hint,
    x=rungs,
    y=values,
    x_label="rung",
)

# the picture, saved in this folder
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path="catalog-timeoffset.png")
''',
    "CodeShare": '''
import matplotlib.pyplot as plt
from robot_metrics import Alias, CodeShare, Collection, View
from robot_metrics import catalog, summarize
from robot_metrics.datasets import onedof_round8
from robot_metrics.views import plot

# round 8's real plays, every station
runs = onedof_round8.real_runs()

# moteus codes 97, 98, 99, 102: the limiter or a fault
limiter_codes = CodeShare(codes=catalog.LIMITER_CODES)
limiter = Alias("limiter share", limiter_codes, unit="%", scale=100.0)
health = Collection([limiter])

# score every run, grouped by station
by_station = summarize(health, runs, by=("station",))
stations, values = [], []
for (station,), group in sorted(by_station.items()):
    stations.append(station)
    values.append(group.values["limiter share"])
values                          # % of samples

view = View(
    title="Rows the limiter or a fault shaped, per station",
    unit="%",
    hint=CodeShare.hint,
    x=stations,
    y=values,
    x_label="station (round 8, every rung)",
)

# the picture, saved in this folder
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path="catalog-codeshare.png")
''',
    "Repeatability": '''
import matplotlib.pyplot as plt
from robot_metrics import Repeat, Repeatability, View
from robot_metrics.datasets import onedof_round8
from robot_metrics.views import plot

# round 8's real plays, CAN 11's only, and CAN 11's lever
runs = onedof_round8.real_runs()
can11_runs = [run for run in runs if run.labels["station"] == "can11"]
lever_m = onedof_round8.stations()["can11"]["tip_length_m"]

repeatability = Repeatability(k=3.0)
rungs, values = [], []
for rung in (0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
    repeats = []
    for run in can11_runs:
        is_repeat = run.labels["kind"] == "repeat"
        if run.labels["rung"] == rung and is_repeat:
            target = run.streams["target"]
            measured = run.streams["state"]
            repeats.append(Repeat(("can11", rung), target, measured))

    # add up the rung's 5 repeats; rad at the joint -> mm at the tip
    tally = repeatability.fold(repeats)
    rp_tip_mm = repeatability.compute(tally) * lever_m * 1e3
    rungs.append(rung)
    values.append(rp_tip_mm)
values                          # mm at the tip, per rung

view = View(
    title="CAN 11 · ISO 9283 RP (k=3) at the tip, 5 repeats a rung",
    unit="mm",
    hint=Repeatability.hint,
    x=rungs,
    y=values,
    x_label="rung",
)

# the picture, saved in this folder
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path="catalog-repeatability.png")
''',
    "PerformanceGain": '''
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from robot_msgs.std.header import Header
from robot_msgs.trajectory import JointTrajectory
from robot_metrics import Alias, ErrorStat, HeldOutPair
from robot_metrics import PerformanceGain, Run, View
from robot_metrics.views import plot

# a sealed sim's held-out runs, in this repo's tests/data/
# (run from a clone's root)
sealed_file = pathlib.Path("tests/data/sealed_ladder_held_out.json")
sealed = json.loads(sealed_file.read_text())


def held_out(rows):
    # the held-out plateaus as runs: feed-forward vs measured torque
    out = []
    for row in rows:
        t = np.asarray(row["t"])
        zeros = np.zeros((t.size, 1))
        speed = np.asarray(row["speed"])[:, None]
        feedforward = np.asarray(row["feedforward"])[:, None]
        torque = np.asarray(row["torque"])[:, None]
        header = Header(stamp=0.0, frame_id="output")
        target = JointTrajectory(
            t=t,
            joint_names=("servo",),
            positions=zeros,
            velocities=speed,
            effort=feedforward,
            header=header,
        )
        state = JointTrajectory(
            t=t,
            joint_names=("servo",),
            positions=zeros,
            velocities=speed,
            effort=torque,
            header=header,
        )
        streams = {"target": target, "state": state}
        run = Run(id=row["id"], streams=streams, provenance="sim")
        out.append(run)
    return tuple(out)


residual_rms = ErrorStat(statistic="rms", field="effort")
residual = Alias("feed-forward residual", residual_rms, unit="N·m")

nominal_runs = held_out(sealed["runs"]["nominal"])
fitted_runs = held_out(sealed["runs"]["fitted"])
pair = HeldOutPair(nominal=nominal_runs, fitted=fitted_runs)

# score both sides of the pair
gain = PerformanceGain(metric=residual)
tally = gain.update(gain.empty(), pair)

# residual RMS: (nominal, fitted)
gain.sides(tally)

# nominal - fitted: above zero, the fit helped
gain.compute(tally)

view = View(
    title="Residual RMS on held-out runs: nominal vs fitted",
    unit="N·m",
    hint="kpi_trend",
    x=["nominal", "fitted"],
    y=list(gain.sides(tally)),
    x_label="the same held-out runs (sealed sim), played with",
)

# the picture, saved in this folder
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path="catalog-performancegain.png")
''',
}
