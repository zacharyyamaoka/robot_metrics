"""The landing page's code: the API at a glance and the gallery, as cells the builder runs for real.

Each entry is (label, caption, code). ``build_site.py`` runs them in one ``docs/session.py`` Session per section, so
the code on the page is exactly the code that ran and each statement's output is the output it printed. ``MEDIA``
(the site's image folder) is the one name the builder puts in a namespace.

WHY written this way (Zach, /rate bad 2026-11-18, on the metrics-API gallery: "you are doing too much per line ... I
am not seeing the API design at a glance"): the API at a glance comes first; then one statement per line, each
followed by its own output; intermediates named; no comprehension inside an f-string; lines under ~90 characters; a
blank line between ideas.
"""

from __future__ import annotations

GLANCE = ("API at a glance", "A real run of a 1-DOF pendulum, its arm's metrics from the catalog, the same fold offline "
          "and live, then a gate kept outside compute(). Every output is from the run that built this page.", '''
from robot_metrics import Objective, catalog, summarize
from robot_metrics.datasets import onedof_round8

runs = onedof_round8.real_runs()        # 252 real plays: robot_msgs streams + labels
run = runs[32]
run

arm = catalog.pendulum_arm(tip_length_m=0.339).collection()
arm

offline = arm.values(arm.fold(run))     # offline: the whole recording, one fold
offline["tip stop error"]               # mm, at the tip

state = arm.empty()                     # live: the same fold, 7 samples at a time
for chunk in run.chunks(7):
    state = arm.update(state, chunk)
live = arm.values(arm.close(state))
live == offline

can11_runs = [run for run in runs if run.labels["station"] == "can11"]
strata = summarize(arm, can11_runs, by=("station", "rung"))
every_stop = Objective("tip stop error", "<=", 1.0, unit="mm", share=1.0, on="items")
every_stop.judge(strata[("can11", 0.9)]).message
''')

PENDULUM = [
    ("six stations, one catalog entry", "Round 8 of the 1-DOF test rack: six pendulum stations, six speed rungs each, "
     "five repeats plus two held-out plays per rung. Each station is the same part type with its own lever.", '''
from robot_metrics import View, catalog, highest_passing, summarize
from robot_metrics.datasets import onedof_round8
from robot_metrics.views import plot

stations = onedof_round8.stations()
stations["can11"]

runs = onedof_round8.real_runs()
len(runs)

gates = catalog.pendulum_gates()
list(gates)
'''),
    ("every station by rung, every gate", "summarize() folds each run, merges each (station, rung) stratum and reads "
     "every alias by its own across rule. The gates judge each stratum; a station reaches the highest rung with every "
     "rung below it passing.", '''
strata_by_station = {}
for station, facts in sorted(stations.items()):
    arm = catalog.pendulum_arm(facts["tip_length_m"]).collection()
    station_runs = [run for run in runs if run.labels["station"] == station]
    strata_by_station[station] = summarize(arm, station_runs, by=("station", "rung"))
len(strata_by_station)

passed_by_gate = {name: {} for name in gates}
for strata in strata_by_station.values():
    for key, stratum in strata.items():
        for name, gate in gates.items():
            passed_by_gate[name][key] = gate.judge(stratum).verdict == "pass"

for name, passed in passed_by_gate.items():
    reached = highest_passing(passed)
    print(f"{name:<22} {reached}")
'''),
    ("CAN 11, rung by rung", "The long-lever station (339 mm to the tip). The every-stop gate reads the tail the "
     "median hides.", '''
can11 = strata_by_station["can11"]

for (station, rung), stratum in sorted(can11.items()):
    tracking_deg = stratum.values["servo/tracking RMS"]
    tail = stratum.tails["tip stop error"]
    tracking_text = f"tracking {tracking_deg:.3f}°"
    middle_text = f"stops median {tail.median:.3f} · p95 {tail.p95:.3f}"
    worst_text = f"worst {tail.worst:.3f} mm (n={tail.n})"
    print(f"rung {rung:<3g} | {tracking_text} | {middle_text} · {worst_text}")
'''),
    ("views", "One View per picture, drawn by the stock plot(): every stop at the tip by rung against Zach's 1 mm, "
     "and every station's tracking against the agent's 0.5° gate.", '''
can11_arm = catalog.pendulum_arm(stations["can11"]["tip_length_m"]).collection()

states_by_rung = {}
for (station, rung), stratum in sorted(can11.items()):
    states_by_rung[f"rung {rung:g}"] = stratum.merged

stops_view = can11_arm.view(
    "tip stop error",
    states_by_rung,
    title="CAN 11 · every stop at the tip, by rung (round 8, real)",
    target=1.0,
    target_label="1 mm (Zach)",
)
figure, axes = plot(stops_view, path=MEDIA / "pendulum-can11-tip-stops.png")

rungs, tracking_values, station_names = [], [], []
for station, strata in sorted(strata_by_station.items()):
    for (_, rung), stratum in sorted(strata.items()):
        rungs.append(rung)
        tracking_values.append(stratum.values["servo/tracking RMS"])
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
figure, axes = plot(tracking_view, path=MEDIA / "pendulum-tracking-by-rung.png")
'''),
]

FIVEBAR = [
    ("a rigid chain from the catalog", "The bam_fb five-bar's speed ladder of 2026-11-10: the same five rungs played "
     "on the real arm and on its sim twin. Its metrics are a rigid chain: each motor's RMS and the tool's error.", '''
from robot_metrics import View, catalog, summarize
from robot_metrics.datasets import fivebar_ladder
from robot_metrics.views import plot

real_runs = fivebar_ladder.runs("real")
sim_runs = fivebar_ladder.runs("sim")
real_runs[0]

chain = catalog.rigid_chain(("j2", "c1"))
print(chain.table())
'''),
    ("real vs sim, rung by rung", "One collection over both sources: the same identities, so the numbers compare.",
     '''
collection = chain.collection()
real = summarize(collection, real_runs, by=("rung",))
sim = summarize(collection, sim_runs, by=("rung",))

for (rung,), stratum in sorted(real.items()):
    real_mm = stratum.values["tool tracking RMS"]
    sim_mm = sim[(rung,)].values["tool tracking RMS"]
    worst_mm = stratum.values["tool worst"]
    tool_text = f"tool RMS real {real_mm:.2f} · sim {sim_mm:.2f} mm"
    print(f"{rung:<28} {tool_text} · worst real {worst_mm:.1f} mm")
'''),
    ("views", "Tool tracking RMS against speed, real and sim per feed-forward mode; and one rung's tool path in the "
     "plane, commanded and measured (every 20th sample).", '''
labels_by_rung = {}
for run in real_runs:
    labels_by_rung[run.labels["rung"]] = run.labels

speeds, tool_values, series = [], [], []
for source, strata in (("real", real), ("sim", sim)):
    for (rung,), stratum in sorted(strata.items()):
        rung_labels = labels_by_rung[rung]
        speeds.append(rung_labels["speed_m_s"])
        tool_values.append(stratum.values["tool tracking RMS"])
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
figure, axes = plot(speed_view, path=MEDIA / "fivebar-tool-rms-by-speed.png")

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
figure.savefig(MEDIA / "fivebar-tool-path.png")
'''),
]

UR5E = [
    ("what goes in: robot_msgs streams", "SIMULATED. MuJoCo Menagerie's UR5e follows a joint-space pick-and-place "
     "under its own PD position servos. The arrays become robot_msgs streams, and the streams a Run.", '''
import ur5e_sim
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
     "lag: new rows are configured stock metrics, never new functions.", '''
chain = catalog.rigid_chain(ur5e_sim.JOINTS)

worst_stops = []
for index, joint in enumerate(ur5e_sim.JOINTS):
    worst_stop = StopStat(of="error", statistic="max", joint=index)
    worst_stops.append(Alias(f"{joint}/worst stop", worst_stop, unit="deg", scale=DEGREES))

lag = Alias("lag", TimeOffset(), unit="ms", scale=1e3, across="median")
ur5e = Collection([*chain.collection().aliases.values(), *worst_stops, lag])
ur5e

values = ur5e.values(ur5e.fold(run))
values["tool tracking RMS"]
values["shoulder_lift/worst stop"]
values["lag"]
'''),
    ("two controllers, three speeds", "The same path at half, full and double speed, with the PD servos as shipped "
     "and with model feed-forward added. summarize() by controller and speed; a 1 mm gate on the worst tool error.",
     '''
sweep = []
for controller in ("pd", "pd+ff"):
    for speed in (0.5, 1.0, 2.0):
        sweep.append(ur5e_sim.pick_and_place_run(speed, controller))
len(sweep)

strata = summarize(ur5e, sweep, by=("controller", "speed"))
tool_gate = Objective("tool worst", "<=", 1.0, unit="mm", on="value")

for (controller, speed), stratum in sorted(strata.items()):
    rms_mm = stratum.values["tool tracking RMS"]
    worst_mm = stratum.values["tool worst"]
    lag_ms = stratum.values["lag"]
    verdict = tool_gate.judge(stratum).verdict
    tool_text = f"tool RMS {rms_mm:6.2f} · worst {worst_mm:6.2f} mm"
    print(f"{controller:<5} x{speed:<3g} | {tool_text} | lag {lag_ms:5.1f} ms | {verdict}")
'''),
    ("views", "Tool tracking RMS by speed for both controllers (log scale), and every stop of the shoulder lift "
     "joint at full speed.", '''
speeds, tool_values, controllers = [], [], []
for (controller, speed), stratum in sorted(strata.items()):
    speeds.append(speed)
    tool_values.append(stratum.values["tool tracking RMS"])
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
figure.savefig(MEDIA / "ur5e-tool-rms-by-speed.png")

full_speed = {}
for controller in ("pd", "pd+ff"):
    full_speed[controller] = strata[(controller, 1.0)].merged

stops_view = ur5e.view(
    "shoulder_lift/worst stop",
    full_speed,
    title="UR5e (simulated) · every stop of the shoulder lift, full speed",
)
figure, axes = plot(stops_view, path=MEDIA / "ur5e-shoulder-lift-stops.png")
'''),
]

# ---- the catalog's pictures: one cell per stock metric, after one setup cell, on CAN 11's real round-8 runs --------

CATALOG_SETUP = ("catalog setup", "CAN 11's real round-8 plays, by rung.", '''
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from robot_msgs.std.header import Header
from robot_msgs.trajectory import JointTrajectory
from robot_metrics import DEGREES, Alias, Collection, HeldOutPair, Repeat, Run, View
from robot_metrics import CodeShare, ErrorPercentile, ErrorStat, PerformanceGain
from robot_metrics import Repeatability, StopStat, TimeOffset, catalog, summarize
from robot_metrics.datasets import onedof_round8
from robot_metrics.views import plot

runs = onedof_round8.real_runs()
can11_runs = [run for run in runs if run.labels["station"] == "can11"]
lever_m = onedof_round8.stations()["can11"]["tip_length_m"]
len(can11_runs)
''')

CATALOG = {
    "ErrorStat": '''
statistics = ("rms", "max_abs", "mean_abs")
aliases = []
for statistic in statistics:
    metric = ErrorStat(statistic=statistic, rows="moving")
    aliases.append(Alias(statistic, metric, unit="deg", scale=DEGREES))
tracking = Collection(aliases)
tracking

by_rung = summarize(tracking, can11_runs, by=("rung",))
rungs, values, series = [], [], []
for (rung,), stratum in sorted(by_rung.items()):
    for statistic in statistics:
        rungs.append(rung)
        values.append(stratum.values[statistic])
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
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path=MEDIA / "catalog-errorstat.png")
''',
    "ErrorPercentile": '''
aliases = []
for percentile in (50.0, 95.0, 99.0):
    metric = ErrorPercentile(percentile=percentile, rows="moving")
    aliases.append(Alias(f"p{percentile:g}", metric, unit="deg", scale=DEGREES))
percentiles = Collection(aliases)
percentiles

by_rung = summarize(percentiles, can11_runs, by=("rung",))
rungs, values, series = [], [], []
for (rung,), stratum in sorted(by_rung.items()):
    for alias in aliases:
        rungs.append(rung)
        values.append(stratum.values[alias.name])
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
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path=MEDIA / "catalog-errorpercentile.png")
''',
    "StopStat": '''
tip_scale = 1e3 * lever_m          # rad at the joint -> mm at the tip
tip_stop = Alias("tip stop error", StopStat(), unit="mm", scale=tip_scale)
stops = Collection([tip_stop])

by_rung = summarize(stops, can11_runs, by=("rung",))
states_by_rung = {}
for (rung,), stratum in sorted(by_rung.items()):
    states_by_rung[f"rung {rung:g}"] = stratum.merged

view = stops.view(
    "tip stop error",
    states_by_rung,
    title="CAN 11 · every stop at the tip, by rung",
    target=1.0,
    target_label="1 mm",
)
len(view.y)

figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path=MEDIA / "catalog-stopstat.png")
''',
    "TimeOffset": '''
plan_and_replies = ("plan", "state")      # the command on its own ticks, unaligned
lag = Alias("lag", TimeOffset(), signal=plan_and_replies, unit="ms", scale=1e3, across="median")
timing = Collection([lag])

by_rung = summarize(timing, can11_runs, by=("rung",))
rungs, values = [], []
for (rung,), stratum in sorted(by_rung.items()):
    rungs.append(rung)
    values.append(stratum.values["lag"])
values

view = View(
    title="CAN 11 · reply lag behind the plan, median per rung",
    unit="ms",
    hint=TimeOffset.hint,
    x=rungs,
    y=values,
    x_label="rung",
)
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path=MEDIA / "catalog-timeoffset.png")
''',
    "CodeShare": '''
limiter_codes = CodeShare(codes=catalog.LIMITER_CODES)     # moteus 97, 98, 99, 102
limiter = Alias("limiter share", limiter_codes, unit="%", scale=100.0)
health = Collection([limiter])

by_station = summarize(health, runs, by=("station",))
stations, values = [], []
for (station,), stratum in sorted(by_station.items()):
    stations.append(station)
    values.append(stratum.values["limiter share"])
values

view = View(
    title="Rows the limiter or a fault shaped, per station",
    unit="%",
    hint=CodeShare.hint,
    x=stations,
    y=values,
    x_label="station (round 8, every rung)",
)
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path=MEDIA / "catalog-codeshare.png")
''',
    "Repeatability": '''
repeatability = Repeatability(k=3.0)
rungs, values = [], []
for rung in (0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
    repeats = []
    for run in can11_runs:
        if run.labels["rung"] == rung and run.labels["kind"] == "repeat":
            repeat = Repeat(("can11", rung), run.streams["target"], run.streams["state"])
            repeats.append(repeat)
    state = repeatability.fold(repeats)
    rp_tip_mm = repeatability.compute(state) * lever_m * 1e3
    rungs.append(rung)
    values.append(rp_tip_mm)
values

view = View(
    title="CAN 11 · ISO 9283 RP (k=3) at the tip, 5 repeats per rung",
    unit="mm",
    hint=Repeatability.hint,
    x=rungs,
    y=values,
    x_label="rung",
)
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path=MEDIA / "catalog-repeatability.png")
''',
    "PerformanceGain": '''
sealed_file = pathlib.Path(TESTS_DATA, "sealed_ladder_held_out.json")
sealed = json.loads(sealed_file.read_text())


def held_out(rows):
    """The sealed sim's held-out plateaus as runs: feed-forward vs measured torque."""
    out = []
    for row in rows:
        t = np.asarray(row["t"])
        zeros = np.zeros((t.size, 1))
        speed = np.asarray(row["speed"])[:, None]
        feedforward = np.asarray(row["feedforward"])[:, None]
        torque = np.asarray(row["torque"])[:, None]
        header = Header(stamp=0.0, frame_id="output")
        target = JointTrajectory(
            t=t, joint_names=("servo",), positions=zeros, velocities=speed,
            effort=feedforward, header=header,
        )
        state = JointTrajectory(
            t=t, joint_names=("servo",), positions=zeros, velocities=speed,
            effort=torque, header=header,
        )
        streams = {"target": target, "state": state}
        out.append(Run(id=row["id"], streams=streams, provenance="sim"))
    return tuple(out)


residual_rms = ErrorStat(statistic="rms", field="effort")
residual = Alias("feed-forward residual RMS", residual_rms, unit="N·m")

nominal_runs = held_out(sealed["runs"]["nominal"])
fitted_runs = held_out(sealed["runs"]["fitted"])
pair = HeldOutPair(nominal=nominal_runs, fitted=fitted_runs)

gain = PerformanceGain(metric=residual)
state = gain.update(gain.empty(), pair)
gain.sides(state)
gain.compute(state)

view = View(
    title="Residual RMS on held-out runs: nominal vs fitted (sealed sim)",
    unit="N·m",
    hint="kpi_trend",
    x=["nominal", "fitted"],
    y=list(gain.sides(state)),
    x_label="the same held-out runs, played with",
)
figure, axes = plt.subplots(figsize=(6.4, 3.3), dpi=150)
figure, axes = plot(view, ax=axes, path=MEDIA / "catalog-performancegain.png")
''',
}
