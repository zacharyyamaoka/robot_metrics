"""Build the landing page (GitHub Pages) from real runs: the API at a glance, the catalog, the aliases, the gallery.

    uv run --extra site python website/build_site.py         # writes website/_build/ (index.html, img/, data/)
    python3 website/publish_pages.py                         # commits website/_build/ to gh-pages and pushes it

Every code block on the page is a cell of ``gallery.py`` that this script ran (``docs/session.py``'s Session), and
every image is a ``plot()`` it drew. It needs what the cells read: round 8 and the five-bar ladder (local recordings,
not public: ``ROBOT_METRICS_ROUND8``, ``ROBOT_METRICS_FIVEBAR_LADDER``) and MuJoCo Menagerie's UR5e (``MENAGERIE``).
So the site is built where the recordings are and published as a build artifact on the gh-pages branch; main's
history holds the builder, never the images or the data.

WHY the highlighting and the notebook rhythm (Zach, /rate bad 2026-11-18): Pygments at build time, one statement then
its own output in a quieter band, blank lines between ideas, the API at a glance first.
"""

from __future__ import annotations

import dataclasses
import html
import inspect
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MUJOCO_GL", "egl")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = Path(os.environ.get("ROBOT_METRICS_SITE_OUT") or HERE / "_build")
IMG = OUT / "img"
sys.path.insert(0, str(ROOT / "docs"))
sys.path.insert(0, str(HERE))

from pygments import highlight  # noqa: E402
from pygments.formatters import HtmlFormatter  # noqa: E402
from pygments.lexers import PythonLexer  # noqa: E402
from session import Session  # noqa: E402

import gallery  # noqa: E402
import robot_metrics  # noqa: E402
from robot_metrics import catalog, functional, stock  # noqa: E402

REPO = "https://github.com/zacharyyamaoka/robot_metrics"
SITE = "https://zacharyyamaoka.github.io/robot_metrics/"
VERSION = robot_metrics.__version__
PRIVATE = re.compile(r"/home/|/archive|/Users/|@gmail|\.netrc|wandb_api|gh[pousr]_[A-Za-z0-9]{20}")

# ---- the catalog's words: what the code cannot say about itself --------------------------------------------------

GROUPS = (
    ("Tracking", "How far the measured stream is from the target, row by row.", ("ErrorStat", "ErrorPercentile")),
    ("Stops / dwell", "What happens where the plan holds still: each stop's error, settle, overshoot.", ("StopStat",)),
    ("Timing", "How late the response runs behind the command.", ("TimeOffset",)),
    ("Health / codes", "How often the drive reported a code: limiter, fault.", ("CodeShare",)),
    ("Repeatability", "The same command, played again: how far apart the answers land.", ("Repeatability",)),
    ("Performance gain", "Did a fitted model make the thing that matters better, on runs it never saw?",
     ("PerformanceGain",)),
)
INPUTS = {
    "AlignedPair": "<code>target</code>, <code>measured</code>: two robot_msgs streams on the same instants "
                   "(<code>JointTrajectory</code>, or <code>PoseTrajectory</code> for a tool). "
                   "<code>align()</code> puts a plan on the replies' instants first.",
    "CommandAndState": "<code>command</code> on its own ticks and <code>state</code> as it came back: not aligned "
                       "(finding the offset is the metric's job).",
    "Repeat": "<code>Repeat(command, target, measured)</code>: one real repeat of one command per update; a repeat "
              "of another command is refused.",
    "HeldOutPair": "<code>HeldOutPair(nominal, fitted)</code>: the same held-out runs played with the nominal "
                   "values and with the fitted ones.",
}
READS = {
    "ErrorStat": "the error <code>compare(target, measured).error</code> of one <code>field</code> and "
                 "<code>joint</code>, or a pose's position error",
    "ErrorPercentile": "the same error signal as ErrorStat",
    "StopStat": "the measured joint and the target's position and <b>velocities</b> (a stop is a dwell at zero "
                "commanded speed)",
    "TimeOffset": "two joint streams; <code>robot_msgs.compare.estimate_time_offset</code> on the whole run",
    "CodeShare": "the measured stream's <code>channels[channel]</code> (moteus fault / limiter codes)",
    "Repeatability": "each repeat's stops (the stop kernel), the same stops across repeats",
    "PerformanceGain": "any metric or alias, folded over each side's runs",
}
OUTPUTS = {
    "ErrorStat": "a number in the signal's SI unit (rad, m for a tool, rad/s, N·m)",
    "ErrorPercentile": "a number in the signal's SI unit",
    "StopStat": "a number in the signal's unit (s for <code>settle</code>), a bool for <code>statistic=\"all\"</code>; "
                "its occurrences are every stop",
    "TimeOffset": "seconds; across runs the median of the per-run offsets",
    "CodeShare": "a share in [0, 1]",
    "Repeatability": "the signal's SI unit (rad at a joint): ISO 9283 RP, the worst stop index",
    "PerformanceGain": "the inner metric's unit: nominal minus fitted (positive = the fit helped)",
}
STATE = {
    "ErrorStat": "(Σe², n, max|e|, Σ|e|): four numbers, shared by rms, max_abs and mean_abs",
    "ErrorPercentile": "every |e| seen so far (exact; merge concatenates)",
    "StopStat": "closed stops plus the open dwell, so a stop split across chunks is found exactly",
    "TimeOffset": "the run's chunks until close, then one offset per run",
    "CodeShare": "(hits, rows)",
    "Repeatability": "(command, each repeat's stop errors)",
    "PerformanceGain": "(nominal value, fitted value)",
}
OUT_SHORT = {"ErrorStat": "a number (SI)", "ErrorPercentile": "a number (SI)",
             "StopStat": "a number or a bool; every stop", "TimeOffset": "seconds", "CodeShare": "a share, 0 to 1",
             "Repeatability": "a number (SI): RP", "PerformanceGain": "nominal − fitted"}
FUNCTION = {"ErrorStat": "error_stat", "ErrorPercentile": "error_percentile", "StopStat": "stop_stat",
            "TimeOffset": "time_offset", "CodeShare": "code_share", "Repeatability": "repeatability",
            "PerformanceGain": "performance_gain"}
CHOICES = {
    ("ErrorStat", "statistic"): "rms · max_abs · mean_abs",
    ("ErrorStat", "rows"): "all · moving",
    ("ErrorStat", "field"): "positions · velocities · effort",
    ("ErrorPercentile", "rows"): "all · moving",
    ("ErrorPercentile", "field"): "positions · velocities · effort",
    ("StopStat", "of"): " · ".join(stock.STOP_QUANTITIES),
    ("StopStat", "statistic"): " · ".join(stock.STOP_STATISTICS) + " · any p&lt;q&gt;",
}


# ---- rendering helpers -------------------------------------------------------------------------------------------

LEXER, FORMATTER = PythonLexer(), HtmlFormatter(nowrap=True)
WRAP_OVER = 96


def python(code: str) -> str:
    return highlight(code, LEXER, FORMATTER).rstrip("\n")


def output_block(text: str) -> str:
    wrap = "\n" not in text and len(text) > WRAP_OVER
    return f"<pre class='out{' wrap' if wrap else ''}'>{html.escape(text)}</pre>"


def notebook(cell: dict) -> str:
    """A cell as a notebook: statements highlighted, each output in a quieter band right under its statement.
    Statements with no output share a block, with the source's blank lines between ideas (docs/build_report.py's
    rule, so the report and the site read the same)."""

    blocks, group, group_gap = [], [], False
    for statement in cell["statements"]:
        if not group:
            group_gap = statement["gap_before"] and bool(blocks)
        elif statement["gap_before"]:
            group.append("")
        group.append(statement["code"])
        if statement["output"]:
            blocks.append(f"<pre class='src{' gap' if group_gap else ''}'>{python(chr(10).join(group))}</pre>")
            blocks.append(output_block(statement["output"]))
            group = []
    if group:
        blocks.append(f"<pre class='src{' gap' if group_gap else ''}'>{python(chr(10).join(group))}</pre>")
    return f"<div class='notebook'>{''.join(blocks)}</div>"


def token_css() -> str:
    light = HtmlFormatter(style="default").get_token_style_defs(".src")
    dark_auto = HtmlFormatter(style="github-dark").get_token_style_defs(':root:not([data-theme="light"]) .src')
    dark_forced = HtmlFormatter(style="github-dark").get_token_style_defs(':root[data-theme="dark"] .src')
    return "\n".join([*light, "@media (prefers-color-scheme: dark) {", *dark_auto, "}", *dark_forced])


def figure(name: str, caption: str, alt: str = "") -> str:
    return (f"<figure><img src='img/{name}' alt='{html.escape(alt or caption)}'>"
            f"<figcaption>{caption}</figcaption></figure>")


def esc(value: object) -> str:
    return html.escape(str(value))


# ---- the diagram: run -> metric (fold) -> value -> objective -> verdict, and runs -> summarize -> strata -----------

LANE_ONE = (
    ("Run", "robot_msgs streams", "target · state · plan"),
    ("Metric", "frozen config, state fold", "update(state, chunk)"),
    ("Value", "SI inside, unit at display", "values(state)"),
    ("Objective", "≤ 1 mm on 100 % of stops", "outside compute()"),
    ("Verdict", "pass · fail · not_run", "judge(stratum)"),
)
LANE_TWO = (
    ("Runs × N", "labels: station, rung", "real · sim · replay"),
    ("summarize", "fold each run, merge", "by=station, rung"),
    ("Stratum", "values · tails · shares", "n, p95, worst"),
    ("Gate", "every objective passes", "Gate.judge()"),
    ("Ladder", "a verdict per cell", "highest_passing()"),
)


def _box(x: float, y: float, width: float, height: float, lines: tuple[str, str, str], accent: bool) -> str:
    title, line_two, line_three = (html.escape(line) for line in lines)
    cx = x + width / 2
    return (f"<rect class='dg-box{' dg-accent' if accent else ''}' x='{x}' y='{y}' width='{width}' height='{height}' "
            f"rx='10'/>"
            f"<text class='dg-title' x='{cx}' y='{y + 24}'>{title}</text>"
            f"<text class='dg-line' x='{cx}' y='{y + 44}'>{line_two}</text>"
            f"<text class='dg-mono' x='{cx}' y='{y + 63}'>{line_three}</text>")


def _arrow(x1: float, y1: float, x2: float, y2: float, dashed: bool = False) -> str:
    return (f"<line class='dg-arrow{' dg-dash' if dashed else ''}' x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' "
            f"marker-end='url(#dg-head)'/>")


def diagram(vertical: bool) -> str:
    marker = ("<defs><marker id='dg-head' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
              "orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' class='dg-head'/></marker></defs>")
    parts = []
    if not vertical:
        width, height, box_w, box_h, gap = 1100, 318, 192, 76, 32
        for lane, (label, boxes, top) in enumerate((("one run · offline or live", LANE_ONE, 36),
                                                     ("many runs · a deployment", LANE_TWO, 226))):
            parts.append(f"<text class='dg-lane' x='8' y='{top - 14}'>{label}</text>")
            for index, lines in enumerate(boxes):
                x = 8 + index * (box_w + gap)
                parts.append(_box(x, top, box_w, box_h, lines, accent=(index == 1)))
                if index:
                    parts.append(_arrow(x - gap + 2, top + box_h / 2, x - 3, top + box_h / 2))
        middle = 8 + (box_w + gap) + box_w / 2
        parts.append(_arrow(middle, 36 + box_h + 4, middle, 226 - 4, dashed=True))
        parts.append(f"<text class='dg-note' x='{middle + 10}' y='{36 + box_h + 58}'>the same fold, per run, "
                     f"then merge states</text>")
    else:
        width, box_w, box_h, gap = 360, 300, 76, 26
        x = (width - box_w) / 2
        y = 30
        for lane, (label, boxes) in enumerate((("one run · offline or live", LANE_ONE),
                                                ("many runs · a deployment", LANE_TWO))):
            parts.append(f"<text class='dg-lane' x='{x}' y='{y - 10}'>{label}</text>")
            for index, lines in enumerate(boxes):
                if index:
                    parts.append(_arrow(width / 2, y - gap + 3, width / 2, y - 3))
                parts.append(_box(x, y, box_w, box_h, lines, accent=(index == 1)))
                y += box_h + gap
            y += 30
        height = y
    return (f"<svg class='diagram {'tall' if vertical else 'wide'}' viewBox='0 0 {width} {height}' role='img' "
            f"aria-label='A run goes into a metric, whose state folds chunk by chunk; compute gives a value with a "
            f"unit; an objective outside compute turns it into a verdict. Many runs are summarized by station and "
            f"rung into strata that gates judge.'>{marker}{''.join(parts)}</svg>")


# ---- the catalog -------------------------------------------------------------------------------------------------

def field_comments(cls: type) -> dict[str, str]:
    """The inline comment on each config field's line in the class source (``statistic: str = "rms"  # rms | ...``)."""

    comments = {}
    for line in inspect.getsource(cls).splitlines():
        match = re.match(r"\s+(\w+): [^=]+=[^#]+#\s*(.+)$", line)
        if match:
            comments[match.group(1)] = match.group(2).strip()
    return comments


def config_rows(cls: type) -> str:
    comments = field_comments(cls)
    rows = []
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            default = repr(field.default)
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            default = repr(field.default_factory())  # type: ignore[misc]
        else:
            default = "required"
        choice = CHOICES.get((cls.__name__, field.name), "")
        note = comments.get(field.name, "")
        detail = " — ".join(part for part in (choice, html.escape(note)) if part)
        rows.append(f"<tr><td><code>{field.name}</code></td><td><code>{html.escape(default)}</code></td>"
                    f"<td>{detail}</td></tr>")
    return "".join(rows)


def plain_signature(function) -> str:
    """``(target, measured, **config)``: the parameters' names only (the annotations are all Any)."""

    names = []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind is parameter.VAR_KEYWORD:
            names.append(f"**{parameter.name}")
        elif parameter.default is not parameter.empty:
            names.append(f"{parameter.name}={parameter.default!r}")
        else:
            names.append(parameter.name)
    return f"({', '.join(names)})"


def short(metric) -> str:
    """A metric as its base and the config fields that differ from the defaults: the identity, minus the noise."""

    changed = []
    for field in dataclasses.fields(metric):
        value = getattr(metric, field.name)
        default = field.default if field.default is not dataclasses.MISSING else field.default_factory()  # type: ignore[misc]
        if value != default:
            shown = f"{value:.4g}" if isinstance(value, float) else repr(value)
            changed.append(f"{field.name}={shown}")
    return f"{type(metric).__name__}({', '.join(changed)})"


def used_by(parts: list) -> dict[str, list[str]]:
    """Each base's aliases in the catalog, the part's own rows only (an assembly's child rows are the child's)."""

    found: dict[str, list[str]] = {}
    for part in parts:
        for alias in part.aliases:
            found.setdefault(type(alias.metric).__name__, []).append(f"{part.name}: {alias.name}")
    return found


def catalog_card(cls: type, cell: dict, used: list[str], group: str) -> str:
    name = cls.__name__
    function = FUNCTION[name]
    assert hasattr(functional, function), function
    signature = plain_signature(getattr(functional, function))
    live = cls.live if isinstance(cls.live, bool) else True
    value_type = "number · bool (statistic=\"all\")" if name == "StopStat" else cls.value_type
    references = "".join(f"<li>{esc(reference)}</li>" for reference in cls.references)
    used_text = " ".join(f"<span class='tag'>{esc(item)}</span>" for item in used) if used else (
        "<span class='muted'>none yet: configure it in your part's metrics</span>")
    image = f"catalog-{name.lower()}.png"
    caption = "a sealed sim's held-out runs" if name == "PerformanceGain" else "CAN 11's real round-8 runs"
    return f"""
<article class="card" id="metric-{name.lower()}">
  <p class="eyebrow">{esc(group)}</p>
  <header><h4>{name}</h4><span class="chip {'live' if live else 'end'}">{'runs live' if live else 'end of run'}</span>
    <span class="chip">view: {esc(cls.hint)}</span></header>
  <p class="desc">{esc(cls.description)}</p>
  {figure(image, f"Its stock view, drawn by <code>plot()</code> ({caption})")}
  <div class="io">
    <div><b>In</b> <code>{cls.Input.__name__}</code> · {INPUTS[cls.Input.__name__]}<br><span class="muted">Reads
      {READS[name]}.</span></div>
    <div><b>Out</b> {esc(value_type)} · {OUTPUTS[name]}.<br><span class="muted">State: {esc(STATE[name])}.</span></div>
  </div>
  <p class="twin"><b>Function</b> <code>functional.{function}{esc(signature)}</code></p>
  <p class="twin"><b>In the catalog</b> {used_text}</p>
  <details><summary>Config: {len(dataclasses.fields(cls))} fields</summary><div class="scroll"><table class="config">
    <thead><tr><th>field</th><th>default</th><th>values</th></tr></thead><tbody>{config_rows(cls)}</tbody></table>
    </div></details>
  <details><summary>The code that drew the picture ({cell['seconds']} s)</summary>{notebook(cell)}</details>
  <details><summary>References</summary><ul>{references}</ul></details>
</article>"""


def alias_table(part) -> tuple[str, int, int]:
    """The part's own aliases (an assembly's child rows are listed under the child), each with the number of the
    state it reads in the part's full collection, so an alias that shares a child's state says so."""

    collection = part.collection()
    state_numbers: dict[tuple, int] = {}
    for name in collection.aliases:
        state_numbers.setdefault(collection.key_of[name], len(state_numbers) + 1)
    rows = []
    for name, alias in collection.aliases.items():
        if "/" in name and any(name.startswith(child.name + "/") for child in part.children) and part.part_type != \
                "rigid_body_chain":
            continue
        number = state_numbers[collection.key_of[name]]
        live = "live" if alias.metric.live else "end of run"
        signal = "/".join(alias.signal)
        scale = f" ×{alias.scale:.4g}" if alias.scale not in (1.0,) else ""
        rows.append(f"<tr><td><b>{esc(name)}</b></td><td>{esc(alias.unit or '—')}{esc(scale)}</td>"
                    f"<td><code>{esc(short(alias.metric))}</code></td><td><code>{esc(signal)}</code></td>"
                    f"<td>{esc(alias.across)}</td><td>{live}</td><td class='num'>#{number}</td>"
                    f"<td class='muted note'>{esc(alias.note)}</td></tr>")
    table = (f"<div class='scroll'><table class='aliases'><thead><tr><th>alias</th><th>unit · SI→unit</th>"
             f"<th>= base + frozen config</th><th>reads</th><th>across runs</th><th>runs</th><th>state</th>"
             f"<th>note</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    return table, len(collection.aliases), collection.computations


# ---- the build ---------------------------------------------------------------------------------------------------

def run_cells(cells: list[tuple[str, str, str]], namespace: dict) -> Session:
    session = Session(namespace)
    for label, caption, code in cells:
        started = time.perf_counter()
        session.cell(label, caption, code)
        print(f"  {label}: {time.perf_counter() - started:.1f} s", flush=True)
    return session


def finite(value):
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, float) and value in (float("inf"), float("-inf")):
        return None
    return value


def summaries(pendulum: Session, fivebar: Session, ur5e: Session) -> dict:
    """The curated small summaries published beside the page (never the recordings)."""

    gates = pendulum.namespace["gates"]
    pendulum_rows = []
    for station, strata in sorted(pendulum.namespace["strata_by_station"].items()):
        for (_, rung), stratum in sorted(strata.items()):
            tail = stratum.tails["tip stop error"]
            pendulum_rows.append({
                "station": station, "rung": rung, "runs": stratum.values["runs"],
                "tracking_rms_deg": finite(stratum.values["servo/tracking RMS"]),
                "tracking_peak_deg": finite(stratum.values["servo/tracking peak"]),
                "tip_stop_mm": {"n": tail.n, "median": finite(tail.median), "p95": finite(tail.p95),
                                "worst": finite(tail.worst)},
                "tip_settle_s": finite(stratum.values["tip settle"]),
                "lag_ms": finite(stratum.values["servo/lag"]),
                "limiter_share_pct": finite(stratum.values["servo/limiter share"]),
                "verdicts": {name: gate.judge(stratum).verdict for name, gate in gates.items()},
            })
    fivebar_rows = []
    for source in ("real", "sim"):
        for (rung,), stratum in sorted(fivebar.namespace[source].items()):
            fivebar_rows.append({"source": source, "rung": rung,
                                 **{name: finite(value) for name, value in stratum.values.items()}})
    ur5e_rows = []
    for (controller, speed), stratum in sorted(ur5e.namespace["strata"].items()):
        ur5e_rows.append({"controller": controller, "speed": speed,
                          **{name: finite(value) for name, value in stratum.values.items()}})
    return {
        "pendulum-round8.json": {"what": "1-DOF test rack, round 8 (2026-11-15/17): real plays summarized by station "
                                         "and rung with robot_metrics " + VERSION, "units": {"tracking": "deg",
                                         "tip stop": "mm", "settle": "s", "lag": "ms", "limiter share": "%"},
                                 "rows": pendulum_rows},
        "fivebar-ladder.json": {"what": "bam_fb five-bar speed ladder (2026-11-10), real and its sim twin, by rung",
                                "units": {"j2/c1 tracking": "deg", "tool": "mm"}, "rows": fivebar_rows},
        "ur5e-sim.json": {"what": "SIMULATED: MuJoCo Menagerie UR5e, pick-and-place, two controllers x three speeds",
                          "units": {"joints": "deg", "tool": "mm", "lag": "ms"}, "rows": ur5e_rows},
    }


CSS = """
:root { --bg:#fbfaf7; --panel:#ffffff; --ink:#1d1f23; --muted:#5d6470; --line:#e3e0d8; --accent:#2f6f4f;
  --accent-soft:#e8f2ec; --warn:#9a5b00; --warn-soft:#fbf0dc; --bad:#b03a2e; --code:#f3f1ec; --out-bg:#f3f1ec;
  --out-ink:#4f5661; --out-rule:#d3cec2; --paper:#ffffff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#15171a; --panel:#1d2024; --ink:#e7e5e0;
  --muted:#a3a9b2; --line:#33373d; --accent:#7cc39b; --accent-soft:#1f3329; --warn:#f0b35a; --warn-soft:#3a2d17;
  --bad:#ec8b7f; --code:#24272c; --out-bg:#17191c; --out-ink:#a3a9b2; --out-rule:#3d4249; --paper:#f4f2ee; } }
:root[data-theme="dark"] { --bg:#15171a; --panel:#1d2024; --ink:#e7e5e0; --muted:#a3a9b2; --line:#33373d;
  --accent:#7cc39b; --accent-soft:#1f3329; --warn:#f0b35a; --warn-soft:#3a2d17; --bad:#ec8b7f; --code:#24272c;
  --out-bg:#17191c; --out-ink:#a3a9b2; --out-rule:#3d4249; --paper:#f4f2ee; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; background:var(--bg); color:var(--ink); font:15.5px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
a { color:var(--accent); }
.wrap { max-width:1200px; margin:0 auto; padding:0 16px; }
header.hero { padding:26px 0 14px; border-bottom:1px solid var(--line); }
.eyebrow { color:var(--muted); font-size:13px; letter-spacing:.02em; margin:0 0 6px; }
h1 { font-size:32px; margin:0 0 4px; letter-spacing:-.01em; }
h1 code { background:none; padding:0; font-size:inherit; }
.lede { font-size:17px; max-width:900px; margin:0 0 10px; }
.links { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 0; }
.links a { text-decoration:none; border:1px solid var(--line); background:var(--panel); padding:5px 12px;
  border-radius:999px; font-size:14px; }
nav.toc { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); }
nav.toc .wrap { display:flex; gap:18px; overflow-x:auto; white-space:nowrap; padding-top:9px; padding-bottom:9px; }
nav.toc a { text-decoration:none; color:var(--muted); font-size:14px; } nav.toc a:hover { color:var(--ink); }
section { padding:22px 0 8px; }
h2 { font-size:24px; margin:8px 0 6px; } h3 { font-size:18px; margin:26px 0 8px; } h4 { font-size:16px; margin:0; }
.section-lede { color:var(--muted); max-width:860px; margin:0 0 14px; }
.split { display:grid; grid-template-columns:minmax(0,5fr) minmax(0,7fr); gap:20px; align-items:start; }
.glance { display:grid; grid-template-columns:minmax(0,1fr); gap:10px; margin:2px 0 10px; }
.glance > div { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:10px 14px; }
.glance-code h2 { margin-top:0; }
.glance b.k { display:block; font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--accent); }
.glance p { margin:4px 0 0; font-size:14.5px; }
svg.diagram { width:100%; height:auto; display:block; margin:10px 0 6px; }
svg.diagram.tall { display:none; max-width:420px; margin:10px auto; }
.dg-box { fill:var(--panel); stroke:var(--line); stroke-width:1.2; }
.dg-accent { stroke:var(--accent); stroke-width:1.8; }
.dg-title { fill:var(--ink); font:600 15px system-ui,sans-serif; text-anchor:middle; }
.dg-line { fill:var(--muted); font:13px system-ui,sans-serif; text-anchor:middle; }
.dg-mono { fill:var(--ink); font:12px ui-monospace,SFMono-Regular,Menlo,monospace; text-anchor:middle; }
.dg-lane { fill:var(--accent); font:600 12px system-ui,sans-serif; text-transform:uppercase; letter-spacing:.06em; }
.dg-note { fill:var(--muted); font:italic 12.5px system-ui,sans-serif; }
.dg-arrow { stroke:var(--muted); stroke-width:1.5; } .dg-dash { stroke-dasharray:5 4; } .dg-head { fill:var(--muted); }
.two { display:grid; grid-template-columns:minmax(0,1.25fr) minmax(0,1fr); gap:18px; align-items:start; }
.cell-caption { color:var(--muted); margin:4px 0 8px; font-size:14.5px; }
.notebook { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; margin:6px 0 16px; }
.notebook pre { margin:0; border:0; border-radius:0; overflow-x:auto; max-width:100%; }
pre { font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre; }
pre.src { background:var(--panel); color:var(--ink); padding:8px 14px; }
pre.src.gap { padding-top:20px; }
pre.out { background:var(--out-bg); color:var(--out-ink); border-left:3px solid var(--out-rule);
  padding:5px 14px 6px 11px; font-size:12.2px; }
pre.out.wrap { white-space:pre-wrap; overflow-wrap:anywhere; }
code { background:var(--code); padding:1px 5px; border-radius:4px; font:12.8px ui-monospace,SFMono-Regular,Menlo,monospace; }
pre code { background:none; padding:0; }
table { border-collapse:collapse; width:100%; background:var(--panel); font-size:14px; }
th, td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
th { background:var(--code); font-weight:600; } td.num { font-variant-numeric:tabular-nums; white-space:nowrap; }
.scroll { overflow-x:auto; margin:8px 0 14px; border-radius:8px; }
.api td:first-child { white-space:nowrap; font-weight:600; }
.group { margin:22px 0 6px; } .group h3 { margin:0 0 2px; } .group p { color:var(--muted); margin:0 0 10px; }
.cards { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; margin-top:14px; }
.card .eyebrow { margin:0 0 2px; color:var(--accent); text-transform:uppercase; font-size:11.5px; letter-spacing:.06em;
  font-weight:600; }
.tag { display:inline-block; font-size:12.5px; background:var(--code); border-radius:6px; padding:1px 7px; margin:2px 2px 0 0; }
table.overview td { font-size:14px; } table.overview td:first-child { width:30%; }
table.aliases td.note { min-width:220px; font-size:13px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; min-width:0; }
.card header { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:4px; }
.card header h4 { font:600 16px ui-monospace,SFMono-Regular,Menlo,monospace; margin-right:auto; }
.chip { font-size:12px; border-radius:999px; padding:2px 9px; background:var(--code); color:var(--muted); }
.chip.live { background:var(--accent-soft); color:var(--accent); } .chip.end { background:var(--warn-soft); color:var(--warn); }
.chip.real { background:var(--accent-soft); color:var(--accent); font-weight:600; }
.chip.sim { background:var(--warn-soft); color:var(--warn); font-weight:600; }
.desc { margin:4px 0 10px; }
.io { display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:14px; margin:0 0 10px; }
.io > div { background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
table.config { font-size:13px; margin:0 0 8px; } table.config td:first-child, table.config td:nth-child(2) { white-space:nowrap; }
.twin { font-size:14px; margin:6px 0; overflow-wrap:anywhere; }
.muted { color:var(--muted); }
figure { margin:12px 0; background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:8px; }
figure img { width:100%; height:auto; display:block; }
figcaption { color:#555b63; font-size:13px; padding:6px 4px 0; }
figcaption code { background:#ecebe6; color:#30343a; }
details { margin:8px 0; } summary { cursor:pointer; font-weight:600; font-size:14px; color:var(--accent); }
.example { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:16px 18px; margin:18px 0 26px; }
.example > header { display:flex; flex-wrap:wrap; align-items:center; gap:10px; }
.example > header h3 { margin:0; margin-right:auto; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:12px 0; }
.tile { background:var(--bg); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.tile b { display:block; font-size:20px; } .tile span { color:var(--muted); font-size:13px; }
.figs { display:grid; grid-template-columns:minmax(0,1fr); gap:12px; }
.figs figure img { max-width:920px; margin:0 auto; }
.parts { display:grid; gap:4px; }
.install pre { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 12px; overflow-x:auto; }
footer { border-top:1px solid var(--line); margin-top:30px; padding:18px 0 40px; color:var(--muted); font-size:13.5px; }
@media (max-width: 900px) {
  .split, .two, .cards, .figs { grid-template-columns:minmax(0,1fr); }
}
@media (max-width: 640px) {
  h1 { font-size:28px; } .lede { font-size:16.5px; }
  .io { grid-template-columns:minmax(0,1fr); } svg.diagram.wide { display:none; } svg.diagram.tall { display:block; }
  .example { padding:12px; } .card { padding:12px; }
}
"""


def api_table() -> str:
    rows = (
        ("one recording", "<code>collection.fold(run)</code> · <code>metric(target, measured)</code>",
         "a closed state · a value in SI"),
        ("a live stream", "<code>state = collection.update(state, chunk)</code>, then <code>close(state)</code>",
         "the same state as offline"),
        ("the values", "<code>collection.values(state)</code>", "{alias: value} in each alias's display unit"),
        ("many runs", "<code>summarize(collection, runs, by=(\"station\", \"rung\"))</code>",
         "strata: values, tails (n, median, p95, worst), Wilson shares"),
        ("a target", "<code>Objective(alias, \"&lt;=\", 1.0, share=1.0).judge(stratum)</code>",
         "a Verdict: pass · fail · not_run, with its owner"),
        ("a picture", "<code>collection.view(alias, states)</code> → <code>plot()</code> · W&amp;B · Rerun",
         "(fig, ax), a W&amp;B payload, an .rrd"),
        ("one call", "<code>functional.error_stat(target, measured, statistic=\"rms\")</code>",
         "the class run once: an SI number"),
        ("a stored value", "<code>collection.records(state, run)</code>",
         "MetricRecord: value, unit, base, config, version"),
    )
    body = "".join(f"<tr><td>{have}</td><td>{call}</td><td>{get}</td></tr>" for have, call, get in rows)
    return (f"<div class='scroll'><table class='api'><thead><tr><th>you have</th><th>you call</th><th>you get</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>")


def example_section(anchor: str, title: str, chip: str, chip_class: str, intro: str, tiles: list[tuple[str, str]],
                    figures: list[tuple[str, str]], session: Session, data_file: str) -> str:
    tile_html = "".join(f"<div class='tile'><b>{esc(value)}</b><span>{esc(label)}</span></div>" for value, label in tiles)
    figure_html = "".join(figure(name, caption) for name, caption in figures)
    cells = "".join(f"<h4 class='cell-title'>{index}. {esc(cell['label'])}</h4>"
                    f"<p class='cell-caption'>{esc(cell['caption'])}</p>{notebook(cell)}"
                    for index, cell in enumerate(session.cells, start=1))
    return f"""
<div class="example" id="{anchor}">
  <header><h3>{title}</h3><span class="chip {chip_class}">{chip}</span></header>
  <p>{intro}</p>
  <div class="tiles">{tile_html}</div>
  <div class="figs">{figure_html}</div>
  <details open><summary>The code, as it ran ({sum(float(cell['seconds']) for cell in session.cells):.1f} s)</summary>
    {cells}</details>
  <p class="muted">Numbers as data: <a href="data/{data_file}">data/{data_file}</a></p>
</div>"""


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    IMG.mkdir(parents=True)
    (OUT / "data").mkdir()
    namespace_base = {"MEDIA": IMG, "TESTS_DATA": str(ROOT / "tests" / "data")}

    print("glance", flush=True)
    glance = run_cells([gallery.GLANCE], dict(namespace_base))
    print("catalog", flush=True)
    catalog_session = run_cells([gallery.CATALOG_SETUP], dict(namespace_base))
    for name, code in gallery.CATALOG.items():
        catalog_session.cell(name, "", code)
    catalog_cells = {cell["label"]: cell for cell in catalog_session.cells}
    print("pendulum", flush=True)
    pendulum = run_cells(gallery.PENDULUM, dict(namespace_base))
    print("five-bar", flush=True)
    fivebar = run_cells(gallery.FIVEBAR, dict(namespace_base))
    print("ur5e", flush=True)
    ur5e = run_cells(gallery.UR5E, dict(namespace_base))

    for file_name, payload in summaries(pendulum, fivebar, ur5e).items():
        (OUT / "data" / file_name).write_text(json.dumps(payload, indent=1, default=str) + "\n", encoding="utf-8")

    parts = [catalog.integrated_servo(), catalog.pendulum_arm(tip_length_m=0.339), catalog.rigid_chain(("j2", "c1"))]
    used = used_by(parts)
    stock_classes = {name: getattr(robot_metrics, name) for _, _, names in GROUPS for name in names}
    missing = {cls.__name__ for cls in vars(stock).values() if isinstance(cls, type)
               and issubclass(cls, robot_metrics.Metric) and cls is not robot_metrics.Metric} - set(stock_classes)
    assert not missing, f"stock metrics missing from the catalog: {missing}"

    cards, overview, counts = [], [], []
    for title, blurb, names in GROUPS:
        counts.append(f"{title} {len(names)}")
        for index, name in enumerate(names):
            cls = stock_classes[name]
            cards.append(catalog_card(cls, catalog_cells[name], used.get(name, []), title))
            group_cell = (f"<td rowspan='{len(names)}'><b>{title}</b><br><span class='muted'>{blurb}</span></td>"
                          if index == 0 else "")
            live = "live" if cls.live else "end of run"
            overview.append(f"<tr>{group_cell}<td><a href='#metric-{name.lower()}'><code>{name}</code></a></td>"
                            f"<td><code>{cls.Input.__name__}</code></td><td>{OUT_SHORT[name]}</td><td>{live}</td><td><code>functional.{FUNCTION[name]}</code></td></tr>")
    group_html = [f"<div class='scroll'><table class='overview'><thead><tr><th>group</th><th>metric</th><th>in</th>"
                  f"<th>out</th><th>runs</th><th>function</th></tr></thead><tbody>{''.join(overview)}</tbody></table>"
                  f"</div><div class='cards'>{''.join(cards)}</div>"]

    alias_html = []
    for part, intro in zip(parts, (
            "The servo judged at its output: the plan at each reply's instant vs the reply.",
            "An assembly: the servo plus a 339 mm lever. Every tip row is the servo's state read through the lever "
            "(<code>Alias.through(..., scale_by=\"lever_m\")</code>), so the arm adds no computation of its own.",
            "A rigid chain (the five-bar, or any serial arm): each joint's RMS and the tool's position error from two "
            "PoseTrajectories.")):
        table, alias_count, state_count = alias_table(part)
        children = "".join(f" + the {child.name}'s {len(child.collection().aliases)}" for child in part.children
                           if part.part_type != "rigid_body_chain")
        mix = f" (its {len(part.aliases)}{children})" if children else ""
        alias_html.append(f"<h3><code>{part.part_type}</code> <span class='muted'>· {alias_count} aliases over "
                          f"{state_count} states{mix}</span></h3><p class='section-lede'>{intro}</p>{table}")
    gates = catalog.pendulum_gates()
    gate_rows = "".join(f"<tr><td><b>{esc(name)}</b></td><td>{esc(objective.title)}</td><td>{esc(objective.on)}</td>"
                        f"<td class='muted'>{esc(objective.owner)}</td></tr>"
                        for name, gate in gates.items() for objective in gate.objectives)

    # ---- tiles from the namespaces (nothing typed by hand) -------------------------------------------------------
    pendulum_ns, fivebar_ns, ur5e_ns = pendulum.namespace, fivebar.namespace, ur5e.namespace
    every_stop = robot_metrics.highest_passing(pendulum_ns["passed_by_gate"]["tip stop (every stop)"])
    tracks = robot_metrics.highest_passing(pendulum_ns["passed_by_gate"]["tracks"])
    can11_worst = max(stratum.tails["tip stop error"].worst for stratum in pendulum_ns["can11"].values())
    fivebar_real = fivebar_ns["real"]
    fivebar_best = min(stratum.values["tool tracking RMS"] for stratum in fivebar_real.values())
    fivebar_worst = max(stratum.values["tool worst"] for stratum in fivebar_real.values())
    ur5e_strata = ur5e_ns["strata"]
    pd_rms = ur5e_strata[("pd", 1.0)].values["tool tracking RMS"]
    ff_rms = ur5e_strata[("pd+ff", 1.0)].values["tool tracking RMS"]
    ff_pass = [key for key, stratum in ur5e_strata.items() if ur5e_ns["tool_gate"].judge(stratum).verdict == "pass"]

    pendulum_html = example_section(
        "pendulum", "1-DOF pendulum · six stations", "REAL · round 8", "real",
        "The 1-DOF test rack: six integrated servos, each swinging a lever, each played on a six-rung speed ladder "
        "(five repeats plus two held-out plays per rung). Recordings stay private; the page shows the code that "
        "scored them, its output, and the summary as data.",
        [(str(len(pendulum_ns["runs"])), "real plays scored"),
         (f"{len(pendulum_ns['stations'])} × 6", "stations × rungs"),
         (str(sum(1 for rung in every_stop.values() if rung >= 1.0)), "stations with every stop ≤ 1 mm up to rung 1.0"),
         (f"{can11_worst:.2f} mm", "CAN 11's worst stop at the tip")],
        [("pendulum-can11-tip-stops.png", "Every stop at CAN 11's tip by rung, against 1 mm: the tail the median "
          "hides."), ("pendulum-tracking-by-rung.png", "Tracking RMS by rung for every station, against the 0.5° "
          "gate.")], pendulum, "pendulum-round8.json")
    fivebar_html = example_section(
        "fivebar", "Five-bar arm · real and its sim twin", "REAL + SIM · 2026-11-10", "real",
        "The bam_fb five-bar (two motors, a closed chain) on a five-rung tool-speed ladder, played on the real arm "
        "and on its sim twin. The catalog's rigid chain scores both with the same identities.",
        [(str(len(fivebar_ns["real_runs"])), "real rungs"), (str(len(fivebar_ns["sim_runs"])), "sim rungs"),
         (f"{fivebar_best:.1f} mm", "best real tool tracking RMS"),
         (f"{fivebar_worst:.1f} mm", "worst real tool error")],
        [("fivebar-tool-rms-by-speed.png", "Tool tracking RMS against tool speed, real and sim."),
         ("fivebar-tool-path.png", "One rung's tool path in the plane, commanded and measured.")],
        fivebar, "fivebar-ladder.json")
    ur5e_html = example_section(
        "ur5e", "UR5e · six joints, simulated", "SIMULATED · MuJoCo", "sim",
        "No real UR5e run exists: this arm is MuJoCo Menagerie's UR5e model under its own PD position servos "
        "(<code>website/ur5e_sim.py</code>), following a joint-space pick-and-place at three speeds, as shipped and "
        "with model feed-forward. It shows a six-joint arm going in as plain robot_msgs streams.",
        [(str(len(ur5e_ns["sweep"])), "simulated runs"), (f"{pd_rms:.0f} mm", "tool RMS, PD as shipped, full speed"),
         (f"{ff_rms:.2f} mm", "tool RMS with feed-forward, full speed"),
         (f"{len(ff_pass)} of {len(ur5e_strata)}", "cells with the worst tool error ≤ 1 mm")],
        [("ur5e-tool-rms-by-speed.png", "Tool tracking RMS by speed, both controllers (log scale). SIMULATED."),
         ("ur5e-shoulder-lift-stops.png", "Every stop of the shoulder lift joint at full speed. SIMULATED.")],
        ur5e, "ur5e-sim.json")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>robot_metrics</title>
<meta name="description" content="Performance metrics for robot parts: torchmetrics' shape with numpy inside, over
robot_msgs streams, live or offline, with the targets kept outside compute().">
<style>{CSS}
{token_css()}</style></head>
<body>
<header class="hero"><div class="wrap">
  <p class="eyebrow">v{VERSION} · MIT · Python ≥ 3.11 · runtime: numpy + robot_msgs</p>
  <h1><code>robot_metrics</code></h1>
  <p class="lede">Performance metrics for robot parts. Hand it robot_msgs streams (a whole recording, or a live stream
  chunk by chunk) and it folds them into values with units. A metric is math plus a frozen config; the target stays
  outside, where an objective turns a value into a verdict.</p>
  <div class="links"><a href="{REPO}">GitHub</a><a href="#install">Install</a><a href="#catalog">Catalog</a>
  <a href="#gallery">Gallery</a><a href="{REPO}/blob/main/docs/peps/0001-robot-metrics-v0.md">Design (PEP 0001)</a></div>
</div></header>
<nav class="toc"><div class="wrap"><a href="#glance">At a glance</a><a href="#api">API</a><a href="#catalog">Catalog</a>
<a href="#aliases">Aliases</a><a href="#gallery">Gallery</a><a href="#install">Install</a></div></nav>
<main class="wrap">

<section id="glance">
  <div class="split">
    <div class="glance">
      <div><b class="k">What goes in</b><p>robot_msgs streams, named in a <code>Run</code>: <code>target</code> and
        <code>state</code> (<code>JointTrajectory</code>), the <code>plan</code> on its own ticks, a tool's
        <code>PoseTrajectory</code>, plus labels (station, rung). A whole recording, or the same run in chunks.</p></div>
      <div><b class="k">The functional core</b><p>A metric is a frozen dataclass: identity = base + config. Its state is a
        value: <code>empty</code> → <code>update</code> → <code>merge</code> → <code>close</code> →
        <code>compute</code>. Nothing hidden is mutated; a plain function sits beside each class.</p></div>
      <div><b class="k">Offline = online</b><p>Live is the fold, chunk by chunk; offline is the same fold once. A
        stop split across two chunks is found exactly. Runs merge into rungs, stations, a deployment.</p></div>
      <div><b class="k">What comes out</b><p>Values in SI inside, shown in each alias's unit (deg, mm, ms, %); every
        stop as an occurrence; a record with identity and version. Targets live on <code>Objective</code>s, outside
        <code>compute()</code>: pass · fail · not_run.</p></div>
    </div>
    <div class="glance-code"><h2>{esc(glance.cells[0]['label'])}</h2>
      <p class="cell-caption">{esc(glance.cells[0]['caption'])}</p>{notebook(glance.cells[0])}</div>
  </div>
  {diagram(vertical=False)}{diagram(vertical=True)}
  <div class="two" id="api">
    <div><h3>The API in one table</h3>{api_table()}</div>
    <div><h3>Math only, meaning from the data</h3><p class="cell-caption">A stock metric is math: one
      <code>ErrorStat</code> serves servo tracking, the tool error of a five-bar and a feed-forward residual, by the
      streams its alias hands it. A new row is a configured stock metric, never a new function; aliases with one
      identity compute once.</p>
      <h3>Layering</h3><p class="cell-caption"><code>robot_msgs</code> ← <code>robot_metrics</code> ← your tools
      (diagnostics, sim-to-real, a part catalog). The core imports numpy, robot_msgs and the stdlib; matplotlib,
      wandb and rerun-sdk only inside <code>plot()</code> and the adapters.</p></div>
  </div>
</section>

<section id="catalog">
  <h2>Catalog <span class="muted">· {sum(len(names) for _, _, names in GROUPS)} stock metrics</span></h2>
  <p class="section-lede">Every stock metric, grouped. Each card: its config, what it reads, what it returns,
  whether it runs live, its function twin, and its stock view, drawn by <code>plot()</code> on CAN 11's real
  round-8 runs (the performance gain on a sealed sim). {esc(' · '.join(counts))}.</p>
  {''.join(group_html)}
</section>

<section id="aliases">
  <h2>Aliases, per part type</h2>
  <p class="section-lede">A semantic name is an alias: a label, a display unit and the streams it reads, over a
  configured stock metric. Its identity is the metric's, so aliases with one identity share one state, and every
  <code>StopStat</code> on one signal shares one pass of the stop kernel (the <b>state</b> column).</p>
  {''.join(alias_html)}
  <h3>Gates <span class="muted">· objectives outside compute()</span></h3>
  <p class="section-lede">Round 8's three verdicts per (station, rung) cell. Each objective names its owner; a new
  target recomputes nothing.</p>
  <div class="scroll"><table><thead><tr><th>gate</th><th>objective</th><th>on</th><th>owner</th></tr></thead>
  <tbody>{gate_rows}</tbody></table></div>
</section>

<section id="gallery">
  <h2>Gallery</h2>
  <p class="section-lede">Three arms, each scored by the builder for real. v1: the errors are what they are; the
  point is to see where each arm stands. The 1-DOF pendulum and the five-bar are real recordings; the UR5e is a
  simulation, labelled as one.</p>
  <nav class="parts"><a href="#pendulum">1-DOF pendulum · real</a><a href="#fivebar">Five-bar arm · real + sim</a>
  <a href="#ur5e">UR5e · simulated</a></nav>
  {pendulum_html}{fivebar_html}{ur5e_html}
</section>

<section id="install" class="install">
  <h2>Install</h2>
  <p class="section-lede">Not on PyPI yet: install the tagged release from GitHub. The runtime needs numpy and
  robot_msgs; the extras add <code>plot()</code> (viz), the W&amp;B payload and the Rerun writer.</p>
  <pre>pip install "robot-metrics[viz] @ git+{REPO}@v{VERSION}"</pre>
  <pre>git clone {REPO} &amp;&amp; cd robot_metrics &amp;&amp; uv run --extra test pytest -q</pre>
  <p class="muted">The tests that read the private recordings skip without them. This page is built by
  <code>website/build_site.py</code>; every number and picture on it came from that run.</p>
</section>
</main>
<footer><div class="wrap">robot_metrics v{VERSION} · MIT © 2026 Zach Yamaoka · built {time.strftime('%Y-%m-%d')} from
<a href="{REPO}">{esc(REPO.removeprefix('https://'))}</a> · the UR5e model is MuJoCo Menagerie's (BSD-3-Clause), used
at build time, not redistributed.</div></footer>
</body></html>
"""
    published = [page] + [path.read_text(encoding="utf-8") for path in (OUT / "data").glob("*.json")]
    leaks = sorted({match.group(0) for text in published for match in PRIVATE.finditer(text)})
    if leaks:
        raise SystemExit(f"refusing to write: private material in the page or data: {leaks}")
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    print(f"wrote {OUT / 'index.html'} · {len(list(IMG.glob('*.png')))} images · {size / 1e6:.2f} MB in all")


if __name__ == "__main__":
    main()
