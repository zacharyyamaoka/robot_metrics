"""Build the coverage gallery report from real recordings: every card, the four-tool comparison, the numbers.

    taskset -c 2-5,16,17 nice -n 10 /home/bam/robot_metrics/.venv/bin/python /home/bam/robot_metrics/docs/proposals/coverage-2026-11-19/build_gallery.py

Reads: figures_*.py (cards), tools_matplotlib (run here), tools/rerun|viser|wandb screenshots already captured by
tools_rerun.capture(), tools_viser.capture() and the W&B screenshots in the media folder; the hero video from hero.py.
Writes one self-contained page: /home/bam/robot_metrics/reports/media/coverage-gallery-2026-11-19.html (media-heavy, so
in the gitignored reports/media/ per robot_metrics' rule; the builder and the proposal code are what git keeps).
"""

from __future__ import annotations

import ast
import base64
import html
import importlib
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import core_types  # noqa: E402
from cards import SECTIONS, Card  # noqa: E402

DATE = "2026-11-19"
MEDIA = Path("/home/bam/robot_metrics/reports/media")
ASSETS = MEDIA / f"coverage-gallery-{DATE}"
OUT = MEDIA / f"coverage-gallery-{DATE}.html"
TOOLS = [("matplotlib", "Matplotlib", "static figure, for reports and papers"),
         ("rerun", "Rerun", "time-scrubbable log viewer, 2-D + 3-D"),
         ("viser", "Viser", "live 3-D scene in the browser, with GUI"),
         ("wandb", "Weights & Biases", "hosted dashboards, compares runs")]
RERUN_BIN = "/home/bam/robot_metrics/.venv/bin/rerun"
VISER_SERVE = ("taskset -c 2-5,16,17 nice -n 19 uv run --no-project --python /home/bam/robot_metrics/.venv/bin/python --with viser==1.1.1 "
               "--with plotly==6.3.0 python /home/bam/robot_metrics/docs/proposals/coverage-2026-11-19/tools_viser.py --serve")
# WHY a verdict per view: Zach asked to see each type in each tool; the comparison only pays off if it says which tool
# to reach for. Judged from the captures and each tool lane's notes, not from the tools' marketing.
VERDICT = {
    "limit-bars": {"matplotlib": ("best", "limit box, lines, histogram in one static figure"), "rerun": ("good", "bar charts; limits drawn as bars"),
                   "viser": ("good", "via a Plotly panel, not the 3-D scene"), "wandb": ("weak", "no limit lines; a one-step histogram is a strip")},
    "coverage-heatmap": {"matplotlib": ("best", "masked white, grey unreachable, colour bar"), "rerun": ("good", "filled boxes seen from above"),
                         "viser": ("weak", "works, adds nothing over a static map"), "wandb": ("good", "custom Vega chart, no built-in heatmap")},
    "histogram-over-runs": {"matplotlib": ("good", "ridgeline, static"), "rerun": ("best", "scrub the run timeline, histogram redraws"),
                            "viser": ("weak", "a 3-D surface is hard to read"), "wandb": ("best", "native histogram per step")},
    "coverage-growth": {"matplotlib": ("good", "band + Chao2 in one figure"), "rerun": ("good", "scalars on a run timeline"),
                        "viser": ("weak", "uPlot panel; only worth it live"), "wandb": ("best", "native lines, compares across runs and days")},
    "phase-space-3d": {"matplotlib": ("weak", "static 3-D hides the far side"), "rerun": ("best", "3-D cells + samples arriving run by run"),
                       "viser": ("best", "orbit the cells, toggle samples in the GUI"), "wandb": ("weak", "no axis labels in its 3-D viewer")},
    "tool-paths": {"matplotlib": ("good", "top + side spaghetti, static"), "rerun": ("best", "replay cycle by cycle beside the voxels"),
                   "viser": ("good", "paths + toggleable voxels"), "wandb": ("weak", "point cloud, no transparency")},
    "grasp-poses": {"matplotlib": ("good", "wireframes, static"), "rerun": ("best", "wireframes + opening histogram with its limit"),
                    "viser": ("good", "robot_msgs-style grippers; dense zone turns grey"), "wandb": ("weak", "boxes only")},
    "direction-sphere": {"matplotlib": ("best", "equal-area Mollweide map reads in one look"), "rerun": ("good", "bins on the sphere"),
                         "viser": ("good", "bins on the sphere, orbitable"), "wandb": ("good", "sphere + the custom map")},
}
FIGURE_MODULES = ["figures_joint", "figures_arm", "figures_grasp", "figures_dataset"]
VAULT_NOTE = "Projects/robot_msgs/Robot Dataset Coverage — Prior Art, Metrics and Gallery (2026-11-19)"


def esc(text) -> str:
    return html.escape(str(text))


def collect_cards() -> tuple[list[Card], list[str]]:
    cards, missing = [], []
    for name in FIGURE_MODULES:
        if not (HERE / f"{name}.py").is_file():
            missing.append(name)
            continue
        started = time.time()
        module = importlib.import_module(name)
        built = module.cards()
        print(f"{name}: {len(built)} cards in {time.time() - started:.0f} s")
        cards += built
    return cards, missing


def function_source(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text()
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    return None


def imports_of(path: Path) -> str:
    text = path.read_text()
    lines = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            lines.append(ast.get_source_segment(text, node))
    return "\n".join(lines)


def module_notes(path: Path) -> dict:
    """A tool module's NOTES = {type_id: str} literal, read without importing it (viser is not in this venv)."""

    if not path.is_file():
        return {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "NOTES" for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return {}
    return {}


def code_html(source: str) -> str:
    # WHY plain text + the page's tiny highlighter instead of Pygments spans: 32 highlighted functions cost ~350 KB of
    # markup and pushed the page past the desktop preview's ~1.9 MB; the script below colours the same tokens offline.
    return f"<pre class='code'><code>{esc(source)}</code></pre>"


def as_webp(data_uri: str, quality: int = 82) -> str:
    """A card's PNG re-encoded as WebP when that is smaller: keeps the page under the desktop preview's ~1.9 MB."""

    if not data_uri.startswith("data:image/png"):
        return data_uri
    image = Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=quality, method=6)
    webp = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode()
    return webp if len(webp) < len(data_uri) else data_uri


def picture(path: Path, width: int = 720) -> str | None:
    """A screenshot as an inline WebP (lossy, q 78): screenshots are photos of UIs, so WebP is ~5x smaller than PNG."""

    if not path.is_file():
        return None
    image = Image.open(path).convert("RGB")
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=66, method=6)
    return "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode()


def card_html(card: Card) -> str:
    numbers = "".join(f"<div class='kv'><span>{esc(k)}</span><b>{esc(v)}</b></div>" for k, v in card.numbers.items())
    return f"""<figure class="card{' wide' if card.wide else ''}" id="{esc(card.id)}">
<figcaption><div class="ctitle">{esc(card.title)}</div><div class="q">{esc(card.answers)}</div></figcaption>
<img src="{as_webp(card.image)}" alt="{esc(card.title)}" loading="lazy">
{f'<div class="nums">{numbers}</div>' if numbers else ''}
<details><summary>data and prior art</summary><p><b>Data:</b> {esc(card.source)}</p>
{f'<p><b>Idea from:</b> {esc(card.prior_art)}</p>' if card.prior_art else ''}<p class="muted">#{esc(card.id)} · {esc(card.use_case)}</p></details>
</figure>"""


def tools_section() -> tuple[str, dict]:
    matplotlib_tool = importlib.import_module("tools_matplotlib")
    matplotlib_tool.capture()
    wandb_runs = {}
    runs_file = ASSETS / "tools" / "wandb" / "runs.json"
    if runs_file.is_file():
        wandb_runs = json.loads(runs_file.read_text())
    notes = {}
    for tool, _, _ in TOOLS:
        notes_file = ASSETS / "tools" / tool / "notes.json"
        notes[tool] = json.loads(notes_file.read_text()) if notes_file.is_file() else module_notes(HERE / f"tools_{tool}.py")
    status = {tool: 0 for tool, _, _ in TOOLS}
    blocks = []
    for core in core_types.TYPES:
        function = core.id.replace("-", "_")
        columns = []
        for tool, label, _ in TOOLS:
            image = picture(ASSETS / "tools" / tool / f"{core.id}.png")
            source = function_source(HERE / f"tools_{tool}.py", function)
            if image:
                status[tool] += 1
            note = notes[tool].get(core.id, "")
            link = ""
            if tool == "wandb" and core.id in wandb_runs:
                link = f" · <a href='{esc(wandb_runs[core.id]['run_url'])}'>open the run</a>"
            opener = ""
            rrd = ASSETS / "tools" / "rerun" / f"{core.id}.rrd"
            if tool == "rerun" and rrd.is_file():
                opener = f"<p>Open it: <code>{esc(RERUN_BIN)} {esc(rrd)}</code></p>"
            if tool == "viser":
                opener = f"<p>Serve it live: <code>{esc(VISER_SERVE)} {esc(core.id)}</code></p>"
            body = (f"<img src='{image}' alt='{esc(label)} {esc(core.id)}' loading='lazy' class='zoom'>" if image
                    else "<div class='pending'>not captured</div>")
            details = (f"<details><summary>{esc(label)} code and notes</summary>{opener}"
                       f"{f'<p>{esc(note)}</p>' if note else ''}{code_html(source) if source else ''}</details>")
            columns.append(f"<div class='tool'><div class='tname'>{esc(label)}{link}</div>{body}"
                           f"{f'<p class=tnote>{esc(note)}</p>' if note else ''}{details}</div>")
        payload_source = core.data()["source"]
        blocks.append(f"""<section class="type" id="type-{esc(core.id)}"><h3>{esc(core.title)}</h3>
<p class="q">{esc(core.question)} <span class="muted">· {esc(core.dims)} · data: {esc(payload_source)}</span></p>
<div class="tools">{''.join(columns)}</div></section>""")
    head = "".join(f"<th>{esc(label)}</th>" for _, label, _ in TOOLS)
    rows = "".join("<tr><td><a href='#type-" + esc(core.id) + "'>" + esc(core.title.split(':')[0]) + "</a></td>" + "".join(
        f"<td class='{VERDICT[core.id][tool][0]}' title='{esc(VERDICT[core.id][tool][1])}'>{VERDICT[core.id][tool][0]} · "
        f"<span class='muted'>{esc(VERDICT[core.id][tool][1])}</span></td>" for tool, _, _ in TOOLS) + "</tr>"
        for core in core_types.TYPES)
    verdict = (f"<p><b>Which tool for which view:</b> Matplotlib for the report, W&amp;B for watching coverage grow across runs "
               f"and days, Rerun for replaying one dataset in 2-D and 3-D on a timeline, Viser for an interactive 3-D scene with "
               f"controls (not for charts).</p><table class='verdict'><tr><th>View</th>{head}</tr>{rows}</table>")
    imports = "".join(f"<details><summary>{esc(label)} module imports</summary>{code_html(imports_of(HERE / f'tools_{tool}.py'))}</details>"
                      for tool, label, _ in TOOLS if (HERE / f"tools_{tool}.py").is_file())
    return verdict + "".join(blocks) + f"<div class='imports'>{imports}</div>", status


CSS = """
:root{--ink:#1d1d1f;--muted:#6e6e73;--hair:#e5e5ea;--accent:#1f5fa8;--bg:#ffffff;--soft:#f6f7f9;--limit:#c0392b}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
main{max-width:1240px;margin:0 auto;padding:28px 20px 80px} h1{font-size:28px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:20px;margin:48px 0 6px;padding-top:12px;border-top:1px solid var(--hair)} h3{font-size:16px;margin:28px 0 2px}
.lede{font-size:17px;max-width:900px} .muted{color:var(--muted)} a{color:var(--accent)}
.hero{margin:18px 0 8px} .hero video{width:100%;max-width:980px;border:1px solid var(--hair);border-radius:6px}
.toc{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13.5px;margin:14px 0 4px} .toc a{text-decoration:none}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px 26px;margin-top:14px}
.card{margin:0;min-width:0} .card.wide{grid-column:1/-1} .card.wide img{max-width:940px} .card img{width:100%;height:auto;display:block;border:1px solid var(--hair);border-radius:4px}
.ctitle{font-weight:600;font-size:14.5px} .q{color:var(--muted);font-size:13.5px;margin:1px 0 8px}
.nums{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:6px;font-size:12.5px} .kv span{color:var(--muted);margin-right:5px}
details{margin-top:4px;font-size:12.5px} summary{cursor:pointer;color:var(--muted)} details p{margin:4px 0}
.answers{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px 30px;margin-top:10px} .answers div{font-size:14.5px}
.answers b{display:block;margin-bottom:2px}
.screen ol{margin:6px 0 0 18px;padding:0} .screen li{margin:3px 0}
.tools{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:8px}
.tool{min-width:0} .tool img{width:100%;border:1px solid var(--hair);border-radius:4px;display:block}
.tname{font-weight:600;font-size:13px;margin-bottom:4px} .tname a,.tname code{font-weight:400;font-size:11.5px}
.tnote{font-size:12px;color:var(--muted);margin:4px 0;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
img.zoom,.card img{cursor:zoom-in} #lightbox{position:fixed;inset:0;background:rgba(255,255,255,.96);display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}
#lightbox img{max-width:96vw;max-height:94vh;border:1px solid var(--hair)} .verdict td.best{font-weight:600} .verdict td.weak{color:var(--muted)}
.pending{height:120px;border:1px dashed var(--hair);border-radius:4px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin-top:8px} th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--hair);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12.5px} td code{font-size:12px}
.code{background:var(--soft);border-radius:4px;padding:8px 10px;overflow-x:auto;font-size:11.5px;line-height:1.45}
pre.code{white-space:pre;margin:4px 0} .code .k{color:#8e44ad;font-weight:600} .code .s{color:#2b7a3d} .code .c{color:#8a8a8e;font-style:italic}
.code .n{color:#b35c00} .code .d{color:#1f5fa8} .decision li{margin:4px 0}
@media (max-width:860px){.grid,.answers{grid-template-columns:1fr}.tools{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:520px){main{padding:18px 16px 60px}.tools{grid-template-columns:1fr}}
"""


# the page's offline Python highlighter (comments, strings, keywords, numbers, decorators): see highlight.js
HIGHLIGHT_JS = (HERE / "highlight.js").read_text()


def main() -> None:
    started = time.time()
    cards, missing = collect_cards()
    tools_html, tool_status = tools_section()
    heatmap = core_types.by_id("coverage-heatmap").data()
    growth = core_types.by_id("coverage-growth").data()
    sphere = core_types.by_id("direction-sphere").data()
    grasps = core_types.by_id("grasp-poses").data()
    hero = ASSETS / "hero.mp4"
    hero_html = ""
    if hero.is_file():
        video = base64.b64encode(hero.read_bytes()).decode()
        poster = picture(ASSETS / "hero-last.png", 640)
        hero_html = (f"<div class='hero'><video autoplay muted loop playsinline controls poster='{poster}' "
                     f"src='data:video/mp4;base64,{video}'></video><p class='muted'>CAN 11, every real play in the order it "
                     f"was recorded, one per frame. GIF fallback: <a href='coverage-gallery-{DATE}/hero.gif'>hero.gif</a></p></div>")
    by_section: dict[str, list[Card]] = {key: [] for key in SECTIONS}
    for card in cards:
        by_section.setdefault(card.section, []).append(card)
    toc = "".join(f"<a href='#s-{key}'>{esc(title)}</a>" for key, title in SECTIONS.items() if by_section.get(key))
    sections = "".join(f"<h2 id='s-{key}'>{esc(SECTIONS[key])}</h2><div class='grid'>{''.join(card_html(c) for c in items)}</div>"
                       for key, items in by_section.items() if items)
    openings = np.degrees(grasps["opening_rad"])
    values, counts = np.unique(np.round(openings, 1), return_counts=True)
    mode_share, mode_value = 100.0 * counts.max() / counts.sum(), values[np.argmax(counts)]
    campaign_pct = float(next(c for c in cards if c.id == "q-qd-linear").numbers["rounds 0-9"].split(" %")[0])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Dataset Coverage</title><style>{CSS}</style></head><body><main>
<h1>Robot dataset coverage: a gallery</h1>
<p class="lede"><b>{len(cards)} ways to see whether a robot dataset covers what the robot can do</b>, every one drawn from real
recordings and measured against the robot's limits; then the 8 core views built four times, in Matplotlib, Rerun, Viser and
W&amp;B. Headline from the data: CAN 11's 413 campaign plays visited {campaign_pct:.0f} % of its reachable (q, qd) cells; ten full-turn
plays took it to {heatmap['visited_pct']:.0f} %. More of the same run would not have got there.</p>
{hero_html}
<div class="toc">{toc}<a href="#s-tools">6 · Same view, four tools</a><a href="#s-numbers">7 · The numbers</a><a href="#s-decide">8 · What's next</a></div>

<h2 id="s-answers">Your questions, answered</h2>
<div class="answers">
<div><b>Log or linear colour?</b>Linear, with empty cells masked to white and the ramp starting at a visible tint, so one sample
never looks like none (<a href="#four-scales">four scales, side by side</a>). Log only when counts span 2+ decades and you need
magnitudes. Better still: colour by <a href="#runs-per-cell">runs per cell</a>, not samples, so a long rest pose stops dominating.
And bins are linear everywhere, so a visited % is a share of the real space.</div>
<div><b>Can we calculate it analytically?</b>Yes: the numbers in <a href="#s-numbers">section 7</a>. "Many speeds in every
configuration" is <a href="#conditional-coverage">conditional coverage</a> (% of reachable qd per q bin) and <a href="#speed-fill">speed-envelope
fill</a>. "Many accelerations at every speed" is the same per qd bin. Whether it is <i>enough</i> is <a href="#excitation">%σ per
model parameter</a> (system identification's own test). Whether more of the same helps is the <a href="#rarefaction">Chao2 ceiling</a>.</div>
<div><b>Limits first.</b>Every card draws the limit: the box, the torque-speed envelope, gravity's bite into the acceleration budget
(<a href="#reachable-envelope">the denominator</a>). The reachability agent's map plugs in as <code>Grid.of(..., reachable=mask)</code>.</div>
<div><b>Grasps and the gripper.</b>Our {len(openings)} real repick grasps used claw openings of {openings.min():.0f}–{openings.max():.0f}°
({mode_share:.0f} % at one setting, {mode_value:.0f}°; limit {np.degrees(grasps['opening_limit_rad']):.0f}°) and {sphere['visited_pct']:.1f} % of
approach directions (all from almost straight down). See section 4 and the
<a href="#type-direction-sphere">direction sphere</a>. Coverage there means position × approach × opening, per object.</div>
</div>
<div class="screen"><h3>If you only build one coverage screen</h3><ol>
<li><a href="#limit-bars">Limit bars</a>: each derivative against its limit, one row per joint.</li>
<li><a href="#runs-per-cell">q × qd and qd × qdd heatmaps</a>, runs per cell, white = empty, grey = unreachable.</li>
<li><a href="#conditional-coverage">Conditional coverage</a> bars under them: the number for "fast in every configuration".</li>
<li><a href="#rarefaction">Coverage growth with its Chao2 ceiling</a>: record more of the same, or something different?</li>
<li><a href="#holes">Hole list</a>: the next run's targets, biggest gap first.</li></ol></div>

{sections}

<h2 id="s-tools">6 · Same view, four tools</h2>
<p class="muted">Each core view is drawn from the same arrays (<code>core_types.by_id(id).data()</code>) in all four tools, so
the difference is the tool. Open "code" under any picture for the exact function that drew it.
Captured: {', '.join(f'{label} {tool_status[tool]}/8' for tool, label, _ in TOOLS)}.</p>
{tools_html}

<h2 id="s-numbers">7 · The numbers a dataset analyzer should report</h2>
{numbers_table()}

<h2 id="s-prior">Prior art, briefly</h2>
<p>The ML community's dataset tools carry over once you fix three things: <b>count episodes, not frames</b>, <b>measure against a stated
reference set</b> (here, the reachable set), and <b>draw empty as white</b>. Robotics adds the denominator (limits, torque-speed, braking)
and system identification's test of whether data is informative, not just spread out. Full sweep with sources: the vault note
<i>{esc(VAULT_NOTE)}</i>.</p>

<h2 id="s-decide">8 · What's next</h2>
{decision_html(missing, tool_status)}
<p class="muted">Built {time.strftime('%Y-%m-%d %H:%M')} by docs/proposals/coverage-{DATE}/build_gallery.py in {time.time() - started:.0f} s.</p>
</main><div id="lightbox"><img alt=""></div><script>{HIGHLIGHT_JS}</script></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"{OUT}: {OUT.stat().st_size / 1e6:.2f} MB, {len(cards)} cards, tools {tool_status}, missing {missing}")


def numbers_table() -> str:
    rows = [
        ("Limit use", "span used / limit span, per derivative (p0.1-p99.9)", "limit-bars"),
        ("Visited %", "reachable cells with ≥ 1 sample / reachable cells", "q-qd-linear"),
        ("Enough %", "reachable cells with ≥ k samples (k = 50 here) / reachable cells", "four-scales"),
        ("Evenness", "exp(entropy of counts) / reachable cells: 100 % only if every cell holds the same", "imbalance"),
        ("Conditional coverage", "per q bin: reachable qd cells visited / reachable qd cells; report mean and worst", "conditional-coverage"),
        ("Envelope fill", "per q bin: (max qd − min qd reached) / (allowed span); same for qdd against torque", "speed-fill"),
        ("Gap radius r90", "90th percentile of distance from a random reachable state to the nearest sample (limit spans = 1)", "dispersion"),
        ("Largest holes", "farthest-first empty balls in (q, qd, qdd): centre + radius, ranked", "holes"),
        ("Chao2 ceiling", "observed + (R−1)/R · Q1² / 2Q2 cells, runs as sampling units: where more of the same tops out", "rarefaction"),
        ("Unique per run", "cells only that run visited: zero = droppable", "novelty-per-run"),
        ("Held-out support", "share of held-out samples outside training cells: extrapolation, not interpolation", "held-out-support"),
        ("Excitation %σ", "100 · √C_ii / |θ_i|, C = σ²(WᵀW)⁻¹ from a least-squares fit of the model to measured torque", "excitation"),
    ]
    body = "".join(f"<tr><td><a href='#{anchor}'>{esc(name)}</a></td><td>{esc(formula)}</td></tr>" for name, formula, anchor in rows)
    return (f"<p class='muted'>All in <code>coverage.py</code> (numpy only). Click a name to see it computed on real data.</p>"
            f"<table><tr><th>Number</th><th>Definition</th></tr>{body}</table>")


def decision_html(missing: list[str], tool_status: dict) -> str:
    left = [f"{name}.py not built yet" for name in missing]
    left += [f"{tool}: {8 - count} of 8 views not captured" for tool, count in tool_status.items() if count < 8]
    return f"""<ul class="decision">
<li><b>Done and proved:</b> the gallery above, built from real recordings by one command; the coverage math in
<code>coverage.py</code> (9 tests; visited % matches robot_msgs.coverage's rule on the same cells); robot_metrics' own 99 tests
still pass; all 8 core views captured in all 4 tools (W&amp;B screenshots from the live runs).</li>
<li><b>Decided for you:</b> linear bins and colour, empty = white, runs per cell as the default colour; one coarse 3-D grid
(18 × 10 × 12) for every 3-D number; the qdd limit comes from the torque limit, with the planner's 20 rad/s² dashed; planned TCP paths moved
into base_link through each plan's plane frame; the five-view standard screen above. All proposal code lives in
<code>robot_metrics/docs/proposals/coverage-{DATE}/</code> and nothing in the package core changed (undo: <code>git revert</code> the commit). The W&amp;B
project "Robot dataset coverage gallery" (8 runs, ~18 MB) is registered as a tracker on bam_ws (<code>58fdaae64</code>).</li>
<li><b>Left (next, not blocked):</b> {esc('; '.join(left)) + '; ' if left else ''}move the new numbers into robot_msgs.coverage and build the
standard screen as a robot_metrics view; plug the reachability work's map in as the denominator.</li>
<li><b>Needs you:</b> the W&amp;B account banner says <b>3 % of the 5 GB free storage remains</b>, not from this work (this project is
~18 MB). Prune old projects or upgrade the plan; my recommendation is to prune, since the tracker review found Traj Gen alone at 2.5 GB.</li>
<li><b>Worth a look (found in the data):</b> the claw's limits disagree (planner span 70 mm, open stop ~125 mm, some plans' opening above
their own claw_max); the UR wrist rig's two frame offsets disagree (+3.7° in the reachable file vs +84° in rig.py); a single
held-out rung (CAN 13, round 0) is 31 % outside its training cells.</li>
<li><b>Deliberately not done:</b> no change to robot_metrics, robot_msgs or sim2real code; no audit; the pick-place simulation was left
out (placeholder ±180° limits and a world frame that doesn't match the real base).</li></ul>"""


if __name__ == "__main__":
    main()
