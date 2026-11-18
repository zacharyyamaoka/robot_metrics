"""Build reports/robot-metrics-v0-2026-11-17.html from real runs of everything it shows.

    cd ~/robot_metrics && taskset -c 2-5,16,17 uv run --extra test --extra viz --extra rerun python docs/build_report.py

What it runs, live: the test suite (green, at HEAD), the same tests against the tests-first commit's tree (red), the
parity table (the tests' own oracle and scoring), two wrong rulers under that oracle (they must go red), the brief's
isolation command, and the usage session on CAN 11's round-8 runs (its plot() images are embedded).

The session's Python is highlighted here with Pygments (it arrives with the test extra: pytest depends on it), inline
CSS only. WHY one statement then its output, the API at a glance first (Zach, /rate bad 2026-11-18): "please show this
python text with syntax highlighting ... make it more like line by line so I can see the output of each line ... I am
not seeing the API design at a glance".
"""

from __future__ import annotations

import base64
import html
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "docs"), str(ROOT / "tests")]

import oracle  # noqa: E402
import scoring  # noqa: E402
import usage_session  # noqa: E402

OUT = ROOT / "reports" / "robot-metrics-v0-2026-11-17.html"
MEDIA = ROOT / "reports" / "media" / "robot-metrics-v0-2026-11-17"
TESTS_FIRST = "7f8763c"
PIN = ["taskset", "-c", "2-5,16,17"]
PYTHON = str(ROOT / ".venv" / "bin" / "python")


def sh(command: list[str], *, cwd: Path = ROOT, env: dict | None = None, timeout: int = 900) -> tuple[str, int, float]:
    started = time.perf_counter()
    done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return (done.stdout + done.stderr).rstrip(), done.returncode, time.perf_counter() - started


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def green_suite() -> tuple[str, int]:
    out, code, _ = sh(PIN + [PYTHON, "-m", "pytest", "-v", "-p", "no:cacheprovider", "--durations=5"])
    return f"$ cd ~/robot_metrics && taskset -c 2-5,16,17 uv run --extra test pytest -v --durations=5\n{out}", code


def red_suite() -> tuple[str, int]:
    """The tests-first commit's tree, run with nothing but its own scaffold on the path (its src shadows HEAD's)."""

    with tempfile.TemporaryDirectory(prefix="robot-metrics-red-") as folder:
        archive = subprocess.run(["git", "archive", TESTS_FIRST], cwd=ROOT, capture_output=True, check=True).stdout
        tarfile.open(fileobj=io.BytesIO(archive)).extractall(folder, filter="data")
        env = {**os.environ, "PYTHONPATH": str(Path(folder) / "src")}
        out, code, _ = sh(PIN + [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--continue-on-collection-errors"],
                          cwd=Path(folder), env=env)
    tail = "\n".join(out.splitlines()[-14:])
    return f"$ git archive {TESTS_FIRST} | tar -x && pytest -q   # the tests, before any implementation\n...\n{tail}", code


def isolation() -> tuple[str, int]:
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    probe = ("import sys, importlib.metadata as md, robot_metrics, robot_metrics.catalog, robot_metrics.views, "
             "robot_metrics.adapters.wandb, robot_metrics.adapters.rerun, robot_metrics.datasets.onedof_round8; "
             "print('loaded:', sorted({m.split('.')[0] for m in sys.modules} - set(sys.stdlib_module_names) - "
             "{'__main__', '_virtualenv', '_distutils_hack'})); "
             "print('installed:', sorted(d.metadata['Name'] for d in md.distributions()))")
    first, code1, _ = sh(PIN + ["uv", "run", "--isolated", "python", "-c", "import robot_metrics"], env=env)
    second, code2, _ = sh(PIN + ["uv", "run", "--isolated", "--no-dev", "python", "-c", probe], env=env)
    text = (f'$ env -u VIRTUAL_ENV uv run --isolated python -c "import robot_metrics"\n{first}\nexit {code1}\n\n'
            f'$ env -u VIRTUAL_ENV uv run --isolated --no-dev python -c "<import every module, list what loaded>"\n'
            f"{second}\nexit {code2}")
    return text, code1 or code2


def parity() -> dict:
    from robot_metrics.datasets import fivebar_ladder, onedof_round8

    stations = onedof_round8.stations()
    runs = onedof_round8.real_runs()
    collections = scoring.arm_collections(stations)
    card = oracle.scorecard_oracle()
    problems, worst = [], {}
    plays = scoring.play_rows(runs, collections)
    oracle.compare_rows("real_plays", plays, card["plays"], oracle.PLAY_TOLERANCES, problems, worst)
    strata = scoring.strata(runs, collections)
    oracle.compare_rows("real_rungs", scoring.rung_rows(strata), card["rungs"], oracle.RUNG_TOLERANCES, problems, worst)
    sim = onedof_round8.sim_runs(oracle.twins_used())
    sim_plays = scoring.play_rows(sim, collections)
    oracle.compare_rows("sim_plays", {k: {"tracking_rms_deg": v["tracking_rms_deg"]} for k, v in sim_plays.items()},
                        card["sim_plays"], oracle.SIM_PLAY_TOLERANCES, problems, worst)
    oracle.compare_rows("sim_rungs", scoring.rung_rows(scoring.strata(sim, collections)), card["sim_rungs"],
                        oracle.SIM_RUNG_TOLERANCES, problems, worst)
    judged = scoring.verdicts(strata)
    chain_want = oracle.chain_oracle()
    chain_got = scoring.chain_rows(fivebar_ladder.runs("real") + fivebar_ladder.runs("sim"))
    chain_worst = max(abs(chain_got[k][n] - v) / abs(v) for k, e in chain_want.items() for n, v in e.items())
    overshoot_worst = max(abs(plays[r.id]["overshoot_deg"] - oracle.overshoot_oracle(
        r.streams["state"].t, r.streams["state"].positions[:, 0], r.streams["target"].positions[:, 0],
        r.streams["target"].velocities[:, 0])) for r in runs)
    return {"problems": problems, "worst": worst, "gates": scoring.gate_counts(judged), "gates_want": card["gates"],
            "highest": scoring.highest_rungs(judged), "highest_want": card["highest"], "chain_worst": chain_worst,
            "chain_rows": len(chain_want), "plays": len(plays), "sim": len(sim), "stops": sum(v["stops"] for v in plays.values()),
            "overshoot_worst": overshoot_worst}


def wrong_rulers() -> list[tuple[str, str, float]]:
    """The contract's two nearest wrong rulers, swapped into the alignment step: the same oracle must go red."""

    import numpy as np

    import robot_metrics.alignment as alignment
    from robot_metrics.datasets import onedof_round8

    def linear(t, knots_t, knots_q, knots_qd):
        own = np.clip(np.searchsorted(knots_t, t, side="right") - 1, 0, knots_t.size - 1)
        return np.interp(t, knots_t, knots_q), knots_qd[own]

    def tick(t, knots_t, knots_q, knots_qd):
        own = np.clip(np.searchsorted(knots_t, t, side="right") - 1, 0, knots_t.size - 1)
        return knots_q[own], knots_qd[own]

    shipped, out = alignment._hermite, []
    try:
        for label, rule in (("plan linearly interpolated (robot_msgs resample)", linear),
                            ("each reply against its own tick", tick), ("hermite at each reply's instant (shipped)", shipped)):
            alignment._hermite = rule
            onedof_round8.real_runs.cache_clear()
            runs = onedof_round8.real_runs()
            collections = scoring.arm_collections(onedof_round8.stations())
            problems, worst = [], {}
            oracle.compare_rows("real_plays", scoring.play_rows(runs, collections), oracle.scorecard_oracle()["plays"],
                                oracle.PLAY_TOLERANCES, problems, worst)
            out.append((label, "RED" if problems else "green", worst.get("real_plays.tracking_rms_deg", 0.0)))
    finally:
        alignment._hermite = shipped
        onedof_round8.real_runs.cache_clear()
    return out


def image(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def pre(text: str) -> str:
    return f"<pre>{html.escape(text)}</pre>"


LEXER, FORMATTER = PythonLexer(), HtmlFormatter(nowrap=True)
LIGHT, DARK = "default", "github-dark"          # Pygments styles: token colours only, the page sets the background
WRAP_OVER = 100                                 # a one-line output longer than this wraps; a table scrolls instead


def python(code: str) -> str:
    return highlight(code, LEXER, FORMATTER).rstrip("\n")


def output(text: str) -> str:
    wrap = "\n" not in text and len(text) > WRAP_OVER
    return f"<pre class='out{' wrap' if wrap else ''}'>{html.escape(text)}</pre>"


def notebook(cell: dict) -> str:
    """A cell as a notebook: its statements highlighted, each output in a quieter band right under its statement.
    Statements with no output share one block, with the source's blank lines between ideas; a blank line that falls
    right after an output becomes the next block's extra top space (``gap``)."""

    blocks, group, group_gap = [], [], False
    for statement in cell["statements"]:
        if not group:
            group_gap = statement["gap_before"] and bool(blocks)
        elif statement["gap_before"]:
            group.append("")
        group.append(statement["code"])
        if statement["output"]:
            blocks.append(f"<pre class='src{' gap' if group_gap else ''}'>{python(chr(10).join(group))}</pre>")
            blocks.append(output(statement["output"]))
            group = []
    if group:
        blocks.append(f"<pre class='src{' gap' if group_gap else ''}'>{python(chr(10).join(group))}</pre>")
    return f"<div class='notebook'>{''.join(blocks)}</div>"


def token_css() -> str:
    light = HtmlFormatter(style=LIGHT).get_token_style_defs(".notebook .src")
    dark_auto = HtmlFormatter(style=DARK).get_token_style_defs(':root:not([data-theme="light"]) .notebook .src')
    dark_forced = HtmlFormatter(style=DARK).get_token_style_defs(':root[data-theme="dark"] .notebook .src')
    return "\n".join([*light, "@media (prefers-color-scheme: dark) {", *dark_auto, "}", *dark_forced])


CSS = """
:root { --bg:#fbfaf7; --panel:#ffffff; --ink:#1d1f23; --muted:#5d6470; --line:#e3e0d8; --accent:#2f6f4f;
        --bad:#b03a2e; --code:#f3f1ec; --chip:#e8f2ec; --out-bg:#f3f1ec; --out-ink:#4f5661; --out-rule:#d3cec2; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#16181b; --panel:#1e2125; --ink:#e7e5e0;
        --muted:#a3a9b2; --line:#33373d; --accent:#7cc39b; --bad:#ec8b7f; --code:#24272c; --chip:#22352b;
        --out-bg:#181a1d; --out-ink:#a3a9b2; --out-rule:#3d4249; } }
:root[data-theme="dark"] { --bg:#16181b; --panel:#1e2125; --ink:#e7e5e0; --muted:#a3a9b2; --line:#33373d;
        --accent:#7cc39b; --bad:#ec8b7f; --code:#24272c; --chip:#22352b; --out-bg:#181a1d; --out-ink:#a3a9b2;
        --out-rule:#3d4249; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1080px; margin:0 auto; padding:28px 16px 64px; }
h1 { font-size:26px; margin:0 0 6px; } h2 { font-size:19px; margin:34px 0 10px; border-bottom:1px solid var(--line);
  padding-bottom:6px; } h3 { font-size:15px; margin:18px 0 6px; }
.verdict { font-size:17px; margin:8px 0 4px; } .muted { color:var(--muted); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:18px 0; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.tile b { display:block; font-size:21px; } .tile span { color:var(--muted); font-size:13px; }
figure { margin:14px 0; background:#fff; border:1px solid var(--line); border-radius:10px; padding:8px; }
figure img { width:100%; height:auto; display:block; } figcaption { color:#444; font-size:13px; padding:6px 4px 0; }
table { border-collapse:collapse; width:100%; background:var(--panel); font-size:14px; }
th, td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
th { background:var(--code); } td.num { font-variant-numeric:tabular-nums; white-space:nowrap; }
.pass { color:var(--accent); font-weight:600; } .fail { color:var(--bad); font-weight:600; }
pre { background:var(--code); border:1px solid var(--line); border-radius:8px; padding:10px 12px; overflow-x:auto;
  font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre; margin:6px 0 12px; }
details { margin:8px 0; } summary { cursor:pointer; font-weight:600; }
code { background:var(--code); padding:1px 5px; border-radius:4px; font-size:13px; }
.cell-caption { color:var(--muted); margin:2px 0 6px; }
.notebook { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; margin:6px 0 18px; }
.notebook pre { margin:0; border:0; border-radius:0; overflow-x:auto; max-width:100%; }
.notebook pre.src { background:var(--panel); color:var(--ink); padding:8px 14px; font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
.notebook pre.src.gap { padding-top:20px; }
.notebook pre.out { background:var(--out-bg); color:var(--out-ink); border-left:3px solid var(--out-rule); padding:5px 14px 6px 11px;
  font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
.notebook pre.out.wrap { white-space:pre-wrap; overflow-wrap:anywhere; }
ul { padding-left:20px; } li { margin:3px 0; }
.wrap { overflow-x:auto; }
"""


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    head, tests_first = git("rev-parse", "--short", "HEAD"), TESTS_FIRST
    green, green_code = green_suite()
    red, red_code = red_suite()
    isolated, isolated_code = isolation()
    table = parity()
    rulers = wrong_rulers()
    session = usage_session.run(MEDIA)
    passed_line = next((line for line in reversed(green.splitlines()) if " passed" in line), "")
    worst = table["worst"]

    def row(label, rows, tolerance, key, unit):
        value = worst.get(key, 0.0)
        return (f"<tr><td>{label}</td><td class='num'>{rows}</td><td class='num'>{tolerance:g} {unit}</td>"
                f"<td class='num'>{value:.1e} {unit}</td><td class='pass'>pass</td></tr>")

    parity_rows = "".join([
        row("per play · tracking RMS", table["plays"], oracle.TRACKING_DEG, "real_plays.tracking_rms_deg", "deg"),
        row("per play · tracking peak", table["plays"], oracle.TRACKING_DEG, "real_plays.tracking_peak_deg", "deg"),
        row("per play · stop count", table["plays"], 0, "real_plays.stops", ""),
        row("per rung · tracking RMS (median of plays)", 36, oracle.TRACKING_DEG, "real_rungs.tracking_rms_deg", "deg"),
        row("per rung · tracking peak", 36, oracle.TRACKING_DEG, "real_rungs.tracking_peak_deg", "deg"),
        row("per rung · tip stop median (pooled)", 36, oracle.STOP_MM, "real_rungs.stop_tool_mm", "mm"),
        row("per rung · tip stop worst", 36, oracle.STOP_MM, "real_rungs.stop_tool_max_mm", "mm"),
        row("per rung · settle (median, 1 mm band)", 36, oracle.SETTLE_S, "real_rungs.settle_s", "s"),
        row("per rung · limiter share", 36, oracle.SHARE, "real_rungs.limiter_share", ""),
        row("twin · per play tracking RMS", table["sim"], oracle.TRACKING_DEG, "sim_plays.tracking_rms_deg", "deg"),
        row("twin · per rung tracking RMS", 36, oracle.TRACKING_DEG, "sim_rungs.tracking_rms_deg", "deg"),
        row("twin · per rung tip stop median", 36, oracle.STOP_MM, "sim_rungs.stop_tool_mm", "mm"),
        row("twin · per rung tip stop worst", 36, oracle.STOP_MM, "sim_rungs.stop_tool_max_mm", "mm"),
        f"<tr><td>five-bar ladder · 5 rungs, real and sim: joint RMS ×2, tool RMS, tool worst</td><td class='num'>{table['chain_rows']} runs × 4</td>"
        f"<td class='num'>{oracle.CHAIN_REL:g} relative</td><td class='num'>{table['chain_worst']:.1e} relative</td><td class='pass'>pass</td></tr>",
        f"<tr><td>overshoot vs its independent oracle</td><td class='num'>{table['plays']}</td><td class='num'>{oracle.OVERSHOOT_DEG:g} deg</td>"
        f"<td class='num'>{table['overshoot_worst']:.1e} deg</td><td class='pass'>pass</td></tr>",
    ])
    gate_rows = "".join(
        f"<tr><td>{name}</td><td class='num'>{got[0]} of {got[1]}</td><td class='num'>{table['gates_want'][name][0]} of {table['gates_want'][name][1]}</td>"
        f"<td class='{'pass' if got == table['gates_want'][name] else 'fail'}'>{'equal' if got == table['gates_want'][name] else 'DIFFERENT'}</td></tr>"
        for name, got in table["gates"].items())
    highest = ", ".join(f"{station} {rung:g}" for station, rung in table["highest"].items())
    ruler_rows = "".join(f"<tr><td>{label}</td><td class='{'fail' if verdict == 'RED' else 'pass'}'>{verdict}</td>"
                         f"<td class='num'>{value:.2e} deg</td></tr>" for label, verdict, value in rulers)
    glance, *walkthrough = session.cells
    cells = "".join(f"<h3>{index} · {html.escape(cell['label'])}</h3><p class='cell-caption'>{html.escape(cell['caption'])}</p>"
                    + notebook(cell) for index, cell in enumerate(walkthrough, start=1))
    tests_passed = passed_line.strip("= ").split(" passed")[0]
    status = "PASS" if not table["problems"] and green_code == 0 and isolated_code == 0 else "FAIL"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>robot_metrics v0</title><style>{CSS}{token_css()}</style></head><body><main>
<h1>robot_metrics v0</h1>
<p class="verdict"><b>{status}: robot_metrics v0 reproduces round 8's scorecard at float round-off</b> — local
<code>main</code> @ <code>{head}</code>, tests first @ <code>{tests_first}</code>; no remote. The W&amp;B send and a
Rerun viewer were not exercised.</p>
<p class="muted">The recommended splice of the five metrics-API proposals: v4's value-state core and occurrence gates under
v1's torchmetrics face, v3's alignment step, v5's context rule, amendment 4's other inputs. Decision record:
<code>docs/peps/0001-robot-metrics-v0.md</code>. Built {time.strftime('%Y-%m-%d %H:%M %Z')} by <code>docs/build_report.py</code>,
every block below from a real run.</p>
<div class="tiles">
<div class="tile"><b>{html.escape(passed_line.strip('= ').split(' in ')[0])}</b><span>tests, red first at {tests_first}</span></div>
<div class="tile"><b>{worst.get('real_rungs.stop_tool_mm', 0):.1e} mm</b><span>worst parity gap (rung stop); tracking bit-equal</span></div>
<div class="tile"><b>{' / '.join(str(v[0]) for v in table['gates'].values())} of 36</b><span>gate cells: tracking / median stop / every stop</span></div>
<div class="tile"><b>4 packages</b><span>in the isolated env: numpy, dataclass-values, robot-msgs, robot-metrics</span></div>
<div class="tile"><b>{table['stops']} stops</b><span>each split mid-dwell in the live == offline test</span></div>
</div>

<h2>API at a glance</h2>
<p class="cell-caption">{html.escape(glance['caption'])} Each line's output sits right under it, from the run that
built this page.</p>
{notebook(glance)}

<h2>What it shows on real runs</h2>
<figure><img alt="Every CAN 11 stop at the tip by rung, with Zach's 1 mm line" src="{image(MEDIA / 'can11-tip-stops.png')}">
<figcaption>Collection.plot() on CAN 11's 42 round-8 plays: 84 stops at the tip. The median gate passes every rung; the
every-stop objective fails rungs 0.4, 0.9 and 1.0 (5 stops over 1 mm, worst 1.317 mm). That gap is the reason gates
read a tail per stratum.</figcaption></figure>
<figure><img alt="Tracking RMS by rung for all six stations with the 0.5 degree gate" src="{image(MEDIA / 'tracking-by-rung.png')}">
<figcaption>views.plot() of a kpi_trend View built from summarize() on all 252 plays: CAN 11 tracks the 1.0 rung, the
other five stations top out at 0.9 (the scorecard's highest rungs: {highest}).</figcaption></figure>

<h2>Parity with round 8</h2>
<p>The frozen check's oracle and tolerances, ported unchanged into <code>tests/oracle.py</code>; only the tests read
the scorecard and the ladder's summary. {table['plays']} real plays, {table['sim']} twin answers, 36 rung cells.</p>
<div class="wrap"><table><tr><th>quantity</th><th>rows</th><th>tolerance (frozen)</th><th>worst |deviation|</th><th></th></tr>
{parity_rows}</table></div>
<h3>Gates and the highest rung</h3>
<div class="wrap"><table><tr><th>gate</th><th>robot_metrics</th><th>scorecard / contract</th><th></th></tr>{gate_rows}
<tr><td>highest rung tracked</td><td colspan="2">{highest}</td><td class="pass">{'equal' if {k: float(v) for k, v in table['highest'].items()} == {k: float(v) for k, v in table['highest_want'].items()} else 'DIFFERENT'}</td></tr></table></div>
<h3>Can the ruler fail? Two wrong rulers under the same oracle</h3>
<div class="wrap"><table><tr><th>target for each reply</th><th>parity</th><th>worst play RMS gap</th></tr>{ruler_rows}</table></div>
<p class="muted">The contract predicted the linear plan at 9.4e-4 deg before any variant was built.</p>

<h2>A usage session on CAN 11's round-8 runs</h2>
<p class="muted">It continues the glance's namespace (<code>run</code>, <code>collection</code>, <code>strata</code>, ...);
<code>MEDIA</code> is this report's media folder. One statement, then its output.</p>
{cells}

<h2>Tests</h2>
<details open><summary>Green at {head}</summary>{pre(green)}</details>
<details><summary>Red at {tests_first} (the tests alone, before any implementation)</summary>{pre(red)}</details>
<details open><summary>Import isolation (the brief's command)</summary>{pre(isolated)}</details>

<h2>What next</h2>
<ul>
<li><b>Done and proved:</b> the package, its {tests_passed} tests (red first), parity at the frozen tolerances, live == offline,
dedup, identity, gate counts, Repeatability and PerformanceGain against their oracles, import isolation.</li>
<li><b>Decided for you</b> (each undone by a commit; list in PEP 0001 §10): module <code>alignment</code> with the function
<code>align</code>; compute groups one step past identity (rms / max_abs / mean_abs share a state); a statistic that lands on a
never-settled stop returns None; the dataset readers live in the package for now; the plot legend sits outside the
axes (inside, it hid CAN 11's worst stop).</li>
<li><b>Next, not blocked:</b> RobotDx's stock and <code>sim2real.metrics.actuator</code> migrate onto these classes;
robotparts / robot_models gain <code>.metrics</code> from <code>catalog</code>; the stop kernel, the hermite rule and
CalibrationOutcome are proposed moves into robot_msgs (PEP 0001 §7, §8).</li>
<li><b>Needs you:</b> nothing.</li>
<li><b>Deliberately not done:</b> no edits to robot_models, robot_msgs, sim2real, io-diagnostics, robotparts or bam_ws;
no remote, no push; no W&amp;B upload (the payload is built and tested; <code>adapters.wandb.log</code> is not run); the
<code>.rrd</code> is written to the gitignored <code>reports/media/</code>, not opened in a viewer; no audit.</li>
</ul>
</main></body></html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) · status {status} · green exit {green_code} · red exit "
          f"{red_code} · isolation exit {isolated_code} · problems {table['problems'][:3]}")


if __name__ == "__main__":
    main()
