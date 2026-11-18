"""Every block of Python shown to a reader runs as shown: the README, the landing page and the v0 report.

What "runs as shown" means here (Zach, /rate bad 2026-11-18: "I cannot read this pseudo code because it's not valid
... you use this pendulum_arm but I have no idea what that is? where did you import it from"):

- a script is what a reader sees as one piece: the README's block, the page's API at a glance, one catalog card, one
  gallery arm (its numbered steps, top to bottom), the report's glance, the report's usage session;
- it imports or defines every name it uses (checked statically on the sources, so CI checks it without recordings);
- it runs top to bottom in a fresh ``python -I`` with nothing pre-imported, from a clone's root, and each statement
  prints exactly the output the page shows under it (checked on the code extracted from the built page itself);
- a script that reads the private recordings, MuJoCo Menagerie or matplotlib skips cleanly where they are absent;
- every notebook block on a built page names its script (``data-script``), so nothing shown escapes this file.
"""

from __future__ import annotations

import builtins
import html.parser
import importlib.util
import json
import os
import re
import subprocess
import sys
import symtable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SITE_PAGE = Path(os.environ.get("ROBOT_METRICS_SITE_OUT") or ROOT / "website" / "_build") / "index.html"
REPORT_PAGE = ROOT / "reports" / "robot-metrics-v0-2026-11-17.html"
# WHY normalized: a timing printed by the code is the one output that differs run to run
VARIES = (re.compile(r"\d+\.\d+(?= ms per)"),)

# A notebook harness, stdlib only: run a script one top-level statement at a time in a fresh namespace and keep what
# each printed, a bare expression echoing its repr (docs/session.py's rule, the one the pages were built with).
RUNNER = r'''
import ast, builtins, contextlib, io, json, sys
path, out = sys.argv[1], sys.argv[2]
tree = ast.parse(open(path, encoding="utf-8").read())
namespace = {"__name__": "__main__", "__builtins__": builtins}
outputs = []
for node in tree.body:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        if isinstance(node, ast.Expr):
            value = eval(compile(ast.Expression(node.value), path, "eval"), namespace)
            if value is not None:
                print(repr(value))
        else:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    outputs.append(buffer.getvalue().rstrip())
json.dump(outputs, open(out, "w", encoding="utf-8"))
'''


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module      # WHY: a dataclass in the module looks its module up here
    spec.loader.exec_module(module)
    return module


def source_scripts() -> dict[str, str]:
    """Each script as its source holds it: {script id: code}."""

    gallery = _load("shown_gallery", ROOT / "website" / "gallery.py")
    usage = _load("shown_usage_session", ROOT / "docs" / "usage_session.py")
    scripts = {}
    readme_blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    for number, block in enumerate(readme_blocks, start=1):
        scripts[f"readme-{number}"] = block
    scripts["site-glance"] = gallery.GLANCE[2]
    for name, code in gallery.CATALOG.items():
        scripts[f"site-catalog-{name.lower()}"] = code
    for arm in ("PENDULUM", "FIVEBAR", "UR5E"):
        scripts[f"site-{arm.lower()}"] = "\n".join(cell[2] for cell in getattr(gallery, arm))
    scripts["report-glance"] = usage.GLANCE[2]
    scripts["report-session"] = "\n".join(cell[2] for cell in usage.CELLS)
    return scripts


class _Notebooks(html.parser.HTMLParser):
    """The notebook blocks of a built page: [(data-script or None, [(\"src\" | \"out\", text), ...])]."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.notebooks: list[tuple[str | None, list[tuple[str, str]]]] = []
        self._depth = 0           # div nesting inside the current notebook, 0 outside one
        self._pre: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "div" and self._depth:
            self._depth += 1
        elif tag == "div" and "notebook" in classes:
            self._depth = 1
            self.notebooks.append((attributes.get("data-script"), []))
        elif tag == "pre" and self._depth:
            self._pre = "out" if "out" in classes else "src"
            self._text = []

    def handle_endtag(self, tag):
        if tag == "pre" and self._pre:
            self.notebooks[-1][1].append((self._pre, "".join(self._text)))
            self._pre = None
        elif tag == "div" and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if self._pre:
            self._text.append(data)


def notebooks(page: Path) -> list[tuple[str | None, list[tuple[str, str]]]]:
    parser = _Notebooks()
    parser.feed(page.read_text(encoding="utf-8"))
    return parser.notebooks


def displayed() -> dict[str, tuple[str, list[str]]]:
    """Each script as a built page shows it: {script id: (code, the outputs shown, in order)}."""

    found: dict[str, tuple[list[str], list[str]]] = {}
    for page in (SITE_PAGE, REPORT_PAGE):
        if not page.is_file():
            continue
        for script, blocks in notebooks(page):
            if script is None:
                continue
            code, outputs = found.setdefault(script, ([], []))
            for kind, text in blocks:
                (outputs if kind == "out" else code).append(text)
    shown = {script: ("\n".join(code), outputs) for script, (code, outputs) in found.items()}
    readme_blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    for number, block in enumerate(readme_blocks, start=1):
        outputs = [line.removeprefix("# ") for line in block.splitlines() if line.startswith("# ")]
        shown[f"readme-{number}"] = (block, outputs)
    return shown


def undefined_names(code: str) -> set[str]:
    """Names the code reads that nothing in it imports or defines and builtins do not supply."""

    module = symtable.symtable(code, "<shown>", "exec")
    defined = {symbol.get_name() for symbol in module.get_symbols()
               if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}
    known = defined | set(dir(builtins))
    missing = {symbol.get_name() for symbol in module.get_symbols() if symbol.is_referenced()} - known
    tables = list(module.get_children())
    while tables:
        table = tables.pop()
        tables += table.get_children()
        for symbol in table.get_symbols():
            if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in known:
                missing.add(symbol.get_name())
    return missing


def needs(code: str) -> str | None:
    """Why this script cannot run on this machine, or None."""

    if "onedof_round8" in code:
        from robot_metrics.datasets import onedof_round8

        if not onedof_round8.available():
            return "round 8's recordings are absent (ROBOT_METRICS_ROUND8; not public)"
    if "fivebar_ladder" in code:
        from robot_metrics.datasets import fivebar_ladder

        if not fivebar_ladder.available():
            return "the five-bar ladder's recordings are absent (ROBOT_METRICS_FIVEBAR_LADDER; not public)"
    if "plot" in code or "matplotlib" in code:
        if importlib.util.find_spec("matplotlib") is None:
            return "matplotlib is not installed (the viz extra)"
    if "ur5e_sim" in code:
        if importlib.util.find_spec("mujoco") is None:
            return "mujoco is not installed (the site extra)"
        ur5e_sim = _load("shown_ur5e_sim", ROOT / "website" / "ur5e_sim.py")
        try:
            ur5e_sim.model_path()
        except FileNotFoundError:
            return "MuJoCo Menagerie's UR5e is absent (MENAGERIE)"
    if "rerun" in code and importlib.util.find_spec("rerun") is None:
        return "rerun-sdk is not installed (the rerun extra)"
    return None


SOURCES = source_scripts()


@pytest.mark.parametrize("script", sorted(SOURCES))
def test_every_name_a_shown_script_uses_is_imported_or_defined_in_it(script):
    assert undefined_names(SOURCES[script]) == set(), f"{script} reads names it never imports or defines"


@pytest.mark.parametrize("page", [SITE_PAGE, REPORT_PAGE], ids=["site", "report"])
def test_every_notebook_on_a_built_page_names_its_script(page):
    if not page.is_file():
        pytest.skip(f"{page.name} is not built here (the site is built where the recordings are)")
    blocks = notebooks(page)
    assert blocks, f"no notebook blocks found on {page}"
    unnamed = [index for index, (script, _) in enumerate(blocks) if script is None]
    assert not unnamed, f"notebooks {unnamed} on {page.name} carry no data-script, so nothing checks them"
    unknown = {script for script, _ in blocks} - set(SOURCES)
    assert not unknown, f"{page.name} shows scripts this check does not know: {unknown}"
    prefix = "site-" if page == SITE_PAGE else "report-"
    missing = {script for script in SOURCES if script.startswith(prefix)} - {script for script, _ in blocks}
    assert not missing, f"{page.name} does not show {missing}"


def _normalized(outputs: list[str]) -> list[str]:
    for pattern in VARIES:
        outputs = [pattern.sub("<varies>", output) for output in outputs]
    return outputs


@pytest.mark.parametrize("script", sorted(SOURCES))
def test_a_shown_script_runs_alone_in_a_fresh_interpreter_and_prints_what_is_shown(script, tmp_path):
    shown = displayed().get(script)
    code, expected = shown if shown else (SOURCES[script], None)
    reason = needs(code)
    if reason:
        pytest.skip(reason)
    # WHY a clone's root: the examples read tests/data/ and import website/ur5e_sim.py by relative path, as a reader
    # who cloned the repo would; images they save land here, not in the repo
    clone = tmp_path / "clone"
    clone.mkdir()
    for folder in ("tests", "website"):
        (clone / folder).symlink_to(ROOT / folder, target_is_directory=True)
    script_file = tmp_path / f"{script}.py"
    script_file.write_text(code, encoding="utf-8")
    out_file = tmp_path / "outputs.json"
    env = {**os.environ, "MPLBACKEND": "Agg", "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    done = subprocess.run([sys.executable, "-I", "-c", RUNNER, str(script_file), str(out_file)], cwd=clone, env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"{script} failed as shown:\n{done.stderr[-3000:]}"
    if expected is None:
        return
    printed = [output for output in json.loads(out_file.read_text(encoding="utf-8")) if output]
    assert _normalized(printed) == _normalized(expected), f"{script}: the page shows other outputs than it prints"
