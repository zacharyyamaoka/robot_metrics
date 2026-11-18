"""Test 7 · import isolation: the core is numpy + robot_msgs + stdlib; matplotlib, wandb and rerun only inside their
adapters; never io-diagnostics, sim2real, robot_models or robotparts.

Held red: planted violations must fail the same checker, so a checker that passes everything cannot pass this file.
The isolated run uses ``env -u VIRTUAL_ENV uv run --isolated`` (bam_ws CLAUDE.md: clear VIRTUAL_ENV first, or both
the red and the green result are lying).
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "robot_metrics"
CORE_ROOTS = {"numpy", "robot_msgs", "robot_metrics"}
# WHY only these files may name a viewer library: plot() and the W&B / Rerun adapters are optional extras.
ADAPTERS = {"views.py": {"matplotlib"}, "adapters/wandb.py": {"wandb"}, "adapters/rerun.py": {"rerun"}}
NEVER = ("robot_models", "sim2real", "robotparts", "robot_parts", "robotdx", "io_diagnostics", "bam_rack_twin",
         "traj_compare", "torch", "torchmetrics", "pandas", "scipy")
# the frozen check's rule: no package file may open an oracle file
ORACLE_READS = re.compile(r"scorecards|latest_real|summary\.json|check_scenario|oracle_residual_rms|_truth")


def _imports(source: str) -> list[tuple[str, bool]]:
    """(root module, imported at module level?) for every import in a file."""

    tree = ast.parse(source)
    top = {id(node) for node in tree.body}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name.split(".")[0], id(node) in top) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            root = "robot_metrics" if node.level else (node.module or "").split(".")[0]
            found.append((root, id(node) in top))
    return found


def violations(src: Path = SRC) -> list[str]:
    bad = []
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(src).as_posix()
        text = path.read_text(encoding="utf-8")
        if ORACLE_READS.search(text):
            bad.append(f"{rel}: names an oracle file")
        for root, at_module_level in _imports(text):
            if root in sys.stdlib_module_names or root == "__future__" or root in CORE_ROOTS:
                continue
            if root in ADAPTERS.get(rel, set()) and not at_module_level:
                continue
            bad.append(f"{rel}: imports {root}" + (" at module level" if root in ADAPTERS.get(rel, set()) else ""))
    return bad


def test_package_imports_only_its_layer():
    assert violations() == []


def _planted(tmp_path: Path, rel: str, text: str) -> Path:
    root = tmp_path / "robot_metrics"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")
    return root


@pytest.mark.parametrize("rel, text, expected", [
    ("stock.py", "import sim2real\n", "stock.py: imports sim2real"),
    ("collection.py", "from robot_models import IntegratedServo\n", "collection.py: imports robot_models"),
    ("metric.py", "def f():\n    import robotdx\n", "metric.py: imports robotdx"),
    ("views.py", "import matplotlib\n", "views.py: imports matplotlib at module level"),
    ("stock.py", "def f():\n    import matplotlib\n", "stock.py: imports matplotlib"),
    ("datasets/onedof_round8.py", "PATH = 'scorecards/latest_real.json'\n", "datasets/onedof_round8.py: names an oracle file"),
])
def test_checker_goes_red_on_a_planted_violation(tmp_path, rel, text, expected):
    assert violations(_planted(tmp_path, rel, text)) == [expected]


CHECK_MODULES = (
    "import sys, robot_metrics, robot_metrics.catalog, robot_metrics.functional, robot_metrics.views, "
    "robot_metrics.adapters.wandb, robot_metrics.adapters.rerun, robot_metrics.datasets.onedof_round8, "
    "robot_metrics.datasets.fivebar_ladder\n"
    f"bad = sorted(m for m in {('matplotlib', 'wandb', 'rerun') + NEVER!r} if m in sys.modules)\n"
    "assert not bad, bad\n"
    "print('robot_metrics', robot_metrics.__version__, 'imports clean')\n"
)


def test_importing_the_package_pulls_in_no_viewer_or_upper_layer():
    subprocess.run([sys.executable, "-c", CHECK_MODULES], check=True)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_core_imports_in_an_isolated_environment():
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    done = subprocess.run(["uv", "run", "--isolated", "--no-dev", "--project", str(ROOT), "python", "-c", CHECK_MODULES],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "imports clean" in done.stdout
