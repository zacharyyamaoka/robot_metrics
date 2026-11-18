"""The README's example runs as written, and its printed results are the ones it shows (a public README stays true)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_the_readme_example_runs_and_shows_its_real_output():
    block = re.search(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S).group(1)
    tree = ast.parse(block)
    namespace: dict = {}
    shown = []
    for node in tree.body:
        if isinstance(node, ast.Expr):
            shown.append(eval(compile(ast.Expression(node.value), "README", "eval"), namespace))
        else:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "README", "exec"), namespace)
    expected = [ast.literal_eval(line.removeprefix("# ")) for line in block.splitlines() if line.startswith("# ")]
    assert shown == expected
