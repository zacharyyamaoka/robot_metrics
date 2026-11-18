"""A REPL transcript harness: run code cells in one namespace and keep exactly what each printed.

    session = Session({})
    session.cell("label", "caption", "print(1 + 1)")
    session.cells          # [{"label", "caption", "code", "output", "seconds"}]
    session.transcript()   # the cells as a terminal would show them (>>> / ...)

The code shown is exactly the code that ran; a cell's last expression is echoed like the REPL does. Ported from the
metrics-API proposals' ``_session.py`` (robot_models f0daa34).
"""

from __future__ import annotations

import ast
import contextlib
import io
import time
from typing import Any


class Session:
    def __init__(self, namespace: dict[str, Any]) -> None:
        self.namespace = namespace
        self.cells: list[dict[str, str]] = []

    def cell(self, label: str, caption: str, code: str) -> None:
        code = code.strip("\n")
        tree = ast.parse(code)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        buffer = io.StringIO()
        started = time.perf_counter()
        with contextlib.redirect_stdout(buffer):
            exec(compile(tree, f"<cell {label}>", "exec"), self.namespace)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), f"<cell {label}>", "eval"), self.namespace)
                if value is not None:
                    print(repr(value))
        self.cells.append({"label": label, "caption": caption, "code": code, "output": buffer.getvalue().rstrip(),
                           "seconds": f"{time.perf_counter() - started:.2f}"})

    @staticmethod
    def prompt(code: str) -> str:
        return "\n".join(("... " if line.startswith((" ", ")", "]", "}")) else ">>> ") + line
                         for line in code.splitlines())

    def transcript(self) -> str:
        parts = []
        for number, cell in enumerate(self.cells, start=1):
            parts.append(f"# {number} · {cell['label']}   ({cell['seconds']} s)\n{self.prompt(cell['code'])}\n{cell['output']}")
        return "\n\n".join(parts)
