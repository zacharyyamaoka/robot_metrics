"""The real recordings the gallery reads, cut down to what coverage needs and cached outside the repo.

- `pendulum_runs()`: every real play of the 1-DOF rack campaign (rounds 0-9 and the Nov 18 full-turn envelope runs),
  one entry per board per file: t, q, qd, tau (measured), q_target, round, station, kind. /archive is a slow HDD, so the
  first call reads ~1100 files once and writes a cache (`COVERAGE_GALLERY_CACHE`, default ~/.cache/robot_metrics_coverage).

Read-only on /archive; nothing here writes next to the data.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

CAMPAIGN = Path(os.environ.get("BAM_ONEDOF_CAMPAIGN") or "/archive/datasets/bam_rig/onedof_2026-11-15")
CACHE = Path(os.environ.get("COVERAGE_GALLERY_CACHE") or Path.home() / ".cache/robot_metrics_coverage")
ENVELOPE = CAMPAIGN / "envelope_1118"

_KIND = re.compile(r"^s\d+_(?P<boards>(?:can\d+_)+)(?P<kind>.+)$")


def _runs_files() -> list[Path]:
    files = []
    for folder in sorted(CAMPAIGN.glob("round*/real/*/runs")) + sorted(ENVELOPE.glob("real/*/runs")):
        files += sorted(path for path in folder.glob("*.npz"))
    return files


def _round_of(path: Path) -> str:
    relative = path.relative_to(CAMPAIGN).parts
    return relative[0] if relative[0].startswith("round") else f"envelope/{relative[2]}"


def _read(path: Path) -> list[dict]:
    with np.load(path) as archive:
        columns = {name: np.asarray(archive[name]) for name in archive.files}
    stem = path.stem
    match = _KIND.match(stem)
    kind = match.group("kind") if match else stem
    boards = sorted({name.rsplit("_can", 1)[1] for name in columns if re.match(r"^q_can\d+$", name)}, key=int)
    entries = []
    if not boards:     # single-board layout: plain q, qd, tau
        board = re.search(r"can(\d+)", stem)
        boards_columns = [(board.group(1) if board else "?", "")]
    else:
        boards_columns = [(board, f"_can{board}") for board in boards]
    for board, suffix in boards_columns:
        if f"q{suffix}" not in columns:
            continue
        q = columns[f"q{suffix}"].astype(float)
        keep = np.isfinite(q)
        if keep.sum() < 10:
            continue
        entry = {"file": str(path), "round": _round_of(path), "station": f"can{board}", "kind": kind,
                 "t": columns["t"][keep].astype(float), "q": q[keep]}
        for name in ("qd", "tau", "q_target", "qd_target"):
            values = columns.get(f"{name}{suffix}")
            entry[name] = values[keep].astype(float) if values is not None else None
        entries.append(entry)
    return entries


def pendulum_runs(refresh: bool = False) -> list[dict]:
    cache = CACHE / "pendulum_runs.pkl"
    if cache.is_file() and not refresh:
        return pickle.loads(cache.read_bytes())
    entries = []
    for path in _runs_files():
        try:
            entries += _read(path)
        except Exception as error:          # a half-written or foreign file never stops the gallery
            print(f"skip {path}: {error}")
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(entries, protocol=pickle.HIGHEST_PROTOCOL))
    return entries


def envelope_config(name: str = "challenge2") -> dict:
    return json.loads((ENVELOPE / name / "config.json").read_text())


if __name__ == "__main__":
    runs = pendulum_runs(refresh=True)
    print(len(runs), "board-runs;", sum(len(r["q"]) for r in runs), "samples")
