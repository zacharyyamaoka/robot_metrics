"""Hero video: CAN 11's (q, qd) coverage filling in, one real play per frame, in recording order.

Left: runs per cell so far (white = never visited, grey = unreachable). Right: visited % of reachable cells, as recorded.
The jump at the end is the Nov 18 full-turn runs. Every frame is computed from the recordings (data.pendulum_runs).

    taskset -c 2-5,16,17 /home/bam/robot_metrics/.venv/bin/python hero.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

import pendulum as pend
import style
from style import ACCENT, LIMIT, MUTED, UNREACHABLE

OUT = Path("/home/bam/robot_metrics/reports/media/coverage-gallery-2026-11-19")
FPS = 30
HOLD_FRAMES = 60


def main() -> dict:
    env = pend.envelope()
    runs = pend.can11_runs()
    ordered = runs["campaign"] + runs["envelope"]
    rows, run_of, _ = pend.samples(ordered)
    grid = pend.grid(rows, env, run_of)
    shape = grid.counts.shape
    q_edges = (grid.axes[0].edges() - env["bottom"]) * pend.DEG
    qd_edges = grid.axes[1].edges() * pend.DEG
    reach = grid.reachable
    runs_so_far = np.zeros(shape)
    visited_pct = []
    for cells in grid.run_cells:
        flat = runs_so_far.ravel()
        flat[cells] += 1
        visited_pct.append(100.0 * ((runs_so_far > 0) & reach).sum() / reach.sum())
    total = np.zeros(shape)
    frames_dir = Path(tempfile.mkdtemp(prefix="coverage-hero-"))
    vmax = float(np.percentile(grid.runs_per_cell[grid.runs_per_cell > 0], 98))
    frame_index = 0
    schedule = list(range(len(grid.run_cells))) + [len(grid.run_cells) - 1] * HOLD_FRAMES
    for step, run_index in enumerate(schedule):
        if step < len(grid.run_cells):
            total.ravel()[grid.run_cells[run_index]] += 1
        fig, (left, right) = style.figure(9.6, 3.6, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
        style.heatmap(left, total, q_edges, qd_edges, reach, vmax=vmax, colorbar_label="runs through the cell", figure=fig)
        left.set_xlabel("q from bottom (deg)")
        left.set_ylabel("qd (deg/s)")
        name = ordered[run_index]["round"].replace("envelope/", "Nov 18 full turn · ").replace("round", "round ")
        left.set_title(f"CAN 11 · play {run_index + 1} of {len(ordered)} · {name}", loc="left")
        x = np.arange(1, run_index + 2)
        right.plot(x, visited_pct[: run_index + 1], color=ACCENT, linewidth=2)
        right.scatter([run_index + 1], [visited_pct[run_index]], color=ACCENT, s=25, zorder=3)
        right.axhline(100, color=LIMIT, linewidth=1)
        right.axvline(len(runs["campaign"]) + 0.5, color=MUTED, linewidth=0.7, linestyle=":")
        right.text(len(runs["campaign"]) - 6, 6, "full-turn runs →", ha="right", fontsize=7.5, color=MUTED)
        right.set_xlim(0, len(ordered) + 5)
        right.set_ylim(0, 105)
        right.set_xlabel("plays recorded")
        right.set_ylabel("% of reachable (q, qd) cells")
        right.set_title(f"{visited_pct[run_index]:.0f} % visited", loc="left", fontsize=12, color=ACCENT)
        fig.tight_layout()
        fig.savefig(frames_dir / f"{frame_index:05d}.png", dpi=100, facecolor="white")
        style.plt.close(fig)
        frame_index += 1
    OUT.mkdir(parents=True, exist_ok=True)
    mp4, gif = OUT / "hero.mp4", OUT / "hero.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "30",
                    "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    "fps=10,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse",
                    str(gif)], check=True)
    still = OUT / "hero-last.png"
    (frames_dir / f"{frame_index - 1:05d}.png").replace(still)
    return {"mp4": str(mp4), "gif": str(gif), "still": str(still), "frames": frame_index,
            "seconds": frame_index / FPS, "final_pct": visited_pct[-1], "campaign_pct": visited_pct[len(runs["campaign"]) - 1]}


if __name__ == "__main__":
    print(main())
