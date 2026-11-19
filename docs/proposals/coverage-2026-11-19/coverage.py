"""Coverage of a robot dataset, as numbers: how much of the space the robot CAN reach did the data actually visit?

Every number here is relative to the limits (Zach, Daily Note 2026-11-19: "When we talk about coverage yes it needs to
be relative to kinda like the limits"). The limits arrive as `Axis` boxes plus an optional `reachable` mask (a torque-
speed envelope, a workspace from FK, the other agent's reachability map); the samples arrive as an (N, D) array cut from
robot_msgs streams. numpy only, so it can move into robot_metrics' core unchanged if Zach picks it.

    axes = (Axis("q", -3.06, 3.31, 24, "rad", 57.3, "deg"), Axis("qd", -10, 10, 20, "rad/s", 57.3, "deg/s"))
    grid = Grid.of(samples, axes, run_ids=run_of_each_sample)
    grid.visited_pct()            # share of reachable cells with at least one sample
    grid.sufficient_pct(10)       # ... with at least 10 samples (enough to fit, not just a fly-by)
    grid.effective_pct()          # exp(entropy) / reachable: how evenly the samples spread
    grid.conditional(given=0)     # per q cell: share of its reachable qd cells visited ("fast in many configurations")
    dispersion(samples, axes)     # radius of the largest empty ball, in limit-normalized units
    rarefaction(grid)             # visited % as runs are added: is more of the same data still buying coverage?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Axis:
    """One dimension of the space, its limit and its binning. SI inside; `display_scale` and `display_unit` only at display."""

    name: str
    low: float
    high: float
    bins: int
    unit: str = ""
    display_scale: float = 1.0
    display_unit: str = ""

    def edges(self) -> np.ndarray:
        # WHY linear bins (Zach, 2026-11-19, on sim2real's log-spaced qd/qdd grid: "I feel it would be better to just have
        # a linear scale?"): equal cells mean equal-sized holes, so a visited % is a share of the space, not of a warped one.
        return np.linspace(self.low, self.high, self.bins + 1)

    def centers(self) -> np.ndarray:
        edges = self.edges()
        return 0.5 * (edges[:-1] + edges[1:])

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Map the limit span onto [0, 1], so distances in different units are comparable."""

        return (np.asarray(values, float) - self.low) / (self.high - self.low)

    def label(self) -> str:
        unit = self.display_unit or self.unit
        return f"{self.name} ({unit})" if unit else self.name


@dataclass
class Grid:
    """Sample counts per cell of a box of `Axis` bins, and which cells the robot can reach at all."""

    axes: tuple[Axis, ...]
    counts: np.ndarray
    reachable: np.ndarray
    runs_per_cell: np.ndarray | None = None      # how many distinct runs touched each cell (for rarefaction / Chao2)
    run_cells: list[np.ndarray] = field(default_factory=list)   # flat cell indices each run touched

    @classmethod
    def of(cls, samples: np.ndarray, axes: tuple[Axis, ...], reachable: np.ndarray | None = None,
           run_ids: np.ndarray | None = None) -> "Grid":
        samples = np.asarray(samples, float)
        if samples.ndim != 2 or samples.shape[1] != len(axes):
            raise ValueError(f"samples must be (N, {len(axes)}), got {samples.shape}")
        counts, _ = np.histogramdd(samples, bins=[axis.edges() for axis in axes])
        shape = tuple(axis.bins for axis in axes)
        reachable = np.ones(shape, bool) if reachable is None else np.asarray(reachable, bool)
        grid = cls(axes, counts, reachable)
        if run_ids is not None:
            run_ids = np.asarray(run_ids)
            cell_of_sample = grid.cell_index(samples)
            inside = cell_of_sample >= 0
            runs_per_cell = np.zeros(int(np.prod(shape)), int)
            for run in np.unique(run_ids):
                cells = np.unique(cell_of_sample[inside & (run_ids == run)])
                grid.run_cells.append(cells)
                runs_per_cell[cells] += 1
            grid.runs_per_cell = runs_per_cell.reshape(shape)
        return grid

    def cell_index(self, samples: np.ndarray) -> np.ndarray:
        """Flat cell index of each sample, -1 for a sample outside the limit box."""

        index = np.zeros(len(samples), int)
        outside = np.zeros(len(samples), bool)
        for column, axis in enumerate(self.axes):
            position = np.floor(axis.normalize(samples[:, column]) * axis.bins).astype(int)
            outside |= (position < 0) | (position >= axis.bins)
            index = index * axis.bins + np.clip(position, 0, axis.bins - 1)
        index[outside] = -1
        return index

    # --- the numbers -------------------------------------------------------------------------------------------------

    @property
    def reachable_cells(self) -> int:
        return int(self.reachable.sum())

    def visited_pct(self) -> float:
        return 100.0 * float(((self.counts > 0) & self.reachable).sum()) / max(self.reachable_cells, 1)

    def sufficient_pct(self, at_least: int) -> float:
        """Share of reachable cells holding at least `at_least` samples: visited is not the same as enough to fit."""

        return 100.0 * float(((self.counts >= at_least) & self.reachable).sum()) / max(self.reachable_cells, 1)

    def effective_pct(self) -> float:
        """exp(Shannon entropy of the counts) as a share of reachable cells: 100 % only if every cell holds the same.

        The ML community's "effective number of classes" (Hill number of order 1). A dataset that visits every cell
        once but sits in one cell for an hour has a high visited % and a low effective %."""

        counts = self.counts[self.reachable].astype(float)
        total = counts.sum()
        if total == 0:
            return 0.0
        share = counts[counts > 0] / total
        return 100.0 * math.exp(float(-(share * np.log(share)).sum())) / max(self.reachable_cells, 1)

    def imbalance_ratio(self) -> float:
        """Most-visited cell over the median visited cell: the class-imbalance ratio, cells as classes."""

        visited = self.counts[(self.counts > 0) & self.reachable]
        return float(visited.max() / np.median(visited)) if visited.size else float("nan")

    def conditional(self, given: int) -> np.ndarray:
        """For each cell of axis `given`: the share of its reachable cells on the other axes that were visited.

        Answers "are we going many velocities in each configuration" (given=q on a q x qd grid) and "many
        accelerations at each velocity" (given=qd on a qd x qdd grid). NaN where the slice has no reachable cell."""

        moved_counts = np.moveaxis(self.counts, given, 0).reshape(self.counts.shape[given], -1)
        moved_reach = np.moveaxis(self.reachable, given, 0).reshape(self.counts.shape[given], -1)
        reachable = moved_reach.sum(axis=1)
        visited = ((moved_counts > 0) & moved_reach).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(reachable > 0, visited / reachable, np.nan)

    def holes(self, top: int = 5) -> list[tuple[tuple[float, ...], int]]:
        """The biggest empty reachable regions, as (center of the region in SI, its size in cells), largest first.

        Connected components of empty reachable cells (face neighbours): what the next run should fill."""

        empty = (self.counts == 0) & self.reachable
        label = np.full(empty.shape, -1, int)
        regions: list[tuple[tuple[float, ...], int]] = []
        centers = [axis.centers() for axis in self.axes]
        for start in zip(*np.nonzero(empty)):
            if label[start] >= 0:
                continue
            stack, members = [start], []
            label[start] = len(regions)
            while stack:
                cell = stack.pop()
                members.append(cell)
                for dim in range(empty.ndim):
                    for step in (-1, 1):
                        neighbour = list(cell)
                        neighbour[dim] += step
                        if 0 <= neighbour[dim] < empty.shape[dim]:
                            neighbour = tuple(neighbour)
                            if empty[neighbour] and label[neighbour] < 0:
                                label[neighbour] = len(regions)
                                stack.append(neighbour)
            middle = np.mean(np.array(members), axis=0)
            regions.append((tuple(float(np.interp(m, np.arange(len(c)), c)) for m, c in zip(middle, centers)),
                            len(members)))
        return sorted(regions, key=lambda region: -region[1])[:top]

    def chao2_reachable_estimate(self) -> float:
        """Incidence-based Chao2 (ecology's unseen-species estimator, runs as sampling units): how many cells this kind
        of run would eventually visit if you kept recording it. Below the reachable count = more of the same data will
        never close the gap; you need a different kind of run."""

        if self.runs_per_cell is None:
            raise ValueError("Chao2 needs run_ids")
        runs = len(self.run_cells)
        seen = int((self.runs_per_cell > 0).sum())
        once = int((self.runs_per_cell == 1).sum())
        twice = int((self.runs_per_cell == 2).sum())
        correction = (runs - 1) / runs if runs > 1 else 1.0
        if twice == 0:
            return seen + correction * once * (once - 1) / 2.0
        return seen + correction * once * once / (2.0 * twice)


def rarefaction(grid: Grid, shuffles: int = 32, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visited % of reachable cells after the first k runs, k = 1..R, averaged over random run orders.

    Ecology's species-accumulation curve. Still climbing: record more of the same. Flat below 100 %: the remaining
    cells need a different trajectory, not more repeats. Returns (k, mean %, 10th-90th percentile band as (2, R))."""

    if not grid.run_cells:
        raise ValueError("rarefaction needs run_ids")
    reachable_flat = grid.reachable.ravel()
    runs = len(grid.run_cells)
    rng = np.random.default_rng(seed)
    curves = np.zeros((shuffles, runs))
    for shuffle in range(shuffles):
        seen = np.zeros(reachable_flat.size, bool)
        for k, run in enumerate(rng.permutation(runs)):
            seen[grid.run_cells[run]] = True
            curves[shuffle, k] = 100.0 * (seen & reachable_flat).sum() / max(reachable_flat.sum(), 1)
    return np.arange(1, runs + 1), curves.mean(axis=0), np.percentile(curves, [10, 90], axis=0)


def _nearest_distance(probes: np.ndarray, points: np.ndarray, chunk: int = 512) -> np.ndarray:
    nearest = np.empty(len(probes))
    points_sq = (points ** 2).sum(axis=1)
    for start in range(0, len(probes), chunk):
        block = probes[start:start + chunk]
        squared = (block ** 2).sum(axis=1)[:, None] - 2.0 * block @ points.T + points_sq[None, :]
        nearest[start:start + chunk] = np.sqrt(np.maximum(squared.min(axis=1), 0.0))
    return nearest


def gap_distances(samples: np.ndarray, axes: tuple[Axis, ...], inside=None, probes: int = 20000,
                  max_points: int = 20000, seed: int = 0) -> np.ndarray:
    """Distance from random reachable points to the nearest sample, in limit-normalized units (each axis span = 1).

    Its maximum is the dataset's dispersion (LaValle: the radius of the largest empty ball); its quantiles say "90 %
    of the reachable space is within r of something we recorded". `inside(points_SI) -> bool` drops unreachable probes."""

    rng = np.random.default_rng(seed)
    normalized = np.column_stack([axis.normalize(samples[:, i]) for i, axis in enumerate(axes)])
    normalized = normalized[np.all((normalized >= 0) & (normalized <= 1), axis=1)]
    if len(normalized) > max_points:
        normalized = normalized[rng.choice(len(normalized), max_points, replace=False)]
    candidates = rng.random((probes * 3, len(axes)))
    if inside is not None:
        si = np.column_stack([axis.low + candidates[:, i] * (axis.high - axis.low) for i, axis in enumerate(axes)])
        candidates = candidates[inside(si)]
    return _nearest_distance(candidates[:probes], normalized)


def largest_holes(samples: np.ndarray, axes: tuple[Axis, ...], inside=None, count: int = 5, probes: int = 20000,
                  seed: int = 0) -> list[tuple[np.ndarray, float]]:
    """The `count` biggest empty balls, biggest first, as (centre in SI, radius in limit-normalized units).

    Farthest-first over reachable probes (Gonzalez's k-center traversal): take the probe farthest from every sample,
    then treat its ball as filled and repeat. The list doubles as a collection plan: record a run through hole 1 first.
    Bin-free, so one connected empty region on a fine grid still splits into its separate bulges."""

    rng = np.random.default_rng(seed)
    normalized = np.column_stack([axis.normalize(samples[:, i]) for i, axis in enumerate(axes)])
    normalized = normalized[np.all((normalized >= 0) & (normalized <= 1), axis=1)]
    if len(normalized) > 20000:
        normalized = normalized[rng.choice(len(normalized), 20000, replace=False)]
    candidates = rng.random((probes * 3, len(axes)))
    if inside is not None:
        si = np.column_stack([axis.low + candidates[:, i] * (axis.high - axis.low) for i, axis in enumerate(axes)])
        candidates = candidates[inside(si)]
    candidates = candidates[:probes]
    gap = _nearest_distance(candidates, normalized)
    holes = []
    for _ in range(count):
        best = int(np.argmax(gap))
        radius = float(gap[best])
        centre = candidates[best]
        holes.append((np.array([axis.low + centre[i] * (axis.high - axis.low) for i, axis in enumerate(axes)]), radius))
        gap = np.minimum(gap, np.linalg.norm(candidates - centre, axis=1))      # its ball now counts as covered
    return holes


def dispersion(samples: np.ndarray, axes: tuple[Axis, ...], inside=None, **options) -> float:
    return float(gap_distances(samples, axes, inside, **options).max())


def excitation(regressor: np.ndarray) -> dict[str, float]:
    """How well the data pins a linear-in-parameters model tau = regressor @ theta (system identification's view).

    Columns are scaled to unit norm first (the condition number of raw columns mostly measures their units).
    condition: <~100 is well excited; a huge value means two parameters cannot be told apart from this data.
    log_det: D-optimality, log det of the normalized information matrix; higher is better, comparable across datasets
    of the same model."""

    regressor = np.asarray(regressor, float)
    regressor = regressor[np.all(np.isfinite(regressor), axis=1)]
    norms = np.linalg.norm(regressor, axis=0)
    scaled = regressor / np.where(norms > 0, norms, 1.0)
    singular = np.linalg.svd(scaled, compute_uv=False)
    information = scaled.T @ scaled
    sign, log_det = np.linalg.slogdet(information)
    return {"condition": float(singular[0] / max(singular[-1], 1e-300)),
            "log_det": float(log_det) if sign > 0 else float("-inf"),
            "weakest_direction_sv": float(singular[-1])}


def limit_use(values: np.ndarray, low: float, high: float) -> dict[str, float]:
    """How much of one limit span the data used: span share, the extremes as a share of each limit, time near a limit."""

    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    span = high - low
    near = 0.05 * span
    return {"span_pct": 100.0 * (values.max() - values.min()) / span,
            "min_pct_of_low": 100.0 * values.min() / low if low < 0 else float("nan"),
            "max_pct_of_high": 100.0 * values.max() / high if high > 0 else float("nan"),
            "time_near_limit_pct": 100.0 * float(np.mean((values < low + near) | (values > high - near)))}


# --- sphere and rotation bins for grasps ------------------------------------------------------------------------------

def sphere_cell(directions: np.ndarray, rings: int = 12, sectors: int = 24) -> np.ndarray:
    """Equal-area cell of unit vectors: `rings` bands uniform in z (Archimedes: equal z bands have equal area) by
    `sectors` uniform in azimuth. Flat index ring * sectors + sector."""

    directions = np.asarray(directions, float)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    ring = np.clip(((directions[:, 2] + 1.0) / 2.0 * rings).astype(int), 0, rings - 1)
    azimuth = np.arctan2(directions[:, 1], directions[:, 0])
    sector = np.clip(((azimuth + np.pi) / (2 * np.pi) * sectors).astype(int), 0, sectors - 1)
    return ring * sectors + sector
