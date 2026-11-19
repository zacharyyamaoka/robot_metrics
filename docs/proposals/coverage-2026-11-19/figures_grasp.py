"""Section 4 (grasps): 6-DOF grasp coverage, position x orientation x gripper opening, per object, against the gripper.

Zach (Daily Note 2026-11-19): "we have a dataset that has a bunch of 6 DOF grasps... I imagine we can use like the thin
grippers to show where a lot of them are? heatmap works for like 2 DOF possibly... well we can use histograms as well";
"When we start looking at the gripper hand I want to make sure the dataset has like good coverage of picking at these
6 DOF locations with different gripper hand positions."

Real data only (grasp_data.py): GraspNet-1Billion's analytic labels on real scanned YCB objects, GraspClutter6D's 200
objects, and our own real full-arm repicks; bam_sim grasp_success is used once and says "simulated" in its source.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import grasp_data as gd
import style
from cards import Card
from coverage import Axis, Grid, limit_use, sphere_cell
from style import ACCENT, ACCENT_DARK, CATEGORY, HAIR, INK, LIMIT, MUTED, UNREACHABLE

DEG = 180.0 / np.pi
RINGS, SECTORS = 8, 16          # 128 equal-area direction cells: every cell holds 1-4 of GraspNet's 300 label views
PLANNER_SPAN_MM = 70.0          # the repick planner's claw span (look.item.claw_span_mm): what it plans an item into
GRASPNET_GRIPPER_MM = 100.0     # graspnetAPI's evaluation gripper (eval_scene max_width=0.1)
WIDE_ORANGE = "#e8590c"
OBJECT_COLOUR = {0: CATEGORY[0], 2: CATEGORY[1], 5: CATEGORY[2], 7: CATEGORY[3]}


# --- shared data ------------------------------------------------------------------------------------------------------

def _context():
    views = gd.generate_views()
    objects = {object_id: gd.graspnet_object(object_id) for object_id in gd.GRASPNET_OBJECTS}
    runs = gd.repicks()
    plans = [plan for run in runs for plan in run["plans"]]
    return {"views": views, "objects": objects, "clutter": gd.graspclutter_objects(), "runs": runs, "plans": plans,
            "sim": gd.sim_grasps()}


def _claw_max_open_mm(ctx) -> float:
    """The claw's widest pad gap, at the open stop (first_grasp.opening.max_open_mm, from claw_linkage.json)."""

    values = [plan.get("max_open_mm") for plan in ctx["plans"] if plan.get("max_open_mm")]
    return float(np.median(values)) if values else 124.79


def _graspnet_source(ctx, ids=None) -> str:
    chosen = [ctx["objects"][i] for i in (ids or ctx["objects"])]
    labels = sum(o["total_labels"] for o in chosen)
    valid = sum(o["total_valid"] for o in chosen)
    names = ", ".join(f"{o['id']:03d} {o['name']}" for o in chosen)
    return (f"GraspNet-1Billion grasp_label (real scanned YCB objects, analytic labels), {len(chosen)} objects ({names}): "
            f"{labels / 1e6:.0f} M labels = surface points x 300 approach views x 12 rolls x 4 depths; "
            f"{valid / 1e6:.2f} M valid (friction coefficient ≤ 0.4, collision-free: graspnetAPI's loadGrasp defaults)")


def _clutter_source(ctx) -> str:
    rows = ctx["clutter"]
    return (f"GraspClutter6D grasp_label (real objects, same label layout), all {len(rows)} objects: "
            f"{sum(r['labels'] for r in rows) / 1e9:.2f} B labels, {sum(r['valid'] for r in rows) / 1e6:.0f} M valid "
            f"(friction ≤ 0.4, collision-free)")


def _closed(plan) -> bool:
    return plan.get("outcome") in ("success", "missed", "suspect")


def _repick_source(ctx, extra: str = "") -> str:
    plans = ctx["plans"]
    stamps = sorted(plan["stamp_utc"][:10] for plan in plans if plan.get("stamp_utc"))
    runs = len({plan["run"] for plan in plans})
    closed = sum(_closed(plan) for plan in plans)
    success = sum(plan.get("outcome") == "success" for plan in plans)
    return (f"our real full-arm repicks (rack 2026-11-10 campaign, pick_loop_full_arm_repick_real_*, {stamps[0]} to "
            f"{stamps[-1]}): {len(plans)} planned 6-DOF grasps in {runs} runs, {closed} closed on the item, {success} "
            f"succeeded{extra}")


def _from_directions(ctx) -> np.ndarray:
    """Where the gripper comes FROM, per GraspNet view, in the object frame: the view point itself (approach = -view)."""

    return ctx["views"].astype(float)


def _sphere_edges():
    longitude = np.linspace(-np.pi, np.pi, SECTORS + 1)
    latitude = np.arcsin(np.linspace(-1.0, 1.0, RINGS + 1))   # rings uniform in z = equal area
    return longitude, latitude


def _sphere_axes(fig, where):
    ax = fig.add_subplot(*where, projection="mollweide")
    ax.grid(color=HAIR, linewidth=0.4)
    ax.set_xticklabels([])
    ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))
    ax.set_yticklabels(["-60°", "-30°", "0°", "+30°", "+60°"], fontsize=6, color=MUTED)
    return ax


def _sphere_mesh(ax, cells: np.ndarray, cmap=style.EMPTY_WHITE, vmax=None, vmin=None):
    longitude, latitude = _sphere_edges()
    grid = cells.reshape(RINGS, SECTORS).astype(float)
    shown = np.ma.masked_where(grid <= 0, grid)
    vmax = vmax or max(float(np.percentile(grid[grid > 0], 98)) if (grid > 0).any() else 1.0, 1.0)
    if vmin is None:
        vmin = min(vmax, float(grid[grid > 0].min())) if (grid > 0).any() else 0.0
    mesh = ax.pcolormesh(longitude, latitude, shown, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", edgecolors=HAIR,
                         linewidth=0.2)
    outline = np.ma.masked_where(grid > 0, np.zeros_like(grid))
    ax.pcolormesh(longitude, latitude, outline, cmap=ListedColormap(["#ffffff"]), shading="flat", edgecolors=HAIR,
                  linewidth=0.2)
    return mesh


def _colorbar(fig, mesh, ax, label, **options):
    bar = fig.colorbar(mesh, ax=ax, **({"fraction": 0.046, "pad": 0.03} | options))
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=6.5, colors=MUTED)
    bar.set_label(label, fontsize=7, color=MUTED)
    return bar


def _gini(values) -> float:
    values = np.sort(np.asarray(values, float))
    n = len(values)
    return float(2.0 * np.sum(np.arange(1, n + 1) * values) / (n * values.sum()) - (n + 1) / n)


def _effective(values) -> float:
    share = np.asarray(values, float)
    share = share[share > 0] / share.sum()
    return float(np.exp(-(share * np.log(share)).sum()))


def _cross(a, b):
    """z of the 2-D cross product (numpy 2 dropped 2-vector np.cross)."""

    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull, counter-clockwise (Andrew's monotone chain), numpy only."""

    points = np.unique(np.asarray(points, float), axis=0)
    if len(points) < 3:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def half(sequence):
        chain = []
        for point in sequence:
            while len(chain) >= 2 and _cross(chain[-1] - chain[-2], point - chain[-2]) <= 0:
                chain.pop()
            chain.append(point)
        return chain

    lower, upper = half(points), half(points[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _inside_hull(hull: np.ndarray, points: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    inside = np.ones(len(points), bool)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        inside &= _cross(b - a, points - a) >= -tolerance
    return inside


def _gripper_segments(center, rotation, width, depth, tail=0.03, palm=0.02) -> np.ndarray:
    """graspnetAPI plot_gripper_pro_max as four thin lines (robot_msgs viz._gripper's glyph): x = approach, y = jaws;
    the palm sits `palm` behind the grasp point, the fingers reach `depth` past it, the tail sticks out behind the palm."""

    count = len(center)
    half = np.asarray(width) / 2.0
    local = np.zeros((count, 6, 3))
    local[:, 0, 0] = -palm - tail
    local[:, 1, 0] = -palm
    local[:, 2, 0], local[:, 2, 1] = -palm, -half
    local[:, 3, 0], local[:, 3, 1] = -palm, half
    local[:, 4, 0], local[:, 4, 1] = depth, -half
    local[:, 5, 0], local[:, 5, 1] = depth, half
    world = np.einsum("nij,nkj->nki", rotation, local) + np.asarray(center)[:, None, :]
    return np.stack([world[:, [a, b]] for a, b in ((0, 1), (2, 3), (2, 4), (3, 5))], axis=1)


def _equal_3d(ax, points, scale_mm: float = 50.0):
    """Equal axes, no panes or ticks; a `scale_mm` bar under the object stands in for the axes."""

    low, high = points.min(axis=0), points.max(axis=0)
    middle, half = (low + high) / 2, (high - low).max() / 2
    for setter, centre in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), middle):
        setter(centre - half, centre + half)
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.set_axis_off()
    start = np.array([middle[0] - scale_mm / 2, middle[1] - half, middle[2] - half])
    ax.plot([start[0], start[0] + scale_mm], [start[1]] * 2, [start[2]] * 2, color=INK, linewidth=1.2)
    ax.text(start[0] + scale_mm / 2, start[1], start[2] - 0.12 * half, f"{scale_mm:.0f} mm", fontsize=6.5, color=INK,
            ha="center", va="top")


# --- GraspNet-1Billion: where on the object, from where, how wide ----------------------------------------------------

def grasp_wireframes(ctx) -> Card:
    SHOWN = 150          # WHY ~150: past that the wireframes hide the object they sit on (tried 220)
    rng = np.random.default_rng(7)
    views = ctx["views"]
    fig = style.plt.figure(figsize=(10.0, 2.8))
    numbers = {}
    for index, (object_id, obj) in enumerate(ctx["objects"].items()):
        ax = fig.add_subplot(1, 4, index + 1, projection="3d")
        chosen = rng.choice(len(obj["point"]), SHOWN, replace=False)
        view = obj["view"][chosen]
        angle = obj["angle"][chosen] * np.pi / gd.NUM_ANGLES
        rotation = gd.grasp_rotation(views[view], angle)
        width = obj["width_mm"][chosen].astype(float) / 1000.0
        depth = gd.DEPTHS_M[obj["depth"][chosen]]
        segments = _gripper_segments(obj["points"][obj["point"][chosen]], rotation, width, depth, tail=0.012) * 1000.0
        fits = width * 1000.0 <= PLANNER_SPAN_MM
        colours = np.repeat(np.where(fits, ACCENT, WIDE_ORANGE), 4)
        points_mm = obj["points"] * 1000.0
        ax.scatter(*points_mm.T, s=1.0, color=MUTED, alpha=0.6, linewidths=0, depthshade=False)
        ax.add_collection3d(Line3DCollection(segments.reshape(-1, 2, 3), colors=colours, linewidths=0.45, alpha=0.32))
        _equal_3d(ax, np.concatenate([points_mm, segments.reshape(-1, 3)]))
        ax.view_init(elev=24, azim=-58)
        share = 100.0 * np.mean(obj["width_mm"].astype(float) <= PLANNER_SPAN_MM)
        ax.set_title(f"{obj['name']}: {obj['total_valid'] / 1e3:.0f} k valid", fontsize=8.5, pad=-2)
        numbers[f"{obj['name']}: fits a 70 mm span"] = f"{share:.0f} % of {obj['total_valid'] / 1e3:.0f} k"
    handles = [style.plt.Line2D([], [], color=ACCENT, linewidth=2.5), style.plt.Line2D([], [], color=WIDE_ORANGE, linewidth=2.5)]
    fig.legend(handles, ["opening ≤ 70 mm (our planner's claw span)", "wider than 70 mm"], loc="upper center", ncols=2,
               bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.08, wspace=0.0)
    return Card("grasp-wireframes", "grasp", f"Thin-gripper wireframes over the object: {SHOWN} random valid grasps each, at low opacity",
                "Where on each object do the dataset's grasps pile up, and would our claw open wide enough for them?",
                style.to_data_uri(fig, colors=40), _graspnet_source(ctx) + ". Object frame (z up); palm 2 cm behind the "
                "grasp point, fingers to the label depth, a short tail behind the palm (graspnetAPI's gripper as lines); "
                "grasps drawn at random, so dense regions hold more of the dataset",
                "Contact-GraspNet's line-wireframe grasp plots; graspnetAPI plot_gripper_pro_max; robot_msgs viz._gripper",
                numbers, "grasp 6-DOF", wide=True)


def approach_sphere(ctx) -> Card:
    from_directions = _from_directions(ctx)
    cell = sphere_cell(from_directions, RINGS, SECTORS)
    views_per_cell = np.bincount(cell, minlength=RINGS * SECTORS)
    fig = style.plt.figure(figsize=(8.8, 4.3))
    numbers = {}
    meshes = []
    per_cell = {}
    for object_id, obj in ctx["objects"].items():
        per_view = obj["valid_per_view"].sum(axis=0).astype(float)
        per_cell[object_id] = np.bincount(cell, weights=per_view, minlength=RINGS * SECTORS) / np.maximum(views_per_cell, 1)
    vmax = float(np.percentile(np.concatenate([v[v > 0] for v in per_cell.values()]), 98))
    for index, (object_id, obj) in enumerate(ctx["objects"].items()):
        ax = _sphere_axes(fig, (2, 2, index + 1))
        meshes.append(_sphere_mesh(ax, per_cell[object_id], vmax=vmax))
        visited = 100.0 * np.mean(per_cell[object_id] > 0)
        rich = 100.0 * np.mean(per_cell[object_id] >= 100)
        ax.set_title(f"{obj['name']}: {visited:.0f} % of directions", fontsize=8.5, loc="left")
        numbers[f"{obj['name']}"] = f"{visited:.0f} % of 128 cells, {rich:.0f} % with ≥ 100 per view"
    fig.subplots_adjust(left=0.03, right=0.88, top=0.93, bottom=0.04, hspace=0.25, wspace=0.08)
    bar_ax = fig.add_axes((0.905, 0.15, 0.015, 0.7))
    bar = fig.colorbar(meshes[0], cax=bar_ax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=6.5, colors=MUTED)
    bar.set_label("valid grasps per label view in the cell", fontsize=7, color=MUTED)
    fig.text(0.03, 0.005, "Mollweide (equal-area); top = gripper above the object coming down, bottom = from below; "
             "8 rings uniform in z × 16 sectors; white = no valid grasp", fontsize=7, color=MUTED)
    return Card("approach-sphere", "grasp", "Approach-direction coverage on an equal-area sphere map, one per object",
                "From which directions around each object can it be grasped, and which directions are white holes?",
                style.to_data_uri(fig, colors=56), _graspnet_source(ctx) + "; each cell's count is averaged over the "
                "label views that fall in it (1-4), so the Fibonacci view lattice's unevenness does not show as coverage",
                "Equal-area sphere binning (coverage.sphere_cell, HEALPix's idea); Mollweide maps of approach/normal "
                "directions in grasp-dataset papers", numbers, "grasp 6-DOF", wide=True)


# per object: two orthographic views (horizontal axis, vertical axis, depth axis, +1 if the viewer sits on the +depth
# side, label), picked by hand so each shows the object's telling face (the box's big face, the mug's handle, the
# banana's curve, the can's side and top)
SURFACE_VIEWS = {0: ((1, 2, 0, 1, "big face (y, z)"), (0, 1, 2, 1, "top (x, y)")),
                 2: ((0, 2, 1, -1, "side (x, z)"), (0, 1, 2, 1, "top (x, y)")),
                 5: ((0, 1, 2, 1, "top (x, y)"), (0, 2, 1, -1, "side (x, z)")),
                 7: ((0, 2, 1, -1, "side (x, z)"), (0, 1, 2, 1, "top (x, y)"))}


def surface_directions(ctx) -> Card:
    fig, axes = style.figure(9.4, 4.1, nrows=2, ncols=4)
    numbers = {}
    mesh = None
    for column, (object_id, obj) in enumerate(ctx["objects"].items()):
        points = obj["points"] * 1000.0
        directions = (obj["valid_per_view"] > 0).sum(axis=1)
        share = 100.0 * directions / gd.NUM_VIEWS
        for row, (h, v, depth, toward, name) in enumerate(SURFACE_VIEWS[object_id]):
            ax = axes[row, column]
            # painter's order: the far side first, so the near surface is what you see
            order = np.argsort(points[:, depth] * toward)
            empty = order[share[order] == 0]
            full = order[share[order] > 0]
            ax.scatter(points[empty, h], points[empty, v], s=5, facecolors="white", edgecolors="#9a9aa0", linewidths=0.4)
            mesh = ax.scatter(points[full, h], points[full, v], s=5, c=share[full], cmap=style.EMPTY_WHITE, vmin=0, vmax=60,
                              linewidths=0)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(obj["name"], fontsize=8.5, loc="left")
            ax.set_xlabel(f"{name}, mm", fontsize=6.5, color=MUTED, labelpad=1)
        graspable = 100.0 * np.mean(directions > 0)
        numbers[f"{obj['name']}"] = (f"{graspable:.0f} % of {len(points)} points graspable; median {np.median(share):.0f} % "
                                     f"of directions")
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    _colorbar(fig, mesh, axes, "% of the 300 approach directions with a valid grasp at this point", fraction=0.02, pad=0.01)
    return Card("surface-directions", "grasp", "Orientation coverage per surface point: object points coloured by how "
                "many approach directions grasp there",
                "Zach's ask as a picture: at each location on the object, from how many directions is there a valid "
                "grasp? (White dots: none.)", style.to_data_uri(fig, colors=48),
                _graspnet_source(ctx) + "; colour clipped at 60 % (a few points reach 70-80 %)",
                "Per-point graspability maps (Dex-Net 2.0 / GraspNet's per-point grasp quality, 'graspness' in "
                "GSNet, Wang et al. 2021)", numbers, "grasp 6-DOF", wide=True)


def height_elevation(ctx) -> Card:
    from_directions = _from_directions(ctx)
    elevation = np.degrees(np.arcsin(np.clip(from_directions[:, 2], -1, 1)))
    elevation_axis = Axis("viewpoint elevation", -90.0, 90.0, 12, "deg")
    fig, axes = style.figure(10.0, 2.7, ncols=4, sharey=True)
    numbers = {}
    mesh = None
    for ax, (object_id, obj) in zip(axes, ctx["objects"].items()):
        z = obj["points"][:, 2] * 1000.0
        height_axis = Axis("height on the object", float(z.min()) - 1e-6, float(z.max()) + 1e-6, 10, "mm")
        z_bin = np.clip(((z - height_axis.low) / (height_axis.high - height_axis.low) * 10).astype(int), 0, 9)
        e_bin = np.clip(((elevation + 90.0) / 180.0 * 12).astype(int), 0, 11)
        pair_cell = z_bin[:, None] * 12 + e_bin[None, :]
        total = np.bincount(pair_cell.ravel(), minlength=120).reshape(10, 12)
        good = np.bincount(pair_cell[obj["valid_per_view"] > 0], minlength=120).reshape(10, 12)
        share = np.where(total > 0, 100.0 * good / np.maximum(total, 1), 0.0)
        grid = Grid((height_axis, elevation_axis), good.astype(float), total > 0)
        mesh = style.heatmap(ax, share, height_axis.edges(), elevation_axis.edges(), total > 0, vmax=60, colorbar_label=None)
        per_height = grid.conditional(given=0)
        ax.set_title(obj["name"], fontsize=8.5, loc="left")
        ax.set_xlabel("height of the grasp point (mm)", fontsize=7)
        ax.set_yticks([-90, -45, 0, 45, 90])
        numbers[f"{obj['name']}: elevations per height"] = (f"mean {100 * np.nanmean(per_height):.0f} %, worst "
                                                             f"{100 * np.nanmin(per_height):.0f} %; "
                                                             f"{100 * np.mean(share[total > 0] >= 10):.0f} % of cells ≥ 10 % valid")
    axes[0].set_ylabel("gripper elevation (deg)\n+90 = from above")
    fig.tight_layout(w_pad=0.5)
    _colorbar(fig, mesh, axes, "% of (point, view) labels valid", fraction=0.02, pad=0.01)
    return Card("height-elevation", "grasp", "Height on the object × gripper elevation: share of labels that are valid "
                "grasps, white = none",
                "At each height on the object, from which elevations can the hand come? (Conditional coverage, "
                "location × orientation.)", style.to_data_uri(fig, colors=128),
                _graspnet_source(ctx) + ". Grey = no label pair in the cell; colour clipped at 60 %; 'elevations per "
                "height' = share of the 12 × 15° elevation bands with any valid grasp, per height band (Grid.conditional)",
                "Conditional / stratified coverage (TFDV slicing); grasp-quality vs approach-angle plots in Dex-Net",
                numbers, "grasp 6-DOF", wide=True)


def width_limits(ctx) -> Card:
    max_open = _claw_max_open_mm(ctx)
    edges = gd.WIDTH_EDGES_MM
    fig, axes = style.figure(6.6, 4.0, nrows=3, sharex=True)
    graspnet_widths = np.concatenate([o["width_mm"].astype(float) for o in ctx["objects"].values()])
    bottom = np.zeros(len(edges) - 1)
    for object_id, obj in ctx["objects"].items():
        counts = np.histogram(obj["width_mm"].astype(float), edges)[0] / 1e3
        axes[0].bar(edges[:-1], counts, width=np.diff(edges), bottom=bottom, align="edge", color=OBJECT_COLOUR[object_id],
                    label=obj["name"], linewidth=0)
        bottom += counts
    axes[0].legend(loc="upper left", ncols=4, fontsize=6.5)
    axes[0].set_ylabel("k grasps", fontsize=7)
    clutter = np.sum([row["width_counts"] for row in ctx["clutter"]], axis=0)
    axes[1].bar(edges[:-1], clutter / 1e6, width=np.diff(edges), align="edge", color=ACCENT, linewidth=0)
    axes[1].set_ylabel("M grasps", fontsize=7)
    opening = np.array([plan["open_gap_mm"] for plan in ctx["plans"] if plan.get("open_gap_mm")], float)
    item = np.array([plan["item_width_mm"] for plan in ctx["plans"] if plan.get("item_width_mm")], float)
    axes[2].hist(item, np.arange(0, 152, 2), color="#9db8d9", label="item width (look)")
    axes[2].hist(opening, np.arange(0, 152, 2), color=ACCENT, label="planned opening at approach")
    axes[2].legend(loc="upper right", fontsize=6.5)
    axes[2].set_ylabel("attempts", fontsize=7)
    clutter_fit = 100.0 * clutter[edges[:-1] < PLANNER_SPAN_MM].sum() / clutter.sum()
    labels = [f"GraspNet-1Billion, 4 objects: {100 * np.mean(graspnet_widths <= PLANNER_SPAN_MM):.0f} % ≤ 70 mm",
              f"GraspClutter6D, 200 objects: {clutter_fit:.0f} % ≤ 70 mm",
              f"our repicks: {100 * np.mean(opening <= PLANNER_SPAN_MM):.1f} % of openings ≤ 70 mm"]
    for ax, label in zip(axes, labels):
        ax.axvspan(max_open, 152, color=UNREACHABLE, zorder=-3)
        ax.axvline(max_open, color=LIMIT, linewidth=1.2)
        ax.axvline(PLANNER_SPAN_MM, color=LIMIT, linewidth=1, linestyle=(0, (3, 2)))
        ax.axvline(GRASPNET_GRIPPER_MM, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(label, loc="left", fontsize=8)
        ax.set_xlim(0, 152)
    # the limit names sit above the top panel, clear of the bars (the title ends before 70 mm)
    top_of = axes[0].get_xaxis_transform()
    axes[0].text(PLANNER_SPAN_MM + 1, 1.04, "70 mm planner span", fontsize=6.5, color=LIMIT, transform=top_of)
    axes[0].text(GRASPNET_GRIPPER_MM - 1, 1.18, "GraspNet's eval gripper 100", fontsize=6.5, color=MUTED, ha="right",
                 transform=top_of)
    axes[0].text(max_open + 1, 1.04, f"open stop {max_open:.0f} mm", fontsize=6.5, color=LIMIT, transform=top_of)
    axes[2].set_xlabel("jaw opening (mm), linear")
    fig.tight_layout(h_pad=0.5)
    return Card("width-limits", "grasp", "Grasp width histograms against our claw: two datasets and our own picks, one axis",
                "How much of each dataset's grasps would our claw open wide enough for, and how much of the claw's "
                "range do our picks use?", style.to_data_uri(fig),
                _graspnet_source(ctx) + ". " + _clutter_source(ctx) + ". " + _repick_source(ctx) + ". Red: the claw's "
                f"open stop ({max_open:.1f} mm, claw_linkage.json via first_grasp.max_open_mm) and the planner's 70 mm span "
                "(dashed); grey beyond the stop",
                "TFDV feature histograms against a schema range; gripper-stroke checks in grasp-transfer papers "
                "(Contact-GraspNet's 8 cm Panda width)",
                {"GraspNet ≤ 70 mm": f"{100 * np.mean(graspnet_widths <= PLANNER_SPAN_MM):.1f} %",
                 "GraspClutter6D ≤ 70 mm": f"{clutter_fit:.1f} %",
                 "our openings": f"{opening.min():.0f}-{opening.max():.0f} mm "
                                 f"({limit_use(opening, 0.0, max_open)['span_pct']:.0f} % of the claw's range)"},
                "gripper", wide=True)


def width_elevation(ctx) -> Card:
    max_open = _claw_max_open_mm(ctx)
    from_directions = _from_directions(ctx)
    elevation = np.degrees(np.arcsin(np.clip(from_directions[:, 2], -1, 1)))
    width_edges = np.arange(0.0, 155.0, 5.0)
    elevation_edges = np.linspace(-90.0, 90.0, 13)
    fig, axes = style.figure(10.0, 2.7, ncols=4, sharey=True)
    numbers = {}
    mesh = None
    shares = {}
    for object_id, obj in ctx["objects"].items():
        counts, _, _ = np.histogram2d(obj["width_mm"].astype(float), elevation[obj["view"]], [width_edges, elevation_edges])
        # share of the object's own grasps, so four objects with 0.4-0.9 M grasps share one colour scale
        shares[object_id] = 100.0 * counts / counts.sum()
    positive = np.concatenate([share[share > 0] for share in shares.values()])
    norm = Normalize(vmin=0.0, vmax=float(np.percentile(positive, 98)))
    for ax, (object_id, obj) in zip(axes, ctx["objects"].items()):
        width = obj["width_mm"].astype(float)
        grasp_elevation = elevation[obj["view"]]
        mesh = style.heatmap(ax, shares[object_id], width_edges, elevation_edges, None, norm=norm, colorbar_label=None)
        ax.axvspan(max_open, 155, color=UNREACHABLE, alpha=0.6, zorder=3, linewidth=0)
        ax.axvline(PLANNER_SPAN_MM, color=LIMIT, linewidth=1, linestyle=(0, (3, 2)), zorder=4)
        ax.axvline(max_open, color=LIMIT, linewidth=1.1, zorder=4)
        ax.set_title(obj["name"], fontsize=8.5, loc="left")
        ax.set_xlabel("width (mm)", fontsize=7)
        ax.set_yticks([-90, -45, 0, 45, 90])
        above = grasp_elevation >= 60.0
        numbers[f"{obj['name']}: from above (≥ 60°), ≤ 70 mm"] = (
            f"{100 * np.mean(width[above] <= PLANNER_SPAN_MM):.0f} % of {above.sum() / 1e3:.0f} k")
    axes[0].set_ylabel("gripper elevation (deg)\n+90 = from above")
    fig.tight_layout(w_pad=0.5)
    _colorbar(fig, mesh, axes, "% of the object's\nvalid grasps", fraction=0.02, pad=0.01)
    return Card("width-elevation", "grasp", "Width × gripper elevation per object (white = no grasp), with the claw's "
                "span and open stop",
                "Which approach elevations need a wider hand than ours? Colour left of the dashed line = grasps our "
                "planner's span could take from that elevation.", style.to_data_uri(fig, colors=128),
                _graspnet_source(ctx) + f"; colour = share of the object's valid grasps per 5 mm × 15° cell, linear, "
                f"one scale for all four, clipped at the p98 cell ({norm.vmax:.1f} %); dashed red = 70 mm planner span, solid = {max_open:.0f} mm claw open stop (greyed "
                "beyond). The 'from above' number matters because every one of our real picks comes straight down",
                "Joint 2-D histograms of grasp parameters (Dex-Net / GraspNet dataset statistics figures)", numbers,
                "gripper", wide=True)


def _rose(ax, centres, counts, width, colour, mirror=False):
    """Linear rose: bar length = count / max; a tick on the rim marks every bin with at least one grasp, so a bin of 1
    still reads as covered next to a bin of 700 (white rim = never)."""

    length = counts / counts.max()
    for offset, alpha in ((0.0, 1.0), (np.pi, 0.45)) if mirror else ((0.0, 1.0),):
        ax.bar(centres + offset, length, width=width * 0.92, color=colour, alpha=alpha, linewidth=0)
        ax.bar(centres[counts > 0] + offset, 0.07, bottom=1.08, width=width * 0.92, color=colour, alpha=alpha, linewidth=0)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([])
    ax.tick_params(labelsize=6, colors=MUTED, pad=-1)
    ax.grid(color=HAIR, linewidth=0.4)
    ax.spines["polar"].set_color(HAIR)


def roll_rose(ctx) -> Card:
    fig = style.plt.figure(figsize=(10.0, 2.75))
    numbers = {}
    bins = gd.NUM_ANGLES
    for index, (object_id, obj) in enumerate(ctx["objects"].items()):
        ax = fig.add_subplot(1, 5, index + 1, projection="polar")
        counts = np.bincount(obj["angle"], minlength=bins).astype(float)
        _rose(ax, (np.arange(bins) + 0.5) * np.pi / bins, counts, np.pi / bins, OBJECT_COLOUR[object_id], mirror=True)
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(["0°", "90°", "180°", "270°"])
        evenness = _effective(counts) / bins
        ax.set_title(f"{obj['name']}: roll\nevenness {100 * evenness:.0f} %", fontsize=8, pad=4)
        numbers[f"{obj['name']} roll evenness"] = f"{100 * evenness:.0f} % (exp entropy / 12 rolls)"
    ax = fig.add_subplot(1, 5, 5, projection="polar")
    psi = np.array([plan["psi_rad"] for plan in ctx["plans"] if plan.get("psi_rad") is not None], float)
    edges = np.radians(np.arange(-180, 181, 10))
    counts = np.histogram(psi, edges)[0].astype(float)
    _rose(ax, edges[:-1] + np.radians(5), counts, np.radians(10), ACCENT_DARK)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "±180°", "-90°"])
    visited = 100.0 * np.mean(counts > 0)
    ax.set_title(f"our repicks: yaw ψ\n{visited:.0f} % of 10° bins", fontsize=8, pad=4)
    numbers["our yaw ψ bins visited"] = (f"{visited:.0f} % of 36; p5-p95 {np.percentile(psi * DEG, 5):.0f}° to "
                                         f"{np.percentile(psi * DEG, 95):.0f}°")
    fig.text(0.005, 0.01, "GraspNet: roll of the jaws about the approach axis, 0° = jaws level; a two-finger grasp at roll r "
             "is the one at r + 180°, so the faded half repeats it. Ours: the grasp's yaw about the vertical (asymmetric "
             "claw: full circle). Rim ticks = bins with any grasp", fontsize=6.6, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.1, wspace=0.35)
    return Card("roll-rose", "grasp", "Roll about the approach axis as a rose: four objects, and our picks' yaw",
                "Is every hand roll represented, or do the grasps all hold the jaws one way?", style.to_data_uri(fig, colors=48),
                _graspnet_source(ctx) + ". " + _repick_source(ctx) + " (ψ = plans' grasp.psi_rad). Bar length = count / "
                "the panel's max, linear",
                "Wind roses / circular histograms (Fisher 1993, Statistical Analysis of Circular Data)", numbers,
                "grasp 6-DOF", wide=True)


# --- across objects: GraspClutter6D's 200 ----------------------------------------------------------------------------

def object_imbalance(ctx) -> Card:
    rows = sorted(ctx["clutter"], key=lambda row: -row["valid"])
    valid = np.array([row["valid"] for row in rows], float)
    items = {}
    for plan in ctx["plans"]:
        items[plan.get("item") or "?"] = items.get(plan.get("item") or "?", 0) + 1
    ours = np.array(sorted(items.values(), reverse=True), float)
    fig, (left, right) = style.figure(9.0, 3.0, ncols=2, gridspec_kw={"width_ratios": [2.2, 1]})
    left.bar(np.arange(len(valid)), valid / 1e6, width=1.0, color=ACCENT, linewidth=0)
    left.axhline(np.median(valid) / 1e6, color=MUTED, linewidth=0.8, linestyle=":")
    left.text(len(valid) * 0.98, np.median(valid) / 1e6, "median", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    left.set_xlim(-1, len(valid))
    left.set_xlabel("GraspClutter6D object, sorted by valid grasps")
    left.set_ylabel("valid grasps (M), linear")
    left.set_title("valid grasps per object", loc="left")
    for values, colour, name in ((valid, ACCENT, f"GraspClutter6D, {len(valid)} objects"),
                                 (ours, WIDE_ORANGE, f"our repicks, {len(ours)} items")):
        ascending = np.sort(values)
        lorenz = np.concatenate([[0], np.cumsum(ascending) / ascending.sum()])
        right.plot(np.linspace(0, 1, len(lorenz)), lorenz, color=colour, linewidth=1.4,
                   label=f"{name}: Gini {_gini(values):.2f}", marker="o" if len(values) < 10 else None, markersize=3)
    right.plot([0, 1], [0, 1], color=HAIR, linewidth=0.8)
    right.set_xlabel("share of objects (fewest grasps first)")
    right.set_ylabel("share of grasps")
    right.set_title("Lorenz curves", loc="left")
    right.legend(loc="upper left", fontsize=6.5)
    right.set_aspect("equal")
    fig.tight_layout()
    top_item = max(items, key=items.get)
    return Card("object-imbalance", "grasp", "Per-object grasp counts: sorted bars and Lorenz curves (class imbalance)",
                "Is every object represented about equally, or do a few objects own the dataset?",
                style.to_data_uri(fig), _clutter_source(ctx) + ". Ours: items named in the plans (item_id), "
                + _repick_source(ctx),
                "ML class-imbalance plots (long-tail recognition, LVIS); Lorenz curve / Gini coefficient; Hill number "
                "exp(entropy) as the effective number of classes",
                {"GraspClutter6D Gini": f"{_gini(valid):.2f}",
                 "effective objects": f"{_effective(valid):.0f} of {len(valid)} (exp entropy)",
                 "max / median": f"{valid.max() / np.median(valid):.1f} ×",
                 "ours": f"{top_item} = {100 * items[top_item] / sum(items.values()):.0f} % of {sum(items.values())} plans"},
                "dataset")


def per_object_coverage(ctx) -> Card:
    rows = ctx["clutter"]
    surface = np.array([100.0 * row["points_graspable"] / row["points"] for row in rows])
    directions = np.array([100.0 * row["directions_per_point_median"] for row in rows])
    valid = np.array([row["valid"] for row in rows], float)
    edges = gd.WIDTH_EDGES_MM
    narrow = np.array([row["width_counts"][edges[:-1] < PLANNER_SPAN_MM].sum() / max(row["width_counts"].sum(), 1)
                       for row in rows])
    fig, ax = style.figure(5.0, 3.6)
    fits = narrow >= 0.5
    sizes = 5 + 55 * np.sqrt(valid / valid.max())
    ax.scatter(surface[fits], directions[fits], s=sizes[fits], color=ACCENT, alpha=0.55, linewidths=0,
               label=f"most grasps ≤ 70 mm ({int(fits.sum())})")
    ax.scatter(surface[~fits], directions[~fits], s=sizes[~fits], color=WIDE_ORANGE, alpha=0.55, linewidths=0,
               label=f"most grasps wider ({int((~fits).sum())})")
    ax.set_xlim(0, 103)
    ax.set_ylim(0, max(40.0, directions.max() * 1.08))
    ax.set_xlabel("% of surface points with any valid grasp")
    ax.set_ylabel("median surface point: % of the 300\napproach directions with a valid grasp")
    ax.legend(loc="upper left", fontsize=6.5, title="dot area ∝ √ valid grasps", title_fontsize=6.5)
    thin = int((directions < 5).sum())
    return Card("per-object-coverage", "grasp", "One dot per object: where it can be held (surface share) × from how many "
                "directions at a typical point",
                "Which objects are thinly covered, in where you can hold them or in how many ways at each place?",
                style.to_data_uri(fig), _clutter_source(ctx) + "; colour: does the majority of the object's valid grasps "
                "fit our planner's 70 mm span. 'Directions' = the 300 label approach views with a valid grasp at any of "
                "the 12 rolls × 4 depths",
                "Per-class coverage scatter (TFDV slice comparison); GSNet's per-point 'graspness' (Wang et al. 2021), "
                "summarised per object",
                {"objects whose median point has < 5 % of directions": f"{thin} of {len(rows)}",
                 "median object": f"{np.median(surface):.0f} % of surface, median point {np.median(directions):.0f} % of "
                                  f"directions",
                 "objects mostly too wide for 70 mm": f"{int((~fits).sum())} of {len(rows)}"}, "grasp 6-DOF")


# --- our real repicks: where, from where, how wide, which hand pose --------------------------------------------------

def _plane_to_base(plan_frame, x, z, y=0.0):
    rows = np.array(plan_frame["B2P_rows"], float)
    origin = np.array(plan_frame["origin_in_base_m"], float)
    return rows.T @ np.array([x, y, z], float) + origin


def repick_positions(ctx) -> Card:
    plans = ctx["plans"]
    tcp = np.array([plan["tcp_base_m"] for plan in plans], float) * 1000.0
    outcome = np.array([plan.get("outcome") or "none" for plan in plans])
    groups = [("success", outcome == "success", ACCENT, "o", "succeeded"),
              ("failed", np.isin(outcome, ["missed", "suspect"]), WIDE_ORANGE, "o", "closed, then missed / suspect"),
              ("not closed", ~np.isin(outcome, ["success", "missed", "suspect"]), "#9a9aa0", "x",
               "planned, never closed (hover-only, refused, stopped)")]
    fig, (top, side) = style.figure(9.4, 3.3, ncols=2, gridspec_kw={"width_ratios": [1.15, 1]})
    frame = plans[0]["plane_frame"]
    corners = []
    for run in ctx["runs"]:
        area = run.get("area")
        if not area:
            continue
        (x0, x1), (z0, z1) = area["x_m"], area["z_m"]
        box = np.array([_plane_to_base(frame, x, z) for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1), (x0, z0))]) * 1000
        top.fill(box[:, 0], box[:, 1], color="#eef2f8", linewidth=0, zorder=-2)
        corners.append(box)
    corners = np.concatenate(corners)
    low, high = corners.min(axis=0), corners.max(axis=0)
    top.add_patch(style.plt.Rectangle((low[0], low[1]), high[0] - low[0], high[1] - low[1], fill=False, edgecolor=LIMIT,
                                      linewidth=1.0, linestyle=(0, (3, 2))))
    for name, mask, colour, marker, label in groups:
        kwargs = {"s": 7, "alpha": 0.55} if marker == "o" else {"s": 9, "alpha": 0.6, "linewidths": 0.6}
        top.scatter(tcp[mask, 0], tcp[mask, 1], color=colour, marker=marker, label=f"{label} ({int(mask.sum())})",
                    **({"linewidths": 0} | kwargs) if marker == "o" else kwargs)
        side.scatter(tcp[mask, 0], tcp[mask, 2], color=colour, marker=marker,
                     **({"linewidths": 0} | kwargs) if marker == "o" else kwargs)
    top.set_aspect("equal")
    top.set_xlabel("base x (mm)")
    top.set_ylabel("base y (mm)")
    top.set_title("top view: grasp TCPs on the belt; pale = each run's ±60 mm look area", loc="left", fontsize=8)
    handles, labels = top.get_legend_handles_labels()
    handles.append(style.plt.Line2D([], [], color=LIMIT, linewidth=1.0, linestyle=(0, (3, 2))))
    labels.append("the runs' look areas, outer box")
    side.set_xlabel("base x (mm)")
    side.set_ylabel("base z, height (mm)")
    side.set_title("side view: grasp height (the belt tilts along x)", loc="left", fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    x_span = tcp[:, 0].max() - tcp[:, 0].min()
    y_span = tcp[:, 1].max() - tcp[:, 1].min()
    closed = np.isin(outcome, ["success", "missed", "suspect"])
    # coverage of the look-area box by closed grasps, in 25 mm cells (Grid over the box the runs could have picked in)
    box_axes = (Axis("x", float(low[0]), float(high[0]), max(int(round((high[0] - low[0]) / 25)), 1), "mm"),
                Axis("y", float(low[1]), float(high[1]), max(int(round((high[1] - low[1]) / 25)), 1), "mm"))
    box_grid = Grid.of(tcp[closed][:, :2], box_axes)
    densest = np.sort(box_grid.counts.ravel())[::-1]
    top_share = 100.0 * densest[:4].sum() / max(densest.sum(), 1)
    return Card("repick-positions", "grasp", "Our real grasp positions, top and side view, coloured by outcome, over the "
                "look areas",
                "Where on the belt have we actually closed the claw, and did it work there?", style.to_data_uri(fig),
                _repick_source(ctx) + ". Base frame (plans' tcp_base_m); dashed red = the union of the runs' look areas "
                "(pick_loop.json area, the planner's work area)",
                "Grasp-position scatter over the workspace (pick-and-place reports, Amazon Picking Challenge "
                "post-mortems)",
                {"TCP span": f"{x_span:.0f} × {y_span:.0f} mm on the belt, {np.ptp(tcp[:, 2]):.0f} mm in height",
                 "closed / succeeded": f"{int(closed.sum())} / {int((outcome == 'success').sum())} "
                                       f"({100 * (outcome == 'success').sum() / max(closed.sum(), 1):.0f} %)",
                 "look-area box visited by closed grasps": f"{box_grid.visited_pct():.0f} % of "
                                                          f"{box_grid.reachable_cells} cells (25 mm)",
                 "closed grasps in the 4 densest cells": f"{top_share:.0f} %"},
                "grasp 6-DOF", wide=True)


def sphere_comparison(ctx) -> Card:
    cell_views = sphere_cell(_from_directions(ctx), RINGS, SECTORS)
    union = np.zeros(RINGS * SECTORS)
    for obj in ctx["objects"].values():
        union += np.bincount(cell_views, weights=obj["valid_per_view"].sum(axis=0).astype(float),
                             minlength=RINGS * SECTORS) > 0
    approach = np.array([plan["approach_base"] for plan in ctx["plans"]], float)
    from_ours = -approach
    ours = np.bincount(sphere_cell(from_ours, RINGS, SECTORS), minlength=RINGS * SECTORS).astype(float)
    tilt = np.degrees(np.arccos(np.clip(from_ours[:, 2] / np.linalg.norm(from_ours, axis=1), -1, 1)))
    fig = style.plt.figure(figsize=(8.8, 2.6))
    left = _sphere_axes(fig, (1, 2, 1))
    mesh_left = _sphere_mesh(left, union, vmin=0.5, vmax=4.0)
    left.set_title(f"GraspNet, 4 objects: {100 * np.mean(union > 0):.0f} % of cells", fontsize=8.5, loc="left")
    right = _sphere_axes(fig, (1, 2, 2))
    mesh_right = _sphere_mesh(right, ours, vmax=max(ours.max(), 1), vmin=1.0)
    right.annotate(f"all {len(from_ours)} grasps come straight down\n(tilt ≤ {tilt.max():.1f}°)", xy=(0.0, np.radians(80)),
                   xytext=(np.radians(40), np.radians(15)), fontsize=7, color=INK, ha="left",
                   arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7})
    right.set_title(f"our repicks: {int((ours > 0).sum())} of {RINGS * SECTORS} cells", fontsize=8.5, loc="left")
    fig.subplots_adjust(left=0.02, right=0.9, top=0.88, bottom=0.08, wspace=0.12)
    bar = _colorbar(fig, mesh_left, left, "objects with a valid grasp (of 4)", orientation="horizontal", fraction=0.06,
                    pad=0.04)
    bar.set_ticks([1, 2, 3, 4])
    _colorbar(fig, mesh_right, right, "planned grasps", orientation="horizontal", fraction=0.06, pad=0.04)
    return Card("sphere-comparison", "grasp", "The same equal-area direction map for a 6-DOF dataset and for our picks",
                "Do our real picks cover approach directions the way a 6-DOF grasp dataset does?",
                style.to_data_uri(fig, colors=56), _graspnet_source(ctx) + ". " + _repick_source(ctx) + " (plans' "
                "grasp.approach_base, base frame, z up). Top of the map = coming straight down",
                "Equal-area sphere binning (coverage.sphere_cell); comparing a deployment's pose distribution to the "
                "training set's (covariate-shift checks)",
                {"GraspNet directions": f"{100 * np.mean(union > 0):.0f} % of 128 cells",
                 "ours": f"{int((ours > 0).sum())} cell ({100 * np.mean(ours > 0):.1f} %)",
                 "our tilt from vertical": f"max {tilt.max():.1f}°, median {np.median(tilt):.1f}°"}, "grasp 6-DOF", wide=True)


def claw_gap(ctx) -> Card:
    DWELL = 0.0005
    max_open = _claw_max_open_mm(ctx)
    gaps = np.concatenate([run["gap_mm"] for run in ctx["runs"]])
    runs_with = sum(1 for run in ctx["runs"] if len(run["gap_mm"]))
    opening = np.array([plan["open_gap_mm"] for plan in ctx["plans"] if plan.get("open_gap_mm")], float)
    stall = np.array([plan["stall_gap_mm"] for plan in ctx["plans"] if plan.get("stall_gap_mm")], float)
    edges = np.arange(-2.0, 142.0, 1.0)
    fig, (top, bottom) = style.figure(6.6, 3.3, nrows=2, sharex=True)
    counts = np.histogram(gaps, edges)[0]
    top.fill_between(edges[:-1], 0, counts / counts.max(), step="post", color=ACCENT, linewidth=0)
    # the strip under the bars: the 1 mm bins the claw HELD (WHY not "any sample": a continuous stroke passes through
    # every gap on the way, so 'ever visited' is always 100 %); white = only swept through
    dwelt = counts >= DWELL * counts.sum()
    top.bar(edges[:-1][dwelt], 0.07, bottom=-0.11, width=1.0, align="edge", color=ACCENT_DARK, linewidth=0)
    top.set_ylim(-0.13, 1.05)
    top.set_title(f"every claw sample's gap: {len(gaps) / 1e6:.1f} M samples, {runs_with} runs; strip = gaps it held "
                  f"(≥ {100 * DWELL:.2f} % of samples)", loc="left", fontsize=8)
    top.set_yticks([])
    bottom.hist(opening, edges, color=ACCENT, label="opening at approach")
    bottom.hist(stall, edges, color=WIDE_ORANGE, label="where the jaws stalled on the item")
    bottom.legend(loc="upper left", fontsize=6.5)
    bottom.set_ylabel("attempts")
    bottom.set_title("per attempt", loc="left", fontsize=8)
    for ax in (top, bottom):
        ax.axvspan(max_open, edges[-1], color=UNREACHABLE, zorder=-3)
        ax.axvspan(edges[0], 0, color=UNREACHABLE, zorder=-3)
        ax.axvline(0, color=LIMIT, linewidth=1.2)
        ax.axvline(max_open, color=LIMIT, linewidth=1.2)
        ax.axvline(PLANNER_SPAN_MM, color=LIMIT, linewidth=1, linestyle=(0, (3, 2)))
    bottom.set_xlabel("pad gap (mm): red = closed and the open stop, dashed = the planner's 70 mm span")
    bottom.set_xlim(edges[0], edges[-1])
    fig.tight_layout(h_pad=0.4)
    # WHY robust extremes (as limit-bars does): one glitched sample would otherwise claim the whole stroke
    low, high = np.percentile(gaps, [0.1, 99.9])
    used = limit_use(np.clip(gaps, low, high), 0.0, max_open)
    over = 100.0 * np.mean(opening > PLANNER_SPAN_MM)
    inside = (edges[:-1] >= 0) & (edges[:-1] < max_open)
    return Card("claw-gap", "grasp", "Claw gap against its limits: time at each gap, and per attempt open vs stalled",
                "How much of the claw's stroke do we use, and do we ever open past the planner's span?",
                style.to_data_uri(fig), _repick_source(ctx, f"; claw_timeline.npz gap_mm from the {runs_with} runs that "
                f"logged it; opening = first_grasp.opening.gap_mm, stall = close.stall_gap_mm"),
                "Actuator stroke-usage histograms; TFDV numeric feature vs schema range", {
                    "stroke used (p0.1-p99.9)": f"{max(low, 0.0):.0f}-{high:.0f} mm = {used['span_pct']:.0f} % of "
                                                f"0-{max_open:.0f} mm",
                    "1 mm gaps held": f"{int(dwelt[inside].sum())} of {int(inside.sum())}",
                    "samples within 5 % of a limit": f"{used['time_near_limit_pct']:.1f} %",
                    "openings past 70 mm": f"{over:.1f} % of {len(opening)}",
                    "median stall gap": f"{np.median(stall):.1f} mm"}, "gripper", wide=True)


def hand_configuration(ctx) -> Card:
    max_open = _claw_max_open_mm(ctx)
    plans = [plan for plan in ctx["plans"] if plan.get("w_rad") and plan.get("open_gap_mm")]
    limits = plans[0]["limits"]
    w1_max = limits["w1_abs_max_rad"] * DEG
    w23_max = limits["w23_abs_max_rad"] * DEG
    w = np.array([plan["w_rad"] for plan in plans], float) * DEG
    opening = np.array([plan["open_gap_mm"] for plan in plans], float)
    roll_axis = Axis("W3", -w23_max, w23_max, 33, "deg")
    open_axis = Axis("opening", 0.0, max_open, 25, "mm")
    w1_axis = Axis("W1", -w1_max, w1_max, 32, "deg")
    w2_axis = Axis("W2", -w23_max, w23_max, 33, "deg")
    roll_grid = Grid.of(np.column_stack([w[:, 2], opening]), (roll_axis, open_axis))
    wrist_grid = Grid.of(np.column_stack([w[:, 0], w[:, 1]]), (w1_axis, w2_axis))
    fig, (left, right) = style.figure(9.4, 3.2, ncols=2)
    style.heatmap(left, roll_grid.counts, roll_axis.edges(), open_axis.edges(), None, figure=fig, colorbar_label="grasps")
    left.axhline(PLANNER_SPAN_MM, color=LIMIT, linewidth=1, linestyle=(0, (3, 2)))
    left.set_xlabel("W3, wrist roll at the grasp (deg)")
    left.set_ylabel("claw opening at approach (mm)")
    left.set_title(f"roll × opening: {roll_grid.visited_pct():.1f} % of the limit box", loc="left", fontsize=8.5)
    style.heatmap(right, wrist_grid.counts, w1_axis.edges(), w2_axis.edges(), None, figure=fig, colorbar_label="grasps")
    right.set_xlabel("W1 (deg)")
    right.set_ylabel("W2 (deg)")
    right.set_title(f"W1 × W2: {wrist_grid.visited_pct():.1f} % of the limit box", loc="left", fontsize=8.5)
    for ax in (left, right):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(LIMIT)
            spine.set_linewidth(1.2)
    fig.tight_layout()
    roll = limit_use(w[:, 2], -w23_max, w23_max)
    return Card("hand-configuration", "grasp", "Hand configurations at the grasp: wrist roll × claw opening, and W1 × W2, "
                "inside the joint limits",
                "Which wrist poses and claw openings have we picked with, out of everything the hand can do?",
                style.to_data_uri(fig), _repick_source(ctx) + f". The frame of each panel IS the limit (red): W1 ±{w1_max:.0f}°, "
                f"W2/W3 ±{w23_max:.0f}° (plans' limits), opening 0-{max_open:.0f} mm; dashed = the 70 mm planner span; "
                "~10° × 5 mm and ~10° × 10° cells. The box is the joint limits, not a reachability map: some of it no "
                "top-down grasp can use, so 100 % is not the goal",
                "sim2real's coverage plane applied to the hand (joint pairs against their limits); configuration-space "
                "occupancy maps", {
                    "roll × opening cells visited": f"{roll_grid.visited_pct():.1f} % of {roll_grid.reachable_cells}",
                    "W1 × W2 cells visited": f"{wrist_grid.visited_pct():.1f} % of {wrist_grid.reachable_cells}",
                    "W3 span used": f"{roll['span_pct']:.0f} % of ±{w23_max:.0f}°"}, "gripper", wide=True)


# --- bam_sim grasp_success (SIMULATED) --------------------------------------------------------------------------------

def sim_support(ctx) -> Card:
    rows = ctx["sim"]
    train = [row for row in rows if row["run"] in ("run1", "run2")]
    held = [row for row in rows if row["run"] == "run3_heldout"]
    def spots(chosen):
        """Distinct pick positions (1 mm) and how many episodes share each: every position is replayed once per
        failure variant, so the episodes sit on top of each other."""

        xy = np.round(np.array([row["pick_xyz"][:2] for row in chosen], float) * 1000.0)
        unique, count = np.unique(xy, axis=0, return_counts=True)
        return xy, unique, count

    train_xy, train_spots, train_count = spots(train)
    held_xy, held_spots, held_count = spots(held)
    hull = _hull(train_spots)
    inside_spots = _inside_hull(hull, held_spots)
    inside = _inside_hull(hull, held_xy)
    fig, (left, middle, right) = style.figure(10.0, 3.2, ncols=3, gridspec_kw={"width_ratios": [1.5, 1, 1]})
    closed = np.vstack([hull, hull[:1]])
    left.fill(closed[:, 0], closed[:, 1], color="#eef2f8", linewidth=0, zorder=-2)
    left.plot(closed[:, 0], closed[:, 1], color=ACCENT, linewidth=0.8)
    for unique, count, colour, marker, name in ((train_spots, train_count, ACCENT, "o", "train (run1 + run2)"),
                                                (held_spots, held_count, WIDE_ORANGE, "D", "held-out (run3)")):
        left.scatter(unique[:, 0], unique[:, 1], s=6 + 3 * count, color=colour, marker=marker, alpha=0.75, linewidths=0,
                     label=f"{name}: {len(unique)} positions (size = episodes)")
    left.set_xlabel("pick x along the belt (mm)")
    left.set_ylabel("pick y across (mm)")
    left.set_title("distinct pick positions; blue hull = train support", loc="left", fontsize=8)
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=1, fontsize=6.5)
    objects = sorted({row["object"] for row in rows})
    variants = sorted({row["variant"] for row in rows})
    numbers = {"held-out positions inside the train hull": f"{int(inside_spots.sum())} of {len(held_spots)} "
                                                           f"({100 * inside.mean():.0f} % of {len(held)} episodes)"}
    for ax, chosen, name in ((middle, train, "train"), (right, held, "held-out")):
        table = np.zeros((len(variants), len(objects)))
        passed = np.zeros_like(table)
        for row in chosen:
            table[variants.index(row["variant"]), objects.index(row["object"])] += 1
            passed[variants.index(row["variant"]), objects.index(row["object"])] += row["passed"]
        style.heatmap(ax, table, np.arange(len(variants) + 1), np.arange(len(objects) + 1), None, vmax=12,
                      colorbar_label=None)
        for i in range(len(variants)):
            for j in range(len(objects)):
                if table[i, j]:
                    ax.text(i + 0.5, j + 0.5, f"{int(passed[i, j])}/{int(table[i, j])}", ha="center", va="center",
                            fontsize=5.2, color="white" if table[i, j] > 6 else INK)
        ax.set_xticks(np.arange(len(variants)) + 0.5)
        ax.set_xticklabels(variants, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(np.arange(len(objects)) + 0.5)
        ax.set_yticklabels([o.replace("_", " ") for o in objects], fontsize=6.5)
        empty = int((table == 0).sum())
        ax.set_title(f"{name}: object × mode\npassed / episodes, {empty} empty", loc="left", fontsize=8)
        numbers[f"{name} empty (object, mode) cells"] = f"{empty} of {table.size}"
    right.set_yticklabels([])
    fig.tight_layout(w_pad=0.6)
    return Card("sim-support", "grasp", "SIMULATED: held-out picks against the training support, and the object × "
                "failure-mode table",
                "Does the held-out set sit inside the training data's support, and is every object × outcome mode present "
                "in both?", style.to_data_uri(fig),
                f"bam_sim grasp_success, SIMULATED (run1, run2 = train, {len(train)} episodes; run3_heldout, {len(held)} "
                "episodes, seed 20260732): pick_xyz, object_type, variant and outcome.passed from each episode's json. "
                "Outcomes come from injected failure variants (drop, slip, miss, drag...), so pass/fail follows the "
                "variant, not the pose; every episode is a top-down pick with the jaws along y",
                "Train/test support checks (convex hull / domain of applicability, Netzeva 2005); the confusion-matrix "
                "layout for label coverage", numbers, "grasp 6-DOF", wide=True)


def cards() -> list[Card]:
    ctx = _context()
    builders = [grasp_wireframes, approach_sphere, surface_directions, height_elevation, width_limits, width_elevation,
                roll_rose, object_imbalance, per_object_coverage, repick_positions, sphere_comparison, claw_gap,
                hand_configuration, sim_support]
    return [build(ctx) for build in builders]


CONTEXT = _context
