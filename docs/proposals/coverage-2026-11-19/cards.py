"""The gallery's one contract: every figure module returns a list of `Card`s, and `build_gallery.py` lays them out.

A figure module is `figures_<section>.py` with a `cards() -> list[Card]` function. It reads real data only (say where in
`source`), computes with `coverage.py`, draws with `style.py`, and never writes files.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Card:
    id: str                  # kebab-case, unique across the gallery: the anchor (#id) and the pick list's key
    section: str             # one of SECTIONS' keys
    title: str               # what the view is, in plain words ("q x qd heatmap, linear, white = empty")
    answers: str             # the one question it answers at a glance ("are we fast in every configuration?")
    image: str               # data:image/png;base64,... from style.to_data_uri (or a data:video/mp4 for the hero)
    source: str              # the real data it drew and how much ("CAN 11, 8 full-turn runs, 2026-11-18, 29 k samples")
    prior_art: str = ""      # where the idea comes from ("TensorBoard histograms, offset mode")
    numbers: dict[str, str] = field(default_factory=dict)   # computed numbers shown under the image, already formatted
    use_case: str = ""       # 1-DOF joint, N-DOF arm, Cartesian, grasp 6-DOF, gripper, dataset
    wide: bool = False       # spans two columns


SECTIONS = {
    "limits": "0 · Limits first: what can the robot reach?",
    "joint1": "1 · One joint: q, qd, qdd against its envelope",
    "jointN": "2 · Many joints: every axis at once",
    "cartesian": "3 · Cartesian: where the tool went",
    "grasp": "4 · Grasps: 6-DOF poses and the gripper",
    "dataset": "5 · The dataset as a whole: the ML view",
}
