"""Readers for the recorded datasets the v0 is proven on. Readers, not metrics: they turn files into robot_msgs streams
and ``Run`` records, and compute nothing.

- ``onedof_round8``: the 1-DOF rack's round 8 (2026-11-15/17), 252 real plays and the twin's answer to each.
- ``fivebar_ladder``: the bam_fb five-bar's real speed ladder of 2026-11-10 and its sim twin.

WHY in the package for now: a usage session needs real runs in five lines. The run files' own reader belongs with the
recorder (data_recorder / sim2real tools/run_files.py); when one exists, these become thin calls to it (PEP 0001).
Both read the recordings read-only (not public: ``ROBOT_METRICS_ROUND8`` / ``ROBOT_METRICS_FIVEBAR_LADDER`` point at a
copy) and never open a scoring file: the parity oracle stays in the tests. ``available()`` says whether they are there.
"""
