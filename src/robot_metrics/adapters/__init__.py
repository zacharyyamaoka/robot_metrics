"""Sinks for a ``View``: ``wandb`` (a payload; ``log`` sends only its summary scalars, the rows to a local jsonl) and
``rerun`` (log calls; ``write_rrd`` writes them). W&B holds summary statistics only, since 2026-11-21.

Each imports its SDK inside the function that needs it, so importing robot_metrics never pulls a viewer in.
"""
