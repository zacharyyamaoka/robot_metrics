"""Sinks for a ``View``: ``wandb`` (a payload; ``log`` sends it) and ``rerun`` (log calls; ``write_rrd`` writes them).

Each imports its SDK inside the function that needs it, so importing robot_metrics never pulls a viewer in.
"""
