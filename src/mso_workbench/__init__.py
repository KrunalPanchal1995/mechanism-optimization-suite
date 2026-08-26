"""
mso_workbench
===============
Cooperative pause/inspect machinery for long-running pipelines (currently
``src/MUQ-SAC/run_script.py``). Pure Python, dependency-free -- must import
cleanly from the anaconda interpreter that actually runs these pipelines
(the same cross-interpreter convention as ``mso_io.notebook_gen``).

The whole feature is inert unless a pause is actually requested (see
``control.py``): a call to :func:`breakpoint.breakpoint` costs one
``os.path.exists`` check when nothing is asking it to pause, and does
nothing at all when ``MSO_WORKBENCH=0`` is set in the environment.
"""
from __future__ import annotations

__version__ = "0.1.0"
