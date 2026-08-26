"""
mso_studio
==========
Unified PySide6 GUI integrator for the Mechanism Optimization Suite.

Two-layer architecture (see the project's GUI guidelines doc):

* :mod:`mso_studio.core` -- pure Python, zero GUI imports. Adapters that
  shell out to the suite's existing science backends (``src/mso_compare``,
  ``src/Sensitivity_Analysis``, ``src/MUQ-SAC``, ``src/rmip``) running in the
  anaconda ``python3.9`` interpreter, and turn their output files into
  :class:`~mso_studio.core.runrecord.RunRecord` objects.
* :mod:`mso_studio.gui` -- all Qt code, and only Qt code. Reads
  ``RunRecord``s, never re-derives results from raw output files itself.

Run with ``run_mso_studio.py`` at the repo root, using the dedicated
``.venv-gui`` interpreter (PySide6 is not installed in the shared anaconda
environment, by design -- see the env-isolation policy in project memory).
"""
__version__ = "0.1.0"
