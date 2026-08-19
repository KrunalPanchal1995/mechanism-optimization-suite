"""
rmip/optimization/__init__.py
==============================
Optimization integration package.

Provides the MSOInterface that bridges RMIP's FusionEngine output to
KP's existing OptimizationTool_v4.py (MSO: Multi-Stage Optimization with
PyGAD + L-BFGS-B).

Design decision D7: RMIP does NOT import OptimizationTool_v4 directly.
Communication is via a JSON file contract:
  - RMIP writes  mso_input.json
  - MSO tool writes mso_output/optimized_params.json
  - RMIP reads the output and applies updated params to the DB.
"""

from rmip.optimization.mso_interface import MSOInterface

__all__ = ["MSOInterface"]
