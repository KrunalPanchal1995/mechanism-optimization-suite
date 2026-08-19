"""
rmip/analysis/__init__.py
=========================
Exports all Phase 2 + Phase 3 analysis classes.

Phase 2
-------
  M3  NetworkVisualizer   — interactive Dash/Cytoscape graph viewer
  M6  ComparisonEngine    — kinetic/thermo cross-mechanism diff

Phase 3
-------
  M4  SparsityDetector          — dead-end nodes, orphan chains, coverage
  M8  SensitivityFluxIntegrator — attach SA + flux data to graph nodes
      GraphComparator           — classify species/reactions, build UnionGraphSpec
      MechanismBenchmarker      — IDT/LFS simulation + scoring vs experiment
"""
from rmip.analysis.visualization  import NetworkVisualizer
from rmip.analysis.comparison     import ComparisonEngine
from rmip.analysis.sparsity       import SparsityDetector
from rmip.analysis.sensitivity    import SensitivityFluxIntegrator
from rmip.analysis.graph_comparator import GraphComparator
from rmip.analysis.benchmarking   import MechanismBenchmarker

__all__ = [
    # Phase 2
    "NetworkVisualizer",
    "ComparisonEngine",
    # Phase 3
    "SparsityDetector",
    "SensitivityFluxIntegrator",
    "GraphComparator",
    "MechanismBenchmarker",
]
