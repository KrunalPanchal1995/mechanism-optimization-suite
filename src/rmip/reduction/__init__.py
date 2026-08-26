"""
rmip/reduction
==============
Phase 5B — Mechanism reduction algorithms and visualizers.

Provides DRG, DRGEP, PFA, DRGPSA, and DRGSA reduction classes that compute
species importance relative to a user-defined seed set (target species) and
return a sweep of ``ReductionStep`` objects across a range of ε thresholds.

All reducers store the r_AB interaction matrix and BFS propagation levels in
the ``ReductionResult``, enabling downstream coefficient and trim visualization.

Usage::

    from rmip.reduction.drg    import DRGReducer
    from rmip.reduction.drgep  import DRGEPReducer
    from rmip.reduction.pfa    import PFAReducer
    from rmip.reduction.drgpsa import DRGPSAReducer, DRGSAReducer

    reducer = DRGReducer("project.db")
    result  = reducer.reduce(mech_id=1, G=G, target_species=["H2", "O2", "H2O"])
    result.save("reduction_steps.json")

    # Coefficient visualization
    from rmip.reduction.coefficient_viz import CoefficientVisualizer
    viz = CoefficientVisualizer(result, G)
    viz.run(port=8055)

    # Trim animation
    from rmip.reduction.trim_viz import TrimVisualizer
    tv = TrimVisualizer(result, G)
    tv.run(port=8056)
"""

from rmip.reduction.drg    import DRGReducer
from rmip.reduction.drgep  import DRGEPReducer
from rmip.reduction.pfa    import PFAReducer
from rmip.reduction.drgpsa import DRGPSAReducer, DRGSAReducer

__all__ = [
    "DRGReducer",
    "DRGEPReducer",
    "PFAReducer",
    "DRGPSAReducer",
    "DRGSAReducer",
]
