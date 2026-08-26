"""
mso_workbench.stages
=======================
The stage registry for ``src/MUQ-SAC/run_script.py``: every point where a
``breakpoint()`` call is (or will be) instrumented, in pipeline order, with
a short human-readable description for the Workbench UI's timeline.

Not every stage below has a live ``breakpoint()`` call yet -- ``wired``
records which ones do. Unwired stages are intentionally listed anyway so
the UI's timeline can show the whole pipeline shape even before every
boundary is instrumented; ``instrument.py``'s own module docstring is the
place that tracks exactly which line each wired stage's call sits at.

Invariant this registry exists to preserve: no stage id below may sit
*inside* a ``multiprocessing.Pool``/``ProcessPoolExecutor`` lifetime
(``DesignMatrix.Worker`` at DesignMatrix.py:1231 creates one short-lived
pool per reaction per curve-class B/C; ``simulation_manager2_0.Worker``
holds a longer-lived pool + executor during ``make_dir_in_parallel``) --
pausing the parent there would leave pool workers orphaned rather than
cleanly idle. ``test_no_stage_inside_a_pool_lifetime`` in this package's
tests asserts this holds for every registered stage's line reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    run_script_ref: str  # "file.py:line" -- documentation only, not executable
    wired: bool
    description: str


STAGES: List[Stage] = [
    Stage("config.loaded", "Config loaded", "run_script.py:120", True,
          "target.opt parsed into optInputs."),
    Stage("targets.loaded", "Targets loaded", "run_script.py:333", True,
          "Experimental targets parsed; targets.csv written."),
    Stage("uncertainty.extracted", "Uncertainty extracted", "run_script.py:358", True,
          "Per-reaction uncertainty data built (unsrt.pkl)."),
    Stage("manipulation.built", "Manipulation dict built", "run_script.py:415", False,
          "Per-reaction Cholesky/zeta/nominal arrays assembled."),
    Stage("sampling.partial.pre", "Partial-PRS sampling (pre)", "run_script.py:722", False,
          "Before PartialPRS.partial_DesignMatrix() for this case."),
    Stage("sampling.partial.post", "Partial-PRS sampling (post)", "run_script.py:734", False,
          "Partial design matrix in hand for this case."),
    Stage("sampling.pre_design_matrix", "Sampling (pre)", "run_script.py:767", True,
          "About to call DesignMatrix(...).getSamples() -- the full-PRS "
          "sample-generation entry point."),
    Stage("sampling.post_design_matrix", "Sampling (post)", "run_script.py:771", True,
          "Design matrix built; about to generate perturbed-mechanism YAMLs."),
    Stage("perturbation.written", "Perturbed mechanisms written", "run_script.py:828", False,
          "Perturbed_Mech/*.yaml on disk for every design-matrix row."),
    Stage("simulation.pre", "Simulations (pre)", "run_script.py:860", True,
          "Before SSM.make_dir_in_parallel -- the long pole."),
    Stage("simulation.post", "Simulations (post)", "run_script.py:871", False,
          "All simulation directories built/run."),
    Stage("prs.pre_fit", "PRS fit (pre)", "run_script.py:1393", False,
          "Before the per-case response-surface fit loop."),
    Stage("resample.pre", "Adaptive resample (pre)", "run_script.py:1286", False,
          "Before D-optimal/outlier resampling for a case."),
    Stage("resample.post", "Adaptive resample (post)", "run_script.py:1328", False,
          "New samples selected and simulated for this resample pass."),
    Stage("prs.post_fit", "PRS fit (post)", "run_script.py:1573", False,
          "Every case's response surface trained."),
    Stage("optimization.pre", "Optimization (pre)", "run_script.py:1577", True,
          "Before Optimizer.run_optimization_with_selected_PRS."),
    Stage("optimization.post", "Optimization (post)", "run_script.py:1638", False,
          "solution_zeta.save / posterior_cov.pkl / Optimized_mech.yaml written."),
]

STAGE_BY_ID = {s.id: s for s in STAGES}
STAGE_ORDER = [s.id for s in STAGES]
WIRED_STAGE_IDS = [s.id for s in STAGES if s.wired]


def is_known_stage(stage_id: str) -> bool:
    return stage_id in STAGE_BY_ID
