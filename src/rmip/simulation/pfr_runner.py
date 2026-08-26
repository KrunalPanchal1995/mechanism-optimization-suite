"""
rmip/simulation/pfr_runner.py
===============================
Future.md Part 2, Feature 6 — PFR (Plug-Flow Reactor) runner.

Same bridge-based orchestration as ``jsr_runner.JSRRunner`` — see that
module's docstring for the overall design. PFR-specific geometry
(length/diameter/mass flow rate/axial resolution) is passed through
``extra_add`` and defaults match ``combustion_target_class.py``'s
``pfr_*`` addendum defaults (Phase-6 additions).

Usage::

    from rmip.simulation.pfr_runner import PFRRunner
    from rmip.types import SimCondition

    runner = PFRRunner(mechanism_id=1, db_path="project.db",
                        base_target_opt="DATASET/H2_O2_TEST/target.opt",
                        run_script_path="src/MUQ-SAC/run_script.py")
    result = runner.run_single(
        SimCondition(T_K=1200, P_atm=10.0, phi=1.0, fuel="CH4:1.0"),
        solver="cantera", work_dir="pfr_work/case1",
        pfr_length=0.1, pfr_diameter=0.02, pfr_mdot=1e-3,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rmip.simulation.muqsac_bridge import MuqsacBridge
from rmip.types import PFRResult, SimCondition

log = logging.getLogger("rmip.simulation.pfr_runner")


class PFRRunner:
    def __init__(
        self,
        mechanism_id: int,
        db_path: str,
        base_target_opt: str,
        run_script_path: str,
    ) -> None:
        self.mechanism_id = mechanism_id
        self._db_path = db_path
        self._base_target_opt = base_target_opt
        self._bridge = MuqsacBridge(db_path, run_script_path)

    def run_single(
        self,
        condition: SimCondition,
        solver: str,
        work_dir: str,
        pfr_length: float = 0.1,
        pfr_diameter: float = 0.02,
        pfr_mdot: float = 1e-3,
        pfr_n_points: int = 200,
        flux_analysis: bool = False,
        flux_element: str = "C",
        extra_add: Optional[Dict[str, Any]] = None,
        run_timeout: Optional[float] = None,
        store: bool = True,
    ) -> PFRResult:
        """Run one PFR simulation via the MUQ-SAC pipeline and return a PFRResult."""
        add = dict(extra_add or {})
        add.setdefault("pfr_length", pfr_length)
        add.setdefault("pfr_diameter", pfr_diameter)
        add.setdefault("pfr_mdot", pfr_mdot)
        add.setdefault("pfr_n_points", pfr_n_points)

        target_opt = self._bridge.build_target_files(
            base_target_opt=self._base_target_opt,
            mechanism_id=self.mechanism_id,
            target_type="PFR",
            solver=solver,
            condition=condition,
            work_dir=work_dir,
            extra_add=add,
            flux_analysis=flux_analysis,
            flux_element=flux_element,
        )

        # Remove any stale output JSON left by a previous run so a failed run
        # can't be silently reported using old results (M21).
        stale = Path(work_dir) / "output" / "pfr_full.json"
        if stale.exists():
            stale.unlink()

        proc = self._bridge.run(target_opt, timeout=run_timeout)
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
            log.warning("MUQ-SAC pipeline failed (rc=%d) for PFR case in %s: %s",
                        proc.returncode, work_dir, stderr_tail)
            result = PFRResult(
                condition=condition, mechanism_id=self.mechanism_id, solver=solver,
                length_m=0.0, residence_time_s=None, species_mole_fractions={},
                converged=False,
                error_msg=f"MUQ-SAC pipeline failed (rc={proc.returncode}): {stderr_tail}",
            )
        else:
            result = self._bridge.parse_pfr_output(
                work_dir, self.mechanism_id, solver, condition)
        if store:
            self._bridge.store_reactor_result(result, tracked_species=add.get("species"))
        return result

    def run_batch(
        self,
        conditions: List[SimCondition],
        solver: str,
        work_dir_base: str,
        **kwargs: Any,
    ) -> List[PFRResult]:
        """Run a sequence of PFR conditions, one work_dir per case."""
        results = []
        for i, cond in enumerate(conditions):
            work_dir = f"{work_dir_base}/case-{i}"
            results.append(self.run_single(cond, solver, work_dir=work_dir, **kwargs))
        return results
