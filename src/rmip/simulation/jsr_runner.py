"""
rmip/simulation/jsr_runner.py
===============================
Future.md Part 2, Feature 6 — JSR (Jet/Perfectly-Stirred Reactor) runner.

Thin orchestration layer over ``muqsac_bridge.MuqsacBridge``: builds the
target/addendum files, invokes the MUQ-SAC pipeline (``run_script.py``),
parses the result, and stores it — mirroring ``IDTRunner``'s
``run_single``/``run_batch`` API shape (simulation/idt_runner.py) but for
all three solvers (cantera/CHEMKIN_PRO/FlameMaster) via the bridge rather
than a native Cantera reactor network.

Usage::

    from rmip.simulation.jsr_runner import JSRRunner
    from rmip.types import SimCondition

    runner = JSRRunner(mechanism_id=1, db_path="project.db",
                        base_target_opt="DATASET/H2_O2_TEST/target.opt",
                        run_script_path="src/MUQ-SAC/run_script.py")
    result = runner.run_single(
        SimCondition(T_K=1200, P_atm=10.0, phi=1.0, fuel="CH4:1.0"),
        solver="cantera", residence_time_s=0.1, work_dir="jsr_work/case1",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rmip.simulation.muqsac_bridge import MuqsacBridge
from rmip.types import JSRResult, SimCondition

log = logging.getLogger("rmip.simulation.jsr_runner")


class JSRRunner:
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
        residence_time_s: float,
        work_dir: str,
        max_simulation_time_s: float = 10.0,
        flux_analysis: bool = False,
        flux_element: str = "C",
        extra_add: Optional[Dict[str, Any]] = None,
        run_timeout: Optional[float] = None,
        store: bool = True,
    ) -> JSRResult:
        """Run one JSR simulation via the MUQ-SAC pipeline and return a JSRResult."""
        add = dict(extra_add or {})
        add.setdefault("residenceTime", residence_time_s)
        add.setdefault("MaxSimulationTime", max_simulation_time_s)

        target_opt = self._bridge.build_target_files(
            base_target_opt=self._base_target_opt,
            mechanism_id=self.mechanism_id,
            target_type="JSR",
            solver=solver,
            condition=condition,
            work_dir=work_dir,
            extra_add=add,
            flux_analysis=flux_analysis,
            flux_element=flux_element,
        )

        # Remove any stale output JSON left by a previous run in this work_dir
        # so a failed/aborted run can't be silently reported using old results
        # (M21: parsing stale jsr_full.json and storing it as converged=True).
        stale = Path(work_dir) / "output" / "jsr_full.json"
        if stale.exists():
            stale.unlink()

        proc = self._bridge.run(target_opt, timeout=run_timeout)
        if proc.returncode != 0:
            stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
            log.warning("MUQ-SAC pipeline failed (rc=%d) for JSR case in %s: %s",
                        proc.returncode, work_dir, stderr_tail)
            # Do NOT trust whatever JSON may be on disk; report the failure.
            result = JSRResult(
                condition=condition, mechanism_id=self.mechanism_id, solver=solver,
                residence_time_s=0.0, species_mole_fractions={}, converged=False,
                error_msg=f"MUQ-SAC pipeline failed (rc={proc.returncode}): {stderr_tail}",
            )
        else:
            result = self._bridge.parse_jsr_output(
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
    ) -> List[JSRResult]:
        """Run a sequence of JSR conditions, one work_dir per case."""
        results = []
        for i, cond in enumerate(conditions):
            work_dir = f"{work_dir_base}/case-{i}"
            results.append(self.run_single(cond, solver, work_dir=work_dir, **kwargs))
        return results
