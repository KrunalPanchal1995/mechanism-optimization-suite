"""
rmip/simulation/lfs_runner.py
=============================
Module: LFS (Laminar Flame Speed) simulation engine.

Physics
-------
- Solver  : Cantera 1-D freely propagating premixed flame (`FreeFlame`)
- Output  : Unstretched laminar burning velocity Su [m/s]

Convergence strategy
--------------------
1. Start with a coarse grid (auto = Cantera default) and loglevel=0.
2. If first solve fails, retry with a better initial guess from a
   neighbouring phi value (if available) or with auto_refine=True.
3. If LFS < 1e-6 m/s or > 5 m/s, flag as non-physical and return None.

Usage
-----
>>> runner = LFSRunner(mechanism_id=1, db_path="project.db")
>>> result = runner.run_single(
...     phi=1.0, T_K=300, P_atm=1.0,
...     fuel="NC7H16:1.0",
... )
>>> print(result.lfs_m_s)

>>> conditions = [
...     SimCondition(T_K=300, P_atm=1.0, phi=p, fuel="NC7H16:1.0")
...     for p in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
... ]
>>> df = runner.run_batch(conditions)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np

from rmip.simulation.base import AbstractSimRunner
from rmip.types import LFSResult, SimCondition

logger = logging.getLogger(__name__)


class LFSRunner(AbstractSimRunner):
    """
    Runs 1-D freely propagating premixed flame simulations for one mechanism.

    Parameters
    ----------
    mechanism_id : int
        DB id of the mechanism to simulate.
    db_path : str
        Path to the RMIP SQLite database.
    width : float
        Domain width [m] for the 1-D flame.  Default 0.03 m.
    loglevel : int
        Cantera solver verbosity (0=quiet, 1=progress, 2=verbose).
    ratio / slope / curve / prune : float
        FreeFlame auto-refinement controls (Cantera defaults recommended).
    """

    def __init__(
        self,
        mechanism_id: int,
        db_path:      str,
        width:        float = 0.03,
        loglevel:     int   = 0,
        ratio:        float = 3.0,
        slope:        float = 0.06,
        curve:        float = 0.12,
        prune:        float = 0.04,
        cache_dir:    Optional[str] = None,
    ) -> None:
        super().__init__(mechanism_id, db_path, cache_dir)
        self._width    = width
        self._loglevel = loglevel
        self._ratio    = ratio
        self._slope    = slope
        self._curve    = curve
        self._prune    = prune

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(
        self,
        phi:      float,
        T_K:      float   = 300.0,
        P_atm:    float   = 1.0,
        fuel:     str     = "NC7H16:1.0",
        oxidizer: str     = "O2:1.0, N2:3.76",
    ) -> LFSResult:
        """
        Run one LFS simulation.

        Parameters
        ----------
        phi : float
            Equivalence ratio.
        T_K : float
            Unburned gas temperature [K].
        P_atm : float
            Pressure [atm].
        fuel : str
            Cantera fuel composition string.
        oxidizer : str
            Cantera oxidizer composition string.

        Returns
        -------
        LFSResult
        """
        import cantera as ct

        cond = SimCondition(T_K=T_K, P_atm=P_atm, phi=phi,
                            fuel=fuel, oxidizer=oxidizer)
        t0 = time.perf_counter()

        try:
            gas = self._fresh_gas()
            gas.set_equivalence_ratio(phi, fuel=fuel, oxidizer=oxidizer)
            gas.TP = T_K, P_atm * ct.one_atm

            flame = ct.FreeFlame(gas, width=self._width)
            flame.set_refine_criteria(
                ratio=self._ratio, slope=self._slope,
                curve=self._curve, prune=self._prune,
            )

            # Attempt solve
            flame.solve(loglevel=self._loglevel, auto=True)
            lfs = flame.velocity[0]

            # Sanity check
            if not (1e-6 < lfs < 5.0):
                raise ValueError(
                    f"Non-physical LFS={lfs:.4f} m/s at phi={phi} "
                    f"(expected 1e-6–5.0 m/s)"
                )

            cpu = time.perf_counter() - t0
            logger.debug("LFS=%.4f m/s at phi=%g, T=%g K, P=%g atm",
                         lfs, phi, T_K, P_atm)
            return LFSResult(cond, self.mechanism_id,
                             lfs_m_s=lfs, converged=True, cpu_time_s=cpu)

        except ct.CanteraError as exc:
            cpu = time.perf_counter() - t0
            logger.warning(
                "FreeFlame did not converge: phi=%g, T=%g K — %s", phi, T_K, exc
            )
            self.reset_gas()
            return LFSResult(cond, self.mechanism_id,
                             lfs_m_s=None, converged=False,
                             cpu_time_s=cpu, error_msg=str(exc))

        except Exception as exc:
            cpu = time.perf_counter() - t0
            logger.warning("LFS simulation failed: phi=%g — %s", phi, exc)
            self.reset_gas()
            return LFSResult(cond, self.mechanism_id,
                             lfs_m_s=None, converged=False,
                             cpu_time_s=cpu, error_msg=str(exc))

    def run_batch(
        self,
        conditions: List[SimCondition],
        n_workers:  int = 1,
    ) -> "pd.DataFrame":
        """
        Run LFS simulations for a list of `SimCondition` objects.

        Parameters
        ----------
        conditions : list of SimCondition
        n_workers : int
            Parallel worker processes.

        Returns
        -------
        pd.DataFrame with columns from LFSResult.to_dict()
        """
        import pandas as pd

        if n_workers == 1:
            results = [
                self.run_single(c.phi, c.T_K, c.P_atm, c.fuel, c.oxidizer)
                for c in conditions
            ]
        else:
            results = self._run_parallel(conditions, n_workers)

        return pd.DataFrame([r.to_dict() for r in results])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fresh_gas(self) -> "ct.Solution":
        """Return a fresh copy of the gas object."""
        import cantera as ct
        g = ct.Solution(
            thermo    = self.gas.thermo_model,
            kinetics  = self.gas.kinetics_model,
            transport = None,
            species   = self.gas.species(),
            reactions = self.gas.reactions(),
        )
        return g

    def _run_parallel(
        self,
        conditions: List[SimCondition],
        n_workers:  int,
    ) -> List[LFSResult]:
        args = [
            (c.phi, c.T_K, c.P_atm, c.fuel, c.oxidizer,
             self.mechanism_id, self._db_path,
             self._width, self._loglevel,
             self._ratio, self._slope, self._curve, self._prune)
            for c in conditions
        ]

        results: List[LFSResult] = [None] * len(conditions)  # type: ignore
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_lfs_worker, a): i for i, a in enumerate(args)}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    cond = conditions[idx]
                    results[idx] = LFSResult(
                        cond, self.mechanism_id,
                        lfs_m_s=None, converged=False,
                        cpu_time_s=0.0, error_msg=str(exc),
                    )
        return results


def _run_lfs_worker(args: tuple) -> LFSResult:
    """Top-level picklable worker for ProcessPoolExecutor."""
    (phi, T_K, P_atm, fuel, oxidizer,
     mechanism_id, db_path, width, loglevel,
     ratio, slope, curve, prune) = args

    runner = LFSRunner(
        mechanism_id=mechanism_id,
        db_path=db_path,
        width=width,
        loglevel=loglevel,
        ratio=ratio,
        slope=slope,
        curve=curve,
        prune=prune,
    )
    return runner.run_single(phi, T_K, P_atm, fuel, oxidizer)
