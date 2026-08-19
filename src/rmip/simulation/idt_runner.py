"""
rmip/simulation/idt_runner.py
=============================
Module: IDT (Ignition Delay Time) simulation engine.

Physics
-------
- Reactor   : Constant-volume, adiabatic, homogeneous batch reactor
- IDT probe : Maximum dT/dt (temperature derivative peak)
              — the most common and robust IDT definition in combustion literature.
- Fallback  : If dT/dt detection fails (monotonic rise), the time at which
              T crosses T_initial + 400 K is used.

Usage (Python API)
------------------
>>> runner = IDTRunner(mechanism_id=1, db_path="project.db")
>>> result = runner.run_single(
...     T_K=1200, P_atm=13.5, phi=1.0,
...     fuel="NC7H16:1.0",
...     oxidizer="O2:1.0, N2:3.76",
... )
>>> print(result.idt_s)

Usage (batch)
-------------
>>> conditions = [
...     SimCondition(T_K=800,  P_atm=13.5, phi=1.0, fuel="NC7H16:1.0"),
...     SimCondition(T_K=1000, P_atm=13.5, phi=1.0, fuel="NC7H16:1.0"),
...     SimCondition(T_K=1200, P_atm=13.5, phi=1.0, fuel="NC7H16:1.0"),
... ]
>>> df = runner.run_batch(conditions)
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np

from rmip.simulation.base import AbstractSimRunner
from rmip.types import IDTResult, SimCondition

logger = logging.getLogger(__name__)


class IDTRunner(AbstractSimRunner):
    """
    Runs constant-volume, adiabatic IDT simulations for one mechanism.

    Parameters
    ----------
    mechanism_id : int
        DB id of the mechanism to simulate.
    db_path : str
        Path to the RMIP SQLite database.
    t_end_s : float
        Maximum simulation time [s].  Default 10 s (covers slow LTC regimes).
    n_steps : int
        Number of output time steps for the state array.  Denser → better peak resolution.
    rtol / atol : float
        Reactor network integrator tolerances.
    """

    def __init__(
        self,
        mechanism_id: int,
        db_path:      str,
        t_end_s:      float = 10.0,
        n_steps:      int   = 2000,
        rtol:         float = 1e-9,
        atol:         float = 1e-15,
        cache_dir:    Optional[str] = None,
    ) -> None:
        super().__init__(mechanism_id, db_path, cache_dir)
        self._t_end_s = t_end_s
        self._n_steps = n_steps
        self._rtol    = rtol
        self._atol    = atol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(
        self,
        T_K:      float,
        P_atm:    float,
        phi:      float,
        fuel:     str,
        oxidizer: str = "O2:1.0, N2:3.76",
    ) -> IDTResult:
        """
        Run one IDT simulation.

        Parameters
        ----------
        T_K : float
            Initial temperature [K].
        P_atm : float
            Initial pressure [atm].
        phi : float
            Equivalence ratio.
        fuel : str
            Cantera composition string for the fuel, e.g. ``"NC7H16:1.0"``.
        oxidizer : str
            Cantera composition string for the oxidizer.

        Returns
        -------
        IDTResult
        """
        import cantera as ct

        cond = SimCondition(T_K=T_K, P_atm=P_atm, phi=phi,
                            fuel=fuel, oxidizer=oxidizer)
        t0 = time.perf_counter()

        try:
            gas = self._fresh_gas()
            gas.set_equivalence_ratio(phi, fuel=fuel, oxidizer=oxidizer)
            gas.TP = T_K, P_atm * ct.one_atm

            r   = ct.IdealGasReactor(gas)
            sim = ct.ReactorNet([r])
            sim.rtol = self._rtol
            sim.atol = self._atol

            times, temps = self._integrate(sim, r, self._t_end_s, self._n_steps)
            idt = self._detect_idt(times, temps, T_K)
            cpu = time.perf_counter() - t0

            if idt is None:
                logger.debug("No ignition detected: T=%g K, P=%g atm, phi=%g",
                             T_K, P_atm, phi)
                return IDTResult(cond, self.mechanism_id,
                                 idt_s=None, converged=False,
                                 cpu_time_s=cpu,
                                 error_msg="No ignition within t_end")

            logger.debug("IDT=%.4e s at T=%g K, P=%g atm, phi=%g",
                         idt, T_K, P_atm, phi)
            return IDTResult(cond, self.mechanism_id,
                             idt_s=idt, converged=True, cpu_time_s=cpu)

        except Exception as exc:
            cpu = time.perf_counter() - t0
            logger.warning("IDT simulation failed: T=%g K — %s", T_K, exc)
            self.reset_gas()          # ensure fresh gas on next call
            return IDTResult(cond, self.mechanism_id,
                             idt_s=None, converged=False,
                             cpu_time_s=cpu, error_msg=str(exc))

    def run_batch(
        self,
        conditions: List[SimCondition],
        n_workers:  int = 1,
    ) -> "pd.DataFrame":
        """
        Run IDT simulations for a list of `SimCondition` objects.

        Parameters
        ----------
        conditions : list of SimCondition
        n_workers : int
            Number of parallel worker processes (1 = serial).
            Note: each worker loads its own Cantera gas object.

        Returns
        -------
        pd.DataFrame with columns from IDTResult.to_dict()
        """
        import pandas as pd

        if n_workers == 1:
            results = [
                self.run_single(c.T_K, c.P_atm, c.phi, c.fuel, c.oxidizer)
                for c in conditions
            ]
        else:
            # Parallel execution: use separate runner per worker
            results = self._run_parallel(conditions, n_workers)

        return pd.DataFrame([r.to_dict() for r in results])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fresh_gas(self) -> "ct.Solution":
        """Return a copy of the gas object with reset state."""
        import cantera as ct
        g = ct.Solution(thermo=self.gas.thermo_model,
                        kinetics=self.gas.kinetics_model,
                        transport=None,
                        species=self.gas.species(),
                        reactions=self.gas.reactions())
        return g

    @staticmethod
    def _integrate(sim, reactor, t_end: float, n_steps: int):
        """
        Advance the reactor to t_end, storing times and temperatures.

        Uses an adaptive time-step strategy:
          - First 0.1% of t_end: fine steps (n_steps / 4 points)
          - Remainder: coarser steps
        This captures the rapid pre-ignition induction period accurately.
        """
        import cantera as ct

        t_arr, T_arr = [0.0], [reactor.T]
        dt_fine   = t_end * 0.001 / (n_steps // 4)
        dt_coarse = t_end / n_steps

        t_switch = t_end * 0.001
        t        = 0.0
        while t < t_end:
            dt = dt_fine if t < t_switch else dt_coarse
            t  = min(t + dt, t_end)
            sim.advance(t)
            t_arr.append(t)
            T_arr.append(reactor.T)

            # Early exit: clearly ignited (T > initial + 700 K)
            if T_arr[-1] - T_arr[0] > 700:
                # Refine around the inflection with dense steps
                # (already past the peak → we have what we need)
                break

        return np.array(t_arr), np.array(T_arr)

    @staticmethod
    def _detect_idt(
        times: np.ndarray,
        temps: np.ndarray,
        T_init: float,
    ) -> Optional[float]:
        """
        Detect the ignition delay time.

        Primary   : time at maximum dT/dt
        Fallback  : time at T_init + 400 K threshold
        Returns None if neither criterion is met.
        """
        if len(times) < 3:
            return None

        dTdt = np.gradient(temps, times)
        peak_idx = int(np.argmax(dTdt))

        # Sanity check: peak must represent a meaningful temperature rise
        delta_T = temps[peak_idx] - temps[0]
        if delta_T > 100 and dTdt[peak_idx] > 0:
            return float(times[peak_idx])

        # Fallback: 400 K threshold
        threshold = T_init + 400.0
        idx = np.where(temps >= threshold)[0]
        if len(idx) > 0:
            return float(times[idx[0]])

        return None

    def _run_parallel(
        self,
        conditions: List[SimCondition],
        n_workers:  int,
    ) -> List[IDTResult]:
        """
        Run conditions in parallel via ProcessPoolExecutor.
        Each worker is a standalone function to avoid Cantera thread-safety issues.
        """
        import pandas as pd

        args = [
            (c.T_K, c.P_atm, c.phi, c.fuel, c.oxidizer,
             self.mechanism_id, self._db_path,
             self._t_end_s, self._n_steps, self._rtol, self._atol)
            for c in conditions
        ]

        results: List[IDTResult] = [None] * len(conditions)  # type: ignore
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_idt_worker, a): i for i, a in enumerate(args)}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    cond = conditions[idx]
                    results[idx] = IDTResult(
                        cond, self.mechanism_id,
                        idt_s=None, converged=False,
                        cpu_time_s=0.0, error_msg=str(exc),
                    )
        return results


def _run_idt_worker(args: tuple) -> IDTResult:
    """
    Top-level function for ProcessPoolExecutor workers.
    Must be importable (module-level) to be picklable.
    """
    (T_K, P_atm, phi, fuel, oxidizer,
     mechanism_id, db_path, t_end_s, n_steps, rtol, atol) = args

    runner = IDTRunner(
        mechanism_id=mechanism_id,
        db_path=db_path,
        t_end_s=t_end_s,
        n_steps=n_steps,
        rtol=rtol,
        atol=atol,
    )
    return runner.run_single(T_K, P_atm, phi, fuel, oxidizer)
