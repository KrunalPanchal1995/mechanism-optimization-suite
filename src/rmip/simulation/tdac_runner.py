"""
rmip/simulation/tdac_runner.py
==============================
FEATURE_ROADMAP #11b — Tabulated Dynamic Adaptive Chemistry (TDAC).

A self-contained **adaptive-chemistry batch reactor integrator** (not a CFD
solver).  A constant-UV reactor is integrated while, at a set cadence, the
*locally active* species set is recomputed from the current state by
instantaneous DRGEP around the target species; reactions touching an inactive
species are switched off (``gas.set_multiplier(0, i)``) for the next step, so
each step pays only for the chemistry that currently matters.  Recurring
thermochemical states reuse a **tabulated** active-set (the "T" in TDAC) instead
of re-running DRGEP, and the retrieve/build hit-rate is reported.

The physics of the reduction is the standard DRGEP importance
(Pepiot-Desjardins & Pitsch): direct interaction coefficients from the
instantaneous rates of progress, propagated from the targets by a max-product
("weakest link") path importance, thresholded at ``eps``.

Honest scope
------------
This demonstrates on-the-fly reduction + tabulation and quantifies its accuracy
(IDT vs the full mechanism) and efficiency (mean active-species fraction,
tabulation hit-rate) for a single reactor.  It is not a coupled CFD/ISAT runtime
— per-cell deployment in a flow solver is the documented follow-on.

Usage
-----
>>> from rmip.simulation.tdac_runner import TDACRunner
>>> res = TDACRunner(mechanism_id=1, db_path="project.db").run_single(
...     T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0",
...     target_species=["OH", "H"], eps=0.01)
>>> res.idt_s, res.mean_active_fraction, res.hit_rate
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from rmip.simulation.base import AbstractSimRunner
from rmip.simulation.idt_runner import IDTRunner
from rmip.simulation.rop_runner import _stoich_matrices
from rmip.types import SimCondition, TDACResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure DRGEP on a live gas (no DB) — reusable and unit-testable
# ---------------------------------------------------------------------------

def drgep_coefficients(gas) -> np.ndarray:
    """
    Direct DRGEP interaction coefficients ``r[A, B]`` at the gas's current state,
    shape (n_species, n_species):

        r_AB = Σ_i |ν_{A,i}·ω_i| · 1[B ∈ reaction i]  /  Σ_i |ν_{A,i}·ω_i|

    where ν is the net stoichiometric coefficient and ω the net rate of progress.
    """
    rr, pp = _stoich_matrices(gas)                 # (n_sp, n_rxn)
    nu = pp - rr                                   # net stoich
    omega = np.abs(np.asarray(gas.net_rates_of_progress, dtype=float))
    contrib = np.abs(nu) * omega                   # (n_sp, n_rxn): |ν_{A,i} ω_i|
    participates = (rr + pp) > 0                   # (n_sp, n_rxn): species in rxn

    denom = contrib.sum(axis=1)                    # (n_sp,)
    # numer[A, B] = Σ_i contrib[A, i] * participates[B, i]
    numer = contrib @ participates.T.astype(float)  # (n_sp, n_sp)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom[:, None] > 0, numer / denom[:, None], 0.0)
    np.fill_diagonal(r, 0.0)
    return r


def drgep_active_species(
    gas,
    target_idx: Sequence[int],
    eps: float,
) -> Set[int]:
    """
    DRGEP active-species set: species whose max-product path importance from any
    target is ≥ ``eps`` (targets always included).
    """
    r = drgep_coefficients(gas)
    n = r.shape[0]
    if not np.any(r > 0):
        # frozen state (no rate activity) — cannot reduce, keep everything
        return set(range(n))
    R = np.zeros(n)
    for t in target_idx:
        R[t] = 1.0
    # Bellman-Ford style max-product relaxation (bounded by n passes)
    for _ in range(n):
        changed = False
        for A in np.nonzero(R)[0]:
            cand = R[A] * r[A]
            upd = cand > R
            if upd.any():
                R[upd] = cand[upd]
                changed = True
        if not changed:
            break
    active = set(np.nonzero(R >= eps)[0].tolist())
    active.update(int(t) for t in target_idx)
    return active


def reaction_species_lists(gas) -> List[Set[int]]:
    """For each reaction, the set of species indices participating in it."""
    rr, pp = _stoich_matrices(gas)
    part = (rr + pp) > 0
    return [set(np.nonzero(part[:, i])[0].tolist()) for i in range(gas.n_reactions)]


# ---------------------------------------------------------------------------
# TDAC runner
# ---------------------------------------------------------------------------

class TDACRunner(AbstractSimRunner):
    """Tabulated dynamic adaptive chemistry batch integrator for one mechanism."""

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
        self._rtol = rtol
        self._atol = atol

    def run_single(
        self,
        T_K:            float,
        P_atm:          float,
        phi:            float,
        fuel:           str,
        oxidizer:       str = "O2:1.0, N2:3.76",
        target_species: Optional[Sequence[str]] = None,
        eps:            float = 1e-2,
        update_every:   int = 10,
        bin_T:          float = 50.0,
        bin_top_k:      int = 6,
        save_db:        bool = True,
    ) -> TDACResult:
        """
        Integrate one TDAC reactor and return a :class:`~rmip.types.TDACResult`.

        Parameters
        ----------
        target_species : sequence[str], optional
            DRGEP targets kept active at all times (default: the fuel species +
            the main oxidiser O2).
        eps : float
            DRGEP threshold (smaller ⇒ larger active set ⇒ more accurate).
        update_every : int
            Recompute the active set every this many macro steps.
        bin_T, bin_top_k : float, int
            Tabulation key granularity: temperature bin width [K] and number of
            leading species (by mole fraction) in the state fingerprint.
        """
        import cantera as ct

        cond = SimCondition(T_K=T_K, P_atm=P_atm, phi=phi, fuel=fuel, oxidizer=oxidizer)
        t0 = time.perf_counter()
        gas = self._fresh_gas()
        gas.set_equivalence_ratio(phi, fuel=fuel, oxidizer=oxidizer)
        gas.TP = T_K, P_atm * ct.one_atm

        targets = self._resolve_targets(gas, fuel, target_species)
        target_idx = [gas.species_index(s) for s in targets]
        rxn_species = reaction_species_lists(gas)
        n_rxn = gas.n_reactions

        reactor = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([reactor])
        net.rtol = self._rtol
        net.atol = self._atol

        cache: Dict[tuple, frozenset] = {}
        n_hits = n_builds = 0
        active_counts: List[int] = []
        trace: List[Dict] = []

        t_arr = [0.0]
        T_arr = [reactor.T]
        dt_fine = self._t_end_s * 0.001 / max(self._n_steps // 4, 1)
        dt_coarse = self._t_end_s / self._n_steps
        t_switch = self._t_end_s * 0.001
        t = 0.0
        step = 0
        post_ignition = 0
        converged = True
        try:
            n_species = gas.n_species
            while t < self._t_end_s:
                if step % update_every == 0:
                    if step == 0:
                        # warm-up: the frozen initial state carries only the
                        # initiation flux and would DRGEP down to a broken 3-4
                        # species set; run the first window on the full mechanism
                        # so the radical pool forms before any reduction.
                        active = set(range(n_species))
                    else:
                        key = self._state_key(gas, bin_T, bin_top_k)
                        cached = cache.get(key)
                        if cached is not None:
                            active = set(cached)
                            n_hits += 1
                        else:
                            active = drgep_active_species(gas, target_idx, eps)
                            cache[key] = frozenset(active)
                            n_builds += 1
                    self._apply_active(gas, active, rxn_species, n_rxn)
                    active_counts.append(len(active))
                    if len(trace) < 500:
                        trace.append({"t": t, "T": float(reactor.T),
                                      "n_active": len(active)})

                dt = dt_fine if t < t_switch else dt_coarse
                t = min(t + dt, self._t_end_s)
                net.advance(t)
                t_arr.append(t)
                T_arr.append(reactor.T)
                step += 1
                # keep integrating a short tail past ignition so the burnt-gas
                # plateau (recurring states) exercises the tabulation cache,
                # then stop — no need to run all the way to t_end.
                if T_arr[-1] - T_arr[0] > 700:
                    post_ignition += 1
                    if post_ignition > 30 * update_every:
                        break
        except Exception as exc:                       # noqa: BLE001
            converged = False
            logger.warning("TDAC integration failed at T=%g: %s", T_K, exc)

        idt = IDTRunner._detect_idt(np.asarray(t_arr), np.asarray(T_arr), T_K)
        counts = np.asarray(active_counts) if active_counts else None

        result = TDACResult(
            mechanism_id     = self.mechanism_id,
            condition        = cond,
            eps              = eps,
            update_every     = update_every,
            bin_T            = bin_T,
            n_species_full   = gas.n_species,
            n_reactions_full = n_rxn,
            idt_s            = idt,
            converged        = converged and idt is not None,
            n_updates        = len(active_counts),
            n_cache_hits     = n_hits,
            n_cache_builds   = n_builds,
            mean_active_species = (float(counts.mean()) if counts is not None else None),
            min_active_species  = (int(counts.min()) if counts is not None else None),
            max_active_species  = (int(counts.max()) if counts is not None else None),
            active_trace     = trace,
            target_species   = list(targets),
            cpu_time_s       = time.perf_counter() - t0,
        )
        result.notes.append(
            "On-the-fly DRGEP reduction + state tabulation for one batch reactor; "
            "coupled CFD/ISAT runtime deployment is a documented follow-on.")

        if save_db:
            result.run_id = _insert_tdac_run(self._loader._db_path, result)  # type: ignore[attr-defined]

        logger.info(
            "TDAC mech=%d T=%.0fK: IDT=%s, mean active %.1f/%d (%.0f%%), "
            "tabulation hit-rate %.2f",
            self.mechanism_id, T_K, f"{idt:.3e}" if idt else "None",
            result.mean_active_species or 0.0, gas.n_species,
            100 * (result.mean_active_fraction or 0.0), result.hit_rate or 0.0)
        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _fresh_gas(self):
        import cantera as ct
        return ct.Solution(
            thermo=self.gas.thermo_model, kinetics=self.gas.kinetics_model,
            transport=None, species=self.gas.species(), reactions=self.gas.reactions())

    @staticmethod
    def _resolve_targets(gas, fuel: str, target_species) -> List[str]:
        if target_species:
            return [s for s in target_species if s in gas.species_names]
        tg = []
        for tok in fuel.replace(",", " ").split():
            name = tok.split(":", 1)[0].strip()
            if name in gas.species_names:
                tg.append(name)
        if "O2" in gas.species_names:
            tg.append("O2")
        return tg or [gas.species_names[0]]

    @staticmethod
    def _apply_active(gas, active: Set[int], rxn_species: List[Set[int]], n_rxn: int) -> None:
        """Enable reactions whose species are all active; disable the rest."""
        for i in range(n_rxn):
            gas.set_multiplier(1.0 if rxn_species[i] <= active else 0.0, i)

    @staticmethod
    def _state_key(gas, bin_T: float, top_k: int) -> tuple:
        """Coarse thermochemical fingerprint for tabulation."""
        X = np.asarray(gas.X)
        top = tuple(sorted(np.argsort(X)[-top_k:].tolist()))
        return (int(gas.T // bin_T), top)


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def _insert_tdac_run(db_path: str, result: TDACResult) -> Optional[int]:
    import json
    import sqlite3
    from datetime import datetime, timezone
    from rmip.database.migrations import run_tdac_migrations, _table_exists

    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "tdac_runs"):
            run_tdac_migrations(conn)
        cond = result.condition
        cur = conn.execute(
            """
            INSERT INTO tdac_runs
                (mechanism_id, T_K, P_atm, phi, fuel, oxidizer, eps, update_every,
                 bin_T, n_species_full, n_reactions_full, idt_s, converged,
                 n_updates, n_cache_hits, n_cache_builds, mean_active_species,
                 min_active_species, max_active_species, target_species_json,
                 notes_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result.mechanism_id,
                cond.T_K if cond else None, cond.P_atm if cond else None,
                cond.phi if cond else None, cond.fuel if cond else None,
                cond.oxidizer if cond else None,
                result.eps, result.update_every, result.bin_T,
                result.n_species_full, result.n_reactions_full, result.idt_s,
                1 if result.converged else 0, result.n_updates,
                result.n_cache_hits, result.n_cache_builds,
                result.mean_active_species, result.min_active_species,
                result.max_active_species, json.dumps(result.target_species),
                json.dumps(result.notes),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid
