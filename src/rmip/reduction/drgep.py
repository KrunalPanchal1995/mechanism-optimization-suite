"""
rmip/reduction/drgep.py
========================
Phase 5B — Directed Relation Graph with Error Propagation (DRGEP).

Algorithm (Pepiot-Desjardins & Pitsch, Combustion and Flame 2008):
  Like DRG but uses a path-propagated coefficient instead of the direct r_AB.
  The DRGEP coefficient from target T to species B is the maximum product of
  DRG interaction coefficients along ANY path from any seed species to B:

      EP_AB = max over all paths P from A₀ (seed) to B of ∏ r_{i,i+1}

  This is equivalent to finding the "maximum-weight path" in a fully-connected
  species interaction graph where edge weight is r_AB.  Computed via a modified
  Dijkstra on log-coefficients (additive in log-space → maximise log sum).

Usage::

    from rmip.reduction.drgep import DRGEPReducer

    reducer = DRGEPReducer("project.db")
    result  = reducer.reduce(mech_id=1, G=G, target_species=["H2", "O2", "H2O"])
"""
from __future__ import annotations

import heapq
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from rmip.reduction.drg import DRGReducer
from rmip.types import ReductionResult, ReductionStep

log = logging.getLogger(__name__)

_NEG_INF = float("-inf")


class DRGEPReducer(DRGReducer):
    """
    DRGEP-based mechanism reducer.

    Extends ``DRGReducer`` — overrides ``_compute_drg_coefficients`` to return
    path-propagated EP_AB coefficients for use in the BFS threshold check.
    """

    METHOD = "DRGEP"

    def reduce(
        self,
        mech_id:        int,
        G:              nx.DiGraph,
        target_species: List[str],
        epsilon_range:  Optional[List[float]] = None,
        condition:      Any = None,
        mech_path:      Optional[str] = None,
        save_db:        bool = True,
    ) -> ReductionResult:
        """
        Run the DRGEP reduction sweep.

        Same signature as ``DRGReducer.reduce()``.  The interaction coefficient
        used for the ε threshold is the path-propagated EP_AB rather than the
        direct r_AB.
        """
        if epsilon_range is None:
            epsilon_range = [round(i * 0.01, 6) for i in range(51)]

        all_species, all_reactions = self._load_species_reactions(mech_id)
        rates   = self._get_rates(G, mech_id, condition, mech_path)
        stoich  = self._load_stoichiometry(mech_id)

        # Direct DRG coefficients
        r_direct = super()._compute_drg_coefficients(all_species, stoich, rates)

        # Path-propagated EP coefficients from each seed
        ep_AB = self._compute_ep_coefficients(all_species, target_species, r_direct)

        steps: List[ReductionStep] = []
        for eps in epsilon_range:
            retained_sp = self._ep_retain_at_epsilon(
                all_species, target_species, ep_AB, eps
            )
            retained_rx: Set[str] = set(
                formula for formula, sp_set in self._rxn_formula_to_species.items()
                if retained_sp.intersection(sp_set)
            )
            removed_sp = sorted(all_species - retained_sp)
            removed_rx = sorted(all_reactions - retained_rx)
            # bfs_levels: seeds at 0, retained-but-not-seed at 1 (EP has no BFS layers)
            bfs_levels = {sp: (0 if sp in target_species else 1) for sp in retained_sp}
            steps.append(ReductionStep(
                threshold          = eps,
                retained_species   = sorted(retained_sp),
                retained_reactions = sorted(retained_rx),
                removed_species    = removed_sp,
                removed_reactions  = removed_rx,
                n_species          = len(retained_sp),
                n_reactions        = len(retained_rx),
                bfs_levels         = bfs_levels,
            ))

        run_id = None
        if save_db:
            run_id = self._insert_run(mech_id, target_species, epsilon_range, steps)

        r_AB_nested = self._r_AB_to_nested(r_direct)

        log.info(
            "DRGEP reduction on mech_id=%d: %d ε steps, "
            "full=%d sp/%d rxn → minimal=%d sp/%d rxn",
            mech_id, len(steps),
            len(all_species), len(all_reactions),
            steps[-1].n_species, steps[-1].n_reactions,
        )
        return ReductionResult(
            mechanism_id   = mech_id,
            method         = self.METHOD,
            target_species = list(target_species),
            steps          = steps,
            run_id         = run_id,
            r_AB           = r_AB_nested,
        )

    # ── DRGEP-specific helpers ────────────────────────────────────────────────

    def _compute_ep_coefficients(
        self,
        all_species:    Set[str],
        target_species: List[str],
        r_direct:       Dict[Tuple[str, str], float],
    ) -> Dict[str, float]:
        """
        Compute EP_B = max path coefficient from any seed to species B.

        Uses Dijkstra in log-space (maximise sum of log-coefficients).
        A coefficient of 0 means no path → species B gets EP = 0.

        Returns
        -------
        dict : {species_name → EP coefficient ∈ [0, 1]}
        """
        # Log-space Dijkstra: dist[B] = max(Σ log r_AB along path seed → B)
        # We maximise → negate to use min-heap.
        _NEGINF = float("-inf")
        dist: Dict[str, float] = {sp: _NEGINF for sp in all_species}
        for seed in target_species:
            if seed in dist:
                dist[seed] = 0.0   # log(1) — seed always retained

        # min-heap: (-log_ep, species)
        heap = [(-dist[seed], seed) for seed in target_species if seed in dist]
        heapq.heapify(heap)

        while heap:
            neg_d, A = heapq.heappop(heap)
            d = -neg_d
            if d < dist.get(A, _NEGINF):
                continue
            for B in all_species:
                r = r_direct.get((A, B), 0.0)
                if r <= 0.0:
                    continue
                log_r = math.log(r)   # ≤ 0
                new_d = d + log_r
                if new_d > dist.get(B, _NEGINF):
                    dist[B] = new_d
                    heapq.heappush(heap, (-new_d, B))

        # Convert back from log-space to [0,1]
        return {
            sp: math.exp(d) if d > _NEGINF else 0.0
            for sp, d in dist.items()
        }

    def _ep_retain_at_epsilon(
        self,
        all_species:    Set[str],
        target_species: List[str],
        ep_AB:          Dict[str, float],
        eps:            float,
    ) -> Set[str]:
        """Return retained set: seeds + species with EP > ε."""
        retained = set(sp for sp in target_species if sp in all_species)
        for sp, ep in ep_AB.items():
            if ep > eps:
                retained.add(sp)
        return retained
