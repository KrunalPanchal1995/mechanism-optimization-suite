"""
rmip/reduction/drgpsa.py
=========================
Phase 5B-extended — DRGPSA and DRGSA mechanism reduction algorithms.

DRGPSA — DRG with Preliminary Sensitivity Analysis
    (Niemeyer et al., Combustion and Flame, 2010)

    Same as DRG but the reaction contribution to each r_AB is amplified by
    the absolute sensitivity coefficient of that reaction.  Reactions that
    are both kinetically dominant (large flux) AND sensitivity-important
    (large |s_i|) have higher weight in the interaction matrix, so the
    algorithm preferentially retains species coupled through those reactions.

    Weight formula:
        contrib(A,i) = |ν_A,i| × ω_i × (1 + |s_i|)

    If SA data is unavailable (no sa_json_path and no condition+mech_path for
    on-the-fly Cantera SA), DRGPSA silently degrades to standard DRG.

DRGSA — Direct SA-based species importance
    A complementary algorithm: species importance is defined entirely by the
    maximum |s_i| among all reactions involving that species.  No DRG
    coefficients are computed.  The ε threshold acts on this SA importance
    score rather than on a BFS coefficient.  Seeds are always retained.

Usage::

    from rmip.reduction.drgpsa import DRGPSAReducer, DRGSAReducer

    # With pre-computed SA JSON (from ``rmip sa-run`` or CanteraSensitivityRunner)
    reducer = DRGPSAReducer("project.db", sa_json_path="sa_data.json")
    result  = reducer.reduce(mech_id=1, G=G, target_species=["H2", "O2", "H2O"])

    # On-the-fly SA via Cantera (requires mech_path + condition)
    from rmip.types import SimCondition
    cond = SimCondition(T_K=1200, P_atm=10.0, phi=1.0, fuel="H2")
    reducer = DRGPSAReducer("project.db")
    result  = reducer.reduce(mech_id=1, G=G, target_species=["H2", "O2", "H2O"],
                             condition=cond, mech_path="mech.yaml")

    # DRGSA — pure SA ranking
    reducer = DRGSAReducer("project.db", sa_json_path="sa_data.json")
    result  = reducer.reduce(mech_id=1, G=G, target_species=["H2", "O2", "H2O"])
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from rmip.reduction.drg import DRGReducer
from rmip.types import ReductionResult, ReductionStep

log = logging.getLogger(__name__)


class DRGPSAReducer(DRGReducer):
    """
    DRGPSA — DRG with Preliminary Sensitivity Analysis weighting.

    Parameters
    ----------
    db_path : str
        RMIP SQLite database path.
    sa_json_path : str, optional
        Path to a JSON file in the ``sensitivity_data`` format produced by
        ``CanteraSensitivityRunner`` or the ``rmip sa-run`` CLI command.
        If omitted, SA is computed on-the-fly from Cantera when ``condition``
        and ``mech_path`` are supplied to ``reduce()``.
    sa_data : dict, optional
        Pre-loaded sensitivity data dict (alternative to ``sa_json_path``).
    """

    METHOD = "DRGPSA"

    def __init__(
        self,
        db_path:      str,
        sa_json_path: Optional[str] = None,
        sa_data:      Optional[dict] = None,
    ) -> None:
        super().__init__(db_path)
        self._sa_data: Optional[dict] = sa_data
        if sa_json_path is not None and sa_data is None:
            with open(sa_json_path) as fh:
                self._sa_data = json.load(fh)
        # {formula_string → |s_i|} built on first call to _load_sa_lookup
        self._sa_lookup: Dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

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
        Run the DRGPSA reduction sweep.

        If no SA data was supplied at construction and ``condition`` +
        ``mech_path`` are provided, Cantera is used to compute SA on-the-fly.
        Otherwise, the algorithm degrades to standard DRG (r_AB unweighted).
        """
        if epsilon_range is None:
            epsilon_range = [round(i * 0.01, 6) for i in range(51)]

        # Try on-the-fly SA if not pre-loaded
        if self._sa_data is None and condition is not None and mech_path is not None:
            self._sa_data = self._run_cantera_sa(mech_path, condition)

        # Build {formula → |s_i|} lookup table
        if self._sa_data:
            self._sa_lookup = {
                k: abs(float(v))
                for k, v in self._sa_data.get("coefficients", {}).items()
            }
        else:
            log.warning(
                "DRGPSA: no SA data available — degrading to standard DRG weighting"
            )

        all_species, all_reactions = self._load_species_reactions(mech_id)
        rates  = self._get_rates(G, mech_id, condition, mech_path)
        stoich = self._load_stoichiometry(mech_id)
        r_AB   = self._compute_drg_coefficients(all_species, stoich, rates)

        steps: List[ReductionStep] = []
        for eps in epsilon_range:
            retained_sp, retained_rx, bfs_levels = self._reduce_at_epsilon(
                all_species, all_reactions, stoich, r_AB, target_species, eps
            )
            removed_sp = sorted(all_species - retained_sp)
            removed_rx = sorted(all_reactions - retained_rx)
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

        r_AB_nested = self._r_AB_to_nested(r_AB)

        # Prepare SA coefficients for storage (normalise to [0, 1] if possible)
        sa_out: Optional[Dict[str, float]] = None
        if self._sa_lookup:
            max_sa = max(self._sa_lookup.values()) or 1.0
            sa_out = {k: round(v / max_sa, 6) for k, v in self._sa_lookup.items()}

        log.info(
            "DRGPSA reduction on mech_id=%d: %d ε steps, "
            "full=%d sp/%d rxn → minimal=%d sp/%d rxn",
            mech_id, len(steps),
            len(all_species), len(all_reactions),
            steps[-1].n_species, steps[-1].n_reactions,
        )
        return ReductionResult(
            mechanism_id    = mech_id,
            method          = self.METHOD,
            target_species  = list(target_species),
            steps           = steps,
            run_id          = run_id,
            r_AB            = r_AB_nested,
            sa_coefficients = sa_out,
        )

    # ── Algorithm override ────────────────────────────────────────────────────

    def _compute_drg_coefficients(
        self,
        all_species: Set[str],
        stoich:      Dict[int, Dict[str, float]],
        rates:       Dict[int, float],
    ) -> Dict[Tuple[str, str], float]:
        """
        SA-weighted DRG interaction coefficients.

        For each reaction i involving species A and B:
            contrib(A, i) = |ν_A,i| × ω_i × (1 + |s_i|)

        where s_i = normalised SA coefficient of reaction i.
        This amplifies interactions through SA-sensitive reactions.
        """
        denom: Dict[str, float]             = defaultdict(float)
        numer: Dict[Tuple[str, str], float] = defaultdict(float)

        for rxn_id, sp_coeffs in stoich.items():
            omega   = abs(rates.get(rxn_id, 1.0))
            formula = getattr(self, "_rxn_id_to_formula", {}).get(rxn_id, "")
            sa_amp  = 1.0 + self._sa_lookup.get(formula, 0.0)
            weighted = omega * sa_amp

            sp_names = set(sp_coeffs.keys())
            for sp_A in sp_names:
                contrib = abs(sp_coeffs[sp_A]) * weighted
                denom[sp_A] += contrib
                for sp_B in sp_names:
                    if sp_B != sp_A:
                        numer[(sp_A, sp_B)] += contrib

        r_AB: Dict[Tuple[str, str], float] = {}
        for (A, B), num in numer.items():
            den = denom.get(A, 0.0)
            r_AB[(A, B)] = num / den if den > 0 else 0.0

        return r_AB

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _run_cantera_sa(mech_path: str, condition: Any) -> Optional[dict]:
        try:
            from rmip.simulation.sensitivity_runner import CanteraSensitivityRunner
        except ImportError:
            log.warning("CanteraSensitivityRunner not available — SA skipped")
            return None
        try:
            runner = CanteraSensitivityRunner()
            return runner.run_idt(
                mechanism_path = mech_path,
                T_K    = condition.T_K,
                P_atm  = condition.P_atm,
                phi    = condition.phi,
                fuel   = condition.fuel,
            )
        except Exception as exc:
            log.warning("On-the-fly Cantera SA failed (%s) — SA skipped", exc)
            return None


class DRGSAReducer(DRGReducer):
    """
    DRGSA — Direct SA-based species importance (no DRG graph traversal).

    Each species A gets an importance score:
        I_A = max(|s_i|) for all reactions i involving A

    Species are retained if I_A > ε, plus seeds are always retained.
    No BFS propagation is performed.

    This is useful as a comparison / sanity check against DRGPSA: it shows
    which species are directly SA-important without graph propagation effects.
    """

    METHOD = "DRGSA"

    def __init__(
        self,
        db_path:      str,
        sa_json_path: Optional[str] = None,
        sa_data:      Optional[dict] = None,
    ) -> None:
        super().__init__(db_path)
        self._sa_data: Optional[dict] = sa_data
        if sa_json_path is not None and sa_data is None:
            with open(sa_json_path) as fh:
                self._sa_data = json.load(fh)
        self._sa_lookup: Dict[str, float] = {}

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
        """Run the DRGSA sweep based purely on SA importance scores."""
        if epsilon_range is None:
            epsilon_range = [round(i * 0.01, 6) for i in range(51)]

        if self._sa_data is None and condition is not None and mech_path is not None:
            self._sa_data = DRGPSAReducer._run_cantera_sa(mech_path, condition)

        if self._sa_data:
            self._sa_lookup = {
                k: abs(float(v))
                for k, v in self._sa_data.get("coefficients", {}).items()
            }
        else:
            log.warning("DRGSA: no SA data — all species have importance 0.0")

        all_species, all_reactions = self._load_species_reactions(mech_id)
        stoich = self._load_stoichiometry(mech_id)

        # Compute per-species importance = max |s_i| over reactions involving species
        importance: Dict[str, float] = {sp: 0.0 for sp in all_species}
        for rxn_id, sp_coeffs in stoich.items():
            formula = getattr(self, "_rxn_id_to_formula", {}).get(rxn_id, "")
            sa_val  = self._sa_lookup.get(formula, 0.0)
            for sp in sp_coeffs:
                if sa_val > importance.get(sp, 0.0):
                    importance[sp] = sa_val

        # Normalise importance to [0, 1]
        max_imp = max(importance.values()) if importance else 1.0
        if max_imp > 0:
            importance = {sp: v / max_imp for sp, v in importance.items()}

        steps: List[ReductionStep] = []
        for eps in epsilon_range:
            seeds = set(sp for sp in target_species if sp in all_species)
            retained_sp = seeds | {
                sp for sp, imp in importance.items() if imp > eps
            }
            retained_rx = set(
                formula for formula, sp_set in self._rxn_formula_to_species.items()
                if retained_sp.intersection(sp_set)
            )
            removed_sp = sorted(all_species - retained_sp)
            removed_rx = sorted(all_reactions - retained_rx)
            bfs_levels = {sp: (0 if sp in seeds else 1) for sp in retained_sp}
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

        sa_out: Optional[Dict[str, float]] = None
        if self._sa_lookup:
            max_sa = max(self._sa_lookup.values()) or 1.0
            sa_out = {k: round(v / max_sa, 6) for k, v in self._sa_lookup.items()}

        log.info(
            "DRGSA reduction on mech_id=%d: %d ε steps, "
            "full=%d sp/%d rxn → minimal=%d sp/%d rxn",
            mech_id, len(steps),
            len(all_species), len(all_reactions),
            steps[-1].n_species, steps[-1].n_reactions,
        )
        return ReductionResult(
            mechanism_id    = mech_id,
            method          = self.METHOD,
            target_species  = list(target_species),
            steps           = steps,
            run_id          = run_id,
            r_AB            = None,
            sa_coefficients = sa_out,
        )
