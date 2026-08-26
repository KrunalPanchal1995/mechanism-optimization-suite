"""
rmip/reduction/drg.py
=====================
Phase 5B — Directed Relation Graph (DRG) mechanism reduction.

Algorithm (Lu & Law, Combustion and Flame 2005):
  1. For each reaction i, evaluate rate ω_i.
  2. For each species pair (A, B), compute interaction coefficient
         r_AB = Σ|ν_A,i · ω_i|  (reactions involving BOTH A and B)
              / Σ|ν_A,i · ω_i|  (ALL reactions involving A)
  3. Starting from seed species, BFS propagation: if A is retained and
     r_AB > ε, then B is retained.
  4. Retained reactions: any reaction where at least one species (reactant
     or product) is in the retained set.
  5. Repeat sweep over the ε range → produces a ``ReductionResult``.

Rate sources (in priority order)
---------------------------------
1. Cantera: pass ``condition`` + ``mech_path`` to evaluate k(T,P) × [conc] at
   the simulation condition.
2. Graph flux_score: if reaction nodes have ``flux_score`` attached (via M8
   SensitivityFluxIntegrator), those values are used as proxy rates.
3. Unweighted fallback: ω_i = 1.0 for all reactions (graph-structural DRG).

Usage::

    from rmip.reduction.drg import DRGReducer

    reducer = DRGReducer("project.db")
    result  = reducer.reduce(
        mech_id        = 1,
        G              = G,
        target_species = ["H2", "O2", "H2O"],
    )
    result.save("drg_steps.json")
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from rmip.types import ReductionResult, ReductionStep

log = logging.getLogger(__name__)


class DRGReducer:
    """
    DRG-based mechanism reducer.

    Parameters
    ----------
    db_path : str
        Path to the RMIP SQLite database.
    """

    METHOD = "DRG"

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    # ── Public API ────────────────────────────────────────────────────────────

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
        Run the DRG reduction sweep.

        Parameters
        ----------
        mech_id : int
            DB mechanism id.
        G : nx.DiGraph
            Bipartite graph from ``GraphBuilder.build_graph()``.
        target_species : list[str]
            Seed species names (e.g. ``["H2", "O2", "H2O"]``).
            These are always retained regardless of ε.
        epsilon_range : list[float], optional
            ε thresholds to sweep.  Defaults to 51 values in [0, 0.5].
        condition : SimCondition, optional
            Simulation condition for Cantera rate evaluation.
        mech_path : str, optional
            Path to the CHEMKIN / YAML file required by Cantera.
        save_db : bool
            If True, insert a ``reduction_runs`` row in the DB.

        Returns
        -------
        ReductionResult
        """
        if epsilon_range is None:
            epsilon_range = [round(i * 0.01, 6) for i in range(51)]

        all_species, all_reactions = self._load_species_reactions(mech_id)
        rates   = self._get_rates(G, mech_id, condition, mech_path)
        stoich  = self._load_stoichiometry(mech_id)           # {rxn_id: {sp_name: coeff}}
        r_AB    = self._compute_drg_coefficients(all_species, stoich, rates)

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

        # Convert r_AB for serialisation: {A: {B: r}} nested dict
        r_AB_nested = self._r_AB_to_nested(r_AB)

        log.info(
            "%s reduction on mech_id=%d: %d ε steps, "
            "full=%d sp/%d rxn → minimal=%d sp/%d rxn",
            self.METHOD, mech_id,
            len(steps),
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

    # ── Core algorithm ────────────────────────────────────────────────────────

    def _compute_drg_coefficients(
        self,
        all_species: Set[str],
        stoich: Dict[int, Dict[str, float]],
        rates: Dict[int, float],
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute r_AB for all species pairs that co-appear in at least one reaction.

        Returns
        -------
        dict mapping (A, B) → r_AB ∈ [0, 1]
        """
        # denom[A] = Σ_i |ν_A,i * ω_i|  (all reactions involving A)
        denom: Dict[str, float] = defaultdict(float)
        # numer[(A,B)] = Σ_i |ν_A,i * ω_i| where B also in reaction i
        numer: Dict[Tuple[str, str], float] = defaultdict(float)

        for rxn_id, sp_coeffs in stoich.items():
            omega = abs(rates.get(rxn_id, 1.0))
            sp_names = set(sp_coeffs.keys())
            for sp_A in sp_names:
                contrib = abs(sp_coeffs[sp_A]) * omega
                denom[sp_A] += contrib
                for sp_B in sp_names:
                    if sp_B != sp_A:
                        numer[(sp_A, sp_B)] += contrib

        r_AB: Dict[Tuple[str, str], float] = {}
        for (A, B), num in numer.items():
            den = denom.get(A, 0.0)
            r_AB[(A, B)] = num / den if den > 0 else 0.0

        return r_AB

    def _reduce_at_epsilon(
        self,
        all_species:    Set[str],
        all_reactions:  Set[str],
        stoich:         Dict[int, Dict[str, float]],
        r_AB:           Dict[Tuple[str, str], float],
        target_species: List[str],
        eps:            float,
    ) -> Tuple[Set[str], Set[str], Dict[str, int]]:
        """
        BFS propagation from seed set.

        Returns
        -------
        tuple[Set[str], Set[str], dict[str, int]]
            (retained_species, retained_reactions, bfs_levels)

            ``bfs_levels`` maps each retained species to its BFS propagation
            level: 0 for seeds, 1 for first neighbours reached above ε, etc.
        """
        retained   = set(sp for sp in target_species if sp in all_species)
        bfs_levels = {sp: 0 for sp in retained}
        queue      = deque((sp, 0) for sp in retained)

        while queue:
            A, level = queue.popleft()
            for B in all_species:
                if B in retained:
                    continue
                if r_AB.get((A, B), 0.0) > eps:
                    retained.add(B)
                    bfs_levels[B] = level + 1
                    queue.append((B, level + 1))

        # Lu & Law (2005): a reaction survives DRG reduction only if *every*
        # species it involves is in the retained set. Keeping a reaction when
        # merely *some* participant is retained (an intersection test) leaves
        # reactions that reference removed species -- an inconsistent skeletal
        # mechanism that fails to build / materialize (M15).
        retained_rx = set(
            formula for formula, sp_set in self._rxn_formula_to_species.items()
            if sp_set and sp_set <= retained
        )
        return retained, retained_rx, bfs_levels

    @staticmethod
    def _r_AB_to_nested(
        r_AB: Dict[Tuple[str, str], float],
    ) -> Dict[str, Dict[str, float]]:
        """Convert {(A,B): val} to JSON-serialisable {A: {B: val}}."""
        nested: Dict[str, Dict[str, float]] = {}
        for (A, B), val in r_AB.items():
            nested.setdefault(A, {})[B] = round(val, 6)
        return nested

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _load_species_reactions(self, mech_id: int) -> Tuple[Set[str], Set[str]]:
        """Return (set of species names, set of formula_strings) for mechanism."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            sp_rows = conn.execute(
                "SELECT name FROM species WHERE mechanism_id = ?", (mech_id,)
            ).fetchall()
            rx_rows = conn.execute(
                "SELECT formula_string FROM reactions WHERE mechanism_id = ?",
                (mech_id,)
            ).fetchall()
        return (
            {r["name"] for r in sp_rows},
            {r["formula_string"] for r in rx_rows},
        )

    def _load_stoichiometry(self, mech_id: int) -> Dict[int, Dict[str, float]]:
        """
        Load stoichiometry data from the DB.

        Returns
        -------
        dict : {reaction_id → {species_name → stoich_coeff (absolute value)}}

        Side-effects
        ------------
        Populates ``self._rxn_formula_to_species`` and ``self._rxn_id_to_formula``
        for use in ``_reduce_at_epsilon``.
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT st.reaction_id,
                       r.formula_string,
                       sp.name         AS sp_name,
                       st.stoich_coeff AS coeff
                FROM   stoichiometry st
                JOIN   reactions r  ON r.id  = st.reaction_id
                JOIN   species   sp ON sp.id = st.species_id
                WHERE  r.mechanism_id = ?
                """,
                (mech_id,),
            ).fetchall()

        stoich: Dict[int, Dict[str, float]]         = defaultdict(dict)
        self._rxn_formula_to_species: Dict[str, Set[str]] = defaultdict(set)
        self._rxn_id_to_formula: Dict[int, str]           = {}

        for row in rows:
            rxn_id   = row["reaction_id"]
            formula  = row["formula_string"]
            sp_name  = row["sp_name"]
            coeff    = abs(float(row["coeff"]))

            stoich[rxn_id][sp_name] = coeff
            self._rxn_formula_to_species[formula].add(sp_name)
            self._rxn_id_to_formula[rxn_id] = formula

        return dict(stoich)

    def _get_rates(
        self,
        G:         nx.DiGraph,
        mech_id:   int,
        condition: Any,
        mech_path: Optional[str],
    ) -> Dict[int, float]:
        """
        Return reaction rates {reaction_id → ω}.

        Priority: Cantera > flux_score from graph > unweighted (1.0).
        """
        # 1. Try Cantera
        if condition is not None and mech_path is not None:
            try:
                return self._cantera_rates(mech_path, condition, mech_id)
            except Exception as exc:
                log.warning("Cantera rate evaluation failed (%s) — falling back", exc)

        # 2. Try flux_score from graph nodes
        flux_rates = self._graph_flux_rates(G, mech_id)
        if flux_rates:
            log.info("Using flux_score from graph as DRG rates (%d reactions)", len(flux_rates))
            return flux_rates

        # 3. Unweighted fallback
        log.info("Using unweighted DRG (ω=1.0 for all reactions)")
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id FROM reactions WHERE mechanism_id = ?", (mech_id,)
            ).fetchall()
        return {r[0]: 1.0 for r in rows}

    def _cantera_rates(
        self, mech_path: str, condition: Any, mech_id: int
    ) -> Dict[int, float]:
        """Evaluate net reaction rates via Cantera at the given condition."""
        import cantera as ct
        gas = ct.Solution(mech_path)
        phi = condition.phi
        fuel_str  = condition.fuel   # e.g. "H2:1.0"
        ox_str    = condition.oxidizer
        gas.set_equivalence_ratio(phi, fuel_str, ox_str)
        gas.TP = condition.T_K, condition.P_atm * ct.one_atm

        # Map Cantera reaction index → DB id via formula_string
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            db_rxns = conn.execute(
                "SELECT id, formula_string FROM reactions WHERE mechanism_id = ?",
                (mech_id,)
            ).fetchall()

        ct_formulas = {i: gas.reaction(i).equation for i in range(gas.n_reactions)}
        db_formula_to_id = {r["formula_string"]: r["id"] for r in db_rxns}

        rates: Dict[int, float] = {}
        net_rates = gas.net_rates_of_progress
        for i, rate in enumerate(net_rates):
            formula = ct_formulas[i]
            db_id   = db_formula_to_id.get(formula)
            if db_id is not None:
                rates[db_id] = abs(rate)
        return rates

    def _graph_flux_rates(self, G: nx.DiGraph, mech_id: int) -> Dict[int, float]:
        """Extract flux_score from graph reaction nodes as proxy rates."""
        rates: Dict[int, float] = {}
        with sqlite3.connect(self._db_path) as conn:
            db_formula_to_id = {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT formula_string, id FROM reactions WHERE mechanism_id = ?",
                    (mech_id,)
                ).fetchall()
            }
        for node, data in G.nodes(data=True):
            if data.get("type") != "reaction":
                continue
            fs = data.get("flux_score")
            if fs is None:
                continue
            formula = data.get("formula_string", "")
            db_id   = db_formula_to_id.get(formula)
            if db_id is not None:
                rates[db_id] = float(fs)
        return rates

    def _insert_run(
        self,
        mech_id:        int,
        target_species: List[str],
        epsilon_range:  List[float],
        steps:          List[ReductionStep],
    ) -> Optional[int]:
        """Insert a reduction_runs row and return the new row id."""
        from rmip.database.migrations import _table_exists
        with sqlite3.connect(self._db_path) as conn:
            if not _table_exists(conn, "reduction_runs"):
                log.warning("reduction_runs table missing — run phase5b migrations first")
                return None
            ts  = datetime.now(timezone.utc).isoformat()
            cur = conn.execute(
                """
                INSERT INTO reduction_runs
                    (mechanism_id, method, target_species_json,
                     epsilon_min, epsilon_max, n_steps,
                     n_species_full, n_reactions_full,
                     n_species_minimal, n_reactions_minimal,
                     created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    mech_id,
                    self.METHOD,
                    json.dumps(target_species),
                    min(epsilon_range),
                    max(epsilon_range),
                    len(epsilon_range),
                    steps[0].n_species   if steps else 0,
                    steps[0].n_reactions if steps else 0,
                    steps[-1].n_species   if steps else 0,
                    steps[-1].n_reactions if steps else 0,
                    ts,
                ),
            )
            return cur.lastrowid
