"""
rmip/reduction/pfa.py
=====================
Phase 5B — Path Flux Analysis (PFA) mechanism reduction.

Algorithm (Sun et al., Combustion and Flame 2010):
  PFA uses a combined production+consumption coefficient:

      r_AB = (P_AB + C_AB) / (P_A + C_A)

  where:
    P_AB = Σ_i max(0, ν_A,i · ω_i)   (sum of production fluxes of A
                                        in reactions where B also participates)
    C_AB = Σ_i max(0, -ν_A,i · ω_i)  (sum of consumption fluxes of A
                                        in reactions where B also participates)
    P_A  = total production flux of A across all reactions
    C_A  = total consumption flux of A across all reactions

  The denominator max(P_A + C_A, ε_floor) avoids division by zero.
  BFS propagation and retained-reaction determination are identical to DRG.

  For unsigned rates (flux_score or unweighted), PFA reduces to DRG because
  production ≡ consumption paths use the same |ν·ω| contribution.

Usage::

    from rmip.reduction.pfa import PFAReducer

    reducer = PFAReducer("project.db")
    result  = reducer.reduce(mech_id=1, G=G, target_species=["H2", "O2", "H2O"])
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from rmip.reduction.drg import DRGReducer
from rmip.types import ReductionResult, ReductionStep

log = logging.getLogger(__name__)


class PFAReducer(DRGReducer):
    """
    PFA-based mechanism reducer.

    Extends ``DRGReducer`` — overrides ``_compute_drg_coefficients`` to use the
    combined production+consumption path-flux coefficient.
    """

    METHOD = "PFA"

    def _compute_drg_coefficients(
        self,
        all_species: Set[str],
        stoich:      Dict[int, Dict[str, float]],
        rates:       Dict[int, float],
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute PFA r_AB = (P_AB + C_AB) / (P_A + C_A).

        For signed rates (ω > 0 = forward, ω < 0 = reverse):
          - P_AB: production flux of A in reactions also involving B
          - C_AB: consumption flux of A in reactions also involving B
        For unsigned rates (ω ≥ 0, as from flux_score or unweighted fallback),
        the formula reduces to DRG since P_AB = C_AB = |ν·ω|/2.
        """
        # Separate production and consumption contributions per species per reaction
        # Signs: stoich is stored as |coeff|; we distinguish reactant vs product
        # roles via the _signed_stoich map loaded in _load_stoichiometry_signed.
        signed = self._load_stoichiometry_signed(stoich)

        prod_A:  Dict[str, float] = defaultdict(float)   # Σ max(0, ν_A,i · ω_i)
        cons_A:  Dict[str, float] = defaultdict(float)   # Σ max(0, -ν_A,i · ω_i)
        prod_AB: Dict[Tuple[str, str], float] = defaultdict(float)
        cons_AB: Dict[Tuple[str, str], float] = defaultdict(float)

        for rxn_id, sp_signed_coeffs in signed.items():
            omega   = rates.get(rxn_id, 1.0)   # may be unsigned (≥ 0)
            sp_list = list(sp_signed_coeffs.keys())

            for sp_A in sp_list:
                nu_A       = sp_signed_coeffs[sp_A]
                flux_prod  = max(0.0,  nu_A * omega)
                flux_cons  = max(0.0, -nu_A * omega)

                prod_A[sp_A] += flux_prod
                cons_A[sp_A] += flux_cons

                for sp_B in sp_list:
                    if sp_B == sp_A:
                        continue
                    prod_AB[(sp_A, sp_B)] += flux_prod
                    cons_AB[(sp_A, sp_B)] += flux_cons

        r_AB: Dict[Tuple[str, str], float] = {}
        _eps_floor = 1e-300
        for sp_A in all_species:
            denom = prod_A.get(sp_A, 0.0) + cons_A.get(sp_A, 0.0)
            if denom < _eps_floor:
                continue
            for sp_B in all_species:
                if sp_B == sp_A:
                    continue
                num = prod_AB.get((sp_A, sp_B), 0.0) + cons_AB.get((sp_A, sp_B), 0.0)
                if num > 0:
                    r_AB[(sp_A, sp_B)] = num / denom

        return r_AB

    def _load_stoichiometry_signed(
        self,
        stoich: Dict[int, Dict[str, float]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Load signed stoichiometry: reactants have negative ν, products positive.

        ``stoich`` from ``_load_stoichiometry()`` stores |coeff| for all species.
        We re-query the DB to get role (reactant → negative, product → positive).
        """
        import sqlite3
        result: Dict[int, Dict[str, float]] = {}

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT st.reaction_id,
                       sp.name         AS sp_name,
                       st.stoich_coeff AS coeff,
                       st.role
                FROM   stoichiometry st
                JOIN   reactions r  ON r.id  = st.reaction_id
                JOIN   species   sp ON sp.id = st.species_id
                WHERE  r.id IN ({})
                """.format(",".join("?" * len(stoich)) if stoich else "SELECT -1"),
                list(stoich.keys()) if stoich else [],
            ).fetchall()

        for row in rows:
            rxn_id  = row["reaction_id"]
            sp_name = row["sp_name"]
            coeff   = abs(float(row["coeff"]))
            role    = row["role"]

            if rxn_id not in result:
                result[rxn_id] = {}
            signed_coeff = coeff if role == "product" else -coeff
            result[rxn_id][sp_name] = result[rxn_id].get(sp_name, 0.0) + signed_coeff

        return result
