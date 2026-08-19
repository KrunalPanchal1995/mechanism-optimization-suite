"""
Analytical Sobol sensitivity indices from a 2nd-order PRS.

For a 2nd-order polynomial:
    η(ζ) = c₀ + Σ_i a_i ζ_i + Σ_{i≤j} b_ij ζ_i ζ_j

with ζ_i ~ U(-1, 1) independently, Sobol indices follow the ANOVA
decomposition of Sudret (2008):

  Var(η) = Σ_i a_i²/3  +  Σ_i b_ii² × 4/45  +  Σ_{i<j} b_ij² × 1/9

  First-order:  S_i  = (a_i²/3 + b_ii² × 4/45)                 / Var(η)
  Total-order:  ST_i = (a_i²/3 + b_ii² × 4/45 + Σ_{j≠i} b_ij²/9) / Var(η)

These are computed analytically from PRS coefficients — no additional
simulations are required.

Reference:
  Sudret B. (2008) "Global sensitivity analysis using polynomial chaos
  expansions", Reliab. Eng. Syst. Safety 93(7):964–979.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class AnalyticalSobolPRS:
    """
    Compute first- and total-order Sobol indices analytically from a
    fitted 2nd-order ResponseSurface object.

    Parameters
    ----------
    prs         : ResponseSurface with fitted .coeff (order must be 2)
    comp_to_rxn : {compressed_index: (rxn_key, param_string)}
                  maps each compressed parameter position to its reaction
    rxn_keys    : ordered list of all reaction keys (sets output ordering)
    """

    # Moments of U(-1, 1)
    VAR_ZI    = 1.0 / 3.0      # Var(ζ_i)
    VAR_ZI2   = 4.0 / 45.0     # Var(ζ_i²)   = E[ζ_i⁴] - (E[ζ_i²])² = 1/5 - 1/9
    VAR_CROSS = 1.0 / 9.0      # Var(ζ_i ζ_j) for i ≠ j

    def __init__(
        self,
        prs,
        comp_to_rxn: Dict[int, Tuple[str, str]],
        rxn_keys: List[str],
    ):
        self.prs         = prs
        self.comp_to_rxn = comp_to_rxn
        self.rxn_keys    = rxn_keys
        self.n_comp      = prs.n_comp

        # ── Reconstruct (a, b) from coeff ─────────────────────────────────
        # coeff ordering mirrors MatPolyFitTransform / evaluate_prs:
        #   coeff[0]            = constant
        #   coeff[1 .. n]       = linear coefficients a_i
        #   coeff[n+1 .. end]   = quadratic coefficients, upper-triangular
        #     for i in range(n): for j in range(i, n): b[i,j] = coeff[count]
        coeff = np.asarray(prs.coeff, dtype=float)
        n     = self.n_comp

        self._zero = float(coeff[0])
        self._a    = coeff[1 : 1 + n].copy()

        self._b = np.zeros((n, n), dtype=float)
        count = 1 + n
        for i in range(n):
            for j in range(i, n):
                if count < len(coeff):
                    self._b[i, j] = coeff[count]
                    count += 1

    # ------------------------------------------------------------------
    # Variance decomposition helpers
    # ------------------------------------------------------------------

    def total_variance(self) -> float:
        """Total output variance D = Σ D_i + Σ D_{ij}."""
        a, b = self._a, self._b
        n    = self.n_comp

        D  = float(np.dot(a, a)) * self.VAR_ZI               # linear
        D += float(np.sum(np.diag(b) ** 2)) * self.VAR_ZI2   # diagonal quadratic

        # Off-diagonal interaction terms
        for i in range(n):
            for j in range(i + 1, n):
                D += b[i, j] ** 2 * self.VAR_CROSS

        return D

    def _first_order_per_param(self) -> np.ndarray:
        """Per-compressed-param first-order variance D_i (unnormalised)."""
        return (self._a ** 2 * self.VAR_ZI
                + np.diag(self._b) ** 2 * self.VAR_ZI2)

    def _interaction_per_param(self) -> np.ndarray:
        """Per-param interaction contribution  Σ_{j≠i} D_ij (unnormalised)."""
        n     = self.n_comp
        D_int = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                contrib    = self._b[i, j] ** 2 * self.VAR_CROSS
                D_int[i]  += contrib
                D_int[j]  += contrib
        return D_int

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, Dict]:
        """
        Compute reaction-level Sobol indices.

        For multi-parameter reactions (A1+B1+C1 design), indices are
        aggregated over all compressed parameters of that reaction.

        Returns
        -------
        {rxn_key: {"S1": float, "ST": float, "V_i": float,
                   "V_Ti": float, "active": bool}}
        """
        D      = self.total_variance()
        result = {}

        if D < 1e-15:
            for rxn in self.rxn_keys:
                result[rxn] = {"S1": 0.0, "ST": 0.0,
                               "V_i": 0.0, "V_Ti": 0.0, "active": False}
            return result

        D_i   = self._first_order_per_param()
        D_int = self._interaction_per_param()

        # Group compressed indices by reaction
        rxn_to_comp: Dict[str, List[int]] = {}
        for comp_idx, (rxn_key, _param) in self.comp_to_rxn.items():
            rxn_to_comp.setdefault(rxn_key, []).append(comp_idx)

        for rxn in self.rxn_keys:
            comp_ids = rxn_to_comp.get(rxn, [])
            if not comp_ids:
                result[rxn] = {"S1": 0.0, "ST": 0.0,
                               "V_i": 0.0, "V_Ti": 0.0, "active": False}
                continue

            V_i  = float(np.sum(D_i[comp_ids]))
            V_Ti = float(np.sum(D_i[comp_ids]) + np.sum(D_int[comp_ids]))
            result[rxn] = {
                "S1":    V_i  / D,
                "ST":    V_Ti / D,
                "V_i":   V_i,
                "V_Ti":  V_Ti,
                "active": True,
            }

        return result

    # ------------------------------------------------------------------
    # Convenience output
    # ------------------------------------------------------------------

    def to_dataframe(self, case_index: int) -> pd.DataFrame:
        """
        Return a long-format DataFrame:
          columns: case_index, reaction, S1, ST, active
        """
        data = self.compute()
        rows = [
            {
                "case_index": case_index,
                "reaction":   rxn,
                "S1":         v["S1"],
                "ST":         v["ST"],
                "active":     v["active"],
            }
            for rxn, v in data.items()
        ]
        return pd.DataFrame(rows)
