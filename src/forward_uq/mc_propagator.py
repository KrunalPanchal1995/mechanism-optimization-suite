"""
Monte Carlo forward UQ propagation engine.

Propagates Arrhenius parameter uncertainties through trained polynomial
response surfaces (PRS) using quasi-Monte Carlo sampling, then collects
output distributions and statistics.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Vectorised polynomial feature builder
# ---------------------------------------------------------------------------

def _build_feature_matrix(Z: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Build the polynomial B-matrix for a batch of compressed inputs.

    The column ordering matches exactly the ordering used in
    ResponseSurface.evaluate_prs() and MatPolyFitTransform():

      col 0          : constant 1
      cols 1..n      : linear terms  z_i
      cols n+1..end  : quadratic terms z_i * z_j  (i <= j, row-major)

    Parameters
    ----------
    Z     : (N, n_comp) array of compressed ζ samples
    order : polynomial order (only 1 and 2 are supported)

    Returns
    -------
    B : (N, n_features) feature matrix  — B @ coeff gives η predictions
    """
    N, n = Z.shape

    if order == 1:
        n_feat = 1 + n
        B = np.empty((N, n_feat))
        B[:, 0] = 1.0
        B[:, 1:] = Z
        return B

    if order == 2:
        n_quad = n * (n + 1) // 2
        n_feat = 1 + n + n_quad
        B = np.empty((N, n_feat))
        B[:, 0] = 1.0
        B[:, 1:1 + n] = Z
        col = 1 + n
        for i in range(n):
            for j in range(i, n):           # j >= i : upper-triangular + diagonal
                B[:, col] = Z[:, i] * Z[:, j]
                col += 1
        return B

    raise ValueError(f"order={order} not supported; use 1 or 2.")


# ---------------------------------------------------------------------------
# Main propagator class
# ---------------------------------------------------------------------------

class MCPropagator:
    """
    Monte Carlo forward UQ propagator using PRS surrogates.

    Key idea: once a PRS is trained on ~hundreds of simulations, evaluating it
    on 50 000 quasi-random samples costs milliseconds (polynomial dot-product),
    giving the full prior output distribution without extra simulations.

    Parameters
    ----------
    prs_dict    : {case_index: ResponseSurface}
    unsrt_data  : {rxn_key: reaction_object}  (from Uncertainty.py)
    target_list : list of combustion_target objects
    n_full      : total number of ζ parameters (sum of activeParameters over all rxns)
    design_type : 'A-facto' or 'A1+B1+C1'
    """

    def __init__(
        self,
        prs_dict: Dict,
        unsrt_data: Dict,
        target_list: List,
        n_full: int,
        design_type: str = "A-facto",
    ):
        self.prs_dict    = prs_dict
        self.unsrt_data  = unsrt_data
        self.target_list = target_list
        self.n_full      = n_full
        self.design_type = design_type

    # ------------------------------------------------------------------
    # Prior sampling
    # ------------------------------------------------------------------

    def sample_prior(
        self,
        n_samples: int = 50_000,
        method: str = "sobol",
        seed: int = 42,
    ) -> np.ndarray:
        """
        Draw ζ samples from the prior distribution.

        For 'A-facto' design: ζ_i ~ U(-1, 1) per active reaction,
        sampled jointly using QMC (Sobol / LHS) for low-discrepancy
        coverage of the uncertainty space.

        For 'A1+B1+C1' design: delegates to DesignMatrix.getSamples()
        which generates class-A/B/C Arrhenius-curve-consistent samples.

        Parameters
        ----------
        n_samples : requested number of samples (Sobol rounds up to 2^k)
        method    : 'sobol', 'lhs', or 'numpy'
        seed      : integer seed for reproducibility

        Returns
        -------
        zeta_matrix : (N_actual, n_full) float64 array
        """
        # Import here to keep the module path flexible
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.join(_root, "MUQ-SAC") not in sys.path:
            sys.path.insert(0, os.path.join(_root, "MUQ-SAC"))

        from qmc_sampling import joint_uniform_block

        if self.design_type == "A-facto":
            # Each of the n_full dimensions is ζ_i ~ U(-1, 1)
            # joint_uniform_block returns (pts, n_actual)
            pts, _n_actual = joint_uniform_block(method, n_samples, self.n_full, seed)
            u    = np.asarray(pts, dtype=np.float64)
            zeta = 2.0 * u - 1.0           # transform U(0,1) → U(-1,1)
            return zeta

        # Full A1+B1+C1 design — use the existing DesignMatrix sampler
        import DesignMatrix as DM
        print(f"  Using DesignMatrix.getSamples() for {method} design ...")
        dm = DM.DesignMatrix(self.unsrt_data, self.design_type, n_samples, method)
        samples = np.asarray(dm.getSamples(), dtype=np.float64)
        return samples

    # ------------------------------------------------------------------
    # Batch PRS propagation
    # ------------------------------------------------------------------

    def propagate(self, zeta_matrix: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Evaluate every PRS at all ζ samples in a vectorised batch.

        Parameters
        ----------
        zeta_matrix : (N, n_full) full-space parameter matrix

        Returns
        -------
        predictions : {case_index: np.ndarray of length N}
                      each array contains predicted η values
        """
        predictions: Dict[int, np.ndarray] = {}
        N = len(zeta_matrix)

        for case_index, prs in self.prs_dict.items():
            active_idx = prs._active_idx

            if len(active_idx) == 0:
                # No active parameters for this case — constant at nominal
                eta_nom = float(prs.coeff[0])
                predictions[case_index] = np.full(N, eta_nom)
                continue

            # Compress to selected parameters and batch-evaluate
            Z_comp = zeta_matrix[:, active_idx]          # (N, n_comp)
            B = _build_feature_matrix(Z_comp, order=int(prs.order))
            eta_pred = B @ np.asarray(prs.coeff, dtype=np.float64)
            predictions[case_index] = eta_pred

        return predictions

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_statistics(
        self,
        predictions: Dict[int, np.ndarray],
    ) -> List[Dict]:
        """
        Compute per-case UQ statistics and back-transform to physical units.

        η ↔ physical unit conventions (matching OptimizationTool.py):
          Tig / RCM / JSR : η = ln(τ_μs × 10)   ↔   τ_μs = exp(η) / 10
          Fls             : η = ln(S_u [cm/s])   ↔   S_u = exp(η)
          Flw / Flf       : η = observable       ↔   linear

        Returns
        -------
        List of dicts (one per target case present in predictions).
        Also sets self.stats_list for later access.
        """
        stats: List[Dict] = []

        for i, case in enumerate(self.target_list):
            if i not in predictions:
                continue

            eta    = predictions[i]
            t_type = getattr(case, "target", "Tig") or "Tig"

            # Back-transform
            if any(k in t_type for k in ("Tig", "RCM", "JSR")):
                phys_pred = np.exp(eta) / 10.0           # [μs]
                phys_obs  = float(case.observed)         # [μs] after unit conv
                eta_obs   = np.log(phys_obs * 10.0)
                unit      = "μs"
            elif "Fls" in t_type or "Fsl" in t_type:
                phys_pred = np.exp(eta)                  # [cm/s]
                phys_obs  = float(case.observed)
                eta_obs   = np.log(max(phys_obs, 1e-12))
                unit      = "cm/s"
            else:
                phys_pred = eta.copy()
                phys_obs  = float(case.observed)
                eta_obs   = phys_obs
                unit      = "linear"

            # η-space statistics
            eta_mean = float(np.mean(eta))
            eta_std  = float(np.std(eta))
            pcts     = np.percentile(eta, [5, 25, 50, 75, 95])
            eta_p5, eta_p25, eta_p50, eta_p75, eta_p95 = map(float, pcts)

            # Coverage
            covered_90 = bool(eta_p5  <= eta_obs <= eta_p95)
            covered_50 = bool(eta_p25 <= eta_obs <= eta_p75)

            # Physical-space statistics
            phys_pcts = np.percentile(phys_pred, [5, 50, 95])
            phys_p5, phys_p50, phys_p95 = map(float, phys_pcts)

            try:
                phi = float(case.phi)
            except (TypeError, ValueError):
                phi = float("nan")

            stats.append({
                "case_index":  i,
                "case_id":     case.case_index,
                "dataset_id":  case.dataSet_id,
                "target_type": t_type,
                "T_K":         float(case.temperature),
                "P_Pa":        float(case.pressure),
                "phi":         phi,
                "observed":    phys_obs,
                "std_dvtn":    float(case.std_dvtn),
                "unit":        unit,
                # Log-space
                "eta_obs":     eta_obs,
                "eta_mean":    eta_mean,
                "eta_std":     eta_std,
                "eta_p5":      eta_p5,
                "eta_p25":     eta_p25,
                "eta_p50":     eta_p50,
                "eta_p75":     eta_p75,
                "eta_p95":     eta_p95,
                # Physical-space
                "phys_p5":     phys_p5,
                "phys_p50":    phys_p50,
                "phys_p95":    phys_p95,
                # Coverage
                "covered_90":  covered_90,
                "covered_50":  covered_50,
                # PRS quality flag
                "prs_selection": int(getattr(self.prs_dict.get(i), "selection", 1) or 1),
            })

        self.stats_list = stats
        return stats
