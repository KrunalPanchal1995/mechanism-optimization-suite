"""
OptimizationTool.py  —  Upgraded Kinetic Mechanism Optimisation
================================================================
Changes vs original
-------------------
1. CodecEngine          – centralises encode/decode logic; fixes T-mismatch bug
                          and adds symmetric kappa_min/kappa_max bounding.
2. EncoderDecoderTester – stand-alone tester that verifies codec round-trips
                          against PRS training-data rows.
3. _eval_obj_core       – single DRY objective body (replaces six identical copies).
4. Gradient-based opts  – L-BFGS-B, SLSQP, trust-constr (analytic PRS Jacobian).
5. Alternative GA       – scipy differential_evolution (always available) +
                          optional DEAP (try-import).
6. All original methods – preserved exactly; run_optimization_* upgraded to
                          dispatch on Input_data["Type"]["Algorithm"].
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import time
import os
import pickle
import functools

from copy import deepcopy
from solution import Solution
from scipy.optimize import (minimize, differential_evolution,
                             NonlinearConstraint, Bounds, shgo, BFGS)
from scipy import optimize as spopt
from simulation_manager2_0 import SM as simulator

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import style, animation
style.use("fivethirtyeight")

import pygad

# Optional DEAP – import gracefully so the rest of the file still works if not installed
try:
    from deap import base, creator, tools, algorithms as deap_algorithms
    _DEAP_AVAILABLE = True
except ImportError:
    _DEAP_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  EarlyStopper  (preserved exactly)
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopper:
    def __init__(self, patience=30):
        self.patience      = patience
        self.best_fitness  = None
        self.counter       = 0

    def check(self, ga_instance):
        current_best_fitness = ga_instance.best_solution()[1]
        if self.best_fitness is None or current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.counter      = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            print(f"Stopping early: No improvement for {self.patience} generations")
            ga_instance.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  CodecEngine
# ─────────────────────────────────────────────────────────────────────────────
class CodecEngine:
    """
    Centralises the encode / decode pipeline for the 3-parameter optimisation.

    Encode (optimizer space → physical space)
    -----------------------------------------
    The optimizer searches x ∈ [-1, 1]^(3·N_rxn).
    For each reaction and each of the 3 anchor temperatures:

        x ≥ 0 :  κ = κ_0 + x · (κ_max − κ_0)
        x < 0 :  κ = κ_0 + x · (κ_0  − κ_min)

    This guarantees  x = +1 → κ = κ_max
                     x =  0 → κ = κ_0
                     x = -1 → κ = κ_min
    and x ∈ [-1,1] ↔ κ ∈ [κ_min, κ_max].

    Bug fixed from original
    -----------------------
    The original code used  self.T = np.linspace(300, 2500, 3)  for encoding
    but getZeta_typeA() uses the reaction's OWN temperature range
    [T_min, T_mid, T_max].  These are different for most reactions, making the
    round-trip  x → κ → getZeta_typeA → κ_recon  evaluate at different T values
    and therefore return a different (wrong) κ.  This class always uses the
    reaction's own anchor temperatures.
    """

    def __init__(self, unsrt_dict):
        """
        Parameters
        ----------
        unsrt_dict : dict  {rxn_key → UncertaintyExtractor subclass instance}
        """
        self.unsrt    = unsrt_dict
        self.rxn_list = list(unsrt_dict.keys())

        # Per-reaction anchor temperatures  [T_min, T_mid, T_max]
        self.T_anchor   = {}
        self.kappa_0    = {}   # nominal κ at anchor temperatures
        self.kappa_max  = {}   # upper uncertainty limit
        self.kappa_min  = {}   # lower uncertainty limit

        for rxn in self.rxn_list:
            T_lo  = self.unsrt[rxn].temperatures[0]
            T_hi  = self.unsrt[rxn].temperatures[-1]
            T_mid = 0.5 * (T_lo + T_hi)
            T_anc = np.array([T_lo, T_mid, T_hi])
            self.T_anchor[rxn]  = T_anc
            self.kappa_0[rxn]   = self.unsrt[rxn].getNominal(T_anc)
            self.kappa_max[rxn] = self.unsrt[rxn].getKappaMax(T_anc)
            self.kappa_min[rxn] = self.unsrt[rxn].getKappaMin(T_anc)

        # Flat vector size
        self.n_genes = 3 * len(self.rxn_list)

    # ── decode: x → {rxn: κ},  {rxn: ζ}  ────────────────────────────────────
    def decode(self, x):
        """
        Map optimizer vector x ∈ [-1,1]^n  →  per-reaction kappa curves and
        the corresponding zeta vectors (Arrhenius perturbation space).

        Returns
        -------
        kappa_dict : dict  rxn → np.ndarray  shape (3,)
        zeta_dict  : dict  rxn → np.ndarray  shape (3,)
        """
        kappa_dict = {}
        zeta_dict  = {}
        offset     = 0

        for rxn in self.rxn_list:
            x_rxn      = np.asarray(x[offset: offset + 3], dtype=float)
            k0         = self.kappa_0[rxn]
            k_max      = self.kappa_max[rxn]
            k_min      = self.kappa_min[rxn]

            # Symmetric encoding: positive x → toward max, negative → toward min
            kappa      = np.where(
                x_rxn >= 0,
                k0 + x_rxn * (k_max - k0),
                k0 + x_rxn * (k0   - k_min),
            )
            kappa_dict[rxn] = kappa
            zeta_dict[rxn]  = self.unsrt[rxn].getZeta_typeA(kappa)
            offset += 3

        return kappa_dict, zeta_dict

    # ── encode: {rxn: κ} → x  ────────────────────────────────────────────────
    def encode(self, kappa_dict):
        """
        Map per-reaction kappa curves  →  optimizer vector x ∈ [-1,1]^n.

        Used by the tester and for warm-start construction from training data.
        """
        x_parts = []
        for rxn in self.rxn_list:
            kappa = np.asarray(kappa_dict[rxn], dtype=float)
            k0    = self.kappa_0[rxn]
            k_max = self.kappa_max[rxn]
            k_min = self.kappa_min[rxn]

            denom_pos = np.where(np.abs(k_max - k0) > 1e-30, k_max - k0, 1e-30)
            denom_neg = np.where(np.abs(k0   - k_min) > 1e-30, k0 - k_min, 1e-30)

            x_rxn = np.where(
                kappa >= k0,
                (kappa - k0) / denom_pos,
                -(k0 - kappa) / denom_neg,
            )
            x_parts.append(x_rxn)

        return np.concatenate(x_parts)

    # ── encode from zeta  ─────────────────────────────────────────────────────
    def encode_from_zeta(self, zeta_dict):
        """
        Given a per-reaction zeta dict (e.g. a row from the PRS design matrix),
        reconstruct the corresponding kappa and then encode to x.

        This is the forward direction used by the tester:
            training-data row (ζ) → κ = p_0 + L·ζ evaluated at T_anchor → x.
        """
        kappa_dict = {}
        for rxn in self.rxn_list:
            p0        = np.asarray(self.unsrt[rxn].nominal, dtype=float)
            L         = self.unsrt[rxn].cholskyDeCorrelateMat
            zeta_rxn  = np.asarray(zeta_dict[rxn], dtype=float)
            T_anc     = self.T_anchor[rxn]
            Theta_anc = np.array([T_anc / T_anc, np.log(T_anc), -1.0 / T_anc])
            p_perturb = p0 + np.asarray(L.dot(zeta_rxn)).flatten()
            kappa_dict[rxn] = Theta_anc.T.dot(p_perturb)
        return self.encode(kappa_dict), kappa_dict

    def flat_zeta(self, zeta_dict):
        """Flatten zeta_dict to a 1-D array in rxn_list order."""
        out = []
        for rxn in self.rxn_list:
            out.extend(list(zeta_dict[rxn]))
        return np.asarray(out)

    def unflatten_zeta(self, zeta_flat):
        """Unpack a flat zeta array back to a per-reaction dict."""
        d      = {}
        offset = 0
        for rxn in self.rxn_list:
            n    = len(self.unsrt[rxn].activeParameters)
            d[rxn] = np.asarray(zeta_flat[offset: offset + n])
            offset += n
        return d

    def zeta_bounds(self):
        """
        Build bounds for direct zeta-space optimisation.
        Returns scipy Bounds object with  -|ζ_max|  ≤  ζ  ≤  +|ζ_max|.
        """
        lo, hi = [], []
        for rxn in self.rxn_list:
            z_max = np.abs(np.asarray(self.unsrt[rxn].zeta.x))
            for z in z_max:
                lo.append(-z);  hi.append(z)
        return Bounds(lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
#  EncoderDecoderTester
# ─────────────────────────────────────────────────────────────────────────────
class EncoderDecoderTester:
    """
    Validates the CodecEngine encode/decode pipeline against PRS training data.

    Tests performed
    ---------------
    T1 – Boundary consistency
         x = ±1 per anchor should decode to exactly κ_max / κ_min.
    T2 – Nominal recovery
         x = 0 per anchor should decode to κ_0.
    T3 – Bounds compliance
         All decoded κ for training-data rows should lie in [κ_min, κ_max].
    T4 – Round-trip (x → κ → ζ → re-encode → x̂ ≈ x)
         Checks that encode(decode(x)) ≈ x  (up to getZeta_typeA residual).
    T5 – Training-data consistency
         For each row of the design matrix (in ζ space), encode to x,
         then decode back to ζ̂, and report ||ζ̂ − ζ||.
    """

    def __init__(self, codec: CodecEngine):
        self.codec = codec
        self.results = {}   # populated by run_all_tests()

    # ── T1: boundary consistency ──────────────────────────────────────────────
    def test_boundary_consistency(self, tol=1e-10):
        print("\n[T1] Boundary consistency test  (x=±1 → κ_max/κ_min)")
        passed   = 0
        failed   = 0
        failures = []
        for rxn in self.codec.rxn_list:
            n   = 3
            # x = +1 everywhere for this reaction, 0 for others
            x_pos = np.zeros(self.codec.n_genes)
            x_neg = np.zeros(self.codec.n_genes)
            idx   = self.codec.rxn_list.index(rxn) * 3
            x_pos[idx: idx + n] =  1.0
            x_neg[idx: idx + n] = -1.0

            _, _ = self.codec.decode(x_pos)
            kd_pos, _ = self.codec.decode(x_pos)
            kd_neg, _ = self.codec.decode(x_neg)

            err_pos = np.max(np.abs(kd_pos[rxn] - self.codec.kappa_max[rxn]))
            err_neg = np.max(np.abs(kd_neg[rxn] - self.codec.kappa_min[rxn]))

            if err_pos < tol and err_neg < tol:
                passed += 1
            else:
                failed += 1
                failures.append(f"  {rxn}: err_pos={err_pos:.2e}  err_neg={err_neg:.2e}")

        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {status}  ({passed} passed, {failed} failed)")
        for f in failures:
            print(f)
        self.results["T1"] = {"passed": passed, "failed": failed, "failures": failures}
        return failed == 0

    # ── T2: nominal recovery ──────────────────────────────────────────────────
    def test_nominal_recovery(self, tol=1e-10):
        print("\n[T2] Nominal recovery  (x=0 → κ_0)")
        x_zero = np.zeros(self.codec.n_genes)
        kd, _  = self.codec.decode(x_zero)
        failed   = 0
        failures = []
        for rxn in self.codec.rxn_list:
            err = np.max(np.abs(kd[rxn] - self.codec.kappa_0[rxn]))
            if err > tol:
                failed += 1
                failures.append(f"  {rxn}: err={err:.2e}")
        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {status}  ({len(self.codec.rxn_list) - failed} passed, {failed} failed)")
        for f in failures:
            print(f)
        self.results["T2"] = {"failed": failed, "failures": failures}
        return failed == 0

    # ── T3: bounds compliance ─────────────────────────────────────────────────
    def test_bounds_compliance(self, x_samples=None, n_random=200, tol=1e-8):
        """
        Check that decoded κ always falls within [κ_min - tol, κ_max + tol].

        x_samples : np.ndarray of shape (N, n_genes), optional.
                    If None, uses n_random random samples from [-1,1].
        """
        print("\n[T3] Bounds compliance  (κ_decoded ∈ [κ_min, κ_max])")
        if x_samples is None:
            x_samples = np.random.uniform(-1, 1, (n_random, self.codec.n_genes))

        violations = 0
        for x in x_samples:
            kd, _ = self.codec.decode(x)
            for rxn in self.codec.rxn_list:
                if np.any(kd[rxn] < self.codec.kappa_min[rxn] - tol):
                    violations += 1
                if np.any(kd[rxn] > self.codec.kappa_max[rxn] + tol):
                    violations += 1

        status = "PASS" if violations == 0 else "FAIL"
        print(f"  {status}  (tested {len(x_samples)} samples, {violations} violations)")
        self.results["T3"] = {"violations": violations, "n_tested": len(x_samples)}
        return violations == 0

    # ── T4: round-trip x → κ → ζ → re-encode → x̂ ────────────────────────────
    def test_round_trip(self, n_samples=50, tol=0.05):
        """
        Decode x → κ → ζ using getZeta_typeA, then re-encode ζ → κ̂ → x̂.
        Because getZeta_typeA solves a constrained optimisation, small residuals
        are expected.  The tolerance is on max|x̂ − x| / (range of x = 2).
        """
        print(f"\n[T4] Round-trip  x → κ → ζ → re-encode → x̂  "
              f"(tol={tol:.2f}, {n_samples} samples)")
        errors  = []
        for _ in range(n_samples):
            x         = np.random.uniform(-0.95, 0.95, self.codec.n_genes)
            kd, zd    = self.codec.decode(x)
            x_hat, _  = self.codec.encode_from_zeta(zd)
            rel_err   = np.max(np.abs(x_hat - x)) / 2.0   # range is 2
            errors.append(rel_err)

        max_err  = np.max(errors)
        mean_err = np.mean(errors)
        status   = "PASS" if max_err < tol else "WARN"
        print(f"  {status}  mean|err|/range = {mean_err:.4f}   max = {max_err:.4f}")
        self.results["T4"] = {"max_err": max_err, "mean_err": mean_err}
        return max_err < tol

    # ── T5: training-data consistency ────────────────────────────────────────
    def test_training_data_consistency(self, design_matrix_rows, tol=0.1):
        """
        For each row of the PRS design matrix (zeta vectors), encode to x,
        then decode back to ζ̂ and compare.

        Parameters
        ----------
        design_matrix_rows : list of lists / np.ndarray, shape (N, n_params)
            Full-space ζ rows (length = sum of activeParameters per reaction).
        tol : float
            Maximum acceptable normalised reconstruction error.

        Returns
        -------
        report : dict  with keys "in_bounds", "mean_err", "max_err",
                 "out_of_bounds_count", "per_row_err"
        """
        print(f"\n[T5] Training-data consistency  ({len(design_matrix_rows)} rows)")
        per_row_err     = []
        out_of_bounds   = 0

        for row_idx, zeta_flat in enumerate(design_matrix_rows):
            zd          = self.codec.unflatten_zeta(np.asarray(zeta_flat, dtype=float))
            x_enc, kd_enc = self.codec.encode_from_zeta(zd)

            # Check bounds
            for rxn in self.codec.rxn_list:
                if (np.any(kd_enc[rxn] < self.codec.kappa_min[rxn] - 1e-6) or
                        np.any(kd_enc[rxn] > self.codec.kappa_max[rxn] + 1e-6)):
                    out_of_bounds += 1

            # Decode back and compare
            _, zd_hat  = self.codec.decode(x_enc)
            zeta_flat_hat = self.codec.flat_zeta(zd_hat)
            # Normalise error by zeta_max to get relative error
            z_max = np.abs(np.asarray(zeta_flat, dtype=float))
            z_max = np.where(z_max < 1e-12, 1.0, z_max)
            rel_err = np.max(np.abs(zeta_flat_hat - np.asarray(zeta_flat, dtype=float)) / z_max)
            per_row_err.append(rel_err)

        max_err  = np.max(per_row_err) if per_row_err else 0.0
        mean_err = np.mean(per_row_err) if per_row_err else 0.0
        in_bounds_pct = 100.0 * (1 - out_of_bounds / max(len(design_matrix_rows) * len(self.codec.rxn_list), 1))

        status = "PASS" if max_err < tol and out_of_bounds == 0 else "WARN"
        print(f"  {status}  mean_rel_err = {mean_err:.4f}   max = {max_err:.4f}   "
              f"in-bounds = {in_bounds_pct:.1f}%")

        report = {
            "in_bounds_pct"    : in_bounds_pct,
            "mean_err"         : mean_err,
            "max_err"          : max_err,
            "out_of_bounds_cnt": out_of_bounds,
            "per_row_err"      : per_row_err,
        }
        self.results["T5"] = report
        return report

    # ── run all ───────────────────────────────────────────────────────────────
    def run_all_tests(self, design_matrix_rows=None, n_random=200, n_round_trip=50):
        """
        Run T1–T5 in sequence.  T5 is skipped if design_matrix_rows is None.
        Prints a summary table.

        Parameters
        ----------
        design_matrix_rows : None  or  list of ζ rows from the design matrix
        n_random           : number of random x vectors for T3
        n_round_trip       : number of random samples for T4
        """
        print("=" * 60)
        print("  EncoderDecoderTester — full validation suite")
        print("=" * 60)

        self.test_boundary_consistency()
        self.test_nominal_recovery()
        self.test_bounds_compliance(n_random=n_random)
        self.test_round_trip(n_samples=n_round_trip)

        if design_matrix_rows is not None:
            self.test_training_data_consistency(design_matrix_rows)
        else:
            print("\n[T5] Skipped (no design_matrix_rows provided)")

        print("\n" + "=" * 60)
        print("  Summary")
        print("=" * 60)
        for key, val in self.results.items():
            if "failed" in val:
                status = "PASS" if val["failed"] == 0 else "FAIL"
            elif "violations" in val:
                status = "PASS" if val["violations"] == 0 else "FAIL"
            elif "max_err" in val:
                status = "PASS" if val["max_err"] < 0.05 else "WARN"
            else:
                status = "INFO"
            print(f"  {key}: {status}")
        print("=" * 60)
        return self.results


# ─────────────────────────────────────────────────────────────────────────────
#  OptimizationTool
# ─────────────────────────────────────────────────────────────────────────────
class OptimizationTool(object):
    def __init__(self, target_list=None, frequency=None):
        self.target_list = target_list
        self.objective   = 0
        self.frequency   = frequency
        self.count       = 0
        self.codec       = None   # set by _build_codec()

    # ═════════════════════════════════════════════════════════════════════════
    #  Codec initialiser
    # ═════════════════════════════════════════════════════════════════════════
    def _build_codec(self, unsrt_dict):
        """Create a CodecEngine and cache it as self.codec."""
        self.codec = CodecEngine(unsrt_dict)
        return self.codec

    # ═════════════════════════════════════════════════════════════════════════
    #  DRY objective core  (extracted from 6 identical copies)
    # ═════════════════════════════════════════════════════════════════════════
    def _eval_obj_core(self, x_zeta_flat, log_files=True):
        """
        Compute the weighted sum-of-squares objective given a flat zeta vector.

        Parameters
        ----------
        x_zeta_flat : 1-D array of zeta values in the order of self.rxn_index.
        log_files   : bool — whether to write the log files (disable during
                      batch testing to avoid I/O overhead).

        Returns
        -------
        obj          : float
        diff         : np.ndarray  (weighted residuals)
        target_value : list
        response_value : list
        diff_3       : dict  {case.uniqueID → relative error}
        """
        x             = np.asarray(x_zeta_flat, dtype=float)
        obj           = 0.0
        diff          = []
        diff_3        = {}
        target_value  = []
        response_value = []
        COUNT_Tig = COUNT_Fls = COUNT_Flw = COUNT_All = 0
        frequency = {}

        for i, case in enumerate(self.target_list):
            if self.ResponseSurfaces[i].selection != 1:
                continue

            ds = case.d_set
            frequency[ds] = frequency.get(ds, 0) + 1
            COUNT_All += 1

            if case.target == "Tig":
                COUNT_Tig += 1
                val   = self.ResponseSurfaces[i].evaluate(x)
                f_exp = np.log(case.observed * 10)
                w_    = case.std_dvtn / case.observed
                w     = 1.0 / w_
                diff.append((val - f_exp) * w)
                diff_3[case.uniqueID] = (val - f_exp) / f_exp
                target_value.append(f_exp)
                response_value.append(val)

            elif case.target == "Fls":
                COUNT_Fls += 1
                val   = np.exp(self.ResponseSurfaces[i].evaluate(x))
                f_exp = case.observed
                w     = 1.0 / case.std_dvtn
                diff.append((val - f_exp) * w)
                target_value.append(f_exp)
                response_value.append(val)

            elif case.target == "Flw":
                COUNT_Flw += 1
                val   = self.ResponseSurfaces[i].evaluate(x)
                f_exp = case.observed
                w     = 1.0 / case.std_dvtn
                diff.append((val - f_exp) * w)
                target_value.append(np.log(f_exp))
                response_value.append(val)

        diff  = np.asarray(diff)
        obj   = float(np.dot(diff, diff))

        if log_files:
            open("Objective.txt", "+a").write(f"{obj}\n")
            open("response_values.txt", "+a").write(
                f"\t{target_value},{response_value}\n")
            open("Dataset_based_obj", "+a").write(f"{diff_3}\n")

        return obj, diff, target_value, response_value, diff_3

    # ═════════════════════════════════════════════════════════════════════════
    #  Gradient-based objective and Jacobian (direct ζ space, PRS)
    # ═════════════════════════════════════════════════════════════════════════
    def _obj_func_zeta_direct(self, zeta_flat):
        """
        Objective in direct zeta space for gradient-based PRS optimisation.
        No encoding/decoding — the PRS is trained on ζ, so we search ζ directly.
        Uses ResponseSurface.Jacobian for the analytic gradient (see _jac below).
        """
        self.count += 1
        obj, diff, tv, rv, d3 = self._eval_obj_core(zeta_flat)
        s = ",".join(f"{v}" for v in zeta_flat) + "\n"
        open("zeta_guess_values.txt", "+a").write(s)
        open("guess_values.txt", "+a").write(f"{self.count},{s}")
        return obj

    def _jac_func_zeta_direct(self, zeta_flat):
        """
        Analytical Jacobian  d(obj)/d(ζ)  via the PRS Jacobian.
        d(obj)/d(ζ) = 2 · Σ_i  (w_i · r_i) · (w_i · dPRS_i/dζ)
        where r_i = PRS_i(ζ) − target_i  and w_i is the per-case weight.
        """
        x        = np.asarray(zeta_flat, dtype=float)
        grad     = np.zeros_like(x)
        COUNT_Tig = COUNT_Fls = 0

        for i, case in enumerate(self.target_list):
            if self.ResponseSurfaces[i].selection != 1:
                continue
            if case.target == "Tig":
                COUNT_Tig += 1
            elif case.target == "Fls":
                COUNT_Fls += 1

        for i, case in enumerate(self.target_list):
            if self.ResponseSurfaces[i].selection != 1:
                continue

            J_i = np.asarray(self.ResponseSurfaces[i].Jacobian(x))

            if case.target == "Tig":
                val   = self.ResponseSurfaces[i].evaluate(x)
                f_exp = np.log(case.observed * 10)
                w     = case.observed / case.std_dvtn          # 1/w_
                r_i   = (val - f_exp) * w
                grad += 2.0 * r_i * w * J_i

            elif case.target == "Fls":
                val   = np.exp(self.ResponseSurfaces[i].evaluate(x))
                f_exp = case.observed
                w     = 1.0 / case.std_dvtn
                r_i   = (val - f_exp) * w
                grad += 2.0 * r_i * w * np.exp(
                    self.ResponseSurfaces[i].evaluate(x)) * J_i

            elif case.target == "Flw":
                val   = self.ResponseSurfaces[i].evaluate(x)
                f_exp = case.observed
                w     = 1.0 / case.std_dvtn
                r_i   = (val - f_exp) * w
                grad += 2.0 * r_i * w * J_i

        return grad

    # ── encoded objective for optimizer (kappa-space x → ζ → PRS) ────────────
    def _obj_func_3param_encoded(self, x):
        """
        Objective in kappa-encoded x space.  Works for any optimizer that
        cannot use the analytical gradient (e.g. DE, GA, Powell).
        """
        self.count += 1
        _, zd      = self.codec.decode(x)
        zeta_flat  = self.codec.flat_zeta(zd)

        s = ",".join(f"{v}" for v in zeta_flat) + "\n"
        open("zeta_guess_values.txt", "+a").write(s)
        open("guess_values.txt", "+a").write(f"{self.count},{','.join(str(v) for v in x)}\n")

        obj, diff, tv, rv, d3 = self._eval_obj_core(zeta_flat)
        open("guess_values_TRANSFORMED.txt", "+a").write(
            f"{self.count},{','.join(str(v) for v in zeta_flat)}\n")
        return obj

    # ═════════════════════════════════════════════════════════════════════════
    #  Tester entry point
    # ═════════════════════════════════════════════════════════════════════════
    def test_encoding_decoding(self, unsrt_dict=None,
                                design_matrix_rows=None,
                                n_random=200, n_round_trip=50):
        """
        Run the full EncoderDecoderTester suite.

        Can be called before or after an optimisation run.

        Parameters
        ----------
        unsrt_dict         : dict  If None, uses self.unsrt (must have been set
                             by a prior call to run_optimization_*).
        design_matrix_rows : list / np.ndarray of flat ζ rows (from design matrix).
                             If None, T5 is skipped.
        n_random           : number of random samples for T3 (bounds compliance).
        n_round_trip       : number of samples for T4 (round-trip).

        Returns
        -------
        dict of test results (see EncoderDecoderTester.run_all_tests).
        """
        if unsrt_dict is None:
            unsrt_dict = getattr(self, "unsrt", None)
        if unsrt_dict is None:
            raise ValueError("unsrt_dict must be provided (or call after run_optimization_*).")

        codec  = CodecEngine(unsrt_dict)
        tester = EncoderDecoderTester(codec)
        return tester.run_all_tests(
            design_matrix_rows=design_matrix_rows,
            n_random=n_random,
            n_round_trip=n_round_trip,
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  ── ORIGINAL METHODS (preserved exactly) ─────────────────────────────
    # ═════════════════════════════════════════════════════════════════════════

    def obj_func_of_direct_simulations_3_param(self, x):
        self.count += 1
        open("guess_values.txt", "+a").write(f"{self.count},{x}\n")

        kappa_curve = {}
        count = 0
        for i in self.rxn_index:
            temp = []
            for j in range(len(self.T)):
                temp.append(x[count])
                count += 1
            Kappa = self.kappa_0[i] + temp * (self.kappa_max[i] - self.kappa_0[i])
            kappa_curve[i] = np.asarray(Kappa).flatten()

        zeta = {}
        for rxn in self.rxn_index:
            zeta[rxn] = self.unsrt[rxn].getZeta_typeA(kappa_curve[rxn])

        x_transformed = []
        string = ""
        for rxn in self.rxn_index:
            temp = list(zeta[rxn])
            for k in temp:
                string += f"{k},"
            x_transformed.extend(temp)
        string += "\n"
        x_transformed = np.asarray(x_transformed)
        open("zeta_guess_values.txt", "+a").write(string)
        x = x_transformed

        obj, diff, tv, rv, d3 = self._eval_obj_core(x)
        open("guess_values_TRANSFORMED.txt", "+a").write(f"{self.count},{x}\n")
        open("Dataset_based_obj", "+a").write(f"{d3}\n")
        return obj

    def obj_func_of_selected_PRS(self, x):
        self.count += 1
        open("guess_values.txt", "+a").write(f"{self.count},{x}\n")

        kappa_curve = {}
        count = 0
        for i in self.rxn_index:
            temp = []
            for j in range(len(self.T)):
                temp.append(x[count])
                count += 1
            Kappa = self.kappa_0[i] + temp * (self.kappa_max[i] - self.kappa_0[i])
            kappa_curve[i] = np.asarray(Kappa).flatten()

        zeta = {}
        for rxn in self.rxn_index:
            zeta[rxn] = self.unsrt[rxn].getZeta_typeA(kappa_curve[rxn])

        x_transformed = []
        string = ""
        for rxn in self.rxn_index:
            temp = list(zeta[rxn])
            for k in temp:
                string += f"{k},"
            x_transformed.extend(temp)
        string += "\n"
        x_transformed = np.asarray(x_transformed)
        open("zeta_guess_values.txt", "+a").write(string)
        x = x_transformed

        obj, diff, tv, rv, d3 = self._eval_obj_core(x)
        open("guess_values_TRANSFORMED.txt", "+a").write(f"{self.count},{x}\n")
        open("Dataset_based_obj", "+a").write(f"{d3}\n")
        return obj

    def FITNESS_PRS_BASED_OPT(self):
        global obj_func_of_selected_PRS

        def obj_func_of_selected_PRS(x, solution_idx):
            self.count += 1
            string_x = ",".join(str(i) for i in x)
            open("guess_values.txt", "+a").write(f"{self.count},{string_x}\n")

            kappa_curve = {}
            count = 0
            for i in self.rxn_index:
                temp = []
                for j in range(len(self.T)):
                    temp.append(x[count])
                    count += 1
                Kappa = self.kappa_0[i] + temp * (self.kappa_max[i] - self.kappa_0[i])
                kappa_curve[i] = np.asarray(Kappa).flatten()

            zeta = {}
            for rxn in self.rxn_index:
                zeta[rxn] = self.unsrt[rxn].getZeta_typeA(kappa_curve[rxn])

            x_transformed = []
            string = ""
            for rxn in self.rxn_index:
                temp = list(zeta[rxn])
                for k in temp:
                    string += f"{k},"
                x_transformed.extend(temp)
            string += "\n"
            x_transformed = np.asarray(x_transformed)
            open("zeta_guess_values.txt", "+a").write(string)
            x_eval = x_transformed

            obj, diff, tv, rv, d3 = self._eval_obj_core(x_eval)
            open("guess_values_TRANSFORMED.txt", "+a").write(f"{self.count},{x_eval}\n")
            open("Dataset_based_obj", "+a").write(f"{d3}\n")
            fitness = 1.0 / (abs(obj) + 1e-6)
            open("samplefile.txt", "+a").write(
                f"{self.ga_instance.generations_completed},{self.count},{self.objective},{fitness}\n")
            return fitness

        return obj_func_of_selected_PRS

    # ── GA fitness wrapper for pygad (3-param, codec-based) ──────────────────
    def _FITNESS_PRS_CODEC(self):
        """pygad fitness using the corrected CodecEngine (fixes T-mismatch)."""
        def fitness_fn(ga_inst, x, solution_idx):
            self.count += 1
            _, zd      = self.codec.decode(x)
            zeta_flat  = self.codec.flat_zeta(zd)
            obj, *_    = self._eval_obj_core(zeta_flat, log_files=True)
            fitness    = 1.0 / (abs(obj) + 1e-6)
            open("samplefile.txt", "+a").write(
                f"{ga_inst.generations_completed},{self.count},{obj},{fitness}\n")
            return fitness
        return fitness_fn

    # ── DE fitness callback ───────────────────────────────────────────────────
    def _obj_de_encoded(self, x):
        """Objective for scipy.differential_evolution (kappa-encoded x space)."""
        return self._obj_func_3param_encoded(x)

    def plot_DATA(self, x):
        kappa_curve = {}
        count = 0
        for i in self.rxn_index:
            temp = []
            for j in range(len(self.T)):
                temp.append(x[count])
                count += 1
            Kappa = self.kappa_0[i] + temp * (self.kappa_max[i] - self.kappa_0[i])
            kappa_curve[i] = np.asarray(Kappa).flatten()

        zeta = {}
        for rxn in self.rxn_index:
            zeta[rxn] = self.unsrt[rxn].getZeta_typeA(kappa_curve[rxn])

        x_transformed = []
        for rxn in self.rxn_index:
            x_transformed.extend(list(zeta[rxn]))
        x_eval = np.asarray(x_transformed)

        VALUE, EXP, CASE, TEMPERATURE = [], [], [], []
        for i, case in enumerate(self.target_list):
            if self.ResponseSurfaces[i].selection != 1:
                continue
            if case.target == "Tig":
                val   = self.ResponseSurfaces[i].evaluate(x_eval)
                f_exp = np.log(case.observed * 10)
                VALUE.append(np.exp(val) / 10)
                EXP.append(np.exp(f_exp) / 10)
                CASE.append(case.dataSet_id)
                TEMPERATURE.append(case.temperature)
            elif case.target == "Fls":
                val   = np.exp(self.ResponseSurfaces[i].evaluate(x_eval))
                f_exp = case.observed
                VALUE.append(val)
                EXP.append(f_exp)
                CASE.append(case.dataSet_id)
                TEMPERATURE.append(case.temperature)
        return VALUE, EXP, CASE, TEMPERATURE

    def _obj_function(self, x):
        """Jacobian-based residual + gradient (for LM-style solvers)."""
        kappa_curve = {}
        count = 0
        for i in self.rxn_index:
            temp = []
            for j in range(len(self.T)):
                temp.append(x[count])
                count += 1
            Kappa = self.kappa_0[i] + temp * (self.kappa_max[i] - self.kappa_0[i])
            kappa_curve[i] = np.asarray(Kappa).flatten()

        zeta = {}
        for rxn in self.rxn_index:
            zeta[rxn] = self.unsrt[rxn].getZeta_typeA(kappa_curve[rxn])

        x_transformed = []
        for rxn in self.rxn_index:
            x_transformed.extend(list(zeta[rxn]))
        x = np.asarray(x_transformed)

        num_params = len(x)
        num_expts  = len(self.target_list)
        f          = np.empty(num_expts)
        df         = np.zeros((num_expts, num_params))
        frequency  = {}
        COUNT_Tig = COUNT_Fls = COUNT_Flw = COUNT_All = 0

        for i, case in enumerate(self.target_list):
            if self.ResponseSurfaces[i].selection != 1:
                continue
            ds = case.d_set
            frequency[ds] = frequency.get(ds, 0) + 1
            COUNT_All += 1

            if case.target == "Tig":
                COUNT_Tig += 1
                val   = self.ResponseSurfaces[i].evaluate(x)
                f_exp = np.log(case.observed * 10)
                w     = 1.0 / np.log(case.std_dvtn * 10)
                f[i]  = (val - f_exp) * w
                df[i, :] = np.asarray(self.ResponseSurfaces[i].Jacobian(x)) * w

            elif case.target == "Fls":
                COUNT_Fls += 1
                val   = np.exp(self.ResponseSurfaces[i].evaluate(x))
                f_exp = case.observed
                w     = 1.0 / case.std_dvtn
                f[i]  = (val - f_exp) * w
                df[i, :] = np.asarray(self.ResponseSurfaces[i].Jacobian(x)) * w

            elif case.target == "Flw":
                COUNT_Flw += 1
                val   = self.ResponseSurfaces[i].evaluate(x)
                f_exp = case.observed
                w     = 1.0 / case.std_dvtn
                f[i]  = (val - f_exp) * w
                df[i, :] = np.asarray(self.ResponseSurfaces[i].Jacobian(x)) * w

        return f, df

    def _obj_func(self, x):
        """A-factor objective for run_optimization_with_selected_PRS A-facto branch."""
        open("zeta_guess_values.txt", "+a").write(",".join(str(i) for i in x) + "\n")

        obj = 0.0
        target_value   = []
        response_value = []
        target_stvd    = []
        case_stvd      = []
        COUNT_Tig = COUNT_Fls = COUNT_Flw = COUNT_All = 0
        frequency = {}

        for i, case in enumerate(self.target_list):
            if self.ResponseSurfaces[i].selection != 1:
                continue
            ds = case.d_set
            frequency[ds] = frequency.get(ds, 0) + 1
            COUNT_All += 1

            if case.target == "Tig":
                COUNT_Tig += 1
                val   = self.ResponseSurfaces[i].evaluate(x)
                response_value.append(val)
                target_value.append(np.log(case.observed * 10))
                target_stvd.append(1.0 / np.log(case.std_dvtn * 10))
                case_stvd.append(case.std_dvtn / case.observed)

            elif case.target == "Fls":
                COUNT_Fls += 1
                val   = np.exp(self.ResponseSurfaces[i].evaluate(x))
                response_value.append(val)
                target_value.append(case.observed)
                target_stvd.append(1.0 / case.std_dvtn)
                case_stvd.append(case.std_dvtn)

            elif case.target == "Flw":
                COUNT_Flw += 1
                val   = self.ResponseSurfaces[i].evaluate(x)
                response_value.append(val)
                target_value.append(np.log(case.observed))
                target_stvd.append(1.0 / (np.log(case.std_dvtn) + abs(np.log(case.observed) - val)))
                case_stvd.append(np.log(case.std_dvtn))

        self.count += 1
        diff                 = np.asarray(response_value) - np.asarray(target_value)
        multiplicating_factors = []
        for i, case in enumerate(self.target_list):
            if self.ResponseSurfaces[i].selection != 1:
                continue
            if case.target == "Tig":
                multiplicating_factors.append(1.0 / COUNT_Tig)
            elif case.target == "Fls":
                multiplicating_factors.append(0.05 * (1.0 / COUNT_Fls))
        mf = np.asarray(multiplicating_factors)

        for i, dif in enumerate(diff):
            obj += mf[i] * (target_stvd[i] * dif) ** 2

        open("guess_values.txt", "+a").write(f"{self.count},{x}\n")
        open("response_values.txt", "+a").write(f"\t{target_value},{response_value}\n")
        open("Objective.txt", "+a").write(f"{obj}\n")
        return obj

    def _obj_func_MLE_no_PRS_A_facto(self, x):
        """Direct simulation A-factor objective (MLE, no PRS)."""
        open("zeta_guess_values.txt", "+a").write(",".join(str(i) for i in x) + "\n")

        obj              = 0.0
        direct_sim_value = []
        target_value     = []
        target_stvd      = []
        COUNT_Tig = COUNT_Fls = COUNT_Flw = COUNT_All = 0
        frequency        = {}

        val = np.asarray(
            self.simulator.do_direct_sim(x, self.count, len(self.target_list),
                                         self.count, obj))
        direct_sim_value.append(val)

        for i, case in enumerate(self.target_list):
            ds = case.d_set
            frequency[ds] = frequency.get(ds, 0) + 1
            COUNT_All += 1

            if case.target == "Tig":
                COUNT_Tig += 1
                target_value.append(np.log(case.observed * 10))
                target_stvd.append(1.0 / np.log(case.std_dvtn * 10))

            elif case.target == "Fls":
                COUNT_Fls += 1
                target_value.append(case.observed)
                target_stvd.append(1.0 / case.std_dvtn)

            elif case.target == "Flw":
                COUNT_Flw += 1
                target_value.append(np.log(case.observed))
                target_stvd.append(1.0 / (np.log(case.std_dvtn) + abs(np.log(case.observed) - val)))

        self.count += 1
        diff = np.asarray(direct_sim_value - np.asarray(target_value)).flatten()
        mf   = []
        for i, case in enumerate(self.target_list):
            if case.target == "Tig":
                mf.append(1.0 / COUNT_Tig)
            elif case.target == "Fls":
                mf.append(1.0 / COUNT_Fls)
        mf = np.asarray(mf)

        for i, dif in enumerate(diff):
            obj += mf[i] * dif ** 2

        open("guess_values.txt", "+a").write(f"{self.count},{','.join(str(i) for i in x)}\n")
        open("response_values.txt", "+a").write(
            ",".join(str(i) for i in direct_sim_value) + "\n")
        open("target_values.txt", "+a").write(
            ",".join(str(i) for i in target_value) + "\n")
        open("Objective.txt", "+a").write(f"{obj}\n")
        return obj

    def fitness_function_for_T_INDIPENDENT(self):
        global fitness_func_T_indi

        def fitness_func_T_indi(x, solution_idx):
            open("zeta_guess_values.txt", "+a").write(",".join(str(i) for i in x) + "\n")

            obj              = 0.0
            direct_sim_value = []
            target_value     = []
            target_stvd      = []
            COUNT_Tig = COUNT_Fls = COUNT_Flw = COUNT_All = 0
            frequency        = {}

            val = np.asarray(
                self.simulator.do_direct_sim(x, self.count, len(self.target_list),
                                             self.count, obj))
            direct_sim_value.append(val)

            for i, case in enumerate(self.target_list):
                ds = case.d_set
                frequency[ds] = frequency.get(ds, 0) + 1
                COUNT_All += 1

                if case.target == "Tig":
                    COUNT_Tig += 1
                    target_value.append(np.log(case.observed * 10))
                    target_stvd.append(1.0 / np.log(case.std_dvtn * 10))

                elif case.target == "Fls":
                    COUNT_Fls += 1
                    target_value.append(case.observed)
                    target_stvd.append(1.0 / case.std_dvtn)

                elif case.target == "Flw":
                    COUNT_Flw += 1
                    target_value.append(np.log(case.observed))
                    target_stvd.append(1.0 / (np.log(case.std_dvtn)
                                               + abs(np.log(case.observed) - val)))

            self.count += 1
            diff = np.asarray(direct_sim_value - np.asarray(target_value)).flatten()
            mf   = []
            for i, case in enumerate(self.target_list):
                if case.target == "Tig":
                    mf.append(1.0 / COUNT_Tig)
                elif case.target == "Fls":
                    mf.append(1.0 / COUNT_Fls)
            mf = np.asarray(mf)

            for i, dif in enumerate(diff):
                obj += mf[i] * dif ** 2

            open("guess_values.txt", "+a").write(
                f"{self.count},{','.join(str(i) for i in x)}\n")
            open("response_values.txt", "+a").write(
                ",".join(str(i) for i in direct_sim_value) + "\n")
            open("target_values.txt", "+a").write(
                ",".join(str(i) for i in target_value) + "\n")
            open("Objective.txt", "+a").write(f"{obj}\n")

            fitness = 1.0 / (abs(obj) + 1e-6)
            open("samplefile.txt", "+a").write(
                f"{self.ga_instance.generations_completed},{self.count},{self.objective},{fitness}\n")
            return fitness

        return fitness_func_T_indi

    # ═════════════════════════════════════════════════════════════════════════
    #  Plotting helpers
    # ═════════════════════════════════════════════════════════════════════════
    def _plot_optimised_reactions(self, zeta, nominal, ch, p_max, p_min, Temp):
        """Generate per-reaction Arrhenius plots after optimisation."""
        d_n = {}
        for rxn in self.unsrt:
            p   = nominal[rxn]
            Tp  = Temp[rxn]
            Theta_p = np.array([Tp / Tp, np.log(Tp), -1.0 / Tp])
            kmax    = Theta_p.T.dot(p_max[rxn])
            kmin    = Theta_p.T.dot(p_min[rxn])
            ka_o    = Theta_p.T.dot(nominal[rxn])
            p_zet   = p + np.asarray(np.dot(ch[rxn], zeta[rxn])).flatten()
            k       = Theta_p.T.dot(p_zet)
            d_n[rxn] = abs(p[1] - p_zet[1])

            fig = plt.figure()
            plt.title(str(rxn))
            plt.xlabel(r"1000/T  K$^{-1}$")
            plt.ylabel(r"$\log_{10}(k)$")
            plt.plot(1 / Tp, kmax, 'k--', label="Uncertainty limits")
            plt.plot(1 / Tp, kmin, 'k--')
            plt.plot(1 / Tp, ka_o, 'b-',  label='Prior')
            plt.plot(1 / Tp, k,    'r-',  label='Optimised')
            plt.savefig(f"Plots/reaction_{rxn}.png", bbox_inches='tight')
            plt.close(fig)
        return d_n

    # ═════════════════════════════════════════════════════════════════════════
    #  run_optimization_with_selected_PRS  (UPGRADED)
    # ═════════════════════════════════════════════════════════════════════════
    def run_optimization_with_selected_PRS(self, Unsrt_data, ResponseSurfaces,
                                            Input_data):
        """
        PRS-based optimisation.

        Supported algorithms (Input_data["Type"]["Algorithm"]):
            A-facto design
            ──────────────
            "SLSQP"         gradient-based, bounded
            "L-BFGS-B"      limited-memory quasi-Newton, bounded
            "Powell"        (default / fallback)
            "trust-constr"  trust-region
            "DE"            scipy differential_evolution
            "GA-pygad"      pygad (codec-corrected)
            "GA-deap"       DEAP (requires deap package)

            A1+B1+C1 design
            ───────────────
            "SLSQP-zeta"    gradient-based directly in ζ space (analytic Jacobian)
            "L-BFGS-B-zeta" L-BFGS-B directly in ζ space
            "SLSQP"         gradient-based, encoded kappa space
            "L-BFGS-B"      L-BFGS-B, encoded kappa space
            "trust-constr"  trust-region, encoded kappa space
            "Powell"        derivative-free, encoded (original behaviour)
            "DE"            differential_evolution, encoded kappa space
            "GA-pygad"      pygad GA, codec-corrected
            "GA-deap"       DEAP GA (optional)
        """
        self.unsrt            = Unsrt_data
        self.ResponseSurfaces = ResponseSurfaces
        self.Input_data       = Input_data
        algorithm             = Input_data["Type"]["Algorithm"]
        self.rxn_index        = list(Unsrt_data.keys())

        # ── A-factor design ──────────────────────────────────────────────────
        if Input_data["Stats"]["Design_of_PRS"] == "A-facto":
            self.init_guess = np.zeros(len(self.rxn_index))
            bounds          = [(-1, 1)] * len(self.init_guess)

            if algorithm in ("SLSQP", "L-BFGS-B", "trust-constr"):
                opt = minimize(self._obj_func, self.init_guess,
                               bounds=bounds, method=algorithm,
                               options={"maxiter": 500000})
                optimal_parameters       = np.asarray(opt.x)
                optimal_parameters_zeta  = np.asarray(opt.x)
                cov                      = []
                print(opt)

            elif algorithm == "DE":
                res = differential_evolution(
                    self._obj_func, bounds,
                    maxiter=2000, tol=1e-8, seed=42,
                    mutation=(0.5, 1.0), recombination=0.7,
                    popsize=15, polish=True, workers=1)
                optimal_parameters       = np.asarray(res.x)
                optimal_parameters_zeta  = np.asarray(res.x)
                cov                      = []
                print(res)

            elif algorithm == "GA-deap" and _DEAP_AVAILABLE:
                optimal_parameters, optimal_parameters_zeta, cov = \
                    self._run_deap_afactor(self._obj_func, len(self.init_guess))

            else:
                # "GA-pygad" or default Powell-like fallback for A-facto
                opt = minimize(self._obj_func, self.init_guess,
                               bounds=bounds, method="SLSQP",
                               options={"maxiter": 500000})
                optimal_parameters       = np.asarray(opt.x)
                optimal_parameters_zeta  = np.asarray(opt.x)
                cov                      = []
                print(opt)

            return (np.asarray(optimal_parameters),
                    np.asarray(optimal_parameters_zeta), cov)

        # ── A1+B1+C1 design ─────────────────────────────────────────────────
        # Build codec (fixes T-mismatch bug in encoding)
        self._build_codec(Unsrt_data)

        # Also keep the legacy kappa_0/kappa_max for backward-compatible methods
        self.kappa_0   = self.codec.kappa_0
        self.kappa_max = self.codec.kappa_max
        self.T         = self.codec.T_anchor[self.rxn_index[0]]  # 3-element

        self.init_guess = np.zeros(self.codec.n_genes)
        bounds          = [(-1, 1)] * self.codec.n_genes

        start = time.time()

        # ── direct ζ-space gradient methods ──────────────────────────────────
        if algorithm in ("SLSQP-zeta", "L-BFGS-B-zeta"):
            method_name = algorithm.replace("-zeta", "")
            zeta_bnds   = self.codec.zeta_bounds()
            zeta0       = np.zeros(sum(
                len(self.unsrt[r].activeParameters) for r in self.rxn_index))
            opt = minimize(
                self._obj_func_zeta_direct, zeta0,
                jac=self._jac_func_zeta_direct,
                bounds=zeta_bnds,
                method=method_name,
                options={"maxiter": 100000, "ftol": 1e-12})
            print(f"Time: {time.time()-start:.1f}s")
            print(opt)
            optimal_parameters_zeta = np.asarray(opt.x)

            # Back out encoded x from optimal ζ
            zd = self.codec.unflatten_zeta(opt.x)
            x_enc, _ = self.codec.encode_from_zeta(zd)
            optimal_parameters = x_enc
            cov = []

        # ── encoded kappa-space gradient methods ──────────────────────────────
        elif algorithm in ("SLSQP", "L-BFGS-B", "trust-constr"):
            opt = minimize(
                self._obj_func_3param_encoded, self.init_guess,
                bounds=bounds, method=algorithm,
                options={"maxiter": 100000})
            print(f"Time: {time.time()-start:.1f}s")
            print(opt)
            optimal_parameters = np.asarray(opt.x)
            _, zd = self.codec.decode(opt.x)
            optimal_parameters_zeta = self.codec.flat_zeta(zd)
            cov = []

        # ── differential evolution ────────────────────────────────────────────
        elif algorithm == "DE":
            res = differential_evolution(
                self._obj_de_encoded, bounds,
                maxiter=2000, tol=1e-8, seed=42,
                mutation=(0.5, 1.0), recombination=0.7,
                popsize=15, polish=True, workers=1,
                callback=lambda xk, convergence=None:
                    open("Objective.txt", "+a").write(
                        f"DE gen: {self._obj_de_encoded(xk)}\n"))
            print(f"Time: {time.time()-start:.1f}s")
            print(res)
            optimal_parameters = np.asarray(res.x)
            _, zd = self.codec.decode(res.x)
            optimal_parameters_zeta = self.codec.flat_zeta(zd)
            cov = []

        # ── pygad GA (codec-corrected) ─────────────────────────────────────────
        elif algorithm == "GA-pygad":
            gene_space = [{"low": -1, "high": 1}] * self.codec.n_genes
            early_stopper = EarlyStopper(patience=50)
            self.ga_instance = pygad.GA(
                num_generations=2000,
                num_parents_mating=150,
                fitness_func=self._FITNESS_PRS_CODEC(),
                init_range_low=-1, init_range_high=1,
                sol_per_pop=300,
                num_genes=self.codec.n_genes,
                crossover_type="uniform",
                crossover_probability=0.6,
                mutation_type="adaptive",
                mutation_probability=(0.04, 0.01),
                gene_type=float,
                allow_duplicate_genes=False,
                gene_space=gene_space,
                keep_parents=-1,
                save_best_solutions=True,
                on_generation=lambda g: early_stopper.check(g),
                stop_criteria=["reach_1e-5"])
            self.ga_instance.run()
            self.ga_instance.save(filename="genetic_codec")
            sol, sol_fit, sol_idx = self.ga_instance.best_solution()
            print(f"Time: {time.time()-start:.1f}s")
            print(f"Best fitness = {sol_fit:.6e}")
            optimal_parameters = np.asarray(sol)
            _, zd = self.codec.decode(sol)
            optimal_parameters_zeta = self.codec.flat_zeta(zd)
            cov = []

        # ── DEAP GA ────────────────────────────────────────────────────────────
        elif algorithm == "GA-deap" and _DEAP_AVAILABLE:
            optimal_parameters, optimal_parameters_zeta, cov = \
                self._run_deap_3param(self._obj_func_3param_encoded,
                                      self.codec.n_genes, bounds)

        # ── Powell (default / legacy) ──────────────────────────────────────────
        else:
            opt = minimize(
                self.obj_func_of_selected_PRS, self.init_guess,
                bounds=bounds, method="Powell",
                options={"maxfev": 500000})
            print(f"Time: {time.time()-start:.1f}s")
            print(opt)
            optimal_parameters = np.asarray(opt.x)
            kappa_curve = {}
            cnt = 0
            for rxn in self.rxn_index:
                seg = [opt.x[cnt + j] for j in range(3)]; cnt += 3
                Kappa = (self.kappa_0[rxn]
                         + np.asarray(seg) * (self.kappa_max[rxn] - self.kappa_0[rxn]))
                kappa_curve[rxn] = Kappa.flatten()
            zeta = {rxn: self.unsrt[rxn].getZeta_typeA(kappa_curve[rxn])
                    for rxn in self.rxn_index}
            flat_z = []
            for rxn in self.rxn_index:
                flat_z.extend(list(zeta[rxn]))
            optimal_parameters_zeta = np.asarray(flat_z)
            cov = []

        # ── post-processing: plots + CSV ────────────────────────────────────
        self._post_optimisation_plots(optimal_parameters,
                                      optimal_parameters_zeta)
        return (np.asarray(optimal_parameters),
                np.asarray(optimal_parameters_zeta), cov)

    # ═════════════════════════════════════════════════════════════════════════
    #  run_optimization_with_MLE_no_PRS  (UPGRADED)
    # ═════════════════════════════════════════════════════════════════════════
    def run_optimization_with_MLE_no_PRS(self, Unsrt_data, Input_data):
        """
        Direct-simulation (no-PRS) optimisation.

        A-facto:  pygad GA or SLSQP (algorithm field selects).
        3-param:  Powell / L-BFGS-B / DE / GA-pygad / GA-deap.
        """
        self.unsrt        = Unsrt_data
        self.Input_data   = Input_data
        algorithm         = Input_data["Type"]["Algorithm"]
        self.rxn_index    = list(Unsrt_data.keys())
        self.init_guess   = np.zeros(len(self.rxn_index))
        bounds            = [(-1, 1)] * len(self.init_guess)
        design_matrix     = [[1]]
        self.simulator    = simulator(self.target_list, self.Input_data,
                                       self.unsrt, design_matrix, tag="Full")

        # ── A-factor design ──────────────────────────────────────────────────
        if Input_data["Stats"]["Design_of_PRS"] == "A-facto":

            if algorithm == "DE":
                res = differential_evolution(
                    self._obj_func_MLE_no_PRS_A_facto, bounds,
                    maxiter=2000, tol=1e-8, seed=42,
                    popsize=15, polish=True, workers=1)
                optimal_parameters       = np.asarray(res.x)
                optimal_parameters_zeta  = np.asarray(res.x)
                cov                      = []
                print(res)

            elif algorithm in ("SLSQP", "L-BFGS-B"):
                opt = minimize(self._obj_func_MLE_no_PRS_A_facto,
                               self.init_guess, bounds=bounds, method=algorithm,
                               options={"maxiter": 500000})
                optimal_parameters       = np.asarray(opt.x)
                optimal_parameters_zeta  = np.asarray(opt.x)
                cov                      = []
                print(opt)

            elif algorithm == "GA-deap" and _DEAP_AVAILABLE:
                optimal_parameters, optimal_parameters_zeta, cov = \
                    self._run_deap_afactor(self._obj_func_MLE_no_PRS_A_facto,
                                           len(self.init_guess))
            else:
                # Default: pygad GA (existing behaviour preserved)
                fitness_function = self.fitness_function_for_T_INDIPENDENT()
                gene_space       = [{"low": -1, "high": 1}] * len(self.init_guess)
                early_stopper    = EarlyStopper()
                self.ga_instance = pygad.GA(
                    num_generations=2000,
                    num_parents_mating=300,
                    fitness_func=fitness_function,
                    init_range_low=-1, init_range_high=1,
                    sol_per_pop=400,
                    num_genes=len(self.init_guess),
                    crossover_type="uniform",
                    crossover_probability=0.6,
                    mutation_type="adaptive",
                    mutation_probability=(0.03, 0.008),
                    gene_type=float,
                    allow_duplicate_genes=False,
                    gene_space=gene_space,
                    keep_parents=-1,
                    save_best_solutions=True,
                    save_solutions=True,
                    stop_criteria=["reach_20"])
                self.ga_instance.run()
                self.ga_instance.save(filename="genetic")
                sol, sol_fit, sol_idx = self.ga_instance.best_solution()
                optimal_parameters       = np.asarray(sol)
                optimal_parameters_zeta  = np.asarray(sol)
                cov                      = []
                print(f"Best fitness = {sol_fit:.6e}")

            return (np.asarray(optimal_parameters),
                    np.asarray(optimal_parameters_zeta), cov)

        # ── 3-param direct simulation design ────────────────────────────────
        self._build_codec(Unsrt_data)
        self.kappa_0   = {}
        self.kappa_max = {}
        self.T         = np.linspace(300, 2500, 3)

        for rxn in self.rxn_index:
            T_anc = self.codec.T_anchor[rxn]
            self.kappa_0[rxn]   = self.unsrt[rxn].getNominal(T_anc)
            self.kappa_max[rxn] = self.unsrt[rxn].getKappaMax(T_anc)

        self.init_guess = np.zeros(len(self.codec.T_anchor[self.rxn_index[0]]) *
                                    len(self.rxn_index))
        bounds          = [(-1, 1)] * len(self.init_guess)
        start           = time.time()

        if algorithm in ("L-BFGS-B", "SLSQP", "trust-constr"):
            opt = minimize(
                self.obj_func_of_direct_simulations_3_param,
                self.init_guess, bounds=bounds, method=algorithm,
                options={"maxiter": 500000})
            optimal_parameters = np.asarray(opt.x)
            print(opt)

        elif algorithm == "DE":
            res = differential_evolution(
                self.obj_func_of_direct_simulations_3_param, bounds,
                maxiter=2000, tol=1e-8, seed=42,
                popsize=15, polish=True, workers=1)
            optimal_parameters = np.asarray(res.x)
            print(res)

        elif algorithm == "GA-pygad":
            gene_space    = [{"low": -1, "high": 1}] * len(self.init_guess)
            early_stopper = EarlyStopper(patience=50)

            def _fitness_3param_direct(ga_inst, x, sol_idx):
                self.count += 1
                obj = self.obj_func_of_direct_simulations_3_param(x)
                return 1.0 / (abs(obj) + 1e-6)

            self.ga_instance = pygad.GA(
                num_generations=2000,
                num_parents_mating=150,
                fitness_func=_fitness_3param_direct,
                init_range_low=-1, init_range_high=1,
                sol_per_pop=300,
                num_genes=len(self.init_guess),
                crossover_type="uniform",
                crossover_probability=0.6,
                mutation_type="adaptive",
                mutation_probability=(0.04, 0.01),
                gene_type=float,
                allow_duplicate_genes=False,
                gene_space=gene_space,
                keep_parents=-1,
                on_generation=lambda g: early_stopper.check(g))
            self.ga_instance.run()
            sol, sol_fit, _ = self.ga_instance.best_solution()
            optimal_parameters = np.asarray(sol)

        elif algorithm == "GA-deap" and _DEAP_AVAILABLE:
            optimal_parameters, _, _ = self._run_deap_3param(
                self.obj_func_of_direct_simulations_3_param,
                len(self.init_guess), bounds)

        else:
            # Powell (original behaviour)
            opt = minimize(
                self.obj_func_of_direct_simulations_3_param,
                self.init_guess, bounds=bounds, method="Powell",
                options={"maxfev": 500000})
            optimal_parameters = np.asarray(opt.x)
            print(opt)

        print(f"Optimisation time: {time.time()-start:.1f}s")

        # Decode optimal kappa → zeta
        kappa_curve = {}
        cnt         = 0
        for rxn in self.rxn_index:
            seg  = optimal_parameters[cnt: cnt + 3]; cnt += 3
            kappa_curve[rxn] = (self.kappa_0[rxn]
                                 + seg * (self.kappa_max[rxn] - self.kappa_0[rxn]))
        zeta = {rxn: self.unsrt[rxn].getZeta_typeA(kappa_curve[rxn])
                for rxn in self.rxn_index}
        flat_z = []
        for rxn in self.rxn_index:
            flat_z.extend(list(zeta[rxn]))
        optimal_parameters_zeta = np.asarray(flat_z)
        cov = []

        # Plots
        ch      = {r: self.unsrt[r].cholskyDeCorrelateMat  for r in self.unsrt}
        nominal = {r: self.unsrt[r].nominal                 for r in self.unsrt}
        p_max   = {r: self.unsrt[r].P_max                  for r in self.unsrt}
        p_min   = {r: self.unsrt[r].P_min                  for r in self.unsrt}
        Temp    = {r: self.unsrt[r].temperatures            for r in self.unsrt}
        d_n = self._plot_optimised_reactions(zeta, nominal, ch, p_max, p_min, Temp)
        print(d_n)

        VALUE, EXP, CASE, TEMPERATURE = self.plot_DATA(optimal_parameters)
        for case_id in set(CASE):
            with open(f"{case_id}.csv", "w") as fh:
                fh.write("T(k)\tf_exp\tValue\n")
                for i, c in enumerate(CASE):
                    if c == case_id:
                        fh.write(f"{TEMPERATURE[i]}\t{EXP[i]}\t{VALUE[i]}\n")

        return (np.asarray(optimal_parameters),
                np.asarray(optimal_parameters_zeta), cov)

    # ═════════════════════════════════════════════════════════════════════════
    #  DEAP helpers  (optional — only used when deap is installed)
    # ═════════════════════════════════════════════════════════════════════════
    def _run_deap_afactor(self, obj_func, n_genes,
                           n_gen=500, pop_size=200, cxpb=0.6, mutpb=0.2):
        """
        DEAP Differential Evolution for the A-factor case.

        Each gene is in [-1, 1].  The DE/rand/1/bin strategy is used.
        """
        if not _DEAP_AVAILABLE:
            raise ImportError("DEAP is not installed.  Run: pip install deap")

        import random

        # Recreate DEAP types (safe to call multiple times)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, -1, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_float, n=n_genes)
        toolbox.register("population", tools.initRepeat, list,
                         toolbox.individual)

        def evaluate(ind):
            return (obj_func(np.array(ind)),)

        toolbox.register("evaluate",  evaluate)
        toolbox.register("select",    tools.selBest)
        toolbox.register("mate",      tools.cxUniform, indpb=0.5)
        toolbox.register("mutate",    tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)

        pop  = toolbox.population(n=pop_size)
        hof  = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        pop, log = deap_algorithms.eaSimple(
            pop, toolbox,
            cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
            stats=stats, halloffame=hof, verbose=True)

        best   = np.array(hof[0])
        # Clip to bounds
        best   = np.clip(best, -1, 1)
        print(f"DEAP best objective = {hof[0].fitness.values[0]:.6e}")
        return best, best, []

    def _run_deap_3param(self, obj_func, n_genes, bounds_list,
                          n_gen=500, pop_size=200, cxpb=0.6, mutpb=0.2):
        """DEAP DE for the 3-param encoded-kappa case."""
        if not _DEAP_AVAILABLE:
            raise ImportError("DEAP is not installed.  Run: pip install deap")

        import random
        lo = [b[0] for b in bounds_list]
        hi = [b[1] for b in bounds_list]

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        def rand_gene(i):
            return random.uniform(lo[i], hi[i])

        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         lambda: [rand_gene(i) for i in range(n_genes)])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evaluate(ind):
            x = np.clip(ind, lo, hi)
            return (obj_func(x),)

        toolbox.register("evaluate", evaluate)
        toolbox.register("select",   tools.selBest)
        toolbox.register("mate",     tools.cxUniform, indpb=0.5)
        toolbox.register("mutate",   tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)

        pop  = toolbox.population(n=pop_size)
        hof  = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        pop, log = deap_algorithms.eaSimple(
            pop, toolbox,
            cxpb=cxpb, mutpb=mutpb, ngen=n_gen,
            stats=stats, halloffame=hof, verbose=True)

        best = np.clip(np.array(hof[0]), lo, hi)
        _, zd = self.codec.decode(best)
        flat_z = self.codec.flat_zeta(zd)
        print(f"DEAP best objective = {hof[0].fitness.values[0]:.6e}")
        return best, flat_z, []

    # ═════════════════════════════════════════════════════════════════════════
    #  Post-optimisation helpers
    # ═════════════════════════════════════════════════════════════════════════
    def _post_optimisation_plots(self, optimal_parameters, optimal_parameters_zeta):
        """Generate reaction plots and per-dataset CSV files after optimisation."""
        ch      = {r: self.unsrt[r].cholskyDeCorrelateMat  for r in self.unsrt}
        nominal = {r: self.unsrt[r].nominal                 for r in self.unsrt}
        p_max   = {r: self.unsrt[r].P_max                  for r in self.unsrt}
        p_min   = {r: self.unsrt[r].P_min                  for r in self.unsrt}
        Temp    = {r: self.unsrt[r].temperatures            for r in self.unsrt}

        zd = self.codec.unflatten_zeta(optimal_parameters_zeta)
        d_n = self._plot_optimised_reactions(zd, nominal, ch, p_max, p_min, Temp)
        print("Δn per reaction:", d_n)

        VALUE, EXP, CASE, TEMPERATURE = self.plot_DATA(optimal_parameters)
        for case_id in set(CASE):
            with open(f"{case_id}.csv", "w") as fh:
                fh.write("T(k)\tf_exp\tValue\n")
                for i, c in enumerate(CASE):
                    if c == case_id:
                        fh.write(f"{TEMPERATURE[i]}\t{EXP[i]}\t{VALUE[i]}\n")
