"""
partial_sac_sampler.py
======================
Generate class-A, B, C Arrhenius curve samples for ALL possible subsets of
active Arrhenius parameters {alpha, n, epsilon} using the Partial-Parameter
SAC formulation (Panchal et al., MUQ-SAC extension).

Usage
-----
    python partial_sac_sampler.py

Inputs
------
- MB_R_ALL_ECM_2025.xml  : uncertainty data (temperatures, f(T) per reaction)
- MB_MB2D_LALIT_2024.yaml: mechanism file (nominal Arrhenius parameters)

Outputs
-------
- Plots/SAC_<rxn_tag>/<selection_label>/  : Arrhenius curve plots per case

Selection sets covered (7 total)
----------------------------------
  m=1  : {alpha}, {n}, {eps}            -> Class-A only
  m=2  : {alpha,n}, {alpha,eps}, {n,eps} -> Class-A, B (soft), C
  m=3  : {alpha,n,eps}                   -> Class-A, B, C (original MUQ-SAC)

Constraints
-----------
  - When n is active: |Delta_n| = |(L_r @ zeta_r)_n| < 2  (physical bound)
"""

# ─── standard library ────────────────────────────────────────────────────────
import os
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

# ─── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on any system)
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.linalg import cholesky

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#   CONSTANTS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

M_CONST   = 3.0 / np.log(10.0)   # IUPAC normalization factor
R_GAS     = 1.987                 # cal/(mol·K)  — Ea stored as cal/mol in YAML
MAX_DELTA_N = 2.0                 # |Δn| hard upper bound
N_CLASS_A   = 20                  # samples of each class to generate
N_CLASS_B   = 15
N_CLASS_C   = 15

PARAM_NAMES  = ["alpha", "n", "eps"]   # 0,1,2
PARAM_LABELS = [r"$\alpha$", r"$n$", r"$\varepsilon$"]

# ── pretty names for plot titles ──────────────────────────────────────────────
SELECTION_LABELS = {
    (0,):    r"$\{\alpha\}$",
    (1,):    r"$\{n\}$",
    (2,):    r"$\{\varepsilon\}$",
    (0, 1):  r"$\{\alpha,n\}$",
    (0, 2):  r"$\{\alpha,\varepsilon\}$",
    (1, 2):  r"$\{n,\varepsilon\}$",
    (0, 1, 2): r"$\{\alpha,n,\varepsilon\}$",
}
SELECTION_DIRS = {
    (0,):    "alpha",
    (1,):    "n",
    (2,):    "eps",
    (0, 1):  "alpha_n",
    (0, 2):  "alpha_eps",
    (1, 2):  "n_eps",
    (0, 1, 2): "alpha_n_eps",
}
ALL_SELECTIONS = [
    (0,), (1,), (2,),
    (0, 1), (0, 2), (1, 2),
    (0, 1, 2),
]


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 1 – XML / YAML PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_equation(s: str) -> str:
    """Normalize a reaction equation string for matching."""
    s = re.sub(r"\s+", " ", s.strip())
    s = s.replace("=>", "<=>").replace("= >", "<=>")
    s = s.replace("< =>", "<=>").replace("<= >", "<=>")
    return s


def parse_yaml_mechanism(yaml_path: str) -> dict:
    """Return {normalized_equation: rate_constant_dict} from a Cantera YAML."""
    with open(yaml_path) as fh:
        mech = yaml.safe_load(fh)
    result = {}
    for rxn in mech.get("reactions", []):
        eq  = normalize_equation(rxn.get("equation", ""))
        rc  = rxn.get("rate-constant")
        if rc is None:
            # pressure-dependent reactions may have high-P / low-P limits
            rc = rxn.get("high-P-rate-constant") or rxn.get("low-P-rate-constant")
        if rc is not None:
            result[eq] = rc
    return result


def parse_xml_uncertainty(xml_path: str) -> dict:
    """
    Parse the uncertainty XML file.

    Returns
    -------
    dict  keyed by reaction nametag, value is a sub-dict with:
        temperatures  : np.ndarray
        uncertainties : np.ndarray  (f(T), in ln-scale already – the IUPAC value)
        rxn_equation  : str         (raw equation from XML)
        pressure_limit: str | None
        rIndex        : str
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    reactions = {}

    for child in root:
        tag = child.tag
        if tag not in ("reaction", "PLOG"):
            continue

        rxn_eq    = child.attrib.get("rxn", "")
        r_index   = child.attrib.get("no", "")
        pres_lim  = None
        data_type = "constant;end_points"
        temps_raw = None
        unsrt_raw = None
        perturb   = "all"

        for item in child:
            if item.tag == "perturbation_type":
                perturb = (item.text or "").strip()
            elif item.tag == "data_type":
                data_type = (item.text or "").strip()
            elif item.tag == "temp":
                temps_raw = (item.text or "").strip()
            elif item.tag == "unsrt":
                unsrt_raw = (item.text or "").strip()
            elif item.tag == "sub_type":
                for sub in item:
                    if sub.tag == "pressure_limit":
                        pres_lim = (sub.text or "").strip()

        if temps_raw is None or unsrt_raw is None:
            continue

        t_vals = [float(v) for v in temps_raw.split(",")]
        u_vals = [float(v) for v in unsrt_raw.split(",")]

        fmt = data_type.split(";")
        interp = fmt[1] if len(fmt) > 1 else "array"

        if interp == "end_points":
            T_arr = np.linspace(t_vals[0], t_vals[-1], 200)
            u_arr = np.linspace(u_vals[0], u_vals[-1], 200)
        else:
            T_arr = np.array(t_vals)
            u_arr = np.array(u_vals)

        nametag = rxn_eq if pres_lim is None else f"{rxn_eq}:{pres_lim}"

        reactions[nametag] = {
            "temperatures":  T_arr,
            "uncertainties": u_arr,
            "rxn_equation":  rxn_eq,
            "pressure_limit": pres_lim,
            "rIndex":        r_index,
            "perturbation":  perturb,
        }

    return reactions


def get_nominal_params(rxn_eq: str, yaml_rate_db: dict) -> np.ndarray | None:
    """
    Look up Arrhenius parameters [alpha=ln(A), n, eps=Ea/R] from the YAML DB.
    Ea is in cal/mol in the Cantera YAML; eps = Ea/R_gas.
    Returns None if the reaction is not found.
    """
    norm = normalize_equation(rxn_eq)
    rc = yaml_rate_db.get(norm)
    if rc is None:
        return None
    A  = rc.get("A",  1.0)
    n  = rc.get("b",  0.0)
    Ea = rc.get("Ea", 0.0)
    # guard against A <= 0
    alpha = np.log(max(A, 1e-300))
    eps   = Ea / R_GAS
    return np.array([alpha, n, eps], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 2 – MUQ: COMPUTE FULL 3x3 CHOLESKY MATRIX L
# ═══════════════════════════════════════════════════════════════════════════════

def theta_full(T: np.ndarray) -> np.ndarray:
    """θ(T) = [1, ln T, -1/T], shape (3, len(T))."""
    return np.array([np.ones_like(T), np.log(T), -1.0 / T])


def f_prior_from_L(L: np.ndarray, T: np.ndarray) -> np.ndarray:
    """f_prior(T) = ||L^T θ(T)||_2 for each temperature."""
    Theta = theta_full(T)
    return np.array([np.linalg.norm(L.T @ th) for th in Theta.T])


def muq_objective(params, T: np.ndarray, uncertainties: np.ndarray) -> float:
    """
    MUQ objective: minimise Σ_i [(f(T_i)/M - ||L^T θ(T_i)||_2) / (f(T_i)/M)]^2
    params = [L11, L21, L22, L31, L32, L33]
    """
    L = np.array([
        [params[0],  0.0,       0.0],
        [params[1],  params[2], 0.0],
        [params[3],  params[4], params[5]],
    ])
    f_model = f_prior_from_L(L, T)
    f_target = uncertainties / M_CONST
    diff = (f_target - f_model) / (f_target + 1e-30)
    return float(np.dot(diff, diff))


def compute_full_L(temperatures: np.ndarray, uncertainties: np.ndarray) -> np.ndarray:
    """
    Solve the MUQ optimisation to find the 3x3 lower-triangular Cholesky matrix L
    such that f_prior(T) = ||L^T θ(T)||_2 ≈ f(T)/M.

    Returns L (3x3 lower triangular numpy array).
    """
    f_mean  = np.mean(uncertainties / M_CONST)
    x0 = np.array([f_mean, 0.0, f_mean * 0.1, f_mean * 100.0, 0.0, f_mean * 10.0])

    result = minimize(
        muq_objective, x0,
        args=(temperatures, uncertainties),
        method="SLSQP",
        options={"maxiter": 5000, "ftol": 1e-12},
    )

    lv = result.x
    L = np.array([
        [lv[0], 0.0,   0.0],
        [lv[1], lv[2], 0.0],
        [lv[3], lv[4], lv[5]],
    ])
    return L


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 3 – PRINCIPAL SUBMATRIX & REDUCED L_r
# ═══════════════════════════════════════════════════════════════════════════════

def get_reduced_L(L_full: np.ndarray, indices: tuple) -> tuple:
    """
    Given the full 3x3 Cholesky factor L (from MUQ), compute the reduced
    Cholesky factor L_r for the subset of parameters specified by indices.

    Steps
    -----
    1. Sigma = L @ L^T
    2. Sigma_r = Sigma[indices, :][:, indices]  (principal submatrix)
    3. L_r = cholesky(Sigma_r, lower=True)

    Returns (Sigma, Sigma_r, L_r).
    """
    Sigma   = L_full @ L_full.T
    idx     = list(indices)
    Sigma_r = Sigma[np.ix_(idx, idx)]

    # Ensure positive definiteness (add small regularisation if needed)
    try:
        L_r = cholesky(Sigma_r, lower=True)
    except Exception:
        eps_reg = 1e-12 * np.trace(Sigma_r) / len(idx)
        L_r = cholesky(Sigma_r + eps_reg * np.eye(len(idx)), lower=True)

    return Sigma, Sigma_r, L_r


def theta_S(T: np.ndarray, indices: tuple) -> np.ndarray:
    """
    Reduced basis vector θ_S(T) of shape (m, len(T)).
    Row mapping: 0→1, 1→ln T, 2→-1/T
    """
    full_theta = theta_full(T)  # (3, N)
    return full_theta[list(indices), :]


def f_prior_S(T: np.ndarray, L_r: np.ndarray, indices: tuple) -> np.ndarray:
    """f_prior,S(T) = ||L_r^T θ_S(T)||_2 for each T."""
    thS = theta_S(T, indices)   # (m, N)
    return np.array([np.linalg.norm(L_r.T @ col) for col in thS.T])


def delta_kappa(T: np.ndarray, L_r: np.ndarray, zeta_r: np.ndarray,
                indices: tuple) -> np.ndarray:
    """∆κ_S(T) = θ_S(T)^T L_r ζ_r"""
    thS = theta_S(T, indices)            # (m, N)
    Lz  = L_r @ zeta_r                   # (m,)
    return thS.T @ Lz                    # (N,)


def n_perturbation(L_r: np.ndarray, zeta_r: np.ndarray, indices: tuple) -> float:
    """
    Return the actual perturbation applied to the n parameter (index=1).
    Returns 0.0 if n is not in the selection.
    """
    if 1 not in indices:
        return 0.0
    pos_n = list(indices).index(1)
    Lz = L_r @ zeta_r
    return float(Lz[pos_n])


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 4 – CLASS-A SAMPLER  (all selections)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_class_A(T: np.ndarray, L_r: np.ndarray, indices: tuple,
                   n_samples: int = N_CLASS_A) -> list:
    """
    Class-A samples: ∆κ_S(T) does NOT cross κ₀.

    For each draw alpha_s ∈ U(0,1):
        min_{ζ_r} Σ_i ( alpha_s · f_prior,S(T_i) - θ_S(T_i)^T L_r ζ_r )²

    The analytical solution for an unconstrained least-squares problem in ζ_r:
        A_mat = θ_S^T L_r  ∈ R^{N x m}
        b_vec = alpha_s · f_prior,S(T)  ∈ R^N
        ζ_r = pinv(A_mat) @ b_vec

    Additionally, δn constraint is enforced by capping ζ_r after solving.
    A sign-flip is randomly applied so curves appear on both sides of κ₀.
    """
    fp = f_prior_S(T, L_r, indices)       # (N,)
    thS = theta_S(T, indices)             # (m, N)
    A_mat = (thS.T @ L_r)                 # (N, m)
    A_pinv = np.linalg.pinv(A_mat)        # (m, N)

    zeta_list = []
    rng = np.random.default_rng()

    for _ in range(n_samples):
        alpha_s = rng.uniform(0.05, 1.0)
        sign    = rng.choice([-1.0, 1.0])
        b_vec   = sign * alpha_s * fp
        zeta_r  = A_pinv @ b_vec

        # Enforce |Δn| < MAX_DELTA_N
        zeta_r = _enforce_dn_constraint(zeta_r, L_r, indices)
        zeta_list.append(zeta_r)

    return zeta_list


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 5 – CLASS-B SAMPLER  (m ≥ 2 only)
# ═══════════════════════════════════════════════════════════════════════════════

def _fp_derivative(T: float, L_r: np.ndarray, indices: tuple) -> float:
    """Numerical derivative of f_prior,S(T) w.r.t. T."""
    h = T * 1e-4
    T_arr = np.array([T - h, T + h])
    fp = f_prior_S(T_arr, L_r, indices)
    return (fp[1] - fp[0]) / (2 * h)


def _dk_derivative(T: float, L_r: np.ndarray, zeta_r: np.ndarray,
                   indices: tuple) -> float:
    """Numerical derivative of ∆κ_S(T) w.r.t. T."""
    h = T * 1e-4
    T_arr = np.array([T - h, T + h])
    dk = delta_kappa(T_arr, L_r, zeta_r, indices)
    return (dk[1] - dk[0]) / (2 * h)


# ─── helpers for class-B (analytical approach) ───────────────────────────────

def _dtheta_S_dT(T_val: float, indices: tuple) -> np.ndarray:
    """Analytical dθ_S/dT: d/dT[1]=0, d/dT[lnT]=1/T, d/dT[-1/T]=1/T²."""
    full = np.array([0.0, 1.0 / T_val, 1.0 / T_val ** 2])
    return full[list(indices)]


def _fp_S_deriv(T_val: float, L_r: np.ndarray, indices: tuple) -> float:
    """Analytical df_prior,S/dT = (L_r^T θ_S)·(L_r^T dθ_S/dT) / f_prior,S."""
    th  = theta_S(np.array([T_val]), indices)[:, 0]
    dth = _dtheta_S_dT(T_val, indices)
    LTth  = L_r.T @ th
    LTdth = L_r.T @ dth
    fp = np.linalg.norm(LTth)
    return float(LTth @ LTdth) / fp if fp > 1e-30 else 0.0


def _solve_2x2(T1: float, T2: float, rhs1: float, rhs2: float,
               L_r: np.ndarray, indices: tuple) -> np.ndarray | None:
    """
    Solve the 2×2 linear system:
        row1 @ ζ_r = rhs1,   row1 = (L_r^T θ_S(T1))^T
        row2 @ ζ_r = rhs2,   row2 = (L_r^T θ_S(T2))^T
    Returns ζ_r or None if singular.
    """
    r1 = L_r.T @ theta_S(np.array([T1]), indices)[:, 0]
    r2 = L_r.T @ theta_S(np.array([T2]), indices)[:, 0]
    A  = np.vstack([r1, r2])
    if abs(np.linalg.det(A)) < 1e-14:
        return None
    try:
        return np.linalg.solve(A, np.array([rhs1, rhs2]))
    except np.linalg.LinAlgError:
        return None


def _class_B_m2(T: np.ndarray, L_r: np.ndarray, indices: tuple,
                fp_grid: np.ndarray, T_min: float, T_max: float,
                n_samples: int, rng) -> list:
    """
    Fast analytical class-B for m=2.

    Primary approach (analytical bisection):
      1. Draw r1 ∈ [-1,1], r3 = -sign(r1).
      2. Scan T_u on a grid; at each T_u solve 2×2 system (C1+C3) → ζ_r.
      3. Evaluate C4 residual g(T_u) = dΔκ/dT - r3·f'(T_u).
      4. Bisect on sign-change interval → T_u* → ζ_r*.

    Fallback (SLSQP, used when C4 has no sign change, e.g. flat f_prior):
      Enforce C1 + C3 as strict equalities, minimise C4² + ||Δκ-f_prior||².
      Require the resulting curve to cross κ₀.
    """
    fp_Tmin   = float(f_prior_S(np.array([T_min]), L_r, indices)[0])
    Tu_grid   = np.linspace(T_min * 1.02, T_max * 0.98, 120)
    zeta_list = []

    def _try_analytical() -> list:
        out = []
        for _ in range(n_samples * 40):
            if len(out) >= n_samples:
                break
            r1   = rng.uniform(-0.95, 0.95)
            r3   = -np.sign(r1) if abs(r1) > 1e-6 else 1.0
            rhs1 = r1 * fp_Tmin

            g = np.full(len(Tu_grid), np.nan)
            valid = np.zeros(len(Tu_grid), dtype=bool)
            for k, Tu in enumerate(Tu_grid):
                fp_Tu = float(f_prior_S(np.array([Tu]), L_r, indices)[0])
                zr    = _solve_2x2(T_min, Tu, rhs1, r3 * fp_Tu, L_r, indices)
                if zr is None:
                    continue
                dth   = _dtheta_S_dT(Tu, indices)
                gk    = float((L_r @ zr) @ dth) - r3 * _fp_S_deriv(Tu, L_r, indices)
                g[k]  = gk; valid[k] = True

            ok = np.where(valid)[0]
            if len(ok) < 2:
                continue
            g_ok = g[ok]
            sc   = np.where(np.diff(np.sign(g_ok)) != 0)[0]
            if len(sc) == 0:
                continue
            pick    = rng.choice(sc)
            ia, ib  = ok[pick], ok[pick + 1]
            Ta, Tb  = Tu_grid[ia], Tu_grid[ib]
            ga, gb  = g[ia], g[ib]

            zr_best = None
            for _ in range(50):
                Tm   = 0.5 * (Ta + Tb)
                fp_m = float(f_prior_S(np.array([Tm]), L_r, indices)[0])
                zr_m = _solve_2x2(T_min, Tm, rhs1, r3 * fp_m, L_r, indices)
                if zr_m is None:
                    break
                dth  = _dtheta_S_dT(Tm, indices)
                gm   = float((L_r @ zr_m) @ dth) - r3 * _fp_S_deriv(Tm, L_r, indices)
                if abs(gm) < 1e-10:
                    zr_best = zr_m; break
                if np.sign(gm) == np.sign(ga):
                    Ta, ga = Tm, gm
                else:
                    Tb, gb = Tm, gm
            if zr_best is None:
                Tm_f = 0.5 * (Ta + Tb)
                fp_f = float(f_prior_S(np.array([Tm_f]), L_r, indices)[0])
                zr_best = _solve_2x2(T_min, Tm_f, rhs1, r3 * fp_f, L_r, indices)
            if zr_best is None:
                continue
            dk = delta_kappa(T, L_r, zr_best, indices)
            if not _has_sign_change(dk):
                continue
            out.append(_enforce_dn_constraint(zr_best, L_r, indices))
        return out

    def _try_slsqp() -> list:
        """SLSQP fallback: C1 + C3 strict, minimise C4² + fit objective.
        Useful when df_prior,S/dT ≈ 0 (flat uncertainty profile).
        """
        out = []
        fp_Tmax = float(f_prior_S(np.array([T_max]), L_r, indices)[0])

        def fp_at(t):
            return float(f_prior_S(np.array([t]), L_r, indices)[0])

        def dk_at(t, zr):
            return float(delta_kappa(np.array([t]), L_r, zr, indices)[0])

        attempt = 0
        while len(out) < n_samples and attempt < n_samples * 20:
            attempt += 1
            r1  = rng.uniform(-0.95, 0.95)
            r3  = -np.sign(r1) if abs(r1) > 1e-6 else 1.0
            r2  = rng.uniform(-0.95, 0.95)
            z0  = rng.uniform(-1.0, 1.0, size=len(indices))
            Tu0 = rng.uniform(T_min + 0.2*(T_max-T_min),
                               T_max - 0.2*(T_max-T_min))
            x0  = np.append(z0, Tu0)

            def obj(x):
                zr = x[:len(indices)]
                Tu = float(np.clip(x[-1], T_min, T_max))
                dth = _dtheta_S_dT(Tu, indices)
                Lz  = L_r @ zr
                c4r = float(dth @ Lz) - r3 * _fp_S_deriv(Tu, L_r, indices)
                fit = np.sum((fp_grid - delta_kappa(T, L_r, zr, indices))**2)
                return fit + 20.0 * c4r**2

            def c1(x): return dk_at(T_min, x[:len(indices)]) - r1 * fp_Tmin
            def c3(x):
                Tu = float(np.clip(x[-1], T_min, T_max))
                return dk_at(Tu, x[:len(indices)]) - r3 * fp_at(Tu)

            try:
                res = minimize(obj, x0, method="SLSQP",
                               bounds=[(None,None)]*len(indices) + [(T_min*1.005, T_max*0.995)],
                               constraints=[{"type":"eq","fun":c1},{"type":"eq","fun":c3}],
                               options={"maxiter":2000,"ftol":1e-9})
                zr = res.x[:len(indices)]; Tu = float(res.x[-1])
                dk = delta_kappa(T, L_r, zr, indices)
                c1r = abs(c1(res.x)); c3r = abs(c3(res.x))
                if _has_sign_change(dk) and c1r+c3r < 1.0 and T_min < Tu < T_max:
                    out.append(_enforce_dn_constraint(zr, L_r, indices))
            except Exception:
                pass
        return out

    # ── Run primary analytical approach ───────────────────────────────────
    results = _try_analytical()
    if len(results) >= n_samples:
        return results

    # ── Fallback to SLSQP if analytical couldn't fill quota ───────────────
    additional = _try_slsqp()
    results.extend(additional)
    return results[:n_samples]


def sample_class_B(T: np.ndarray, L_r: np.ndarray, indices: tuple,
                   n_samples: int = N_CLASS_B) -> list:
    """
    Class-B samples: crossover curves tangent to the uncertainty limit.

    m=1: infeasible.
    m=2: fast analytical approach (scan T_u grid + bisection on C4 residual).
    m=3: SLSQP with 4 strict equality constraints (exactly determined).
    """
    m = len(indices)
    if m == 1:
        return []

    T_min, T_max = float(T[0]), float(T[-1])
    fp_grid = f_prior_S(T, L_r, indices)
    rng = np.random.default_rng()

    if m == 2:
        return _class_B_m2(T, L_r, indices, fp_grid, T_min, T_max, n_samples, rng)

    # ── m=3: SLSQP, exactly determined ────────────────────────────────────
    zeta_list = []
    fp_Tmin   = float(f_prior_S(np.array([T_min]), L_r, indices)[0])
    fp_Tmax   = float(f_prior_S(np.array([T_max]), L_r, indices)[0])

    def fp_at(t):
        return float(f_prior_S(np.array([t]), L_r, indices)[0])

    def dk_at(t, zr):
        return float(delta_kappa(np.array([t]), L_r, zr, indices)[0])

    attempt = 0
    while len(zeta_list) < n_samples and attempt < n_samples * 15:
        attempt += 1
        r1 = rng.uniform(-0.95, 0.95)
        r2 = rng.uniform(-0.95, 0.95)
        r3 = -np.sign(r1) if abs(r1) > 1e-6 else 1.0

        z0  = rng.uniform(-1.0, 1.0, size=m)
        Tu0 = rng.uniform(T_min + 0.2 * (T_max - T_min),
                           T_max - 0.2 * (T_max - T_min))
        x0  = np.append(z0, Tu0)

        def c1(x): return dk_at(T_min, x[:m]) - r1 * fp_Tmin
        def c2(x): return dk_at(T_max, x[:m]) - r2 * fp_Tmax
        def c3(x):
            Tu = float(np.clip(x[m], T_min, T_max))
            return dk_at(Tu, x[:m]) - r3 * fp_at(Tu)
        def c4(x):
            Tu = float(np.clip(x[m], T_min, T_max))
            zr = x[:m]
            dth  = _dtheta_S_dT(Tu, indices)
            Lz   = L_r @ zr
            return float(dth @ Lz) - r3 * _fp_S_deriv(Tu, L_r, indices)
        def obj(x):
            return float(np.sum((fp_grid - delta_kappa(T, L_r, x[:m], indices))**2))

        try:
            res = minimize(obj, x0, method="SLSQP",
                           bounds=[(None,None)]*m + [(T_min*1.005, T_max*0.995)],
                           constraints=[{"type":"eq","fun":f} for f in [c1,c2,c3,c4]],
                           options={"maxiter": 1500, "ftol": 1e-9})
            zr = res.x[:m]; Tu = float(res.x[m])
            dk = delta_kappa(T, L_r, zr, indices)
            cres = abs(c1(res.x)) + abs(c2(res.x)) + abs(c3(res.x)) + abs(c4(res.x))
            if _has_sign_change(dk) and T_min < Tu < T_max and cres < 2.0:
                zeta_list.append(_enforce_dn_constraint(zr, L_r, indices))
        except Exception:
            pass

    return zeta_list


def _has_sign_change(arr: np.ndarray) -> bool:
    """Return True if arr changes sign at least once."""
    return bool(np.any(np.diff(np.sign(arr)) != 0))


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 6 – CLASS-C SAMPLER  (m ≥ 2 only)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_class_C(T: np.ndarray, L_r: np.ndarray, indices: tuple,
                   n_samples: int = N_CLASS_C) -> list:
    """
    Class-C samples: crossover via a linearised f_c(T).

    For each (r1, r2) pair:
        f_c(T) = r1·f_prior,S(T_min) + [r2·f_prior,S(T_max) - r1·f_prior,S(T_min)]
                 / (T_max - T_min) · (T - T_min)
    Solve:
        min_{ζ_r} Σ_i ( f_c(T_i) - θ_S(T_i)^T L_r ζ_r )²
    (unconstrained LS via pseudo-inverse)
    """
    m = len(indices)
    if m == 1:
        return []   # no crossover for single-parameter selections

    T_min, T_max = T[0], T[-1]
    fp_Tmin      = float(f_prior_S(np.array([T_min]), L_r, indices)[0])
    fp_Tmax      = float(f_prior_S(np.array([T_max]), L_r, indices)[0])

    thS   = theta_S(T, indices)           # (m, N)
    A_mat = thS.T @ L_r                   # (N, m)
    A_pinv = np.linalg.pinv(A_mat)        # (m, N)

    zeta_list = []
    rng        = np.random.default_rng()

    for _ in range(n_samples * 3):          # allow extra attempts
        r1 = rng.uniform(-1.0, 1.0)
        r2 = rng.uniform(-1.0, 1.0)

        # r1 and r2 should have opposite signs for a crossover f_c
        if np.sign(r1) == np.sign(r2):
            r2 = -r2

        fc = (r1 * fp_Tmin
              + (r2 * fp_Tmax - r1 * fp_Tmin) / (T_max - T_min) * (T - T_min))

        zeta_r = A_pinv @ fc

        dk = delta_kappa(T, L_r, zeta_r, indices)
        if not _has_sign_change(dk):
            continue

        zeta_r = _enforce_dn_constraint(zeta_r, L_r, indices)
        zeta_list.append(zeta_r)

        if len(zeta_list) >= n_samples:
            break

    return zeta_list


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 7 – Δn CONSTRAINT ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _enforce_dn_constraint(zeta_r: np.ndarray, L_r: np.ndarray,
                            indices: tuple) -> np.ndarray:
    """
    If n (index=1) is in the active set, ensure |Δn| = |(L_r @ ζ_r)_{n}| < MAX_DELTA_N.
    Scale ζ_r uniformly if the constraint is violated.
    """
    if 1 not in indices:
        return zeta_r
    pos_n  = list(indices).index(1)
    Lz     = L_r @ zeta_r
    delta_n = abs(Lz[pos_n])
    if delta_n > MAX_DELTA_N:
        scale  = MAX_DELTA_N / (delta_n + 1e-30)
        zeta_r = zeta_r * scale * 0.95   # slight margin
    return zeta_r


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 8 – RECONSTRUCT FULL κ(T) CURVE FROM ζ_r
# ═══════════════════════════════════════════════════════════════════════════════

def kappa_curve(T: np.ndarray, nominal: np.ndarray,
                L_r: np.ndarray, zeta_r: np.ndarray,
                indices: tuple) -> np.ndarray:
    """
    Return the perturbed log-rate-constant curve:
        κ(T) = κ₀(T) + ∆κ_S(T)

    κ₀(T) = θ(T)^T p₀  (computed from full nominal [α, n, ε])
    ∆κ_S  = θ_S(T)^T L_r ζ_r
    """
    kappa_0 = theta_full(T).T @ nominal                    # (N,)
    dk_S    = delta_kappa(T, L_r, zeta_r, indices)         # (N,)
    return kappa_0 + dk_S


def kappa_nominal(T: np.ndarray, nominal: np.ndarray) -> np.ndarray:
    return theta_full(T).T @ nominal


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 9 – PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "nominal":     "#1D9E75",
    "limit":       "#444441",
    "classA":      "#378ADD",
    "classB":      "#D85A30",
    "classC":      "#7F77DD",
    "background":  "#F1EFE8",
}


def plot_sac_curves(
    T: np.ndarray,
    nominal: np.ndarray,
    L_full: np.ndarray,
    L_r: np.ndarray,
    indices: tuple,
    zeta_A: list,
    zeta_B: list,
    zeta_C: list,
    rxn_label: str,
    save_dir: Path,
) -> None:
    """Plot Arrhenius curves for one selection set and save to save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)

    inv_T = 1000.0 / T           # x-axis: 1000/T [K⁻¹]
    kap0  = kappa_nominal(T, nominal)

    # Uncertainty limits (from the FULL L, not reduced L_r)
    fp_full = f_prior_from_L(L_full, T) * M_CONST
    kap_up  = kap0 + fp_full
    kap_lo  = kap0 - fp_full

    sel_label = SELECTION_LABELS.get(indices, str(indices))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(COLORS["background"])

    # ── limits ────────────────────────────────────────────────────────────────
    ax.plot(inv_T, kap_up, color=COLORS["limit"],
            ls="--", lw=1.2, label=r"Uncertainty limits ($\pm f_{\rm prior}$)")
    ax.plot(inv_T, kap_lo, color=COLORS["limit"], ls="--", lw=1.2)

    # ── class-A ───────────────────────────────────────────────────────────────
    first_A = True
    for zr in zeta_A:
        kap = kappa_curve(T, nominal, L_r, zr, indices)
        lbl = "Class-A" if first_A else "_nolegend_"
        ax.plot(inv_T, kap, color=COLORS["classA"],
                lw=0.8, alpha=0.65, label=lbl)
        first_A = False

    # ── class-B ───────────────────────────────────────────────────────────────
    first_B = True
    for zr in zeta_B:
        kap = kappa_curve(T, nominal, L_r, zr, indices)
        lbl = "Class-B" if first_B else "_nolegend_"
        ax.plot(inv_T, kap, color=COLORS["classB"],
                lw=0.9, alpha=0.75, label=lbl)
        first_B = False

    # ── class-C ───────────────────────────────────────────────────────────────
    first_C = True
    for zr in zeta_C:
        kap = kappa_curve(T, nominal, L_r, zr, indices)
        lbl = "Class-C" if first_C else "_nolegend_"
        ax.plot(inv_T, kap, color=COLORS["classC"],
                lw=0.9, alpha=0.75, label=lbl)
        first_C = False

    # ── nominal ───────────────────────────────────────────────────────────────
    ax.plot(inv_T, kap0, color=COLORS["nominal"],
            lw=2.0, label=r"Nominal $\kappa_0$", zorder=10)

    # Annotation for n constraint
    n_note = ""
    if 1 in indices:
        n_note = f"  [|Δn| < {MAX_DELTA_N}]"

    ax.set_xlabel(r"$1000\,/\,T\;\;[\mathrm{K}^{-1}]$", fontsize=11)
    ax.set_ylabel(r"$\kappa = \ln k$", fontsize=11)
    ax.set_title(
        f"{rxn_label}\nActive parameters: {sel_label}{n_note}",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    out_file = save_dir / "arrhenius_curves.pdf"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved → {out_file}")


def plot_summary_panel(
    T: np.ndarray,
    nominal: np.ndarray,
    L_full: np.ndarray,
    L_r_dict: dict,
    zeta_A_dict: dict,
    zeta_B_dict: dict,
    zeta_C_dict: dict,
    rxn_label: str,
    save_dir: Path,
) -> None:
    """
    7-panel summary figure showing all selection sets in one figure.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    inv_T  = 1000.0 / T
    kap0   = kappa_nominal(T, nominal)
    fp_full = f_prior_from_L(L_full, T) * M_CONST
    kap_up  = kap0 + fp_full
    kap_lo  = kap0 - fp_full

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=False, sharey=False)
    axes = axes.flatten()

    for panel_idx, sel in enumerate(ALL_SELECTIONS):
        ax  = axes[panel_idx]
        L_r = L_r_dict[sel]
        ax.set_facecolor(COLORS["background"])
        ax.plot(inv_T, kap_up, color=COLORS["limit"], ls="--", lw=1.0)
        ax.plot(inv_T, kap_lo, color=COLORS["limit"], ls="--", lw=1.0)

        for zr in zeta_A_dict.get(sel, []):
            ax.plot(inv_T, kappa_curve(T, nominal, L_r, zr, sel),
                    color=COLORS["classA"], lw=0.7, alpha=0.55)
        for zr in zeta_B_dict.get(sel, []):
            ax.plot(inv_T, kappa_curve(T, nominal, L_r, zr, sel),
                    color=COLORS["classB"], lw=0.8, alpha=0.65)
        for zr in zeta_C_dict.get(sel, []):
            ax.plot(inv_T, kappa_curve(T, nominal, L_r, zr, sel),
                    color=COLORS["classC"], lw=0.8, alpha=0.65)

        ax.plot(inv_T, kap0, color=COLORS["nominal"], lw=1.8, zorder=10)
        ax.set_title(SELECTION_LABELS.get(sel, str(sel)), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.4)

    # last panel: legend
    ax_leg = axes[-1]
    ax_leg.axis("off")
    handles = [
        plt.Line2D([0], [0], color=COLORS["nominal"],  lw=2.0, label=r"Nominal $\kappa_0$"),
        plt.Line2D([0], [0], color=COLORS["limit"],    lw=1.2, ls="--", label="Uncertainty limits"),
        plt.Line2D([0], [0], color=COLORS["classA"],   lw=1.2, alpha=0.7, label="Class-A"),
        plt.Line2D([0], [0], color=COLORS["classB"],   lw=1.2, alpha=0.8, label="Class-B"),
        plt.Line2D([0], [0], color=COLORS["classC"],   lw=1.2, alpha=0.8, label="Class-C"),
    ]
    ax_leg.legend(handles=handles, loc="center", fontsize=9, frameon=False)
    ax_leg.set_title("Legend", fontsize=9)

    fig.suptitle(f"All selection sets — {rxn_label}", fontsize=11, y=1.01)
    fig.tight_layout()

    out_file = save_dir / "summary_all_selections.pdf"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Summary panel → {out_file}")


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 10 – PER-REACTION DRIVER
# ═══════════════════════════════════════════════════════════════════════════════

def process_reaction(rxn_tag: str, rxn_data: dict,
                     yaml_rate_db: dict, base_plot_dir: Path) -> None:
    """
    Full pipeline for one reaction:
        1. Fetch nominal params
        2. Compute full L via MUQ
        3. For each of the 7 selection subsets:
             - reduce L → L_r
             - sample class-A, B, C
             - plot
        4. Save summary panel
    """
    # ── Nominal parameters ────────────────────────────────────────────────────
    rxn_eq  = rxn_data["rxn_equation"]
    nominal = get_nominal_params(rxn_eq, yaml_rate_db)
    if nominal is None:
        print(f"    [skip] {rxn_tag!r}: reaction not found in YAML mechanism")
        return
    print(f"    Nominal [α, n, ε]: {nominal}")

    T          = rxn_data["temperatures"]
    unsrt_vals = rxn_data["uncertainties"]
    rIndex     = rxn_data["rIndex"]

    # ── Full Cholesky L ───────────────────────────────────────────────────────
    print("    Computing full L (MUQ)…")
    L_full = compute_full_L(T, unsrt_vals)
    Sigma  = L_full @ L_full.T
    print(f"    L_full diagonal: {np.diag(L_full).round(4)}")
    print(f"    Sigma diag:      {np.diag(Sigma).round(4)}")

    # ── Safe reaction tag for directory names ─────────────────────────────────
    safe_tag = re.sub(r"[^\w\-]", "_", rxn_tag)[:50]
    rxn_dir  = base_plot_dir / safe_tag
    rxn_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-selection processing ──────────────────────────────────────────────
    L_r_dict  = {}
    zA_dict   = {}
    zB_dict   = {}
    zC_dict   = {}

    for sel in ALL_SELECTIONS:
        sel_name  = SELECTION_DIRS[sel]
        sel_label = SELECTION_LABELS[sel]
        m         = len(sel)

        _, Sigma_r, L_r = get_reduced_L(L_full, sel)
        L_r_dict[sel] = L_r

        print(f"    [{sel_name}] m={m}  L_r diag: {np.diag(L_r).round(4)}")

        # Class-A (always)
        zA = sample_class_A(T, L_r, sel, n_samples=N_CLASS_A)

        # Class-B (m ≥ 2 only)
        zB = []
        if m >= 2:
            print(f"      Sampling Class-B ({sel_name})…")
            zB = sample_class_B(T, L_r, sel, n_samples=N_CLASS_B)
            print(f"      Got {len(zB)}/{N_CLASS_B} class-B samples")

        # Class-C (m ≥ 2 only)
        zC = []
        if m >= 2:
            zC = sample_class_C(T, L_r, sel, n_samples=N_CLASS_C)
            print(f"      Got {len(zC)}/{N_CLASS_C} class-C samples")

        zA_dict[sel] = zA
        zB_dict[sel] = zB
        zC_dict[sel] = zC

        # ── per-selection plot ─────────────────────────────────────────────────
        plot_sac_curves(
            T         = T,
            nominal   = nominal,
            L_full    = L_full,
            L_r       = L_r,
            indices   = sel,
            zeta_A    = zA,
            zeta_B    = zB,
            zeta_C    = zC,
            rxn_label = f"{rxn_tag}  ({rIndex})",
            save_dir  = rxn_dir / sel_name,
        )

    # ── Summary panel ─────────────────────────────────────────────────────────
    plot_summary_panel(
        T          = T,
        nominal    = nominal,
        L_full     = L_full,
        L_r_dict   = L_r_dict,
        zeta_A_dict = zA_dict,
        zeta_B_dict = zB_dict,
        zeta_C_dict = zC_dict,
        rxn_label  = f"{rxn_tag}  ({rIndex})",
        save_dir   = rxn_dir,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#   SECTION 11 – MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    here = Path(__file__).parent

    xml_path  = here / "MB_R_ALL_ECM_2025.xml"
    yaml_path = here / "MB_MB2D_LALIT_2024.yaml"

    if not xml_path.exists():
        sys.exit(f"ERROR: {xml_path} not found")
    if not yaml_path.exists():
        sys.exit(f"ERROR: {yaml_path} not found")

    print("=" * 65)
    print("  Partial-Parameter SAC — All Arrhenius Subsets")
    print("=" * 65)

    # ── Parse inputs ──────────────────────────────────────────────────────────
    print("\n[1] Parsing YAML mechanism…")
    yaml_rate_db = parse_yaml_mechanism(str(yaml_path))
    print(f"    {len(yaml_rate_db)} reactions found in YAML")

    print("\n[2] Parsing XML uncertainty data…")
    rxn_dict = parse_xml_uncertainty(str(xml_path))
    print(f"    {len(rxn_dict)} reactions found in XML")

    # ── Output directory ──────────────────────────────────────────────────────
    plot_root = here / "Plots" / "SAC_Curves"
    plot_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[3] Output directory: {plot_root}")

    # ── Process each reaction ─────────────────────────────────────────────────
    print("\n[4] Generating samples for each reaction…\n")

    total = len(rxn_dict)
    for idx, (rxn_tag, rxn_data) in enumerate(rxn_dict.items()):
        print(f"\n  [{idx + 1}/{total}]  {rxn_tag}")
        try:
            process_reaction(rxn_tag, rxn_data, yaml_rate_db, plot_root)
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 65)
    print(f"  Done.  Plots written to: {plot_root}")
    print("=" * 65)


if __name__ == "__main__":
    main()
