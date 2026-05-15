#!/usr/bin/env python3
"""
test_benchmark.py
=================
Self-contained benchmark comparing Arrhenius curve sampling methods across
Class-A, -B, -C for m=2 and m=3 active-parameter selections.

No imports from Uncertainty.py or any custom project module.
Allowed imports: numpy, scipy, yaml, xml.etree.ElementTree, matplotlib,
                 time, pathlib, re, warnings, sys.
"""

import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.linalg import cholesky
from scipy.optimize import minimize, shgo

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

M_CONST     = 3.0 / np.log(10.0)   # IUPAC normalisation factor
R_GAS       = 1.987                  # cal/(mol·K)
MAX_DELTA_N = 2.0                    # |Δn| hard upper bound

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MATH HELPERS  (from test_MUQ_SAC.py, verbatim / adapted sig)
# ════════════════════════════════════════════════════════════════════════════

def theta_full(T):
    """theta(T) = [1, ln T, -1/T], shape (3, N)."""
    return np.array([np.ones_like(T), np.log(T), -1.0 / T])


def f_prior_from_L(L, T):
    """f_prior(T) = ||L^T theta(T)||_2 for each temperature."""
    Theta = theta_full(T)
    return np.array([np.linalg.norm(L.T @ th) for th in Theta.T])


def _muq_objective(params, T, uncertainties):
    """MUQ residual for Cholesky fitting."""
    L = np.array([[params[0], 0.0, 0.0],
                  [params[1], params[2], 0.0],
                  [params[3], params[4], params[5]]])
    f_model  = f_prior_from_L(L, T)
    f_target = uncertainties / M_CONST
    diff = (f_target - f_model) / (f_target + 1e-30)
    return float(np.dot(diff, diff))


def compute_full_L(temperatures, uncertainties):
    """Solve MUQ optimisation to find 3x3 lower-triangular Cholesky L."""
    f_mean = np.mean(uncertainties / M_CONST)
    x0 = np.array([f_mean, 0.0, f_mean * 0.1,
                   f_mean * 100.0, 0.0, f_mean * 10.0])
    result = minimize(_muq_objective, x0,
                      args=(temperatures, uncertainties),
                      method="SLSQP",
                      options={"maxiter": 5000, "ftol": 1e-12})
    lv = result.x
    return np.array([[lv[0], 0.0,   0.0],
                     [lv[1], lv[2], 0.0],
                     [lv[3], lv[4], lv[5]]])


def get_reduced_L(L_full, indices):
    """Return (Sigma, Sigma_r, L_r) for the parameter subset."""
    Sigma   = L_full @ L_full.T
    idx     = list(indices)
    Sigma_r = Sigma[np.ix_(idx, idx)]
    try:
        L_r = cholesky(Sigma_r, lower=True)
    except Exception:
        eps_reg = 1e-12 * np.trace(Sigma_r) / len(idx)
        L_r = cholesky(Sigma_r + eps_reg * np.eye(len(idx)), lower=True)
    return Sigma, Sigma_r, L_r


def theta_S(T, indices):
    """Reduced basis theta_S(T) of shape (m, N). Row map: 0->1, 1->lnT, 2->-1/T."""
    return theta_full(T)[list(indices), :]


def f_prior_S(T, L_r, indices):
    """f_prior_S(T) = ||L_r^T theta_S(T)||_2 for each T."""
    thS = theta_S(T, indices)
    return np.array([np.linalg.norm(L_r.T @ col) for col in thS.T])


def delta_kappa(T, L_r, zeta_r, indices):
    """Delta_kappa_S(T) = theta_S(T)^T L_r zeta_r."""
    return theta_S(T, indices).T @ (L_r @ zeta_r)


def kappa_nominal(T, nominal):
    """kappa_0(T) = theta(T)^T p_0."""
    return theta_full(T).T @ nominal


def kappa_curve(T, nominal, L_r, zeta_r, indices):
    """kappa(T) = kappa_0(T) + Delta_kappa_S(T)."""
    return kappa_nominal(T, nominal) + delta_kappa(T, L_r, zeta_r, indices)


def _dtheta_S_dT(T_val, indices):
    """Analytical d(theta_S)/dT: d[1]=0, d[lnT]=1/T, d[-1/T]=1/T^2."""
    full = np.array([0.0, 1.0 / T_val, 1.0 / T_val ** 2])
    return full[list(indices)]


def _fp_full_deriv(T_val, L_full):
    """Analytical d(f_prior_full)/dT = (L^T theta).(L^T d_theta/dT) / ||...||."""
    th    = theta_full(np.array([T_val]))[:, 0]
    dth   = np.array([0.0, 1.0 / T_val, 1.0 / T_val ** 2])
    LTth  = L_full.T @ th
    LTdth = L_full.T @ dth
    fp    = np.linalg.norm(LTth)
    return float(np.dot(LTth, LTdth)) / fp if fp > 1e-30 else 0.0


def _fp_S_deriv(T_val, L_r, indices):
    """Analytical d(f_prior_S)/dT using reduced L_r."""
    th    = theta_S(np.array([T_val]), indices)[:, 0]
    dth   = _dtheta_S_dT(T_val, indices)
    LTth  = L_r.T @ th
    LTdth = L_r.T @ dth
    fp    = np.linalg.norm(LTth)
    return float(np.dot(LTth, LTdth)) / fp if fp > 1e-30 else 0.0


def _has_sign_change(arr):
    """Return True if arr changes sign at least once."""
    return bool(np.any(np.diff(np.sign(arr)) != 0))


def _enforce_dn_constraint(zeta_r, L_r, indices, A_system=None, b_system=None):
    """Enforce |Delta_n| < MAX_DELTA_N; QR-augmented solve when system provided."""
    if 1 not in indices:
        return zeta_r
    pos_n   = list(indices).index(1)
    Lz      = L_r @ zeta_r
    delta_n = Lz[pos_n]
    if abs(delta_n) <= MAX_DELTA_N:
        return zeta_r
    if A_system is None or b_system is None:
        scale = (0.95 * MAX_DELTA_N) / (abs(delta_n) + 1e-30)
        return zeta_r * scale
    else:
        e_n    = np.zeros(len(indices)); e_n[pos_n] = 1.0
        row_dn = e_n @ L_r
        A_aug  = np.vstack([A_system, row_dn])
        b_aug  = np.append(b_system, np.sign(delta_n) * 0.95 * MAX_DELTA_N)
        zeta_r, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
        return zeta_r


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PARSING HELPERS  (from test_MUQ_SAC.py verbatim)
# ════════════════════════════════════════════════════════════════════════════

def normalize_equation(s):
    """Normalize a reaction equation string for matching."""
    s = re.sub(r"\s+", " ", s.strip())
    s = s.replace("=>", "<=>").replace("= >", "<=>")
    s = s.replace("< =>", "<=>").replace("<= >", "<=>")
    return s


def parse_yaml_mechanism(yaml_path):
    """Return {normalized_equation: rate_constant_dict} from a Cantera YAML."""
    with open(yaml_path) as fh:
        mech = yaml.safe_load(fh)
    result = {}
    for rxn in mech.get("reactions", []):
        eq  = normalize_equation(rxn.get("equation", ""))
        rc  = rxn.get("rate-constant")
        if rc is None:
            rc = (rxn.get("high-P-rate-constant")
                  or rxn.get("low-P-rate-constant"))
        if rc is not None:
            result[eq] = rc
    return result


def parse_xml_uncertainty(xml_path):
    """Parse uncertainty XML; returns dict keyed by reaction nametag."""
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
        temps_raw = None; unsrt_raw = None
        for item in child:
            if item.tag == "temp":
                temps_raw = (item.text or "").strip()
            elif item.tag == "unsrt":
                unsrt_raw = (item.text or "").strip()
            elif item.tag == "data_type":
                data_type = (item.text or "").strip()
            elif item.tag == "sub_type":
                for sub in item:
                    if sub.tag == "pressure_limit":
                        pres_lim = (sub.text or "").strip()
        if temps_raw is None or unsrt_raw is None:
            continue
        t_vals = [float(v) for v in temps_raw.split(",")]
        u_vals = [float(v) for v in unsrt_raw.split(",")]
        fmt    = data_type.split(";")
        interp = fmt[1] if len(fmt) > 1 else "array"
        if interp == "end_points":
            T_arr = np.linspace(t_vals[0], t_vals[-1], 200)
            u_arr = np.linspace(u_vals[0], u_vals[-1], 200)
        else:
            T_arr = np.array(t_vals); u_arr = np.array(u_vals)
        nametag = rxn_eq if pres_lim is None else f"{rxn_eq}:{pres_lim}"
        reactions[nametag] = {
            "temperatures":   T_arr,
            "uncertainties":  u_arr,
            "rxn_equation":   rxn_eq,
            "pressure_limit": pres_lim,
            "rIndex":         r_index,
        }
    return reactions


def get_nominal_params(rxn_eq, yaml_rate_db):
    """Look up [alpha=ln(A), n, eps=Ea/R] from YAML DB; None if missing."""
    norm = normalize_equation(rxn_eq)
    rc   = yaml_rate_db.get(norm)
    if rc is None:
        return None
    A  = rc.get("A",  1.0); n  = rc.get("b",  0.0); Ea = rc.get("Ea", 0.0)
    return np.array([np.log(max(A, 1e-300)), n, Ea / R_GAS], dtype=float)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SOLVE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _solve_2x2(T1, T2, rhs1, rhs2, L_r, indices):
    """Solve 2x2 [row(T1); row(T2)] @ zeta_r = [rhs1; rhs2]."""
    r1 = L_r.T @ theta_S(np.array([T1]), indices)[:, 0]
    r2 = L_r.T @ theta_S(np.array([T2]), indices)[:, 0]
    A  = np.vstack([r1, r2])
    if abs(np.linalg.det(A)) < 1e-14:
        return None
    try:
        return np.linalg.solve(A, np.array([rhs1, rhs2]))
    except np.linalg.LinAlgError:
        return None


def _solve_3x3_B(Tu, r1, r3, T_min, L_r, L_full, indices):
    """Solve exact 3x3 [C1; C3; C4] system at fixed T_u for m=3."""
    row1 = L_r.T @ theta_S(np.array([T_min]), indices)[:, 0]
    row2 = L_r.T @ theta_S(np.array([Tu]),    indices)[:, 0]
    row3 = L_r.T @ _dtheta_S_dT(Tu, indices)
    A    = np.vstack([row1, row2, row3])
    fp_min  = float(f_prior_from_L(L_full, np.array([T_min]))[0])
    fp_Tu   = float(f_prior_from_L(L_full, np.array([Tu]))[0])
    fpd_Tu  = _fp_full_deriv(Tu, L_full)
    b = np.array([r1 * fp_min, r3 * fp_Tu, r3 * fpd_Tu])
    try:
        if abs(np.linalg.det(A)) < 1e-14:
            return None
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def _solve_3x3_C1C2C3(Tu, r1, r2, r3, T_min, T_max, L_r, L_full, indices):
    """Solve exact 3x3 [C1; C2; C3] system at fixed T_u for m=3."""
    row1 = L_r.T @ theta_S(np.array([T_min]), indices)[:, 0]
    row2 = L_r.T @ theta_S(np.array([T_max]), indices)[:, 0]
    row3 = L_r.T @ theta_S(np.array([Tu]),    indices)[:, 0]
    A    = np.vstack([row1, row2, row3])
    fp_min = float(f_prior_from_L(L_full, np.array([T_min]))[0])
    fp_max = float(f_prior_from_L(L_full, np.array([T_max]))[0])
    fp_Tu  = float(f_prior_from_L(L_full, np.array([Tu]))[0])
    b = np.array([r1 * fp_min, r2 * fp_max, r3 * fp_Tu])
    try:
        if abs(np.linalg.det(A)) < 1e-14:
            return None
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — QR PRE-COMPUTATION  (from class_B_m2_QR_C2.py verbatim)
# ════════════════════════════════════════════════════════════════════════════

def _precompute_fixed_block(row_min, row_max):
    """Pre-compute QR of fixed 2-row block [row_min; row_max]."""
    A_fixed = np.vstack([row_min, row_max])
    Q2, R2  = np.linalg.qr(A_fixed, mode='reduced')
    return {'Q2': Q2, 'R2': R2, 'row_min': row_min, 'row_max': row_max}


def _qr_solve_with_precomputed(pre, row_Tu, b):
    """Solve 3x2 LS system via pre-computed partial QR update; returns zeta_r or None."""
    A_full = np.vstack([pre['row_min'], pre['row_max'], row_Tu])
    Q3, R3 = np.linalg.qr(A_full, mode='reduced')
    if abs(R3[0, 0]) < 1e-14 or abs(R3[1, 1]) < 1e-14:
        return None
    rhs = Q3.T @ b
    return np.linalg.solve(R3, rhs)


def _g_C4(Tu, zr, r3, L_r, L_full, indices):
    """C4 residual: g = (L_r d_theta_S/dT)^T zeta_r - r3*fp'(T_u)."""
    dth = _dtheta_S_dT(Tu, indices)
    lhs = float((L_r @ zr) @ dth)
    rhs = r3 * _fp_full_deriv(Tu, L_full)
    return lhs - rhs


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — NEW SAMPLERS
# ════════════════════════════════════════════════════════════════════════════

def sample_class_A(T, L_r, L_full, indices, n_samples=1, rng=None):
    """Class-A samples: Delta_kappa proportional to f_prior_S, both signs."""
    if rng is None:
        rng = np.random.default_rng()
    fp    = f_prior_S(T, L_r, indices)
    thS   = theta_S(T, indices)
    A_mat  = thS.T @ L_r
    A_pinv = np.linalg.pinv(A_mat)
    zeta_list = []
    for _ in range(n_samples):
        alpha_s = rng.uniform(0.05, 1.0)
        sign    = rng.choice([-1.0, 1.0])
        b_vec   = sign * alpha_s * fp
        zeta_r  = A_pinv @ b_vec
        zeta_r  = _enforce_dn_constraint(zeta_r, L_r, indices)
        zeta_list.append(zeta_r)
    return zeta_list


def _class_B_m2(T, L_r, L_full, indices, T_min, T_max, n_samples, rng):
    """Fast analytical class-B for m=2 (C1+C3 exact solve + bisect C4; no C2)."""
    fp_Tmin   = float(f_prior_S(np.array([T_min]), L_r, indices)[0])
    Tu_grid   = np.linspace(T_min * 1.02, T_max * 0.98, 120)
    zeta_list = []
    attempt   = 0
    while len(zeta_list) < n_samples and attempt < n_samples * 40:
        attempt += 1
        r1   = rng.uniform(-0.95, 0.95)
        r3   = -np.sign(r1) if abs(r1) > 1e-6 else 1.0
        rhs1 = r1 * fp_Tmin
        g     = np.full(len(Tu_grid), np.nan)
        valid = np.zeros(len(Tu_grid), dtype=bool)
        for k, Tu in enumerate(Tu_grid):
            fp_Tu = float(f_prior_S(np.array([Tu]), L_r, indices)[0])
            zr    = _solve_2x2(T_min, Tu, rhs1, r3 * fp_Tu, L_r, indices)
            if zr is None:
                continue
            dth   = _dtheta_S_dT(Tu, indices)
            g[k]  = float((L_r @ zr) @ dth) - r3 * _fp_S_deriv(Tu, L_r, indices)
            valid[k] = True
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
        ga      = g[ia]
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
                Tb = Tm
        if zr_best is None:
            Tm_f = 0.5 * (Ta + Tb)
            fp_f = float(f_prior_S(np.array([Tm_f]), L_r, indices)[0])
            zr_best = _solve_2x2(T_min, Tm_f, rhs1, r3 * fp_f, L_r, indices)
        if zr_best is None:
            continue
        dk = delta_kappa(T, L_r, zr_best, indices)
        if not _has_sign_change(dk):
            continue
        zeta_list.append(_enforce_dn_constraint(zr_best, L_r, indices))
    return zeta_list


def _class_B_m2_QR_C2(T, L_r, L_full, indices, T_min, T_max, n_samples, rng):
    """Class-B m=2 with C1+C2+C3+C4 via thin QR least-squares + bisection."""
    row_min = L_r.T @ theta_S(np.array([T_min]), indices)[:, 0]
    row_max = L_r.T @ theta_S(np.array([T_max]), indices)[:, 0]
    fp_Tmin = float(f_prior_from_L(L_full, np.array([T_min]))[0])
    fp_Tmax = float(f_prior_from_L(L_full, np.array([T_max]))[0])
    pre     = _precompute_fixed_block(row_min, row_max)
    Tu_grid = np.linspace(T_min * 1.02, T_max * 0.98, 120)
    zeta_list = []
    attempt   = 0
    while len(zeta_list) < n_samples and attempt < n_samples * 40:
        attempt += 1
        r1 = rng.uniform(-0.95, 0.95)
        r2 = rng.uniform(-0.95, 0.95)
        r3 = -np.sign(r1) if abs(r1) > 1e-6 else 1.0
        b_fixed = np.array([r1 * fp_Tmin, r2 * fp_Tmax])
        g     = np.full(len(Tu_grid), np.nan)
        valid = np.zeros(len(Tu_grid), dtype=bool)
        for k, Tu in enumerate(Tu_grid):
            row_Tu = L_r.T @ theta_S(np.array([Tu]), indices)[:, 0]
            fp_Tu  = float(f_prior_from_L(L_full, np.array([Tu]))[0])
            b_full = np.array([b_fixed[0], b_fixed[1], r3 * fp_Tu])
            zr = _qr_solve_with_precomputed(pre, row_Tu, b_full)
            if zr is None:
                continue
            g[k]     = _g_C4(Tu, zr, r3, L_r, L_full, indices)
            valid[k] = True
        ok = np.where(valid)[0]
        if len(ok) < 2:
            continue
        g_ok = g[ok]
        sc   = np.where(np.diff(np.sign(g_ok)) != 0)[0]
        if len(sc) == 0:
            continue
        pick   = rng.choice(sc)
        ia, ib = ok[pick], ok[pick + 1]
        Ta, Tb = Tu_grid[ia], Tu_grid[ib]
        ga     = g[ia]
        zr_best = None
        for _ in range(50):
            Tm     = 0.5 * (Ta + Tb)
            row_Tm = L_r.T @ theta_S(np.array([Tm]), indices)[:, 0]
            fp_Tm  = float(f_prior_from_L(L_full, np.array([Tm]))[0])
            b_m    = np.array([b_fixed[0], b_fixed[1], r3 * fp_Tm])
            zr_m   = _qr_solve_with_precomputed(pre, row_Tm, b_m)
            if zr_m is None:
                break
            gm = _g_C4(Tm, zr_m, r3, L_r, L_full, indices)
            if abs(gm) < 1e-10:
                zr_best = zr_m; break
            if np.sign(gm) == np.sign(ga):
                Ta, ga = Tm, gm
            else:
                Tb = Tm
        if zr_best is None:
            Tm_f   = 0.5 * (Ta + Tb)
            row_Tf = L_r.T @ theta_S(np.array([Tm_f]), indices)[:, 0]
            fp_Tf  = float(f_prior_from_L(L_full, np.array([Tm_f]))[0])
            b_f    = np.array([b_fixed[0], b_fixed[1], r3 * fp_Tf])
            zr_best = _qr_solve_with_precomputed(pre, row_Tf, b_f)
        if zr_best is None:
            continue
        dk = delta_kappa(T, L_r, zr_best, indices)
        if not _has_sign_change(dk):
            continue
        zeta_list.append(_enforce_dn_constraint(zr_best, L_r, indices))
    return zeta_list


def _class_B_m3_noC2(T, L_r, L_full, indices, T_min, T_max, n_samples, rng):
    """m=3 class-B no-C2: grid scan with exact 3x3 [C1,C3,C4] analytical solve."""
    Tu_grid   = np.linspace(T_min * 1.04, T_max * 0.96, 80)
    zeta_list = []
    attempt   = 0
    while len(zeta_list) < n_samples and attempt < n_samples * 30:
        attempt += 1
        r1 = rng.uniform(-0.95, 0.95)
        r3 = -np.sign(r1) if abs(r1) > 1e-6 else 1.0
        for Tu in rng.permutation(Tu_grid):
            zr = _solve_3x3_B(float(Tu), r1, r3, T_min, L_r, L_full, indices)
            if zr is None:
                continue
            dk = delta_kappa(T, L_r, zr, indices)
            if _has_sign_change(dk):
                zeta_list.append(_enforce_dn_constraint(zr, L_r, indices))
                break
    return zeta_list


def _class_B_m3_noC2_bisect(T, L_r, L_full, indices, T_min, T_max, n_samples, rng):
    """m=3 class-B no-C2 bisect: underdetermined lstsq C1+C3 + bisect on C4."""
    fp_Tmin   = float(f_prior_from_L(L_full, np.array([T_min]))[0])
    Tu_grid   = np.linspace(T_min * 1.02, T_max * 0.98, 80)
    zeta_list = []
    attempt   = 0
    while len(zeta_list) < n_samples and attempt < n_samples * 40:
        attempt += 1
        r1 = rng.uniform(-0.95, 0.95)
        r3 = -np.sign(r1) if abs(r1) > 1e-6 else 1.0
        g     = np.full(len(Tu_grid), np.nan)
        valid = np.zeros(len(Tu_grid), dtype=bool)
        for k, Tu in enumerate(Tu_grid):
            row1 = L_r.T @ theta_S(np.array([T_min]), indices)[:, 0]
            row2 = L_r.T @ theta_S(np.array([Tu]),    indices)[:, 0]
            A_k  = np.vstack([row1, row2])
            fp_Tu = float(f_prior_from_L(L_full, np.array([Tu]))[0])
            b_k   = np.array([r1 * fp_Tmin, r3 * fp_Tu])
            zr_k, _, _, _ = np.linalg.lstsq(A_k, b_k, rcond=None)
            g[k]     = _g_C4(Tu, zr_k, r3, L_r, L_full, indices)
            valid[k] = True
        ok = np.where(valid)[0]
        if len(ok) < 2:
            continue
        g_ok = g[ok]
        sc   = np.where(np.diff(np.sign(g_ok)) != 0)[0]
        if len(sc) == 0:
            continue
        pick   = rng.choice(sc)
        ia, ib = ok[pick], ok[pick + 1]
        Ta, Tb = Tu_grid[ia], Tu_grid[ib]
        ga     = g[ia]
        zr_best = None; A_best = None; b_best = None
        for _ in range(50):
            Tm   = 0.5 * (Ta + Tb)
            row1 = L_r.T @ theta_S(np.array([T_min]), indices)[:, 0]
            row2 = L_r.T @ theta_S(np.array([Tm]),    indices)[:, 0]
            A_m  = np.vstack([row1, row2])
            fp_m = float(f_prior_from_L(L_full, np.array([Tm]))[0])
            b_m  = np.array([r1 * fp_Tmin, r3 * fp_m])
            zr_m, _, _, _ = np.linalg.lstsq(A_m, b_m, rcond=None)
            gm   = _g_C4(Tm, zr_m, r3, L_r, L_full, indices)
            if abs(gm) < 1e-10:
                zr_best = zr_m; A_best = A_m; b_best = b_m; break
            if np.sign(gm) == np.sign(ga):
                Ta, ga = Tm, gm
            else:
                Tb = Tm
        if zr_best is None:
            Tm_f  = 0.5 * (Ta + Tb)
            row1  = L_r.T @ theta_S(np.array([T_min]), indices)[:, 0]
            row2  = L_r.T @ theta_S(np.array([Tm_f]),  indices)[:, 0]
            A_best = np.vstack([row1, row2])
            fp_f  = float(f_prior_from_L(L_full, np.array([Tm_f]))[0])
            b_best = np.array([r1 * fp_Tmin, r3 * fp_f])
            zr_best, _, _, _ = np.linalg.lstsq(A_best, b_best, rcond=None)
        zr_best = _enforce_dn_constraint(zr_best, L_r, indices,
                                          A_system=A_best, b_system=b_best)
        dk = delta_kappa(T, L_r, zr_best, indices)
        if not _has_sign_change(dk):
            continue
        zeta_list.append(zr_best)
    return zeta_list


def _class_B_m3_with_C2(T, L_r, L_full, indices, T_min, T_max, n_samples, rng):
    """m=3 class-B with C2: exact 3x3 [C1,C2,C3] solve then bisect C4."""
    Tu_grid   = np.linspace(T_min * 1.02, T_max * 0.98, 120)
    zeta_list = []
    attempt   = 0
    while len(zeta_list) < n_samples and attempt < n_samples * 40:
        attempt += 1
        r1 = rng.uniform(-0.95, 0.95)
        r2 = rng.uniform(-0.95, 0.95)
        r3 = -np.sign(r1) if abs(r1) > 1e-6 else 1.0
        g     = np.full(len(Tu_grid), np.nan)
        valid = np.zeros(len(Tu_grid), dtype=bool)
        for k, Tu in enumerate(Tu_grid):
            zr = _solve_3x3_C1C2C3(Tu, r1, r2, r3, T_min, T_max,
                                     L_r, L_full, indices)
            if zr is None:
                continue
            g[k]     = _g_C4(Tu, zr, r3, L_r, L_full, indices)
            valid[k] = True
        ok = np.where(valid)[0]
        if len(ok) < 2:
            continue
        g_ok = g[ok]
        sc   = np.where(np.diff(np.sign(g_ok)) != 0)[0]
        if len(sc) == 0:
            continue
        pick   = rng.choice(sc)
        ia, ib = ok[pick], ok[pick + 1]
        Ta, Tb = Tu_grid[ia], Tu_grid[ib]
        ga     = g[ia]
        zr_best = None
        for _ in range(50):
            Tm   = 0.5 * (Ta + Tb)
            zr_m = _solve_3x3_C1C2C3(Tm, r1, r2, r3, T_min, T_max,
                                       L_r, L_full, indices)
            if zr_m is None:
                break
            gm = _g_C4(Tm, zr_m, r3, L_r, L_full, indices)
            if abs(gm) < 1e-10:
                zr_best = zr_m; break
            if np.sign(gm) == np.sign(ga):
                Ta, ga = Tm, gm
            else:
                Tb = Tm
        if zr_best is None:
            zr_best = _solve_3x3_C1C2C3(0.5 * (Ta + Tb), r1, r2, r3,
                                          T_min, T_max, L_r, L_full, indices)
        if zr_best is None:
            continue
        dk = delta_kappa(T, L_r, zr_best, indices)
        if not _has_sign_change(dk):
            continue
        zeta_list.append(_enforce_dn_constraint(zr_best, L_r, indices))
    return zeta_list


def sample_class_C(T, L_r, L_full, indices, n_samples=1, rng=None):
    """Class-C: LS fit to linear crossing target f_c(T) = r1->r2 ramp."""
    if rng is None:
        rng = np.random.default_rng()
    if len(indices) < 2:
        return []
    T_min, T_max = float(T[0]), float(T[-1])
    fp_Tmin = float(f_prior_S(np.array([T_min]), L_r, indices)[0])
    fp_Tmax = float(f_prior_S(np.array([T_max]), L_r, indices)[0])
    thS    = theta_S(T, indices)
    A_mat  = thS.T @ L_r
    A_pinv = np.linalg.pinv(A_mat)
    zeta_list = []
    for _ in range(n_samples * 5):
        if len(zeta_list) >= n_samples:
            break
        r1 = rng.uniform(-1.0, 1.0)
        r2 = rng.uniform(-1.0, 1.0)
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
    return zeta_list


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ORIGINAL SLSQP / SHGO SAMPLERS  (adapted from Uncertainty.py)
# ════════════════════════════════════════════════════════════════════════════

def orig_get_covariance(temperatures, uncertainties):
    """Compute full 3x3 Cholesky L via SLSQP (equivalent to compute_full_L)."""
    return compute_full_L(temperatures, uncertainties)


def orig_get_uncorrelated(L, temperatures, uncertainties):
    """Find zeta_unc via Nelder-Mead; returns (zeta_unc, P_max, P_min, kmax, kmin)."""
    def obj_zeta(guess):
        T     = temperatures
        Theta = np.array([T / T, np.log(T), -1.0 / T])
        QtLZ  = np.array([th @ L @ guess for th in Theta.T])
        f     = uncertainties - QtLZ
        return float(np.dot(f, f))
    guess  = np.array([0.5, 0.1, 0.5])
    result = minimize(obj_zeta, guess, method="Nelder-Mead",
                      options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9})
    return result.x, None, None, None, None


def _orig_signs(kleft_fact, kright_fact):
    """Return (sign_C2, sign_C4, kmiddle_fact) per Uncertainty.py constraint analysis."""
    sign_C2      = -1.0 if (kleft_fact > 0 and kright_fact > 0) else 1.0
    sign_C4      = -1.0 if kleft_fact > 0 else 1.0
    kmiddle_fact = abs(kleft_fact)
    return sign_C2, sign_C4, kmiddle_fact


def orig_get_B2_m3(temperatures, L, nominal, kleft_fact, kright_fact, uncertainties):
    """ORIGINAL SHGO Class-B sampler for m=3; returns zeta_r shape (3,)."""
    zeta_unc, *_ = orig_get_uncorrelated(L, temperatures, uncertainties)
    T_min = float(temperatures[0]); T_max = float(temperatures[-1])
    sign_C2, sign_C4, kmiddle = _orig_signs(kleft_fact, kright_fact)

    def _dk_unc(Tv):
        return float(theta_full(np.array([Tv]))[:, 0] @ L @ zeta_unc)
    def _dk_z(Tv, z):
        return float(theta_full(np.array([Tv]))[:, 0] @ L @ z)
    def _ddk_unc(Tv):
        dth = np.array([0.0, 1.0 / Tv, 1.0 / Tv ** 2])
        return float(dth @ L @ zeta_unc)
    def _ddk_z(Tv, z):
        dth = np.array([0.0, 1.0 / Tv, 1.0 / Tv ** 2])
        return float(dth @ L @ z)

    def obj_b2(z):
        T = temperatures
        Theta = np.array([T / T, np.log(T), -1.0 / T])
        QtLZ  = np.array([th @ L @ z[:3] for th in Theta.T])
        return float(np.dot(uncertainties - QtLZ, uncertainties - QtLZ))

    def c1(z): return kleft_fact  * _dk_unc(T_min) - _dk_z(T_min, z[:3])
    def c3(z): return kright_fact * _dk_unc(T_max) - _dk_z(T_max, z[:3])
    def c2(z):
        Tu = float(np.clip(z[-1], T_min + 1, T_max - 1))
        return sign_C2 * kmiddle * _dk_unc(Tu) - _dk_z(Tu, z[:3])
    def c4(z):
        Tu = float(np.clip(z[-1], T_min + 1, T_max - 1))
        return sign_C4 * kmiddle * _ddk_unc(Tu) - _ddk_z(Tu, z[:3])

    constraints = [{'type':'eq','fun':c1},{'type':'eq','fun':c2},
                   {'type':'eq','fun':c3},{'type':'eq','fun':c4}]
    bounds = [(-10000,10000)]*3 + [(200,3500)]
    x0     = np.array([0.1, 0.1, 0.1, (T_min + T_max) / 2])
    try:
        sol = shgo(obj_b2, bounds, sampling_method='sobol', constraints=constraints,
                   options={'maxiter': 100, 'f_tol': 1e-6})
        zr = sol.x[:3]
    except Exception as e:
        print(f"  Warning: SHGO B2_m3 failed ({e}), falling back to SLSQP")
        try:
            sol = minimize(obj_b2, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': 2000, 'ftol': 1e-9})
            zr = sol.x[:3]
        except Exception as e2:
            print(f"  Warning: SLSQP B2_m3 fallback failed ({e2}); returning x0")
            zr = x0[:3]
    return np.asarray(zr)


def orig_get_C2_m3(temperatures, L, nominal, kleft_fact, kright_fact, uncertainties):
    """ORIGINAL SLSQP Class-C sampler for m=3; returns zeta_r shape (3,)."""
    zeta_unc, *_ = orig_get_uncorrelated(L, temperatures, uncertainties)
    T_min = float(temperatures[0]); T_max = float(temperatures[-1])
    th_min     = theta_full(np.array([T_min]))[:, 0]
    th_max     = theta_full(np.array([T_max]))[:, 0]
    dk_unc_min = float(th_min @ L @ zeta_unc)
    dk_unc_max = float(th_max @ L @ zeta_unc)
    FT2 = kleft_fact  * dk_unc_min   # value at T_min (variable labelled T2 in Uncertainty.py)
    FT1 = kright_fact * dk_unc_max   # value at T_max (variable labelled T1 in Uncertainty.py)
    T2, T1 = T_max, T_min
    slope    = (FT2 - FT1) / (T2 - T1)
    constant = FT2 - slope * T2
    Yu       = slope * temperatures + constant

    def obj_c2(z):
        Theta = np.array([temperatures / temperatures, np.log(temperatures),
                           -1.0 / temperatures])
        QtLZ  = np.array([th @ L @ z for th in Theta.T])
        f     = Yu - QtLZ
        return float(np.dot(f, f))

    guess = np.zeros(3)
    try:
        result = minimize(obj_c2, guess, method='SLSQP',
                          options={'maxiter': 2000, 'ftol': 1e-9})
        return result.x
    except Exception as e:
        print(f"  Warning: SLSQP C2_m3 failed ({e})")
        return guess


def orig_get_B2_m2(temperatures, L_full, nominal, kleft_fact, kright_fact,
                    uncertainties, indices=(0, 1)):
    """ORIGINAL SHGO Class-B adapted for m=2 (Rules A-F); returns zeta_r shape (2,)."""
    _, _, L_r    = get_reduced_L(L_full, indices)
    zeta_unc, *_ = orig_get_uncorrelated(L_full, temperatures, uncertainties)
    zeta_unc_S   = zeta_unc[list(indices)]
    T_min = float(temperatures[0]); T_max = float(temperatures[-1])
    sign_C2, sign_C4, kmiddle = _orig_signs(kleft_fact, kright_fact)

    def _dk_u(Tv):
        thS = theta_S(np.array([Tv]), indices)[:, 0]
        return float(thS @ L_r @ zeta_unc_S)
    def _dk_z(Tv, z):
        thS = theta_S(np.array([Tv]), indices)[:, 0]
        return float(thS @ L_r @ z)
    def _ddk_u(Tv):
        dthS = _dtheta_S_dT(Tv, indices)
        return float(dthS @ L_r @ zeta_unc_S)
    def _ddk_z(Tv, z):
        dthS = _dtheta_S_dT(Tv, indices)
        return float(dthS @ L_r @ z)

    def obj_b2_m2(z):
        thS_all = theta_S(temperatures, indices)
        QtLZ    = np.array([thS @ L_r @ z[:2] for thS in thS_all.T])
        f       = uncertainties - QtLZ
        return float(np.dot(f, f))

    def c1(z): return kleft_fact  * _dk_u(T_min) - _dk_z(T_min, z[:2])
    def c3(z): return kright_fact * _dk_u(T_max) - _dk_z(T_max, z[:2])
    def c2(z):
        Tu = float(np.clip(z[-1], T_min + 1, T_max - 1))
        return sign_C2 * kmiddle * _dk_u(Tu) - _dk_z(Tu, z[:2])
    def c4(z):
        Tu = float(np.clip(z[-1], T_min + 1, T_max - 1))
        return sign_C4 * kmiddle * _ddk_u(Tu) - _ddk_z(Tu, z[:2])

    constraints = [{'type':'eq','fun':c1},{'type':'eq','fun':c2},
                   {'type':'eq','fun':c3},{'type':'eq','fun':c4}]
    bounds = [(-10000,10000)]*2 + [(200,3500)]
    x0     = np.array([0.1, 0.1, (T_min + T_max) / 2])
    try:
        sol = shgo(obj_b2_m2, bounds, sampling_method='sobol',
                   constraints=constraints,
                   options={'maxiter': 100, 'f_tol': 1e-6})
        zr = sol.x[:2]
    except Exception as e:
        print(f"  Warning: SHGO B2_m2 failed ({e}), falling back to SLSQP")
        try:
            sol = minimize(obj_b2_m2, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': 2000, 'ftol': 1e-9})
            zr = sol.x[:2]
        except Exception as e2:
            print(f"  Warning: SLSQP B2_m2 fallback failed ({e2}); returning x0")
            zr = x0[:2]
    return np.asarray(zr)


def orig_get_C2_m2(temperatures, L_full, nominal, kleft_fact, kright_fact,
                    uncertainties, indices=(0, 1)):
    """ORIGINAL SLSQP Class-C adapted for m=2 (Rules A-F); returns zeta_r shape (2,)."""
    _, _, L_r    = get_reduced_L(L_full, indices)
    zeta_unc, *_ = orig_get_uncorrelated(L_full, temperatures, uncertainties)
    zeta_unc_S   = zeta_unc[list(indices)]
    T_min = float(temperatures[0]); T_max = float(temperatures[-1])
    thS_min    = theta_S(np.array([T_min]), indices)[:, 0]
    thS_max    = theta_S(np.array([T_max]), indices)[:, 0]
    dk_unc_min = float(thS_min @ L_r @ zeta_unc_S)
    dk_unc_max = float(thS_max @ L_r @ zeta_unc_S)
    FT2 = kleft_fact  * dk_unc_min
    FT1 = kright_fact * dk_unc_max
    T2, T1 = T_max, T_min
    slope    = (FT2 - FT1) / (T2 - T1)
    constant = FT2 - slope * T2
    Yu       = slope * temperatures + constant

    def obj_c2_m2(z):
        thS_all = theta_S(temperatures, indices)
        QtLZ    = np.array([thS @ L_r @ z for thS in thS_all.T])
        f       = Yu - QtLZ
        return float(np.dot(f, f))

    guess = np.zeros(len(indices))
    try:
        result = minimize(obj_c2_m2, guess, method='SLSQP',
                          options={'maxiter': 2000, 'ftol': 1e-9})
        return result.x
    except Exception as e:
        print(f"  Warning: SLSQP C2_m2 failed ({e})")
        return guess


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ANCHOR LOGIC + RNG WRAPPER + DISPATCHERS
# ════════════════════════════════════════════════════════════════════════════

class _ForcedRng:
    """Wraps a base rng; injects specified values on first uniform() calls."""
    def __init__(self, base_rng, forced=None):
        self._rng   = base_rng
        self._queue = list(forced) if forced else []

    def uniform(self, low=-1.0, high=1.0, size=None):
        if self._queue:
            val = float(self._queue.pop(0))
            return float(np.clip(val, low + 1e-9, high - 1e-9))
        return self._rng.uniform(low, high, size)

    def choice(self, a, *args, **kwargs):
        return self._rng.choice(a, *args, **kwargs)

    def permutation(self, x):
        return self._rng.permutation(x)

    def integers(self, *args, **kwargs):
        return self._rng.integers(*args, **kwargs)


def prepare_anchors(n_samples, anchor_min_input=None, anchor_max_input=None,
                    threshold=20, rng=None):
    """
    Return (r1_list, r2_list) each of length n_samples.
    If n_samples <= threshold and anchor provided: list[0] = clamped anchor.
    Otherwise all entries are random U(-0.95, 0.95).
    """
    if rng is None:
        rng = np.random.default_rng()
    r1_list = rng.uniform(-0.95, 0.95, size=n_samples).tolist()
    r2_list = rng.uniform(-0.95, 0.95, size=n_samples).tolist()
    if n_samples <= threshold and anchor_min_input is not None:
        r1_list[0] = float(np.clip(anchor_min_input, -0.95, 0.95))
    if n_samples <= threshold and anchor_max_input is not None:
        r2_list[0] = float(np.clip(anchor_max_input, -0.95, 0.95))
    return r1_list, r2_list


_B_USES_C2  = {"ORIG_m3", "ORIG_m2", "m2_QR_C2", "m3_C2"}
_B_LABELS   = {
    "ORIG_m3":  "ORIGINAL SAC (m=3)",
    "ORIG_m2":  "ORIGINAL SAC (m=2)",
    "m3_noC2":  "m=3, no C2",
    "m2_noC2":  "m=2, no C2",
    "m2_QR_C2": "m=2, QR+C2",
    "m3_noC2b": "m=3, no C2 (bisect)",
    "m3_C2":    "m=3, with C2",
}
_C_LABELS   = {
    "C_ORIG_m3": "ORIGINAL C (m=3)",
    "C_ORIG_m2": "ORIGINAL C (m=2)",
    "C_new_m3":  "New LS (m=3)",
    "C_new_m2":  "New LS (m=2)",
}
_A_LABELS   = {
    "A_m3": "Class-A (m=3)",
    "A_m2": "Class-A (m=2)",
}

METHOD_COLORS = {
    "ORIG_m3":   "#8B0000",
    "ORIG_m2":   "#FF6B6B",
    "m3_noC2":   "#D85A30",
    "m2_noC2":   "#378ADD",
    "m2_QR_C2":  "#7F77DD",
    "m3_noC2b":  "#FAC775",
    "m3_C2":     "#1D9E75",
    "C_ORIG_m3": "#8B0000",
    "C_ORIG_m2": "#FF6B6B",
    "C_new_m3":  "#1D9E75",
    "C_new_m2":  "#378ADD",
    "A_m3":      "#1D9E75",
    "A_m2":      "#378ADD",
}

_NO_C1_METHODS  = {"A_m2", "A_m3"}
_NO_C2_B_IDS    = {"m3_noC2", "m2_noC2", "m3_noC2b"}
_NO_C2_METHODS  = _NO_C1_METHODS | _NO_C2_B_IDS


def _batch_B(method_id, T, L_r, L_full, indices, nominal, uncertainties,
              T_min, T_max, n_samples, rng):
    """Route to appropriate B-sampler (batch mode for timing)."""
    if method_id == "ORIG_m3":
        out = []
        for _ in range(n_samples):
            r1 = float(rng.uniform(-0.95, 0.95)); r2 = float(rng.uniform(-0.95, 0.95))
            try:
                out.append(np.asarray(
                    orig_get_B2_m3(T, L_full, nominal, r1, r2, uncertainties)))
            except Exception as e:
                print(f"  Warning ORIG_m3 sample: {e}")
        return out
    if method_id == "ORIG_m2":
        out = []
        for _ in range(n_samples):
            r1 = float(rng.uniform(-0.95, 0.95)); r2 = float(rng.uniform(-0.95, 0.95))
            try:
                out.append(np.asarray(
                    orig_get_B2_m2(T, L_full, nominal, r1, r2, uncertainties, indices)))
            except Exception as e:
                print(f"  Warning ORIG_m2 sample: {e}")
        return out
    if method_id == "m3_noC2":
        return _class_B_m3_noC2(T, L_r, L_full, indices, T_min, T_max, n_samples, rng)
    if method_id == "m2_noC2":
        return _class_B_m2(T, L_r, L_full, indices, T_min, T_max, n_samples, rng)
    if method_id == "m2_QR_C2":
        return _class_B_m2_QR_C2(T, L_r, L_full, indices, T_min, T_max, n_samples, rng)
    if method_id == "m3_noC2b":
        return _class_B_m3_noC2_bisect(T, L_r, L_full, indices, T_min, T_max, n_samples, rng)
    if method_id == "m3_C2":
        return _class_B_m3_with_C2(T, L_r, L_full, indices, T_min, T_max, n_samples, rng)
    raise ValueError(f"Unknown B method: {method_id}")


def _batch_C(method_id, T, L_r, L_full, indices, nominal, uncertainties,
              T_min, T_max, n_samples, rng):
    """Route to appropriate C-sampler (batch mode for timing)."""
    if method_id == "C_ORIG_m3":
        out = []
        for _ in range(n_samples):
            r1 = float(rng.uniform(-0.95, 0.95)); r2 = float(rng.uniform(-0.95, 0.95))
            try:
                out.append(np.asarray(
                    orig_get_C2_m3(T, L_full, nominal, r1, r2, uncertainties)))
            except Exception as e:
                print(f"  Warning C_ORIG_m3 sample: {e}")
        return out
    if method_id == "C_ORIG_m2":
        out = []
        for _ in range(n_samples):
            r1 = float(rng.uniform(-0.95, 0.95)); r2 = float(rng.uniform(-0.95, 0.95))
            try:
                out.append(np.asarray(
                    orig_get_C2_m2(T, L_full, nominal, r1, r2, uncertainties, indices)))
            except Exception as e:
                print(f"  Warning C_ORIG_m2 sample: {e}")
        return out
    if method_id in ("C_new_m3", "C_new_m2"):
        return sample_class_C(T, L_r, L_full, indices, n_samples, rng)
    raise ValueError(f"Unknown C method: {method_id}")


def sample_class_B_method(T, L_r, L_full, indices, n_samples, method_id,
                           r1_list=None, r2_list=None, rng=None,
                           nominal=None, uncertainties=None,
                           timing_mode=False):
    """Unified dispatcher for Class-B methods with optional anchor injection."""
    T_min, T_max = float(T[0]), float(T[-1])
    if rng is None:
        rng = np.random.default_rng()
    if timing_mode:
        return _batch_B(method_id, T, L_r, L_full, indices, nominal, uncertainties,
                         T_min, T_max, n_samples, rng)
    zeta_list = []
    for idx in range(n_samples):
        r1 = (r1_list[idx] if (r1_list is not None and idx < len(r1_list))
              else float(rng.uniform(-0.95, 0.95)))
        r2 = (r2_list[idx] if (r2_list is not None and idx < len(r2_list))
              else float(rng.uniform(-0.95, 0.95)))
        try:
            if method_id == "ORIG_m3":
                zr = orig_get_B2_m3(T, L_full, nominal, r1, r2, uncertainties)
                zeta_list.append(np.asarray(zr))
            elif method_id == "ORIG_m2":
                zr = orig_get_B2_m2(T, L_full, nominal, r1, r2, uncertainties, indices)
                zeta_list.append(np.asarray(zr))
            else:
                forced = [r1, r2] if method_id in _B_USES_C2 else [r1]
                frng   = _ForcedRng(rng, forced=forced)
                res    = _batch_B(method_id, T, L_r, L_full, indices, nominal,
                                   uncertainties, T_min, T_max, 1, frng)
                if res:
                    zeta_list.append(res[0])
                else:
                    print(f"  Warning: {method_id} sample {idx} produced no result")
        except Exception as e:
            print(f"  Warning: {method_id} sample {idx}: {e}")
    return zeta_list


def sample_class_C_method(T, L_r, L_full, indices, n_samples, method_id,
                           r1_list=None, r2_list=None, rng=None,
                           nominal=None, uncertainties=None,
                           timing_mode=False):
    """Unified dispatcher for Class-C methods with optional anchor injection."""
    T_min, T_max = float(T[0]), float(T[-1])
    if rng is None:
        rng = np.random.default_rng()
    if timing_mode:
        return _batch_C(method_id, T, L_r, L_full, indices, nominal, uncertainties,
                         T_min, T_max, n_samples, rng)
    zeta_list = []
    for idx in range(n_samples):
        r1 = (r1_list[idx] if (r1_list is not None and idx < len(r1_list))
              else float(rng.uniform(-0.95, 0.95)))
        r2 = (r2_list[idx] if (r2_list is not None and idx < len(r2_list))
              else float(rng.uniform(-0.95, 0.95)))
        try:
            if method_id == "C_ORIG_m3":
                zr = orig_get_C2_m3(T, L_full, nominal, r1, r2, uncertainties)
                zeta_list.append(np.asarray(zr))
            elif method_id == "C_ORIG_m2":
                zr = orig_get_C2_m2(T, L_full, nominal, r1, r2, uncertainties, indices)
                zeta_list.append(np.asarray(zr))
            else:
                frng = _ForcedRng(rng, forced=[r1, r2])
                res  = _batch_C(method_id, T, L_r, L_full, indices, nominal,
                                 uncertainties, T_min, T_max, 1, frng)
                if res:
                    zeta_list.append(res[0])
                else:
                    print(f"  Warning: {method_id} sample {idx} produced no result")
        except Exception as e:
            print(f"  Warning: {method_id} sample {idx}: {e}")
    return zeta_list


def sample_class_A_method(T, L_r, L_full, indices, n_samples, method_id, rng=None):
    """Dispatcher for Class-A methods (no anchors)."""
    if rng is None:
        rng = np.random.default_rng()
    return sample_class_A(T, L_r, L_full, indices, n_samples, rng)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TIMING BENCHMARK
# ════════════════════════════════════════════════════════════════════════════

_ORIG_CAP_SECONDS = 120.0


def time_method(curve_class, method_id, T, L_r, L_full, indices,
                nominal, uncertainties, n_samples, n_repeats=3):
    """
    Time n_repeats runs of the sampling call using time.perf_counter.
    ORIG methods capped at _ORIG_CAP_SECONDS per run.
    Returns (mean_seconds, std_seconds, last_zeta_list).
    """
    T_min, T_max = float(T[0]), float(T[-1])
    is_orig = (method_id.startswith("ORIG") or method_id.startswith("C_ORIG"))
    times  = []; last = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed=42 + rep * 17)
        t0  = time.perf_counter()
        try:
            if curve_class == 'A':
                result = sample_class_A_method(T, L_r, L_full, indices,
                                                n_samples, method_id, rng)
            elif curve_class == 'B':
                result = _batch_B(method_id, T, L_r, L_full, indices,
                                   nominal, uncertainties,
                                   T_min, T_max, n_samples, rng)
            else:
                result = _batch_C(method_id, T, L_r, L_full, indices,
                                   nominal, uncertainties,
                                   T_min, T_max, n_samples, rng)
            last = result
        except Exception as e:
            print(f"  Warning: {method_id} timing rep {rep}: {e}")
            result = []
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if is_orig and elapsed > _ORIG_CAP_SECONDS:
            print(f"  Warning: {method_id} n={n_samples} exceeded cap "
                  f"({elapsed:.1f}s); skipping remaining repeats.")
            break
    return float(np.mean(times)), float(np.std(times) if len(times) > 1 else 0.0), last


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ACCURACY MEASUREMENT
# ════════════════════════════════════════════════════════════════════════════

def compute_errors(zeta_list, r1_list, r2_list,
                   method_id, L_r, L_full, indices, T):
    """Compute mean |C1 error| at T_min and |C2 error| at T_max."""
    T_min, T_max = float(T[0]), float(T[-1])
    fp_Tmin = float(f_prior_from_L(L_full, np.array([T_min]))[0])
    fp_Tmax = float(f_prior_from_L(L_full, np.array([T_max]))[0])
    errs_min = []; errs_max = []
    for k, zr in enumerate(zeta_list):
        r1 = r1_list[k] if k < len(r1_list) else 0.0
        r2 = r2_list[k] if k < len(r2_list) else 0.0
        try:
            act_min = float(delta_kappa(np.array([T_min]), L_r, zr, indices)[0])
            errs_min.append(abs(act_min - r1 * fp_Tmin))
            if method_id not in _NO_C2_METHODS:
                act_max = float(delta_kappa(np.array([T_max]), L_r, zr, indices)[0])
                errs_max.append(abs(act_max - r2 * fp_Tmax))
        except Exception:
            pass
    mean_min = float(np.mean(errs_min)) if errs_min else float('nan')
    mean_max = float(np.mean(errs_max)) if errs_max else float('nan')
    return mean_min, mean_max


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — PLOTS
# ════════════════════════════════════════════════════════════════════════════

def _plot_timing(curve_class, methods_in_class, labels_dict, timing_results,
                 n_samples_list, baseline_id):
    """Plot timing and speedup for one curve class."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Timing — Class-{curve_class}", fontsize=13)
    for mid in methods_in_class:
        color = METHOD_COLORS.get(mid, "gray")
        label = labels_dict.get(mid, mid)
        xs, ys, yerrs = [], [], []
        for ns in n_samples_list:
            entry = timing_results.get(curve_class, {}).get(mid, {}).get(ns)
            if entry is not None:
                xs.append(ns); ys.append(entry[0]); yerrs.append(entry[1])
        if xs:
            ax1.errorbar(xs, ys, yerr=yerrs, marker='o', label=label,
                         color=color, linewidth=1.5, capsize=3)
    ax1.set_yscale('log'); ax1.set_xlabel("n_samples"); ax1.set_ylabel("Wall time (s)")
    ax1.set_title("Wall-clock time"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    base_times = {}
    for ns in n_samples_list:
        entry = timing_results.get(curve_class, {}).get(baseline_id, {}).get(ns)
        if entry is not None:
            base_times[ns] = entry[0]

    for mid in methods_in_class:
        color = METHOD_COLORS.get(mid, "gray")
        label = labels_dict.get(mid, mid)
        xs, ys = [], []
        for ns in n_samples_list:
            entry = timing_results.get(curve_class, {}).get(mid, {}).get(ns)
            bt    = base_times.get(ns)
            if entry is not None and bt is not None and entry[0] > 0:
                xs.append(ns); ys.append(bt / entry[0])
        if xs:
            ax2.plot(xs, ys, marker='s', label=label, color=color, linewidth=1.5)
    ax2.axhline(1.0, color='k', ls='--', lw=0.8)
    ax2.set_xlabel("n_samples"); ax2.set_ylabel(f"Speedup vs {baseline_id}")
    ax2.set_title("Speedup"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"timing_class{curve_class}.pdf"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(f"  Saved {fname}")


def _plot_arrhenius(curve_class, methods_in_class, labels_dict,
                    accuracy_samples, T, nominal, L_full,
                    L_r_m2, L_r_m3, INDICES_M2, INDICES_M3,
                    ANCHOR_MIN, ANCHOR_MAX, accuracy_results,
                    r1_acc, r2_acc):
    """Plot sampled Arrhenius curves for each method in a class."""
    n_methods = len(methods_in_class)
    fig, axes = plt.subplots(1, n_methods,
                              figsize=(4.0 * n_methods, 4.5), squeeze=False)
    axes = axes[0]
    inv_T   = 1000.0 / T
    kap0    = kappa_nominal(T, nominal)
    fp_full = f_prior_from_L(L_full, T) * M_CONST

    T_min = float(T[0]); T_max = float(T[-1])
    fp_Tmin = float(f_prior_from_L(L_full, np.array([T_min]))[0])
    fp_Tmax = float(f_prior_from_L(L_full, np.array([T_max]))[0])

    for col, mid in enumerate(methods_in_class):
        ax    = axes[col]
        color = METHOD_COLORS.get(mid, "gray")
        label = labels_dict.get(mid, mid)
        is_m2 = mid.endswith("_m2") or mid in ("m2_noC2", "m2_QR_C2")
        L_r   = L_r_m2 if is_m2 else L_r_m3
        indices = INDICES_M2 if is_m2 else INDICES_M3
        zeta_list = accuracy_samples.get(curve_class, {}).get(mid, [])
        errs      = accuracy_results.get(curve_class, {}).get(mid, (float('nan'), float('nan')))
        mean_min, mean_max = errs

        ax.plot(inv_T, kap0 + fp_full * M_CONST, '--', color='#888888', lw=0.9,
                label='+f_prior')
        ax.plot(inv_T, kap0 - fp_full * M_CONST, '--', color='#888888', lw=0.9)
        ax.plot(inv_T, kap0, '-', color='#1D9E75', lw=2.0, label='nominal')

        n_show = min(10, len(zeta_list))
        for zr in zeta_list[:n_show]:
            kc = kappa_curve(T, nominal, L_r, zr, indices)
            ax.plot(inv_T, kc, '-', color=color, lw=0.8, alpha=0.6)

            # Actual Tmin/Tmax scatter
            ax.scatter(1000.0 / T_min,
                       float(kap0[0]) + float(delta_kappa(np.array([T_min]), L_r, zr, indices)[0]),
                       marker='x', s=40, color=color, zorder=5)
            if mid not in _NO_C2_METHODS:
                ax.scatter(1000.0 / T_max,
                           float(kap0[-1]) + float(delta_kappa(np.array([T_max]), L_r, zr, indices)[0]),
                           marker='x', s=40, color=color, zorder=5)

        # Target anchor markers
        r1_show = r1_acc[0] if r1_acc else ANCHOR_MIN
        ax.scatter(1000.0 / T_min,
                   float(kap0[0]) + r1_show * fp_Tmin,
                   marker='o', s=70, color='blue', zorder=6, label='T_min target')
        if mid not in _NO_C2_METHODS:
            r2_show = r2_acc[0] if r2_acc else ANCHOR_MAX
            ax.scatter(1000.0 / T_max,
                       float(kap0[-1]) + r2_show * fp_Tmax,
                       marker='s', s=70, color='darkorange', zorder=6, label='T_max target')

        title = f"{label}\nT_min err={mean_min:.3f}"
        if mid not in _NO_C2_METHODS and not np.isnan(mean_max):
            title += f"  T_max err={mean_max:.3f}"
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("1000/T  [K⁻¹]", fontsize=8)
        ax.set_ylabel("κ = ln k", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Arrhenius curves — Class-{curve_class}", fontsize=11)
    fig.tight_layout()
    fname = f"arrhenius_class{curve_class}.pdf"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(f"  Saved {fname}")


def _plot_anchor_errors(curve_class, methods_in_class, labels_dict,
                         accuracy_results):
    """Bar chart of mean anchor errors for T_min and T_max."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Anchor errors — Class-{curve_class}", fontsize=12)
    mids   = methods_in_class
    labels = [labels_dict.get(m, m) for m in mids]
    colors = [METHOD_COLORS.get(m, 'gray') for m in mids]
    errs_min = [accuracy_results.get(curve_class, {}).get(m, (float('nan'), float('nan')))[0]
                for m in mids]
    errs_max = [accuracy_results.get(curve_class, {}).get(m, (float('nan'), float('nan')))[1]
                for m in mids]

    for ax, err_vals, title_str in [(ax1, errs_min, "T_min error (|C1|)"),
                                     (ax2, errs_max, "T_max error (|C2|)")]:
        xs = np.arange(len(mids))
        for i, (mid, ev) in enumerate(zip(mids, err_vals)):
            if np.isnan(ev):
                ax.bar(i, 0.0, color=colors[i], hatch='//', alpha=0.5)
            else:
                ax.bar(i, ev, color=colors[i], alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel("Mean |error|")
        ax.set_title(title_str)
        ax.axhline(0.0, color='k', ls='--', lw=0.7)
        ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fname = f"anchor_errors_class{curve_class}.pdf"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Run the full benchmark: timing + accuracy + 9 PDF plots."""
    # ─── 0. Configuration ────────────────────────────────────────────────
    XML_PATH          = "MB_R_ALL_ECM_2025.xml"
    YAML_PATH         = "MB_MB2D_LALIT_2024.yaml"
    N_SAMPLES_TIMING  = [5, 10, 20, 50, 100]
    N_SAMPLES_ACCUR   = 10
    ANCHOR_MIN        = 0.5
    ANCHOR_MAX        = -0.4
    ANCHOR_THRESHOLD  = 20
    N_REPEATS         = 3
    INDICES_M2        = (0, 1)
    INDICES_M3        = (0, 1, 2)
    rng               = np.random.default_rng(seed=42)

    CLASS_B_METHODS = ["ORIG_m3","ORIG_m2","m3_noC2","m2_noC2",
                        "m2_QR_C2","m3_noC2b","m3_C2"]
    CLASS_C_METHODS = ["C_ORIG_m3","C_ORIG_m2","C_new_m3","C_new_m2"]
    CLASS_A_METHODS = ["A_m2","A_m3"]

    CLASS_METHODS = {"A": CLASS_A_METHODS,
                     "B": CLASS_B_METHODS,
                     "C": CLASS_C_METHODS}
    BASELINE_IDS  = {"A": "A_m3", "B": "ORIG_m3", "C": "C_ORIG_m3"}
    ALL_LABELS    = {**_B_LABELS, **_C_LABELS, **_A_LABELS}

    # ─── 1. Parse one reaction ───────────────────────────────────────────
    print("\n" + "="*65)
    print("  MUQ-SAC Arrhenius Sampling Benchmark")
    print("="*65)
    if not Path(XML_PATH).exists():
        sys.exit(f"ERROR: {XML_PATH} not found.  "
                  "Place mechanism files in the working directory.")
    if not Path(YAML_PATH).exists():
        sys.exit(f"ERROR: {YAML_PATH} not found.")

    print(f"\n[1] Parsing YAML: {YAML_PATH}")
    yaml_rate_db = parse_yaml_mechanism(YAML_PATH)
    print(f"    {len(yaml_rate_db)} reactions in YAML")

    print(f"\n[2] Parsing XML: {XML_PATH}")
    rxn_dict = parse_xml_uncertainty(XML_PATH)
    print(f"    {len(rxn_dict)} reactions in XML")

    # Pick first XML tag that has a YAML match
    rxn_tag = None; rxn_data = None; nominal = None
    for tag, data in rxn_dict.items():
        eq  = data["rxn_equation"]
        nom = get_nominal_params(eq, yaml_rate_db)
        if nom is not None:
            rxn_tag = tag; rxn_data = data; nominal = nom
            break
    if rxn_tag is None:
        sys.exit("ERROR: No XML reaction matched in YAML.")

    T            = rxn_data["temperatures"]
    uncertainties = rxn_data["uncertainties"]
    rIndex        = rxn_data["rIndex"]
    T_min, T_max  = float(T[0]), float(T[-1])
    print(f"\n[3] Selected reaction:  {rxn_tag}")
    print(f"    Equation:           {rxn_data['rxn_equation']}")
    print(f"    rIndex:             {rIndex}")
    print(f"    T range:            {T_min:.0f} – {T_max:.0f} K  "
          f"({len(T)} points)")
    print(f"    Nominal [α,n,ε]:    {np.round(nominal,4)}")

    # ─── 2. Covariance matrices ──────────────────────────────────────────
    print("\n[4] Computing L_full (MUQ)…")
    L_full  = compute_full_L(T, uncertainties)
    print(f"    L_full diag: {np.diag(L_full).round(4)}")
    _, _, L_r_m2 = get_reduced_L(L_full, INDICES_M2)
    _, _, L_r_m3 = get_reduced_L(L_full, INDICES_M3)

    # ─── 3. Anchor lists for accuracy run ───────────────────────────────
    r1_acc, r2_acc = prepare_anchors(N_SAMPLES_ACCUR, ANCHOR_MIN, ANCHOR_MAX,
                                      threshold=ANCHOR_THRESHOLD, rng=rng)

    # ─── 4. Timing benchmark ─────────────────────────────────────────────
    print("\n[5] Timing benchmark…")
    timing_results = {"A": {m: {} for m in CLASS_A_METHODS},
                      "B": {m: {} for m in CLASS_B_METHODS},
                      "C": {m: {} for m in CLASS_C_METHODS}}

    for curve_class, methods in CLASS_METHODS.items():
        for ns in N_SAMPLES_TIMING:
            for mid in methods:
                is_m2 = mid.endswith("_m2") or mid in ("m2_noC2","m2_QR_C2")
                L_r     = L_r_m2 if is_m2 else L_r_m3
                indices = INDICES_M2 if is_m2 else INDICES_M3
                print(f"    Timing  class={curve_class}  method={mid:<14}  n={ns}", end='', flush=True)
                mt, ms, _ = time_method(
                    curve_class, mid, T, L_r, L_full, indices,
                    nominal, uncertainties, ns, n_repeats=N_REPEATS)
                timing_results[curve_class][mid][ns] = (mt, ms)
                print(f"  {mt:.3f}s ± {ms:.3f}s")

    # ─── 5. Accuracy run ─────────────────────────────────────────────────
    print("\n[6] Accuracy run (n={})…".format(N_SAMPLES_ACCUR))
    accuracy_results = {}
    accuracy_samples = {}
    for curve_class, methods in CLASS_METHODS.items():
        accuracy_results[curve_class] = {}
        accuracy_samples[curve_class] = {}
        for mid in methods:
            is_m2   = mid.endswith("_m2") or mid in ("m2_noC2","m2_QR_C2")
            L_r     = L_r_m2 if is_m2 else L_r_m3
            indices = INDICES_M2 if is_m2 else INDICES_M3
            acc_rng = np.random.default_rng(seed=99)
            try:
                if curve_class == 'A':
                    zlist = sample_class_A_method(
                        T, L_r, L_full, indices, N_SAMPLES_ACCUR, mid, acc_rng)
                elif curve_class == 'B':
                    zlist = sample_class_B_method(
                        T, L_r, L_full, indices, N_SAMPLES_ACCUR, mid,
                        r1_list=r1_acc, r2_list=r2_acc, rng=acc_rng,
                        nominal=nominal, uncertainties=uncertainties)
                else:
                    zlist = sample_class_C_method(
                        T, L_r, L_full, indices, N_SAMPLES_ACCUR, mid,
                        r1_list=r1_acc, r2_list=r2_acc, rng=acc_rng,
                        nominal=nominal, uncertainties=uncertainties)
            except Exception as e:
                print(f"  Warning: accuracy run for {mid} failed: {e}")
                zlist = []
            accuracy_samples[curve_class][mid] = zlist
            emn, emx = compute_errors(zlist, r1_acc, r2_acc,
                                       mid, L_r, L_full, indices, T)
            accuracy_results[curve_class][mid] = (emn, emx)
            print(f"    {mid:<14}  n_got={len(zlist):2d}  "
                  f"err_Tmin={emn:.4f}  err_Tmax={'N/A' if np.isnan(emx) else f'{emx:.4f}'}")

    # ─── 6. Print summary tables ─────────────────────────────────────────
    print("\n" + "="*65)
    for curve_class, methods in CLASS_METHODS.items():
        print(f"\n── Class-{curve_class} Timing (s) ──")
        header = f"{'Method':<16}" + "".join(f"  n={ns:>4}" for ns in N_SAMPLES_TIMING)
        print(header)
        print("-" * len(header))
        for mid in methods:
            row = f"{mid:<16}"
            for ns in N_SAMPLES_TIMING:
                entry = timing_results[curve_class][mid].get(ns)
                row += f"  {entry[0]:7.3f}" if entry else "     N/A"
            print(row)

    print()
    for curve_class, methods in CLASS_METHODS.items():
        print(f"\n── Class-{curve_class} Accuracy ──")
        print(f"{'Method':<16}  {'err_Tmin':>10}  {'err_Tmax':>10}")
        print("-" * 40)
        for mid in methods:
            emn, emx = accuracy_results[curve_class].get(mid, (float('nan'), float('nan')))
            tmax_str = "N/A" if np.isnan(emx) else f"{emx:.4f}"
            print(f"{mid:<16}  {emn:>10.4f}  {tmax_str:>10}")

    # ─── 7. Produce all plots ─────────────────────────────────────────────
    print("\n[7] Generating plots…")
    for curve_class, methods in CLASS_METHODS.items():
        labels_map = {m: ALL_LABELS.get(m, m) for m in methods}

        _plot_timing(curve_class, methods, labels_map,
                     timing_results, N_SAMPLES_TIMING,
                     BASELINE_IDS[curve_class])

        _plot_arrhenius(curve_class, methods, labels_map,
                        accuracy_samples, T, nominal, L_full,
                        L_r_m2, L_r_m3, INDICES_M2, INDICES_M3,
                        ANCHOR_MIN, ANCHOR_MAX, accuracy_results,
                        r1_acc, r2_acc)

        _plot_anchor_errors(curve_class, methods, labels_map,
                             accuracy_results)

    print("\nDone.  9 PDF plots saved.")


if __name__ == "__main__":
    main()