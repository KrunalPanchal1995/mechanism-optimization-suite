#!/usr/bin/env python3
"""
generate_class_A.py
===================
Generate Class-A Arrhenius perturbation samples from an XML uncertainty file.

Class-A definition
------------------
  Δκ(T) ∝ f_prior_S(T),  both ± signs allowed.
  zeta_r = pinv(Θ_S^T L_r) · (±α · f_prior_S(T)),   α ~ U(0.05, 1.0)

Generates 100 samples per sub-configuration for:
  m = 1  →  A only (1,0,0) | n only (0,1,0) | Ea only (0,0,1)
  m = 2  →  (A,n)(1,1,0)   | (A,Ea)(1,0,1)  | (n,Ea)(0,1,1)
  m = 3  →  (A,n,Ea)(1,1,1)

Input
-----
  XML_PATH  : reaction uncertainty file  (required)
  YAML_PATH : Cantera mechanism file     (optional – needed for nominal params
                                          and kappa(T) output curves)
Output structure
----------------
  output/
    {rxn_safe_name}/
      info.json           ← L_full, Sigma, L_r, Sigma_r, timing per sub-config
      m1/
        A/     zeta_samples.npy   [100 × 1]
        n/     zeta_samples.npy   [100 × 1]
        Ea/    zeta_samples.npy   [100 × 1]
      m2/
        A_n/   zeta_samples.npy   [100 × 2]
        A_Ea/  zeta_samples.npy   [100 × 2]
        n_Ea/  zeta_samples.npy   [100 × 2]
      m3/
        A_n_Ea/ zeta_samples.npy  [100 × 3]

Note: zeta_r vectors are the dimensionless perturbation directions in the
      reduced parameter space.  To recover Δκ(T):
        Δκ(T) = Θ_S(T)^T  L_r  zeta_r
      where Θ_S and L_r come from the matching sub-config in info.json.
"""

import json
import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.linalg import cholesky
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

M_CONST     = 3.0 / np.log(10.0)   # IUPAC normalisation factor
R_GAS       = 1.987                 # cal/(mol·K)
MAX_DELTA_N = 2.0                   # hard upper bound on |Δn|
N_SAMPLES   = 100                   # number of samples per sub-config
RANDOM_SEED = 42

# Parameter index mapping: 0 → ln A,  1 → n,  2 → Ea/R
PARAM_NAMES = {0: "A", 1: "n", 2: "Ea"}

# All sub-configurations keyed by (m_level, folder_name, indices_tuple)
M_CONFIGS = [
    # m = 1
    (1, "A",      (0,)),
    (1, "n",      (1,)),
    (1, "Ea",     (2,)),
    # m = 2
    (2, "A_n",    (0, 1)),
    (2, "A_Ea",   (0, 2)),
    (2, "n_Ea",   (1, 2)),
    # m = 3
    (3, "A_n_Ea", (0, 1, 2)),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ MATH HELPERS  (ported verbatim from benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def theta_full(T):
    """θ(T) = [1, ln T, -1/T],  shape (3, N)."""
    return np.array([np.ones_like(T), np.log(T), -1.0 / T])


def theta_S(T, indices):
    """Reduced basis Θ_S(T) of shape (m, N); row map: 0→1, 1→lnT, 2→-1/T."""
    return theta_full(T)[list(indices), :]

def f_prior(T, L_full):
    """f_prior_S(T) = ‖L_r^T Θ_S(T)‖₂ for each T in array."""
    thS = theta_full(T)
    return np.array([np.linalg.norm(L_full.T @ col) for col in thS.T])


def f_prior_S(T, L_r, indices):
    """f_prior_S(T) = ‖L_r^T Θ_S(T)‖₂ for each T in array."""
    thS = theta_S(T, indices)
    return np.array([np.linalg.norm(L_r.T @ col) for col in thS.T])


def delta_kappa(T, L_r, zeta_r, indices):
    """Δκ_S(T) = Θ_S(T)^T L_r zeta_r."""
    return theta_S(T, indices).T @ (L_r @ zeta_r)


def kappa_nominal(T, nominal):
    """κ₀(T) = θ(T)^T p₀  (nominal log-rate curve)."""
    return theta_full(T).T @ nominal


def _dtheta_S_dT(T_val, indices):
    """Analytical d(θ_S)/dT:  d[1]=0,  d[lnT]=1/T,  d[-1/T]=1/T²."""
    full = np.array([0.0, 1.0 / T_val, 1.0 / T_val ** 2])
    return full[list(indices)]


def _enforce_dn_constraint(zeta_r, L_r, indices):
    """Clamp |Δn| to MAX_DELTA_N by scaling zeta_r if needed."""
    if 1 not in indices:
        return zeta_r
    pos_n   = list(indices).index(1)
    delta_n = (L_r @ zeta_r)[pos_n]
    if abs(delta_n) <= MAX_DELTA_N:
        return zeta_r
    scale = (0.95 * MAX_DELTA_N) / (abs(delta_n) + 1e-30)
    return zeta_r * scale


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ COVARIANCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _muq_objective(params, T, uncertainties):
    """MUQ residual for Cholesky fitting (used by compute_full_L)."""
    L = np.array([[params[0], 0.0, 0.0],
                  [params[1], params[2], 0.0],
                  [params[3], params[4], params[5]]])
    Theta    = theta_full(T)
    f_model  = np.array([np.linalg.norm(L.T @ col) for col in Theta.T])
    f_target = uncertainties / M_CONST
    diff     = (f_target - f_model) / (f_target + 1e-30)
    return float(np.dot(diff, diff))


def compute_full_L(temperatures, uncertainties):
    """Fit 3×3 lower-triangular Cholesky L via SLSQP (MUQ optimisation)."""
    f_mean = np.mean(uncertainties / M_CONST)
    x0     = np.array([f_mean, 0.0, f_mean * 0.1,
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
    """
    Extract (Σ, Σ_r, L_r) for a parameter subset.

    Returns
    -------
    Sigma   : (3, 3) full covariance  Σ = L L^T
    Sigma_r : (m, m) sub-matrix of Σ indexed by `indices`
    L_r     : (m, m) Cholesky factor of Σ_r  (lower triangular)
    """
    Sigma   = L_full @ L_full.T
    idx     = list(indices)
    Sigma_r = Sigma[np.ix_(idx, idx)]
    try:
        L_r = cholesky(Sigma_r, lower=True)
    except Exception:
        eps = 1e-12 * np.trace(Sigma_r) / len(idx)
        L_r = cholesky(Sigma_r + eps * np.eye(len(idx)), lower=True)
    return Sigma, Sigma_r, L_r


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ CLASS-A SAMPLER  (ported verbatim from benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def sample_class_A(T, L_full, L_r, indices, n_samples=100, rng=None, max_attempts=1000):
    """
    Generate `n_samples` Class-A zeta_r vectors for the given parameter subset,
    with rejection sampling to enforce the fp uncertainty bounds at all temperatures.

    Algorithm
    ---------
    1.  Compute f_prior_S(T) = ‖L_r^T Θ_S(T)‖₂  for all T.
    2.  Form A_mat = Θ_S(T)^T L_r  (overdetermined system, shape N×m).
    3.  For each sample:
          a. Draw α ~ U(0, 1), draw sign ∈ {-1, +1}.
             Set target b = sign·α·f_prior_S(T).
             Solve zeta_r = pinv(A_mat) · b  (least-squares fit).
          b. Apply Δn constraint.
          c. Reject if |A_mat @ zeta_r| > fp at ANY temperature; retry.
    4.  Raise RuntimeError if max_attempts is exceeded for any single sample.

    Parameters
    ----------
    T            : temperature array (K), shape (N,)
    L_full       : full (3,3) Cholesky factor, used to compute fp
    L_r          : (m, m) reduced Cholesky factor
    indices      : tuple of active parameter indices, e.g. (0, 1) for (A, n)
    n_samples    : number of valid samples to generate
    rng          : numpy Generator (created fresh if None)
    max_attempts : max draws per sample before raising an error (default 1000)

    Returns
    -------
    zeta_list : list of n_samples zeta_r arrays, each shape (m,)
    """
    if rng is None:
        rng = np.random.default_rng()

    fp     = f_prior(T, L_full)          # shape (N,)  — the bound at every T
    thS    = theta_S(T, indices)
    A_mat  = thS.T @ L_r                 # shape (N, m)
    A_pinv = np.linalg.pinv(A_mat)       # shape (m, N)

    zeta_list = []

    for i in range(n_samples):
        accepted = False
        for attempt in range(max_attempts):
            alpha_s = rng.uniform(0.0, 1.0)
            sign    = rng.choice([-1.0, 1.0])
            b_vec   = sign * alpha_s * fp

            zeta_r  = A_pinv @ b_vec
            zeta_r  = _enforce_dn_constraint(zeta_r, L_r, indices)

            # ── Rejection check ──────────────────────────────────────────────
            # Reconstruct the projected curve and verify it lies within ±fp(T).
            # A_mat @ zeta_r  ≡  Θ_S(T)^T L_r zeta_r  (shape N,)
            curve = A_mat @ zeta_r                      # shape (N,)
            if np.all(np.abs(curve) <= fp):             # passes at every T
                accepted = True
                break
            # else: discard and redraw
            # ─────────────────────────────────────────────────────────────────

        if not accepted:
            raise RuntimeError(
                f"Class-A sample {i}: could not find a valid sample within "
                f"{max_attempts} attempts. "
                f"Check L_r / indices consistency or increase max_attempts."
            )

        zeta_list.append(zeta_r)

    return zeta_list


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ PARSING HELPERS  (ported verbatim from benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_equation(s):
    """Normalise reaction equation string for consistent matching."""
    s = re.sub(r"\s+", " ", s.strip())
    s = s.replace("=>", "<=>").replace("= >", "<=>")
    s = s.replace("< =>", "<=>").replace("<= >", "<=>")
    return s


def parse_xml_uncertainty(xml_path):
    """
    Parse XML uncertainty file.

    Returns
    -------
    dict  keyed by reaction nametag, each value:
        {temperatures, uncertainties, rxn_equation, pressure_limit, rIndex}
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
            T_arr = np.array(t_vals)
            u_arr = np.array(u_vals)
        nametag = rxn_eq if pres_lim is None else f"{rxn_eq}:{pres_lim}"
        reactions[nametag] = {
            "temperatures":   T_arr,
            "uncertainties":  u_arr,
            "rxn_equation":   rxn_eq,
            "pressure_limit": pres_lim,
            "rIndex":         r_index,
        }
    return reactions


def parse_yaml_mechanism(yaml_path):
    """Return {normalised_equation: rate_constant_dict} from a Cantera YAML."""
    with open(yaml_path) as fh:
        mech = yaml.safe_load(fh)
    result = {}
    for rxn in mech.get("reactions", []):
        eq = normalize_equation(rxn.get("equation", ""))
        rc = rxn.get("rate-constant")
        if rc is None:
            rc = (rxn.get("high-P-rate-constant")
                  or rxn.get("low-P-rate-constant"))
        if rc is not None:
            result[eq] = rc
    return result


def get_nominal_params(rxn_eq, yaml_rate_db):
    """Look up [α=ln(A), n, ε=Ea/R] from YAML DB; returns None if missing."""
    norm = normalize_equation(rxn_eq)
    rc   = yaml_rate_db.get(norm)
    if rc is None:
        return None
    A  = rc.get("A",  1.0)
    n  = rc.get("b",  0.0)
    Ea = rc.get("Ea", 0.0)
    return np.array([np.log(max(A, 1e-300)), n, Ea / R_GAS], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ FILE-SYSTEM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_folder_name(rxn_eq: str) -> str:
    """
    Convert a reaction equation string to a valid directory name.

    Examples
    --------
    'H2O2 (+M) <=> 2OH (+M)'  →  'H2O2_pM__2OH_pM'
    'CH4 + OH <=> CH3 + H2O'  →  'CH4_p_OH__CH3_p_H2O'
    """
    s = rxn_eq.strip()
    s = s.replace("(+M)", "_pM").replace("(+m)", "_pm")
    s = re.sub(r'\s*<=>\s*', '__', s)
    s = re.sub(r'\s*=>\s*',  '__', s)
    s = re.sub(r'\s*\+\s*',  '_p_', s)
    s = re.sub(r'\s+',        '_',  s)
    s = re.sub(r'[^\w\-]',   '_',  s)
    s = re.sub(r'_+',         '_',  s)
    return s.strip('_')[:80]   # cap at 80 chars to avoid OS limits


def write_info_json(out_dir: Path, rxn_info: dict, L_full: np.ndarray,
                    Sigma: np.ndarray, config_records: list):
    """
    Write info.json at the reaction root folder.

    JSON schema
    -----------
    {
      "reaction": { name_tag, equation, rIndex, T_min, T_max, n_T },
      "covariance": { L_full [3×3], Sigma [3×3] },
      "configurations": {
        "<m_level>/<folder>": {
          indices, param_labels, m,
          Sigma_r [m×m], L_r [m×m],
          n_samples_requested, n_samples_generated, wall_time_s
        }, ...
      }
    }
    """
    doc = {
        "reaction": rxn_info,
        "covariance": {
            "L_full":  L_full.tolist(),
            "Sigma":   Sigma.tolist(),
        },
        "configurations": {},
    }
    for rec in config_records:
        key = f"m{rec['m']}/{rec['folder']}"
        doc["configurations"][key] = {
            "m":                    rec["m"],
            "indices":              list(rec["indices"]),
            "param_labels":         [PARAM_NAMES[i] for i in rec["indices"]],
            "Sigma_r":              np.array(rec["Sigma_r"]).tolist(),
            "L_r":                  np.array(rec["L_r"]).tolist(),
            "n_samples_requested":  rec["n_requested"],
            "n_samples_generated":  rec["n_generated"],
            "wall_time_s":          round(rec["wall_time_s"], 6),
        }
    info_path = out_dir / "info.json"
    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"    Wrote {info_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves_for_subfolder(sub_dir, T, L_full, L_r, indices, zeta_list,
                               rxn_eq, m_level, folder, curve_class="A"):
    """
    Save one PDF plot inside `sub_dir` showing all sampled curves.

    What is plotted
    ---------------
    Primary plot (always):
      · All Δκ(T) perturbation curves  =  Θ_S(T)^T L_r zeta_r
      · ±f_prior_S(T) envelope  (dashed grey) — theoretical ±1σ boundary

    Secondary plot (if kappa_curves.npy exists in sub_dir):
      · Full log-rate curves  κ(T) = κ₀(T) + Δκ(T)  on a second axes

    X-axis convention: 1000/T  (K⁻¹)  — standard Arrhenius presentation.

    Parameters
    ----------
    sub_dir    : pathlib.Path  – the sub-config folder (where .npy lives)
    T          : (N,) temperature array
    L_r        : (m, m) reduced Cholesky factor for this sub-config
    indices    : tuple of active parameter indices
    zeta_list  : list of zeta_r arrays (each shape (m,))
    rxn_eq     : reaction equation string (for the title)
    m_level    : int, 1 / 2 / 3
    folder     : str, e.g. "A_n" (for subplot/file label)
    curve_class: "A", "B", or "C"
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if not zeta_list:
        return

    inv_T   = 1000.0 / T                      # x-axis: 1000/T
    fp_S    = f_prior(T, L_full)      # ±envelope

    # ── Compute all Δκ curves ────────────────────────────────────────────
    dk_all  = np.vstack([delta_kappa(T, L_r, zr, indices)
                         for zr in zeta_list])   # shape (n_samples, N)

    # ── Load full κ(T) curves if nominal was saved ───────────────────────
    kappa_path = sub_dir / "kappa_curves.npy"
    T_path     = sub_dir / "temperatures.npy"
    has_kappa  = kappa_path.exists() and T_path.exists()

    # ── Figure layout ────────────────────────────────────────────────────
    n_axes  = 2 if has_kappa else 1
    fig, axes = plt.subplots(1, n_axes, figsize=(7 * n_axes, 5), squeeze=False)
    ax_dk   = axes[0, 0]
    ax_k    = axes[0, 1] if has_kappa else None

    param_label = ", ".join(PARAM_NAMES[i] for i in indices)
    rxn_short   = rxn_eq[:50] + ("…" if len(rxn_eq) > 50 else "")
    fig.suptitle(
        f"Class-{curve_class} | m={m_level} [{param_label}]\n{rxn_short}",
        fontsize=9, y=1.01
    )

    # colour palette: one colour per sample
    colours = cm.plasma(np.linspace(0.15, 0.85, len(zeta_list)))

    # ── Left axes: Δκ(T) curves ──────────────────────────────────────────
    for i, dk in enumerate(dk_all):
        ax_dk.plot(inv_T, dk, color=colours[i], alpha=0.45, linewidth=0.7)

    ax_dk.plot(inv_T,  fp_S, color="black", lw=1.4, ls="--",
               label=r"$\pm f_{\mathrm{prior},S}$")
    ax_dk.plot(inv_T, -fp_S, color="black", lw=1.4, ls="--")
    ax_dk.axhline(0, color="black", lw=0.6, ls=":")

    ax_dk.set_xlabel(r"$1000/T$  (K$^{-1}$)", fontsize=10)
    ax_dk.set_ylabel(r"$\Delta\kappa_S(T)$", fontsize=10)
    #ax_dk.set_title(r"Perturbation curves $\Delta\kappa_S$", fontsize=9)
    ax_dk.legend(fontsize=9)
    ax_dk.grid(True, alpha=0.25)

    # ── Right axes: κ(T) curves (only if nominal saved) ──────────────────
    if has_kappa:
        kappa_all = np.load(kappa_path)          # shape (n_samples, N)
        T_saved   = np.load(T_path)
        inv_T_k   = 1000.0 / T_saved

        kappa_nom = kappa_all.mean(axis=0)       # rough nominal approximation
        for i, krow in enumerate(kappa_all):
            ax_k.plot(inv_T_k, krow, color=colours[i], alpha=0.45, linewidth=0.7)

        ax_k.plot(inv_T_k, kappa_nom, color="black", lw=1.5, ls="-",
                  label=r"mean $\kappa(T)$")
        ax_k.set_xlabel(r"$1000/T$  (K$^{-1}$)", fontsize=10)
        ax_k.set_ylabel(r"$\kappa(T) = \ln k(T)$", fontsize=10)
        #ax_k.set_title(r"Full log-rate curves $\kappa(T)$", fontsize=9)
        ax_k.legend(fontsize=9)
        ax_k.grid(True, alpha=0.25)

    # ── Colour bar: sample index ──────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(1, len(zeta_list)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0, -1], shrink=0.85, pad=0.02)
    cbar.set_label("Sample index", fontsize=8)

    plt.tight_layout()
    pdf_path = sub_dir / "curves.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Plot saved → {pdf_path}")

def plot_curves_plain_blue(sub_dir, T, L_r, indices, zeta_list,
                            rxn_eq, m_level, folder, curve_class="A"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not zeta_list:
        return

    kappa_path = sub_dir / "kappa_curves.npy"
    T_path     = sub_dir / "temperatures.npy"
    if not (kappa_path.exists() and T_path.exists()):
        return   # nothing to plot without nominal

    kappa_all = np.load(kappa_path)
    T_saved   = np.load(T_path)
    inv_T_k   = 1000.0 / T_saved
    kappa_nom = kappa_all.mean(axis=0)

    param_label = ", ".join(PARAM_NAMES[i] for i in indices)
    rxn_short   = rxn_eq[:50] + ("…" if len(rxn_eq) > 50 else "")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"Class-{curve_class} | m={m_level} [{param_label}]  (n={len(kappa_all)})\n{rxn_short}",
        fontsize=9
    )

    for krow in kappa_all:
        ax.plot(inv_T_k, krow, color="red", alpha=0.35, linewidth=1.1)

    ax.plot(inv_T_k, kappa_nom, color="black", lw=1.8, ls="-",
            label=r"mean $\kappa(T)$")

    ax.set_xlabel(r"$1000/T$  (K$^{-1}$)", fontsize=11)
    ax.set_ylabel(r"$\kappa(T) = \ln k(T)$", fontsize=11)
    #ax.set_title(r"Full log-rate curves $\kappa(T)$", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    pdf_path = sub_dir / "curves_plain.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Plot saved → {pdf_path}")

def plot_delta_kappa_standalone(sub_dir, T, L_full, L_r, indices, zeta_list,
                                 rxn_eq, m_level, folder, curve_class="A"):
    """Save a full-width Δκ vs 1000/T plot as delta_kappa.pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not zeta_list:
        return

    inv_T  = 1000.0 / T
    fp_S   = f_prior(T, L_full)
    dk_all = np.vstack([delta_kappa(T, L_r, zr, indices) for zr in zeta_list])

    param_label = ", ".join(PARAM_NAMES[i] for i in indices)
    rxn_short   = rxn_eq[:50] + ("…" if len(rxn_eq) > 50 else "")

    fig, ax = plt.subplots(figsize=(7, 5))
    #fig.suptitle(
    #    f"Class-{curve_class} | m={m_level} [{param_label}]  (n={len(zeta_list)})\n{rxn_short}",
    #    fontsize=9
    #)

    for dk in dk_all:
        ax.plot(inv_T, dk, color="red", alpha=0.35, linewidth=1.1)

    ax.plot(inv_T,  fp_S, color="black", lw=1.4, ls="--",
            label=r"$\pm f_{\mathrm{prior},S}$")
    ax.plot(inv_T, -fp_S, color="black", lw=1.4, ls="--")
    ax.axhline(0, color="black", lw=0.6, ls=":")

    ax.set_xlabel(r"$1000/T$  (K$^{-1}$)", fontsize=11)
    ax.set_ylabel(r"$\Delta\kappa_S(T)$", fontsize=11)
    #ax.set_title(r"Perturbation curves $\Delta\kappa_S(T)$", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    pdf_path = sub_dir / "delta_kappa.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Plot saved → {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ─ PER-REACTION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def process_reaction(rxn_tag, rxn_data, output_root, yaml_rate_db=None,
                     n_samples=N_SAMPLES, seed=RANDOM_SEED):
    """
    Run the full Class-A generation pipeline for one reaction.

    Steps
    -----
    1.  Compute L_full (MUQ optimisation) from temperatures + uncertainties.
    2.  For every sub-configuration in M_CONFIGS:
          a.  Extract L_r via principal sub-matrix Cholesky (get_reduced_L).
          b.  Time and run sample_class_A → zeta_list.
          c.  Save zeta_samples.npy to the sub-config folder.
          d.  If YAML nominal available, also save kappa_curves.npy.
    3.  Write info.json at the reaction root with all matrix data + timings.
    """
    T            = rxn_data["temperatures"]
    uncertainties = rxn_data["uncertainties"]
    rxn_eq       = rxn_data["rxn_equation"]
    T_min, T_max = float(T[0]), float(T[-1])

    # ── Reaction output folder ────────────────────────────────────────────
    safe_name  = safe_folder_name(rxn_eq)
    rxn_folder = output_root / safe_name
    rxn_folder.mkdir(parents=True, exist_ok=True)

    rxn_info = {
        "name_tag": rxn_tag,
        "equation": rxn_eq,
        "rIndex":   rxn_data["rIndex"],
        "T_min":    T_min,
        "T_max":    T_max,
        "n_T":      int(len(T)),
    }

    # ── Optional nominal params (YAML) ────────────────────────────────────
    nominal = None
    if yaml_rate_db is not None:
        nominal = get_nominal_params(rxn_eq, yaml_rate_db)

    # ── Compute L_full ────────────────────────────────────────────────────
    print(f"  Computing L_full for: {rxn_eq[:60]}")
    t0     = time.perf_counter()
    L_full = compute_full_L(T, uncertainties)
    t_Lf   = time.perf_counter() - t0
    Sigma_full = L_full @ L_full.T
    print(f"    L_full diag: {np.diag(L_full).round(4)}  ({t_Lf:.2f}s)")

    config_records = []
    rng = np.random.default_rng(seed=seed)

    # ── Loop over all sub-configurations ─────────────────────────────────
    for m_level, folder, indices in M_CONFIGS:
        print(f"    m={m_level}  [{', '.join(PARAM_NAMES[i] for i in indices)}]",
              end="  ...", flush=True)

        # Reduced Cholesky
        Sigma_full_out, Sigma_r, L_r = get_reduced_L(L_full, indices)

        # Time the sampling
        t_start     = time.perf_counter()
        zeta_list   = sample_class_A(T, L_full, L_r, indices, n_samples=n_samples,
                                     rng=np.random.default_rng(seed=seed + hash(folder) % 9999))
        wall_time   = time.perf_counter() - t_start
        n_generated = len(zeta_list)
        print(f"  {n_generated}/{n_samples} samples  {wall_time:.4f}s")

        # Save folder structure
        sub_dir = rxn_folder / f"m{m_level}" / folder
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Save zeta_r samples  →  shape (n_generated, m)
        if n_generated > 0:
            zeta_array = np.vstack(zeta_list)          # (n_generated, m)
            np.save(sub_dir / "zeta_samples.npy", zeta_array)

            # If nominal available, also save kappa curves  →  shape (n_generated, N)
            if nominal is not None:
                kappa_nom = kappa_nominal(T, nominal)  # shape (N,)
                kappa_arr = np.vstack([
                    kappa_nom + delta_kappa(T, L_r, zr, indices)
                    for zr in zeta_list
                ])                                     # (n_generated, N)
                np.save(sub_dir / "kappa_curves.npy", kappa_arr)
                np.save(sub_dir / "temperatures.npy", T)
            
            # ── Plot all curves for this sub-config ───────────────────────
            plot_curves_for_subfolder(
                sub_dir, T, L_full, L_r, indices, zeta_list,
                rxn_eq=rxn_eq, m_level=m_level,
                folder=folder, curve_class="A"   # ← change to "B" or "C"
            )
            
            plot_curves_plain_blue(
                sub_dir, T, L_r, indices, zeta_list,
                rxn_eq=rxn_eq, m_level=m_level,
                folder=folder, curve_class="A"   # ← "B" or "C" accordingly
            )
            
            plot_delta_kappa_standalone(
                sub_dir, T, L_full, L_r, indices, zeta_list,
                rxn_eq=rxn_eq, m_level=m_level,
                folder=folder, curve_class="A"   # ← "B" or "C"
            )

        # Record for info.json
        config_records.append({
            "m":           m_level,
            "folder":      folder,
            "indices":     indices,
            "Sigma_r":     Sigma_r,
            "L_r":         L_r,
            "n_requested": n_samples,
            "n_generated": n_generated,
            "wall_time_s": wall_time,
        })

    # ── Write consolidated info.json ─────────────────────────────────────
    write_info_json(rxn_folder, rxn_info, L_full, Sigma_full, config_records)

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n    ── Timing summary for {safe_name} ──")
    print(f"    {'Config':<14} {'m':>2}  {'Indices':<12} {'Generated':>10} "
          f"{'Time (s)':>10}")
    print(f"    {'-'*52}")
    for rec in config_records:
        idx_str = "(" + ",".join(str(i) for i in rec["indices"]) + ")"
        print(f"    {rec['folder']:<14} {rec['m']:>2}  {idx_str:<12} "
              f"{rec['n_generated']:>10}  {rec['wall_time_s']:>9.4f}s")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ─ MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # ── Configuration ─────────────────────────────────────────────────────
    XML_PATH   = "MB_R_ALL_ECM_2025.xml"
    YAML_PATH  = "MB_MB2D_LALIT_2024.yaml"   # set to None to skip nominal params
    OUTPUT_DIR = Path("output_A_100")
    N_SAMP     = N_SAMPLES
    SEED       = RANDOM_SEED

    # ── Validate paths ────────────────────────────────────────────────────
    if not Path(XML_PATH).exists():
        sys.exit(f"ERROR: XML file not found: {XML_PATH}")

    yaml_rate_db = None
    if YAML_PATH is not None:
        if Path(YAML_PATH).exists():
            print(f"[+] Loading YAML mechanism: {YAML_PATH}")
            yaml_rate_db = parse_yaml_mechanism(YAML_PATH)
            print(f"    {len(yaml_rate_db)} reactions found in YAML")
        else:
            print(f"[!] YAML not found ({YAML_PATH}); skipping nominal/kappa output")

    print(f"[+] Parsing XML: {XML_PATH}")
    rxn_dict = parse_xml_uncertainty(XML_PATH)
    print(f"    {len(rxn_dict)} reactions found in XML\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Process each reaction ─────────────────────────────────────────────
    for rxn_tag, rxn_data in rxn_dict.items():
        print(f"{'='*65}")
        print(f"  Reaction: {rxn_tag}")
        print(f"{'='*65}")
        try:
            process_reaction(
                rxn_tag, rxn_data,
                output_root=OUTPUT_DIR,
                yaml_rate_db=yaml_rate_db,
                n_samples=N_SAMP,
                seed=SEED,
            )
        except Exception as exc:
            print(f"  [ERROR] Skipping {rxn_tag}: {exc}")
        raise AssertionError("Need only one reaction for the analysis!")

    print("\n[✓] Class-A generation complete.")
    print(f"    Output written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()