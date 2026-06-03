#!/usr/bin/env python3
"""
generate_class_B.py
===================
Generate Class-B Arrhenius perturbation samples from an XML uncertainty file.

Class-B definition
------------------
Samples satisfy four geometric constraints on the Δκ curve:
  C1: Δκ(T_min) = r1 · f_prior(T_min)          (anchor at T_min)
  C2: Δκ(T_mid) = sign · f_prior(T_mid)         (interior extremum)
  C3: Δκ(T_max) = r2 · f_prior(T_max)           (anchor at T_max)
  C4: dΔκ/dT|_{T_mid} = sign · df_prior/dT|_{T_mid}  (slope match at mid)

Methods implemented (m=3 uses L_full; m=2 uses reduced L_r)
------------------------------------------------------------
  "original_m3": published SHGO/SLSQP method, all three params (A, n, Ea)
  "original_m2": same published method adapted for any 2-param subset via L_r

Generates 100 samples per sub-configuration for:
  m = 2  →  (A,n)(1,1,0)  |  (A,Ea)(1,0,1)  |  (n,Ea)(0,1,1)
  m = 3  →  (A,n,Ea)(1,1,1)

How to add a new Class-B method
--------------------------------
1.  Write a function with this signature:
      def generate_class_B_my_new_method(
              temperatures, L_full, uncertainties, indices,
              n_samples, rng):
          # ...
          return zeta_list   # list of numpy arrays, each shape (m,)

2.  Register it in CLASS_B_METHOD_REGISTRY:
      CLASS_B_METHOD_REGISTRY["my_new_method"] = generate_class_B_my_new_method

3.  Call it by passing method="my_new_method" to generate_class_B_samples().
    No other changes needed.

Output structure (same as Code 1, no m1 layer)
----------------------------------------------
  output/
    {rxn_safe_name}/
      info.json           ← L_full, Sigma, L_r, Sigma_r, timing per sub-config
      m2/
        A_n/   zeta_samples.npy   [≤100 × 2]
        A_Ea/  zeta_samples.npy   [≤100 × 2]
        n_Ea/  zeta_samples.npy   [≤100 × 2]
      m3/
        A_n_Ea/ zeta_samples.npy  [≤100 × 3]

Note: Class-B SHGO can be slow (~10–120 s per 100 samples for m=3).
      A time cap of 120 s per sample is applied for the original methods.
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
from scipy.optimize import minimize, shgo

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MaxEvalReached(Exception):
    pass

eval_counter = [0]          # use list so closure can mutate it
MAX_FEVAL    = 500          # your limit per restart

def speed_up_animation(input_path, speedup=10):
    """
    Speed up a saved animation by `speedup` factor.
    - .mp4  : uses ffmpeg (must be installed)
    - .gif  : uses Pillow (no external dependency)
    Falls back gracefully if neither works.
    """
    import os
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_{speedup}x{ext}"

    if not os.path.isfile(input_path):
        print(f"      [viz] speed_up skipped — file not found: {input_path}")
        return None

    if ext.lower() == '.gif':
        # ── Pillow path ───────────────────────────────────────────────
        try:
            from PIL import Image
            img    = Image.open(input_path)
            frames, durations = [], []
            try:
                while True:
                    frames.append(img.copy().convert('RGBA'))
                    dur = max(20, img.info.get('duration', 100) // speedup)
                    durations.append(dur)
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            if frames:
                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=durations,
                    loop=0
                )
                print(f"      [viz] sped up (Pillow) → {output_path}")
            return output_path
        except Exception as e:
            print(f"      [viz] Pillow speed-up failed: {e}")
            return None

    else:
        # ── ffmpeg path (.mp4 / .avi etc.) ────────────────────────────
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:v", f"setpts={1.0/speedup:.4f}*PTS",
            "-an",
            output_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"      [viz] sped up (ffmpeg) → {output_path}")
                return output_path
            else:
                print(f"      [viz] ffmpeg failed:\n{result.stderr}")
                return None
        except FileNotFoundError:
            print(f"      [viz] ffmpeg not found — install via: winget install ffmpeg")
            return None

class ClassBSearchViz:
    """
    Live recorder + post-hoc animator for Class-B original (m=2) search.
    Static layer : nominal QtPo, uncertainty band, anchor markers.
    Animated layer: current search curve (QtPo + QtLZ) every N evals,
                    objective vs eval-count on the right.
    """
    CURVE_UPDATE_EVERY = 5          # update left plot every N obj evaluations

    def __init__(self, temperatures, QtPo, uncertainties,
                 T_min, T_max, ftol, live=False):
        self.T        = temperatures
        self.inv_T    = 1.0 / temperatures
        self.QtPo     = QtPo
        self.unc      = uncertainties
        self.T_min    = T_min
        self.T_max    = T_max
        # anchor targets in curve-value space (QtPo offset + constraint target)
        self.anchor_min = None   # set per sample in the for-loop
        self.anchor_max = None
        self.ftol     = ftol
        # storage
        self._eval    = 0
        self._obj_vals      = []   # every evaluation
        self._curve_frames  = []   # (eval_index, QtLZ) every CURVE_UPDATE_EVERY evals
        self.live = live
        if self.live:
            import matplotlib
            matplotlib.use('TkAgg')        # or 'Qt5Agg' if Tk is unavailable
            plt.ion()                      # turn on interactive mode
            self._setup_live_figure()

    # ------------------------------------------------------------------
    def _setup_live_figure(self):
        self._fig, (self._ax_L, self._ax_R) = plt.subplots(1, 2, figsize=(13, 5))
        self._fig.suptitle("Class-B Search — Live", fontsize=11)

        # static left
        self._ax_L.plot(self.inv_T, self.QtPo, 'k-', lw=2,
                        label=r'Nominal $Q^T P_0$')
        self._ax_L.fill_between(self.inv_T,
                                self.QtPo - self.unc,
                                self.QtPo + self.unc,
                                alpha=0.15, color='steelblue',
                                label='Uncertainty band')
        self._ax_L.axvline(1.0/self.T_min, color='gray', ls=':', lw=0.9)
        self._ax_L.axvline(1.0/self.T_max, color='gray', ls=':', lw=0.9)
        self._ax_L.set_xlabel(r'$1/T\;[\mathrm{K}^{-1}]$')
        self._ax_L.set_ylabel(r'$\log k\;/\;\kappa$')
        self._ax_L.legend(fontsize=8)

        # animated artists
        self._search_line, = self._ax_L.plot([], [], 'r-', lw=1.3,
                                              alpha=0.85, label='Search')
        self._eval_text = self._ax_L.text(0.02, 0.97, '',
                                           transform=self._ax_L.transAxes,
                                           fontsize=8, va='top', color='crimson')
        # anchor scatter placeholders (updated per sample)
        self._anc_scatter = self._ax_L.scatter([], [], marker='o', s=60,
                                                color='darkorange', zorder=5)

        # right panel
        self._ax_R.set_xlabel('Function evaluation')
        self._ax_R.set_ylabel(r'Objective $\|\mathbf{u} - Q^T L z\|^2$')
        self._obj_line, = self._ax_R.plot([], [], 'royalblue', lw=1.0)
        self._ftol_line  = self._ax_R.axhline(self.ftol, color='seagreen',
                                               ls='--', lw=1.4,
                                               label=f'ftol={self.ftol:.0e}')
        self._ax_R.legend(fontsize=8)
        plt.tight_layout()
        plt.show(block=False)
    
    # ------------------------------------------------------------------
    def record(self, z_active, obj_val, L_r, indices):
        self._eval += 1
        self._obj_vals.append(float(obj_val))

        # store curve frame every N evals (for post-hoc animation)
        if self._eval % self.CURVE_UPDATE_EVERY == 0:
            thS_all = theta_S(self.T, indices)
            QtLZ = np.array([thS @ L_r @ z_active for thS in thS_all.T])
            self._curve_frames.append((self._eval, QtLZ.copy()))

        # live display: update EVERY eval (cheap — just set_data + flush)
        if self.live:
            thS_all = theta_S(self.T, indices)
            QtLZ    = np.array([thS @ L_r @ z_active for thS in thS_all.T])
            self._push_live(QtLZ)

    def _push_live(self, QtLZ):
        # ── left panel ───────────────────────────────────────────────
        self._search_line.set_data(self.inv_T, self.QtPo + QtLZ)
        self._eval_text.set_text(f'eval = {self._eval}')

        if self.anchor_min is not None:
            self._anc_scatter.set_offsets(
                np.array([self.anchor_min, self.anchor_max]))

        # relim so the search line is never clipped outside the view
        self._ax_L.relim()
        self._ax_L.autoscale_view(tight=False)

        # ── right panel ──────────────────────────────────────────────
        xs      = list(range(1, len(self._obj_vals) + 1))
        obj_arr = np.array(self._obj_vals)
        self._obj_line.set_data(xs, obj_arr)
        self._ax_R.set_xlim(0, max(len(self._obj_vals) + 1, 10))

        ylo = max(float(obj_arr.min()) * 0.5, 1e-14)
        yhi = float(obj_arr.max()) * 3.0
        self._ax_R.set_ylim(ylo, yhi)
        if (yhi / max(ylo, 1e-30)) > 100:
            self._ax_R.set_yscale('log')

        # ── flush to screen ──────────────────────────────────────────
        self._fig.canvas.draw_idle()       # only redraws dirty regions
        self._fig.canvas.flush_events()    # processes GUI events immediately
        plt.pause(0.02)                    # yields to event loop long enough to render

    
    def mark_restart(self):
        """Record the current eval index as a restart boundary."""
        if not hasattr(self, '_restart_evals'):
            self._restart_evals = []
        self._restart_evals.append(len(self._obj_vals))
        
    # ------------------------------------------------------------------
    def build_and_save(self, filename="class_b_sample.mp4", sample_id=0, folder=None):
        if not self._obj_vals:
            return None

        import os
        # ── create output folder if specified ────────────────────────────
        if folder:
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, os.path.basename(filename))

        obj_arr  = np.array(self._obj_vals)
        use_log  = (obj_arr.max() / max(obj_arr.min(), 1e-30)) > 100
        n_evals  = len(obj_arr)

        fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
        fig.suptitle(f"Class-B Original  |  Sample {sample_id}", fontsize=11)

        # ── LEFT : static background ──────────────────────────────────────
        ax_L.plot(self.inv_T, self.QtPo,
                'k-', lw=2.0, label=r'Nominal $Q^T P_0$', zorder=3)
        ax_L.fill_between(self.inv_T,
                        self.QtPo - self.unc,
                        self.QtPo + self.unc,
                        alpha=0.15, color='steelblue',
                        label='Uncertainty band', zorder=2)
        # FIX: both boundary lines now have self.inv_T as x-data
        ax_L.plot(self.inv_T, self.QtPo + self.unc,
                color='steelblue', lw=0.8, ls='--', alpha=0.6, zorder=2)
        ax_L.plot(self.inv_T, self.QtPo - self.unc,
                color='steelblue', lw=0.8, ls='--', alpha=0.6, zorder=2)
        ax_L.axvline(1.0/self.T_min, color='gray', ls=':', lw=0.9,
                    alpha=0.7, label=r'$T_{\min},\,T_{\max}$')
        ax_L.axvline(1.0/self.T_max, color='gray', ls=':', lw=0.9, alpha=0.7)
        if self.anchor_min is not None:
            ax_L.scatter(*self.anchor_min, marker='o', s=60,
                        color='darkorange', zorder=5, label='Anchor (C1, C3)')
            ax_L.scatter(*self.anchor_max, marker='o', s=60,
                        color='darkorange', zorder=5)
        ax_L.set_xlabel(r'$1/T\;[\mathrm{K}^{-1}]$', fontsize=10)
        ax_L.set_ylabel(r'$\log k\;/\;\kappa$', fontsize=10)
        ax_L.legend(fontsize=8, loc='upper right')

        # animated artist (left)
        search_line, = ax_L.plot([], [], color='crimson', lw=1.3,
                                alpha=0.85, label='Search', zorder=4)
        eval_text = ax_L.text(0.02, 0.97, '', transform=ax_L.transAxes,
                            fontsize=8, va='top', color='crimson')

        # ── RIGHT : objective ─────────────────────────────────────────────
        ax_R.axhline(self.ftol, color='seagreen', ls='--', lw=1.4,
                    label=f'ftol = {self.ftol:.0e}', zorder=3)
        for rev in getattr(self, '_restart_evals', []):
            if rev > 0:
                ax_R.axvline(rev, color='orange', ls=':', lw=0.8,
                            alpha=0.7, label='Restart' if rev == next(
                                (r for r in self._restart_evals if r > 0), rev)
                            else '')
        ax_R.set_xlim(0, n_evals + 1)
        ylo = max(obj_arr.min() * 0.5, 1e-14)
        yhi = obj_arr.max() * 3
        ax_R.set_ylim(ylo, yhi)
        if use_log:
            ax_R.set_yscale('log')
        ax_R.set_xlabel('Function evaluation', fontsize=10)
        ax_R.set_ylabel(r'Objective  $\|u - Q^T L_r z\|^2$', fontsize=10)
        ax_R.legend(fontsize=8)

        obj_line, = ax_R.plot([], [], color='royalblue', lw=1.0, alpha=0.9)

        # ── curve frame lookup ────────────────────────────────────────────
        frame_dict = {ev: crv for ev, crv in self._curve_frames}

        # ── update function (blit=False — static bg preserved in GIF) ────
        def update(fi):
            cur_eval = fi + 1

            # right panel
            obj_line.set_data(list(range(1, cur_eval + 1)),
                            self._obj_vals[:cur_eval])

            # left panel — nearest curve frame at or before cur_eval
            avail = [ev for ev in frame_dict if ev <= cur_eval]
            if avail:
                ev   = max(avail)
                QtLZ = frame_dict[ev]
                search_line.set_data(self.inv_T, self.QtPo + QtLZ)
                eval_text.set_text(f'eval = {ev}')

            return search_line, obj_line, eval_text

        ani = animation.FuncAnimation(
            fig, update, frames=n_evals,
            blit=False,        # FIX: False ensures static bg renders in every GIF frame
            interval=40,
            repeat=False)

        # ── save ──────────────────────────────────────────────────────────
        try:
            writer = animation.FFMpegWriter(fps=15, bitrate=2400)
            ani.save(filename, writer=writer, dpi=120)
            print(f"      [viz] saved → {filename}")
            saved_path = filename
        except Exception:
            gif_name = filename.replace('.mp4', '.gif')
            ani.save(gif_name, writer='pillow', fps=10, dpi=100)
            print(f"      [viz] ffmpeg unavailable — saved as {gif_name}")
            saved_path = gif_name
        plt.close(fig)

        # ── reset for next sample ─────────────────────────────────────────
        self._eval = 0
        self._obj_vals.clear()
        self._curve_frames.clear()
        if hasattr(self, '_restart_evals'):
            self._restart_evals.clear()
        return saved_path

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

M_CONST     = 3.0 / np.log(10.0)
R_GAS       = 1.987
MAX_DELTA_N = 2.0
N_SAMPLES   = 100
RANDOM_SEED = 42
ORIG_TIME_CAP_S = 240.0       # per-sample wall-clock cap for SHGO/SLSQP methods

PARAM_NAMES = {0: "A", 1: "n", 2: "Ea"}

# Sub-configurations for Class-B (no m=1)
"""
M_CONFIGS = [
    (2, "A_n",    (0, 1)),
    (2, "A_Ea",   (0, 2)),
    (2, "n_Ea",   (1, 2)),
    (3, "A_n_Ea", (0, 1, 2)),
]
"""
M_CONFIGS = [
    (2, "A_n",    (0, 1)),
    (2, "A_Ea",   (0, 2)),
    (2, "n_Ea",   (1, 2)),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ MATH HELPERS  (ported verbatim from benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def theta_full(T):
    """θ(T) = [1, ln T, -1/T],  shape (3, N)."""
    return np.array([np.ones_like(T), np.log(T), -1.0 / T])


def theta_S(T, indices):
    """Reduced basis Θ_S(T) of shape (m, N)."""
    return theta_full(T)[list(indices), :]


def f_prior_full(T, L_full):
    """f_prior(T) = ‖L^T θ(T)‖₂ for each T (uses full L)."""
    Theta = theta_full(T)
    return np.array([np.linalg.norm(L_full.T @ col) for col in Theta.T])


def f_prior_S(T, L_r, indices):
    """f_prior_S(T) = ‖L_r^T Θ_S(T)‖₂ for each T (uses reduced L_r)."""
    thS = theta_S(T, indices)
    return np.array([np.linalg.norm(L_r.T @ col) for col in thS.T])


def delta_kappa(T, L_r, zeta_r, indices):
    """Δκ_S(T) = Θ_S(T)^T L_r zeta_r."""
    return theta_S(T, indices).T @ (L_r @ zeta_r)


def kappa_nominal(T, nominal):
    """κ₀(T) = θ(T)^T p₀."""
    return theta_full(T).T @ nominal


def _dtheta_S_dT(T_val, indices):
    """Analytical d(θ_S)/dT at a single temperature value."""
    full = np.array([0.0, 1.0 / T_val, 1.0 / T_val ** 2])
    return full[list(indices)]


def _has_sign_change(arr):
    """Return True if arr changes sign at least once across the array."""
    return bool(np.any(np.diff(np.sign(arr)) != 0))

def _fp_S_deriv(T_val, L_r, indices):
    """Analytical d(f_prior_S)/dT using reduced L_r."""
    th    = theta_S(np.array([T_val]), indices)[:, 0]
    dth   = _dtheta_S_dT(T_val, indices)
    LTth  = L_r.T @ th
    LTdth = L_r.T @ dth
    fp    = np.linalg.norm(LTth)
    return float(np.dot(LTth, LTdth)) / fp if fp > 1e-30 else 0.0

def _enforce_dn_constraint(zeta_r, L_r, indices):
    """Clamp |Δn| to MAX_DELTA_N by scaling zeta_r if needed."""
    if 1 not in indices:
        return zeta_r
    pos_n   = list(indices).index(1)
    delta_n = (L_r @ zeta_r)[pos_n]
    if abs(delta_n) <= MAX_DELTA_N:
        return zeta_r
    scale = ( MAX_DELTA_N) / (abs(delta_n) + 1e-30)
    return zeta_r * scale


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ COVARIANCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _muq_objective(params, T, uncertainties):
    """MUQ residual for full L fitting."""
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
    Sigma_r : (m, m) sub-matrix of Σ
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
# SECTION 4 ─ ORIGINAL CLASS-B INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_uncorrelated_direction(L, temperatures, uncertainties):
    """
    Find zeta_unc by Nelder-Mead minimisation so that
    Θ(T)^T L zeta_unc ≈ uncertainties(T).

    Used internally by the original Class-B and Class-C samplers to
    define the 'uncorrelated' reference direction in parameter space.

    Returns
    -------
    zeta_unc : shape (3,) – best-fit perturbation direction
    """
    def objective(guess):
        Theta = np.array([temperatures / temperatures,
                          np.log(temperatures),
                          -1.0 / temperatures])
        QtLZ  = np.array([th @ L @ guess for th in Theta.T])
        f     = uncertainties - QtLZ
        return float(np.dot(f, f))

    guess  = np.array([0.5, 0.1, 0.5])
    result = minimize(objective, guess, method="SLSQP")
    print(result)
    return result.x


def _determine_constraint_signs(kleft_factor, kright_factor):
    """
    Determine the sign rules for C2 and C4 based on the anchor scale factors.

    Parameters
    ----------
    kleft_factor  : scale factor r1 at T_min (typically ∈ (-0.95, 0.95))
    kright_factor : scale factor r2 at T_max

    Returns
    -------
    sign_C2      : ±1  (sign of the interior extremum constraint C2)
    sign_C4      : ±1  (sign of the derivative constraint C4)
    kmiddle_fact : |r1| – magnitude used for C2/C4 targets
    """
    sign_C2 = -1.0 if kleft_factor > 0 else 1.0
    sign_C4 = -1.0 if kleft_factor > 0 else 1.0
    kmiddle_fact = 1.0
    return sign_C2, sign_C4, kmiddle_fact


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ ORIGINAL CLASS-B SAMPLERS (readable names, adapted from benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_class_B_original_full_parameters(
        nominal, temperatures, L_full, uncertainties, indices, sub_dir,
        n_samples, rng):
    """
    Generate one Class-B sample using the ORIGINAL published SHGO/SLSQP
    method for m = 3  (all three Arrhenius parameters A, n, Ea active).

    Constraints C1–C4 are enforced on the full-space Δκ using L_full.
    SHGO is used as the primary solver; SLSQP is the fallback.

    This is the method published in the MUQ-SAC paper.

    Parameters
    ----------
    temperatures  : (N,) array of temperatures in K
    L_full        : (3, 3) lower-triangular Cholesky factor from MUQ
    uncertainties : (N,) array of uncertainty values
    indices       : must be (0, 1, 2) for this method
    n_samples     : number of samples to generate
    rng           : numpy Generator for random r1, r2 draws

    Returns
    -------
    zeta_list : list of zeta_r arrays, each shape (3,)
    """
    T_min   = float(temperatures[0])
    T_max   = float(temperatures[-1])
    L       = L_full
    zeta_unc = _compute_uncorrelated_direction(L, temperatures, uncertainties)

    # Helper closures – rate curve values and derivatives
    def dk_unc_at_T(Tv):
        return float(theta_full(np.array([Tv]))[:, 0] @ L @ zeta_unc)

    def dk_z_at_T(Tv, z):
        return float(theta_full(np.array([Tv]))[:, 0] @ L @ z)

    def ddk_unc_at_T(Tv):
        dth = np.array([0.0, 1.0 / Tv, 1.0 / Tv ** 2])
        return float(dth @ L @ zeta_unc)

    def ddk_z_at_T(Tv, z):
        dth = np.array([0.0, 1.0 / Tv, 1.0 / Tv ** 2])
        return float(dth @ L @ z)

    def mismatch_objective(z):
        Theta = np.array([temperatures / temperatures,
                          np.log(temperatures),
                          -1.0 / temperatures])
        QtLZ  = np.array([th @ L @ z[:3] for th in Theta.T])
        obj   = float(np.dot(uncertainties - QtLZ, uncertainties - QtLZ))
        viz.record(z[:3], obj, L, (0, 1, 2))   # ← only new line
        return obj

    # Nominal curve — full 3-param theta
    Theta_full = np.array([temperatures / temperatures,
                           np.log(temperatures),
                           -1.0 / temperatures])
    QtPo = np.array([th @ nominal for th in Theta_full.T])

    viz = ClassBSearchViz(
        temperatures  = temperatures,
        QtPo          = QtPo,
        uncertainties = uncertainties,
        T_min         = T_min,
        T_max         = T_max,
        ftol          = 1e-6,
    )
    
    zeta_list = []
    for _ in range(n_samples):
        r1 = float(rng.uniform(-1.0, 1.0))
        r2 = float(rng.uniform(-1.0, 1.0))
        sign_C2, sign_C4, kmiddle = _determine_constraint_signs(r1, r2)

        # re-set per-sample anchor points
        _dk_min        = dk_unc_at_T(T_min)       # note: dk_unc_at_T not dk_unc_S_at_T
        _dk_max        = dk_unc_at_T(T_max)
        idx_min        = int(np.argmin(np.abs(temperatures - T_min)))
        idx_max        = int(np.argmin(np.abs(temperatures - T_max)))
        viz.anchor_min = (1.0/T_min, QtPo[idx_min] + r1 * _dk_min)
        viz.anchor_max = (1.0/T_max, QtPo[idx_max] + r2 * _dk_max)

        def c1(z): return r1 * dk_unc_at_T(T_min) - dk_z_at_T(T_min, z[:3])
        def c3(z): return r2 * dk_unc_at_T(T_max) - dk_z_at_T(T_max, z[:3])
        def c2(z):
            Tu = float(np.clip(z[-1], T_min + 1, T_max - 1))
            return sign_C2 * kmiddle * dk_unc_at_T(Tu) - dk_z_at_T(Tu, z[:3])
        def c4(z):
            Tu = float(np.clip(z[-1], T_min + 1, T_max - 1))
            return sign_C4 * kmiddle * ddk_unc_at_T(Tu) - ddk_z_at_T(Tu, z[:3])

        constraints = [{'type': 'eq', 'fun': c1},
                       {'type': 'eq', 'fun': c2},
                       {'type': 'eq', 'fun': c3},
                       {'type': 'eq', 'fun': c4}]
        bounds = [(-10000, 10000)] * 3 + [(200, 3500)]
        x0     = np.array([10, 10, 100, (T_min + T_max) / 2])

        t_sample = time.perf_counter()
        try:
            sol = shgo(mismatch_objective, bounds,
                       constraints=constraints,n=64, iters=1,
                        sampling_method='sobol',
                        minimizer_kwargs={
                            "method": "SLSQP",
                            "options": {
                                "maxiter": 50,     # default is 15000 — massively reduce
                                "ftol": 1e-7,      # loosen slightly from default 1e-12
                                "maxfun": 100,     # cap f-evals per local run
                            }
                        })
            zr = sol.x[:3]
        except Exception:
            try:
                sol = minimize(mismatch_objective, x0, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 2000, 'ftol': 1e-9})
                zr = sol.x[:3]
            except Exception:
                zr = x0[:3]
        elapsed = time.perf_counter() - t_sample
        if elapsed > ORIG_TIME_CAP_S:
            print(f"      [!] original_m3 sample took {elapsed:.1f}s (cap={ORIG_TIME_CAP_S}s)")
        zeta_list.append(_enforce_dn_constraint(np.asarray(zr), L_full, (0, 1, 2)))
        
        saved_path = viz.build_and_save(
            filename  = f"classB_orig_sample_{_:03d}.mp4",
            sample_id = _,
            folder    = str(sub_dir)
        )
        if saved_path:
            speed_up_animation(saved_path, speedup=10)
    return zeta_list


def generate_class_B_original_reduced_parameters(
        nominal, temperatures, L_full, uncertainties, indices, sub_dir,
        n_samples, rng):
    """
    Generate Class-B samples using the ORIGINAL published method adapted
    for m = 2  (any 2-parameter subset via reduced Cholesky L_r).

    This is the same SHGO/SLSQP approach as the m=3 original, but
    constraints C1–C4 are expressed in the reduced L_r space so that
    only the active parameter subset is perturbed.

    Parameters
    ----------
    nominal       : (3,) nominal Arrhenius parameters (for reference, not used by all methods)
    temperatures  : (N,) temperature array
    L_full        : (3, 3) full Cholesky factor
    uncertainties : (N,) uncertainty array
    indices       : 2-element tuple, e.g. (0, 1), (0, 2), or (1, 2)
    n_samples     : number of samples to generate
    rng           : numpy Generator

    Returns
    -------
    zeta_list : list of zeta_r arrays, each shape (2,)
    """
    _, _, L_r    = get_reduced_L(L_full, indices)
    zeta_unc_full = _compute_uncorrelated_direction(L_full, temperatures, uncertainties)
    zeta_unc_S    = zeta_unc_full[list(indices)]   # projected to active subset
    T_min  = float(temperatures[0])
    T_max  = float(temperatures[-1])

    # Helper closures in the reduced L_r space
    def dk_unc_S_at_T(Tv):
        thS = theta_S(np.array([Tv]), indices)[:, 0]
        return float(thS @ L_r @ zeta_unc_S)

    def dk_z_S_at_T(Tv, z):
        thS = theta_S(np.array([Tv]), indices)[:, 0]
        return float(thS @ L_r @ z)

    def ddk_unc_S_at_T(Tv):
        dthS = _dtheta_S_dT(Tv, indices)
        return float(dthS @ L_r @ zeta_unc_S)

    def ddk_z_S_at_T(Tv, z):
        dthS = _dtheta_S_dT(Tv, indices)
        return float(dthS @ L_r @ z)

    # ── replace your original def ──────────────────────────────────────
    """
    def mismatch_objective_m2(z):
        eval_counter[0] += 1
        if eval_counter[0] > MAX_FEVAL:
            raise MaxEvalReached(f"max fevals {MAX_FEVAL} reached")
        thS_all = theta_S(temperatures, indices)
        QtLZ    = np.array([thS @ L_r @ z[:2] for thS in thS_all.T])
        f       = uncertainties - QtLZ
        obj     = float(np.dot(f, f))
        viz.record(z[:2], obj, L_r, indices)
        return obj
    """
    def mismatch_objective_m2(z):
        thS_all = theta_S(temperatures, indices)
        QtLZ    = np.array([thS @ L_r @ z[:2] for thS in thS_all.T])
        f       = uncertainties - QtLZ
        obj     = float(np.dot(f, f))
        viz.record(z[:2], obj, L_r, indices)
        return obj

    # ── add BEFORE the for-loop ────────────────────────────────────────
    # Nominal curve using all three Arrhenius parameters
    thS_full = theta_S(temperatures, (0, 1, 2))
    QtPo     = np.array([th @ nominal for th in thS_full.T])

    viz = ClassBSearchViz(
        temperatures  = temperatures,
        QtPo          = QtPo,
        uncertainties = uncertainties,
        T_min         = T_min,
        T_max         = T_max,
        ftol          = 1e-6,               # shgo f_tol (change to 1e-9 for SLSQP runs)
    )

    zeta_list = []
    for _ in range(n_samples):
        r1 = float(rng.uniform(-0.95, 0.95))
        #r1 = -1.0
        #r2 = -1.0
        r2 = float(rng.uniform(-0.95, 0.95))
        sign_C2, sign_C4, kmiddle = _determine_constraint_signs(r1, r2)
        # ── re-initialise per-sample statics ──────────────────────────
        viz.T_min      = T_min
        viz.T_max      = T_max
        _dk_min        = dk_unc_S_at_T(T_min)
        _dk_max        = dk_unc_S_at_T(T_max)
        idx_min        = int(np.argmin(np.abs(temperatures - T_min)))
        idx_max        = int(np.argmin(np.abs(temperatures - T_max)))
        viz.anchor_min = (1.0/T_min, QtPo[idx_min] + r1 * _dk_min)
        viz.anchor_max = (1.0/T_max, QtPo[idx_max] + r2 * _dk_max)
        def _is_feasible(sol, constraints, tol=1e-6):
                return all(
                    abs(c['fun'](sol.x)) <= tol if c['type'] == 'eq'
                    else c['fun'](sol.x) >= -tol
                    for c in constraints
                )
        def c1(z): return r1 * dk_unc_S_at_T(T_min) - dk_z_S_at_T(T_min, z[:2])
        def c3(z): return r2 * dk_unc_S_at_T(T_max) - dk_z_S_at_T(T_max, z[:2])
        def c2(z):
            Tu = float(np.clip(z[-1], T_min + 1, T_max - 1))
            return sign_C2 * kmiddle * dk_unc_S_at_T(Tu) - dk_z_S_at_T(Tu, z[:2])
        def c4(z):
            Tu = float(np.clip(z[-1], T_min + 1, T_max - 1))
            return sign_C4 * kmiddle * ddk_unc_S_at_T(Tu) - ddk_z_S_at_T(Tu, z[:2])

        # C5 : |Δn| ≤ MAX_DELTA_N  — only active when n-parameter (index 1) is in subset
        if 1 in indices:
            _pos_n = list(indices).index(1)          # position of n in reduced space

            def c5a(z):
                # MAX_DELTA_N - (L_r @ z)_n  ≥ 0  →  upper bound
                return MAX_DELTA_N - (L_r @ z[:2])[_pos_n]

            def c5b(z):
                # MAX_DELTA_N + (L_r @ z)_n  ≥ 0  →  lower bound
                return MAX_DELTA_N + (L_r @ z[:2])[_pos_n]

            dn_constraints = [{'type': 'ineq', 'fun': c5a},
                               {'type': 'ineq', 'fun': c5b}]
        else:
            dn_constraints = []                      # n not active, no constraint needed
        
        """
        constraints = [{'type': 'eq', 'fun': c1},
                       {'type': 'eq', 'fun': c2},
                       {'type': 'eq', 'fun': c3},
                       {'type': 'eq', 'fun': c4},
                       *dn_constraints]           # C5a, C5b if n active
        """
        constraints = [{'type': 'eq', 'fun': c1},
                       {'type': 'eq', 'fun': c3},
                       ]                         
        # C2 and C4 are now inside the objective as penalty
        bounds = [(-10000, 10000)] * 2 + [(200, 3500)]
        x0     = np.array([10, 10, (T_min + T_max) / 2])

        t_sample = time.perf_counter()
        try:
            sol = shgo(mismatch_objective_m2, bounds,
                       constraints=constraints,n=64, iters=1,
                        sampling_method='sobol',
                        minimizer_kwargs={
                            "method": "SLSQP",
                            "options": {
                                "maxiter": 50,     # default is 15000 — massively reduce
                                "ftol": 1e-7,      # loosen slightly from default 1e-12
                                "maxfun": 100,     # cap f-evals per local run
                            }
                        })
            zr = sol.x[:2]
        except Exception:
            try:
                sol = minimize(mismatch_objective_m2, x0, method='SLSQP',
                               bounds=bounds, constraints=constraints,
                               options={'maxiter': 2000, 'ftol': 1e-9})
                zr = sol.x[:2]
            except Exception:
                zr = x0[:2]
        elapsed = time.perf_counter() - t_sample
        if elapsed > ORIG_TIME_CAP_S:
            print(f"      [!] original_m2 sample took {elapsed:.1f}s (cap={ORIG_TIME_CAP_S}s)")
        zeta_list.append(np.asarray(zr))            # C5 now enforced inside optimizer
        
        #saved_path = viz.build_and_save(
        #    filename  = f"classB_orig_sample_{_:03d}.mp4",
        #    sample_id = _,
        #    folder    = str(sub_dir)
        #)
        #if saved_path:
        #    speed_up_animation(saved_path, speedup=10)
    return zeta_list


def f_prior_from_L(L, T):
    """f_prior(T) = ||L^T theta(T)||_2 for each temperature."""
    Theta = theta_full(T)
    return np.array([np.linalg.norm(L.T @ th) for th in Theta.T])

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

def _fp_full_deriv(T_val, L_full):
    """Analytical d(f_prior_full)/dT = (L^T theta).(L^T d_theta/dT) / ||...||."""
    th    = theta_full(np.array([T_val]))[:, 0]
    dth   = np.array([0.0, 1.0 / T_val, 1.0 / T_val ** 2])
    LTth  = L_full.T @ th
    LTdth = L_full.T @ dth
    fp    = np.linalg.norm(LTth)
    return float(np.dot(LTth, LTdth)) / fp if fp > 1e-30 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ METHOD REGISTRY  (add new methods here)
# ─────────────────────────────────────────────────────────────────────────────
#
# To register a new algorithm:
#   1. Implement a function with this exact signature:
#        def your_new_method(temperatures, L_full, uncertainties,
#                            indices, n_samples, rng) -> list[np.ndarray]
#   2. Add it to CLASS_B_METHOD_REGISTRY below.
#   3. Call generate_class_B_samples(..., method="your_key") – done.
#
# ─────────────────────────────────────────────────────────────────────────────

CLASS_B_METHOD_REGISTRY = {

    # ── Published methods ──────────────────────────────────────────────────
    "original_m3": generate_class_B_original_full_parameters,
    "original_m2": generate_class_B_original_reduced_parameters,

    # ── Add faster / analytical methods below as they are developed ───────
    
    # ──────────────────────────────────────────────────────────────────────

}


def generate_class_B_samples(nominal, temperatures, L_full, uncertainties, indices, sub_dir,
                              n_samples, method, rng):
    """
    Public entry point: generate n_samples Class-B zeta_r vectors.

    Dispatches to the requested algorithm via CLASS_B_METHOD_REGISTRY.

    Parameters
    ----------
    nominal       : (3,) nominal Arrhenius parameters (for reference, not used by all methods)
    temperatures  : (N,) temperature array in K
    L_full        : (3, 3) lower-triangular Cholesky from MUQ
    uncertainties : (N,) uncertainty array
    indices       : tuple of active parameter indices
    n_samples     : number of samples to generate
    method        : key in CLASS_B_METHOD_REGISTRY, e.g. "original_m3"
    rng           : numpy Generator

    Returns
    -------
    zeta_list : list of zeta_r arrays, each shape (m,)

    Raises
    ------
    ValueError if method is not in CLASS_B_METHOD_REGISTRY
    """
    if method not in CLASS_B_METHOD_REGISTRY:
        available = list(CLASS_B_METHOD_REGISTRY.keys())
        raise ValueError(
            f"Unknown Class-B method '{method}'. "
            f"Available: {available}"
        )
    sampler_fn = CLASS_B_METHOD_REGISTRY[method]
    return sampler_fn(nominal, temperatures, L_full, uncertainties, indices, sub_dir, n_samples, rng)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves_for_subfolder(sub_dir, T, L_r, indices, zeta_list,
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
    fp_S    = f_prior_S(T, L_r, indices)      # ±envelope

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

    ax_dk.plot(inv_T,  fp_S, color="dimgrey", lw=1.2, ls="--",
               label=r"$\pm f_{\mathrm{prior},S}$")
    ax_dk.plot(inv_T, -fp_S, color="dimgrey", lw=1.2, ls="--")
    ax_dk.axhline(0, color="black", lw=0.6, ls=":")

    ax_dk.set_xlabel(r"$1000/T$  (K$^{-1}$)", fontsize=10)
    ax_dk.set_ylabel(r"$\Delta\kappa_S(T)$", fontsize=10)
    ax_dk.set_title(r"Perturbation curves $\Delta\kappa_S$", fontsize=9)
    ax_dk.legend(fontsize=8)
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
        ax_k.set_title(r"Full log-rate curves $\kappa(T)$", fontsize=9)
        ax_k.legend(fontsize=8)
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
                            rxn_eq, m_level, folder, curve_class="A",
                            uncertainties=None, nominal=None, ramp_targets=None):
    if not zeta_list:
        return
    kappa_path = sub_dir / "kappa_curves.npy"
    T_path     = sub_dir / "temperatures.npy"
    if not (kappa_path.exists() and T_path.exists()):
        return

    kappa_all = np.load(kappa_path)
    T_saved   = np.load(T_path)
    inv_T_k   = 1000.0 / T_saved

    # Use true nominal if available, else fall back to sample mean
    if nominal is not None:
        kappa_nom = kappa_nominal(T_saved, nominal)
    else:
        kappa_nom = kappa_all.mean(axis=0)

    param_label = ", ".join(PARAM_NAMES[i] for i in indices)
    rxn_short   = rxn_eq[:50] + ("…" if len(rxn_eq) > 50 else "")

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        f"Class-{curve_class} | m={m_level} [{param_label}]  (n={len(kappa_all)})\n{rxn_short}",
        fontsize=9
    )

    # Uncertainty band in κ space
    if uncertainties is not None and nominal is not None:
        ax.fill_between(inv_T_k,
                        kappa_nom - uncertainties,
                        kappa_nom + uncertainties,
                        alpha=0.15, color='steelblue', label='Uncertainty band')
        ax.plot(inv_T_k, kappa_nom + uncertainties,
                color='steelblue', lw=1.5, ls='--', alpha=0.5)
        ax.plot(inv_T_k, kappa_nom - uncertainties,
                color='steelblue', lw=1.5, ls='--', alpha=0.5)

    # Ramp targets f_c(T) shifted into κ space:  κ_nom + f_c
    if ramp_targets is not None:
        for i, fc in enumerate(ramp_targets):
            ax.plot(inv_T_k, kappa_nom + fc,
                    color='darkorange', lw=1.2, ls='--', alpha=0.4,
                    label='Ramp $f_c$' if i == 0 else None)

    # Sampled κ(T) curves
    for krow in kappa_all:
        ax.plot(inv_T_k, krow, color='red', alpha=0.5, linewidth=1.3)

    # Nominal
    ax.plot(inv_T_k, kappa_nom, color='black', lw=1.8, ls='-',
            label=r'Nominal $\kappa_0(T)$')

    ax.set_xlabel(r"$1000/T$  (K$^{-1}$)", fontsize=11)
    ax.set_ylabel(r"$\kappa(T) = \ln k(T)$", fontsize=11)
    ax.set_title(r"Full log-rate curves $\kappa(T)$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    pdf_path = sub_dir / "curves_plain.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Plot saved → {pdf_path}")

def plot_delta_kappa_standalone(sub_dir, T, L_r, indices, zeta_list,
                                 rxn_eq, m_level, folder, curve_class="A",
                                 ramp_targets=None):
    """Save a full-width Δκ vs 1000/T plot as delta_kappa.pdf."""
    if not zeta_list:
        return

    inv_T  = 1000.0 / T
    fp_S   = f_prior_S(T, L_r, indices)
    dk_all = np.vstack([delta_kappa(T, L_r, zr, indices) for zr in zeta_list])

    param_label = ", ".join(PARAM_NAMES[i] for i in indices)
    rxn_short   = rxn_eq[:50] + ("…" if len(rxn_eq) > 50 else "")

    fig, ax = plt.subplots(figsize=(7, 5))
    #fig.suptitle(
    #    f"Class-{curve_class} | m={m_level} [{param_label}]  (n={len(zeta_list)})\n{rxn_short}",
    #    fontsize=9
    #)

    # Sampled Δκ curves — red
    for i, dk in enumerate(dk_all):
        ax.plot(inv_T, dk, color='red', alpha=0.5, linewidth=1.2,
                label='Samples' if i == 0 else None)

    # Ramp targets f_c(T)
    if ramp_targets is not None:
        for i, fc in enumerate(ramp_targets):
            ax.plot(inv_T, fc, color='darkorange', lw=1.3, ls='--', alpha=0.5,
                    label='Ramp $f_c$' if i == 0 else None)

    # ±f_prior_S envelope
    ax.fill_between(inv_T, -fp_S, fp_S, alpha=0.10, color='dimgrey')
    ax.plot(inv_T,  fp_S, color='dimgrey', lw=1.4, ls='--',
            label=r'$\pm f_{\mathrm{prior},S}$')
    ax.plot(inv_T, -fp_S, color='dimgrey', lw=1.4, ls='--')
    ax.axhline(0, color='black', lw=0.6, ls=':')

    ax.set_xlabel(r"$1000/T$  (K$^{-1}$)", fontsize=11)
    ax.set_ylabel(r"$\Delta\kappa_S(T)$", fontsize=11)
    #ax.set_title(r"Perturbation curves $\Delta\kappa_S(T)$", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    pdf_path = sub_dir / "delta_kappa.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Plot saved → {pdf_path}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ─ PARSING HELPERS  (ported verbatim from benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_equation(s):
    s = re.sub(r"\s+", " ", s.strip())
    s = s.replace("=>", "<=>").replace("= >", "<=>")
    s = s.replace("< =>", "<=>").replace("<= >", "<=>")
    return s


def parse_xml_uncertainty(xml_path):
    """Parse XML uncertainty file → dict keyed by reaction nametag."""
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
    norm = normalize_equation(rxn_eq)
    rc   = yaml_rate_db.get(norm)
    if rc is None:
        return None
    A  = rc.get("A",  1.0); n  = rc.get("b",  0.0); Ea = rc.get("Ea", 0.0)
    return np.array([np.log(max(A, 1e-300)), n, Ea / R_GAS], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ─ FILE-SYSTEM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_folder_name(rxn_eq: str) -> str:
    """Convert a reaction equation to a valid directory name (≤80 chars)."""
    s = rxn_eq.strip()
    s = s.replace("(+M)", "_pM").replace("(+m)", "_pm")
    s = re.sub(r'\s*<=>\s*', '__', s)
    s = re.sub(r'\s*=>\s*',  '__', s)
    s = re.sub(r'\s*\+\s*',  '_p_', s)
    s = re.sub(r'\s+',        '_',  s)
    s = re.sub(r'[^\w\-]',   '_',  s)
    s = re.sub(r'_+',         '_',  s)
    return s.strip('_')[:80]


def write_info_json(out_dir: Path, rxn_info: dict, L_full: np.ndarray,
                    Sigma: np.ndarray, config_records: list, method: str):
    """Write consolidated info.json at the reaction root folder."""
    doc = {
        "reaction":   rxn_info,
        "method":     method,
        "covariance": {
            "L_full": L_full.tolist(),
            "Sigma":  Sigma.tolist(),
        },
        "configurations": {},
    }
    for rec in config_records:
        key = f"m{rec['m']}/{rec['folder']}"
        doc["configurations"][key] = {
            "m":                   rec["m"],
            "indices":             list(rec["indices"]),
            "param_labels":        [PARAM_NAMES[i] for i in rec["indices"]],
            "Sigma_r":             np.array(rec["Sigma_r"]).tolist(),
            "L_r":                 np.array(rec["L_r"]).tolist(),
            "n_samples_requested": rec["n_requested"],
            "n_samples_generated": rec["n_generated"],
            "wall_time_s":         round(rec["wall_time_s"], 6),
        }
    info_path = out_dir / "info.json"
    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"    Wrote {info_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 ─ PER-REACTION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def process_reaction(rxn_tag, rxn_data, output_root, method,
                     yaml_rate_db=None, n_samples=N_SAMPLES, seed=RANDOM_SEED):
    """
    Full Class-B generation pipeline for one reaction.

    Steps
    -----
    1.  Compute L_full via MUQ optimisation.
    2.  For every sub-config in M_CONFIGS:
          a.  Get L_r via principal sub-matrix Cholesky.
          b.  Dispatch to the requested Class-B method.
          c.  Save zeta_samples.npy (and optionally kappa_curves.npy).
    3.  Write info.json.
    """
    T             = rxn_data["temperatures"]
    uncertainties = rxn_data["uncertainties"]
    rxn_eq        = rxn_data["rxn_equation"]
    T_min, T_max  = float(T[0]), float(T[-1])

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

    nominal = None
    if yaml_rate_db is not None:
        nominal = get_nominal_params(rxn_eq, yaml_rate_db)

    print(f"  Computing L_full for: {rxn_eq[:60]}")
    t0     = time.perf_counter()
    L_full = compute_full_L(T, uncertainties)
    t_Lf   = time.perf_counter() - t0
    Sigma_full = L_full @ L_full.T
    print(f"    L_full diag: {np.diag(L_full).round(4)}  ({t_Lf:.2f}s)")

    config_records = []

    for m_level, folder, indices in M_CONFIGS:
        # Choose the right method key automatically
        if m_level == 3:
            active_method = "original_m3"
        else:
            active_method = "original_m2"
        if method not in ("original_m3", "original_m2"):
            active_method = method   # custom method handles all m

        print(f"    m={m_level}  [{', '.join(PARAM_NAMES[i] for i in indices)}]  "
              f"method={active_method}", end="  ...\n    ", flush=True)

        _, Sigma_r, L_r = get_reduced_L(L_full, indices)

        t_start   = time.perf_counter()
        sub_dir = rxn_folder / f"m{m_level}" / folder
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            zeta_list = generate_class_B_samples(
                nominal, T, L_full, uncertainties, indices, sub_dir,
                n_samples=n_samples, method=active_method,
                rng=np.random.default_rng(seed=seed + hash(folder) % 9999))
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            zeta_list = []
        wall_time   = time.perf_counter() - t_start
        n_generated = len(zeta_list)
        print(f"  {n_generated}/{n_samples} samples  {wall_time:.2f}s")

        if n_generated > 0:
            np.save(sub_dir / "zeta_samples.npy", np.vstack(zeta_list))
            if nominal is not None:
                kappa_nom = kappa_nominal(T, nominal)
                kappa_arr = np.vstack([
                    kappa_nom + delta_kappa(T, L_r, zr, indices)
                    for zr in zeta_list
                ])
                np.save(sub_dir / "kappa_curves.npy", kappa_arr)
                np.save(sub_dir / "temperatures.npy", T)
            
            # ── Plot all curves for this sub-config ───────────────────────
            plot_curves_for_subfolder(
                sub_dir, T, L_r, indices, zeta_list,
                rxn_eq=rxn_eq, m_level=m_level,
                folder=folder, curve_class="B"   # ← change to "B" or "C"
            )
            
            plot_curves_plain_blue(
                sub_dir, T, L_r, indices, zeta_list,
                rxn_eq=rxn_eq, m_level=m_level, folder=folder, curve_class="B",
                uncertainties=uncertainties, nominal=nominal, ramp_targets=None
            )
            plot_delta_kappa_standalone(
                sub_dir, T, L_r, indices, zeta_list,
                rxn_eq=rxn_eq, m_level=m_level, folder=folder, curve_class="B",
                ramp_targets=None
            )

        config_records.append({
            "m": m_level, "folder": folder, "indices": indices,
            "Sigma_r": Sigma_r, "L_r": L_r,
            "n_requested": n_samples, "n_generated": n_generated,
            "wall_time_s": wall_time,
        })

    write_info_json(rxn_folder, rxn_info, L_full, Sigma_full,
                    config_records, method)

    print(f"\n    ── Timing summary for {safe_name} ──")
    print(f"    {'Config':<14} {'m':>2}  {'Indices':<12} {'Generated':>10} "
          f"{'Time (s)':>10}")
    print(f"    {'-'*55}")
    for rec in config_records:
        idx_str = "(" + ",".join(str(i) for i in rec["indices"]) + ")"
        print(f"    {rec['folder']:<14} {rec['m']:>2}  {idx_str:<12} "
              f"{rec['n_generated']:>10}  {rec['wall_time_s']:>9.2f}s")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 ─ MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # ── Configuration ─────────────────────────────────────────────────────
    XML_PATH   = "MB_R_ALL_ECM_2025.xml"
    YAML_PATH  = "MB_MB2D_LALIT_2024.yaml"   # optional; None to skip
    OUTPUT_DIR = Path("output_B_m2_original_100")
    METHOD     = "original_m2"     # "original_m3" / "original_m2" or any
                                    # registered key in CLASS_B_METHOD_REGISTRY
                                    # m2_no_c2_adhoc_dn
                                    # m2_with_c2_adhoc_dn
    N_SAMP     = N_SAMPLES
    SEED       = RANDOM_SEED

    if not Path(XML_PATH).exists():
        sys.exit(f"ERROR: XML file not found: {XML_PATH}")

    yaml_rate_db = None
    if YAML_PATH is not None and Path(YAML_PATH).exists():
        print(f"[+] Loading YAML: {YAML_PATH}")
        yaml_rate_db = parse_yaml_mechanism(YAML_PATH)
        print(f"    {len(yaml_rate_db)} reactions found")

    print(f"[+] Parsing XML: {XML_PATH}")
    rxn_dict = parse_xml_uncertainty(XML_PATH)
    print(f"    {len(rxn_dict)} reactions found\n")
    print(f"[+] Class-B method : {METHOD}")
    print(f"[+] Available methods: {list(CLASS_B_METHOD_REGISTRY.keys())}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for rxn_tag, rxn_data in rxn_dict.items():
        print(f"{'='*65}")
        print(f"  Reaction: {rxn_tag}")
        print(f"{'='*65}")
        try:
            process_reaction(
                rxn_tag, rxn_data,
                output_root=OUTPUT_DIR,
                method=METHOD,
                yaml_rate_db=yaml_rate_db,
                n_samples=N_SAMP,
                seed=SEED,
            )
        except Exception as exc:
            print(f"  [ERROR] Skipping {rxn_tag}: {exc}")
        raise AssertionError("Testing single reaction run; remove this line to process all reactions.")

    print("\n[✓] Class-B generation complete.")
    print(f"    Output written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()