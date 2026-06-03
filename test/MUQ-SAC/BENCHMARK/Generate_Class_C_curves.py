#!/usr/bin/env python3
"""
generate_class_C.py
===================
Generate Class-C Arrhenius perturbation samples from an XML uncertainty file.

Class-C definition
------------------
Samples are fitted to a linear ramp target f_c(T):

  f_c(T) = r1·f_prior_S(T_min) + [r2·f_prior_S(T_max) - r1·f_prior_S(T_min)]
                                   ×  (T - T_min) / (T_max - T_min)

with the constraint  sign(r1) ≠ sign(r2)  (ensures a sign change in Δκ).
zeta_r is found by least-squares:  zeta_r = pinv(Θ_S^T L_r) · f_c(T).

Method implemented (original SLSQP, from published MUQ-SAC paper)
------------------------------------------------------------------
  m = 3  →  "original_m3":  SLSQP fit to f_c using L_full directly
  m = 2  →  "original_m2":  SLSQP fit to f_c in the reduced L_r space

Generates 100 samples per sub-configuration for:
  m = 2  →  (A,n)(1,1,0)  |  (A,Ea)(1,0,1)  |  (n,Ea)(0,1,1)
  m = 3  →  (A,n,Ea)(1,1,1)

How to add a new Class-C method
--------------------------------
1.  Implement:
      def generate_class_C_my_method(
              nominal, temperatures, L_full, uncertainties, indices,
              sub_dir, n_samples, rng) -> list[np.ndarray]
2.  Add to CLASS_C_METHOD_REGISTRY:
      CLASS_C_METHOD_REGISTRY["my_method"] = generate_class_C_my_method
3.  Set METHOD = "my_method" in main() or pass it to process_reaction().

Output structure (same as Codes 1 & 2, no m1 layer)
----------------------------------------------------
  output/
    {rxn_safe_name}/
      info.json
      m2/
        A_n/   zeta_samples.npy   [≤100 × 2]
        A_Ea/  zeta_samples.npy   [≤100 × 2]
        n_Ea/  zeta_samples.npy   [≤100 × 2]
      m3/
        A_n_Ea/ zeta_samples.npy  [≤100 × 3]
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

# ── Interactive backend must be selected BEFORE pyplot is imported.
# TkAgg works on Windows/Linux/macOS with Tk installed.
# Fall back to Qt5Agg if Tk is unavailable, then to the system default.
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    try:
        matplotlib.use('Qt5Agg')
    except Exception:
        pass   # leave whatever default the system chose

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ─────────────────────────────────────────────────────────────────────────────
# ANIMATION UTILITY
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION CLASS  (Class-C counterpart of ClassBSearchViz)
# ─────────────────────────────────────────────────────────────────────────────

class ClassCSearchViz:
    """
    Live recorder + post-hoc animator for Class-C original search.

    Left panel  : nominal curve κ₀(T), uncertainty band, ramp target overlay,
                  animated search curve (QtPo + QtLZ) updated every N evals.
    Right panel : objective value vs. function evaluation count.

    The ramp target is the linear interpolation
        f_c(T) = r1·f_prior(T_min) + slope·(T - T_min)
    whose endpoints (anchor_min, anchor_max) are stored per-sample and
    drawn as orange scatter markers on the left axes.

    Usage pattern (mirrors ClassBSearchViz)
    ----------------------------------------
    1.  Create one instance per reaction sub-config before the sample loop.
    2.  Inside mismatch_objective, call viz.record(z, obj, L_r, indices).
    3.  After each accepted sample, call viz.build_and_save(...).
        build_and_save resets internal state so the object is ready for
        the next sample.
    """

    CURVE_UPDATE_EVERY = 5   # left-panel update interval (eval count)

    def __init__(self, temperatures, QtPo, uncertainties,
                 T_min, T_max, ftol, live=False):
        self.T         = temperatures
        self.inv_T     = 1.0 / temperatures
        self.QtPo      = QtPo
        self.unc       = uncertainties
        self.T_min     = T_min
        self.T_max     = T_max
        self.ftol      = ftol
        # per-sample ramp endpoints (set before each sample in the loop)
        self.anchor_min = None   # (1/T_min, value) tuple
        self.anchor_max = None   # (1/T_max, value) tuple
        self.ramp_line  = None   # (inv_T_array, ramp_values) tuple for overlay
        # internal state
        self._eval         = 0
        self._obj_vals     = []
        self._curve_frames = []  # (eval_index, QtLZ) stored every CURVE_UPDATE_EVERY
        self._restart_evals = []
        self.live = live
        if self.live:
            plt.ion()
            self._setup_live_figure()

    # ------------------------------------------------------------------
    def _setup_live_figure(self):
        self._fig, (self._ax_L, self._ax_R) = plt.subplots(1, 2, figsize=(13, 5))
        self._fig.suptitle("Class-C Search — Live", fontsize=11)

        # static left panel
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
        self._eval_text    = self._ax_L.text(0.02, 0.97, '',
                                              transform=self._ax_L.transAxes,
                                              fontsize=8, va='top', color='crimson')
        self._anc_scatter  = self._ax_L.scatter([], [], marker='o', s=60,
                                                 color='darkorange', zorder=5)

        # right panel
        self._ax_R.set_xlabel('Function evaluation')
        self._ax_R.set_ylabel(r'Objective $\|f_c - Q^T L z\|^2$')
        self._obj_line, = self._ax_R.plot([], [], 'royalblue', lw=1.0)
        self._ftol_line  = self._ax_R.axhline(self.ftol, color='seagreen',
                                               ls='--', lw=1.4,
                                               label=f'ftol={self.ftol:.0e}')
        self._ax_R.legend(fontsize=8)
        plt.tight_layout()
        plt.show(block=False)

    # ------------------------------------------------------------------
    def record(self, z_active, obj_val, L_r, indices):
        """Record one objective evaluation; update live display if enabled."""
        self._eval += 1
        self._obj_vals.append(float(obj_val))

        # store curve frame for post-hoc animation
        if self._eval % self.CURVE_UPDATE_EVERY == 0:
            QtLZ = _compute_QtLZ(self.T, L_r, z_active, indices)
            self._curve_frames.append((self._eval, QtLZ.copy()))

        if self.live:
            QtLZ = _compute_QtLZ(self.T, L_r, z_active, indices)
            self._push_live(QtLZ)

    def _push_live(self, QtLZ):
        # left panel
        self._search_line.set_data(self.inv_T, self.QtPo + QtLZ)
        self._eval_text.set_text(f'eval = {self._eval}')
        
        if self.anchor_min is not None:
            self._anc_scatter.set_offsets(
                np.array([self.anchor_min, self.anchor_max]))
        self._ax_L.relim()
        self._ax_L.autoscale_view(tight=False)

        # right panel
        xs      = list(range(1, len(self._obj_vals) + 1))
        obj_arr = np.array(self._obj_vals)
        self._obj_line.set_data(xs, obj_arr)
        self._ax_R.set_xlim(0, max(len(self._obj_vals) + 1, 10))
        ylo = max(float(obj_arr.min()) * 0.5, 1e-14)
        yhi = float(obj_arr.max()) * 3.0
        self._ax_R.set_ylim(ylo, yhi)
        if (yhi / max(ylo, 1e-30)) > 100:
            self._ax_R.set_yscale('log')

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.02)

    # ------------------------------------------------------------------
    def mark_restart(self):
        """Record the current eval index as a restart boundary."""
        self._restart_evals.append(len(self._obj_vals))

    # ------------------------------------------------------------------
    def build_and_save(self, filename="class_c_sample.mp4",
                       sample_id=0, folder=None):
        """
        Render a two-panel animation from recorded data and save to disk.
        Resets internal state so the object is ready for the next sample.

        Returns the path of the saved file (possibly sped-up), or None.
        """
        if not self._obj_vals:
            return None

        import os
        if folder:
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, os.path.basename(filename))

        obj_arr = np.array(self._obj_vals)
        use_log = (obj_arr.max() / max(obj_arr.min(), 1e-30)) > 100
        n_evals = len(obj_arr)

        fig, (ax_L, ax_R) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
        fig.suptitle(f"Class-C Original  |  Sample {sample_id}", fontsize=11)

        # ── LEFT : static background ──────────────────────────────────
        ax_L.plot(self.inv_T, self.QtPo,
                  'k-', lw=2.0, label=r'Nominal $Q^T P_0$', zorder=3)
        ax_L.fill_between(self.inv_T,
                          self.QtPo - self.unc,
                          self.QtPo + self.unc,
                          alpha=0.15, color='steelblue',
                          label='Uncertainty band', zorder=2)
        ax_L.plot(self.inv_T, self.QtPo + self.unc,
                  color='steelblue', lw=0.8, ls='--', alpha=0.6, zorder=2)
        ax_L.plot(self.inv_T, self.QtPo - self.unc,
                  color='steelblue', lw=0.8, ls='--', alpha=0.6, zorder=2)
        ax_L.axvline(1.0/self.T_min, color='gray', ls=':', lw=0.9,
                     alpha=0.7, label=r'$T_{\min},\,T_{\max}$')
        ax_L.axvline(1.0/self.T_max, color='gray', ls=':', lw=0.9, alpha=0.7)

        # ramp target overlay (static — same for all frames of this sample)
        if self.ramp_line is not None:
            ax_L.plot(self.ramp_line[0],
                      self.QtPo + self.ramp_line[1],   # absolute position
                      color='darkorange', lw=1.2, ls='--',
                      alpha=0.75, label='Ramp target $f_c$', zorder=3)
        if self.anchor_min is not None:
            ax_L.scatter(*self.anchor_min, marker='o', s=60,
                         color='darkorange', zorder=5,
                         label='Ramp anchors (C1, C3)')
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

        # ── RIGHT : objective ─────────────────────────────────────────
        ax_R.axhline(self.ftol, color='seagreen', ls='--', lw=1.4,
                     label=f'ftol = {self.ftol:.0e}', zorder=3)
        for rev in self._restart_evals:
            if rev > 0:
                ax_R.axvline(rev, color='orange', ls=':', lw=0.8, alpha=0.7)
        ax_R.set_xlim(0, n_evals + 1)
        ylo = max(obj_arr.min() * 0.5, 1e-14)
        yhi = obj_arr.max() * 3
        ax_R.set_ylim(ylo, yhi)
        if use_log:
            ax_R.set_yscale('log')
        ax_R.set_xlabel('Function evaluation', fontsize=10)
        ax_R.set_ylabel(r'Objective  $\|f_c - Q^T L_r z\|^2$', fontsize=10)
        ax_R.legend(fontsize=8)

        obj_line, = ax_R.plot([], [], color='royalblue', lw=1.0, alpha=0.9)

        # curve frame lookup (eval → QtLZ snapshot)
        frame_dict = {ev: crv for ev, crv in self._curve_frames}

        def update(fi):
            cur_eval = fi + 1
            obj_line.set_data(list(range(1, cur_eval + 1)),
                              self._obj_vals[:cur_eval])
            avail = [ev for ev in frame_dict if ev <= cur_eval]
            if avail:
                ev   = max(avail)
                QtLZ = frame_dict[ev]
                search_line.set_data(self.inv_T, self.QtPo + QtLZ)
                eval_text.set_text(f'eval = {ev}')
            return search_line, obj_line, eval_text

        ani = animation.FuncAnimation(
            fig, update, frames=n_evals,
            blit=False,       # False → static bg renders in every GIF frame
            interval=40,
            repeat=False)

        # ── save ──────────────────────────────────────────────────────
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

        # ── reset for next sample ──────────────────────────────────────
        self._eval = 0
        self._obj_vals.clear()
        self._curve_frames.clear()
        self._restart_evals.clear()
        self.anchor_min = None
        self.anchor_max = None
        self.ramp_line  = None
        return saved_path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

M_CONST     = 3.0 / np.log(10.0)
R_GAS       = 1.987
MAX_DELTA_N = 2.0
N_SAMPLES   = 100
RANDOM_SEED = 42

PARAM_NAMES = {0: "A", 1: "n", 2: "Ea"}

# Sub-configurations for Class-C (no m=1)
M_CONFIGS = [
    (2, "A_n",    (0, 1)),
    (2, "A_Ea",   (0, 2)),
    (2, "n_Ea",   (1, 2)),
    (3, "A_n_Ea", (0, 1, 2)),
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


def f_prior_S(T, L_r, indices):
    """f_prior_S(T) = ‖L_r^T Θ_S(T)‖₂ for each T."""
    thS = theta_S(T, indices)
    return np.array([np.linalg.norm(L_r.T @ col) for col in thS.T])


def delta_kappa(T, L_r, zeta_r, indices):
    """Δκ_S(T) = Θ_S(T)^T L_r zeta_r."""
    return theta_S(T, indices).T @ (L_r @ zeta_r)


def kappa_nominal(T, nominal):
    """κ₀(T) = θ(T)^T p₀."""
    return theta_full(T).T @ nominal


def _has_sign_change(arr):
    """Return True if arr changes sign at least once."""
    return bool(np.any(np.diff(np.sign(arr)) != 0))


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


def _compute_QtLZ(T, L_r, z, indices):
    """
    Compute the perturbation curve QtLZ = Θ_S(T)^T (L_r z) for viz.
    Works for both m=2 (reduced L_r, subset indices) and
    m=3 (L_full passed as L_r, indices=(0,1,2)).
    """
    thS_all = theta_S(T, indices)                 # (m, N)
    return np.array([thS @ L_r @ z for thS in thS_all.T])   # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ COVARIANCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _muq_objective(params, T, uncertainties):
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
    Sigma   : (3, 3) full covariance
    Sigma_r : (m, m) principal sub-matrix of Σ
    L_r     : (m, m) Cholesky factor of Σ_r (lower triangular)
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
# SECTION 4 ─ INTERNAL HELPER: uncorrelated reference direction
# ─────────────────────────────────────────────────────────────────────────────

def _compute_uncorrelated_direction(L, temperatures, uncertainties):
    """
    Find zeta_unc by Nelder-Mead so that Θ(T)^T L zeta_unc ≈ uncertainties(T).

    This defines the 'uncorrelated' reference direction used by the original
    Class-C method to scale the linear ramp target f_c(T).

    Returns
    -------
    zeta_unc : shape (3,)
    """
    def objective(guess):
        Theta = np.array([temperatures / temperatures,
                          np.log(temperatures),
                          -1.0 / temperatures])
        QtLZ  = np.array([th @ L @ guess for th in Theta.T])
        f     = uncertainties - QtLZ
        return float(np.dot(f, f))

    guess  = np.array([0.5, 0.1, 0.5])
    result = minimize(objective, guess, method="Nelder-Mead",
                      options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9})
    return result.x


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ ORIGINAL CLASS-C SAMPLERS
# ─────────────────────────────────────────────────────────────────────────────
def generate_class_C_curves(
        nominal, temperatures, L_full, uncertainties, indices, sub_dir,
        n_samples, rng):
    """
    Class-C: direct least-squares fit to a linear crossing ramp target.

    f_c(T) = r1·f_prior_S(T_min) + [r2·f_prior_S(T_max) - r1·f_prior_S(T_min)]
                                     × (T - T_min) / (T_max - T_min)

    zeta_r = pinv(Θ_S^T L_r) · f_c(T)  — single closed-form solve per attempt,
    no iterative optimisation.  The recorded "objective" is the LS residual
    ||f_c - Θ_S^T L_r zeta_r||² (should be near-zero for well-posed subsets).

    Parameters
    ----------
    nominal       : (3,) nominal Arrhenius parameters (for viz baseline)
    temperatures  : (N,) temperature array in K
    L_full        : (3, 3) full Cholesky factor
    uncertainties : (N,) uncertainty array  (used for viz band only)
    indices       : tuple of active parameter indices (must have len >= 2)
    sub_dir       : Path-like, directory for per-sample animation output
    n_samples     : number of accepted samples to generate
    rng           : numpy Generator

    Returns
    -------
    zeta_list : list of zeta_r arrays, each shape (m,)
    """
    if len(indices) < 2:
        return []

    _, _, L_r = get_reduced_L(L_full, indices)

    T_min = float(temperatures[0])
    T_max = float(temperatures[-1])
    inv_T     = 1.0 / temperatures
    inv_T_min = 1.0 / T_max    # note: T_max → smallest 1/T
    inv_T_max = 1.0 / T_min    # note: T_min → largest 1/T

    fp_Tmin = float(f_prior_S(np.array([T_min]), L_r, indices)[0])
    fp_Tmax = float(f_prior_S(np.array([T_max]), L_r, indices)[0])

    # Pre-compute pinv once — reused for every attempt
    thS    = theta_S(temperatures, indices)   # (m, N)
    A_mat  = thS.T @ L_r                      # (N, m)  =  Φ_S L_r
    A_pinv = np.linalg.pinv(A_mat)            # (m, N)

    # ── viz setup ────────────────────────────────────────────────────────
    thS_full = theta_S(temperatures, (0, 1, 2))
    QtPo     = np.array([th @ nominal for th in thS_full.T]) \
        if nominal is not None else np.zeros_like(temperatures)

    viz = ClassCSearchViz(
        temperatures  = temperatures,
        QtPo          = QtPo,
        uncertainties = uncertainties,
        T_min         = T_min,
        T_max         = T_max,
        ftol          = 1e-14,        # LS residual reference line in right panel
    )

    # ── main sampling loop ────────────────────────────────────────────
    zeta_list  = []
    sample_idx = 0
    attempt    = 0
    ramp_targets = []
    while len(zeta_list) < n_samples and attempt < n_samples * 5:
        attempt += 1

        r1 = float(rng.uniform(-1.0, 1.0))
        r2 = float(rng.uniform(-1.0, 1.0))
        if np.sign(r1) == np.sign(r2):
            r2 = -r2

        # Build ramp target
        #fc = (r1 * fp_Tmin
        #      + (r2 * fp_Tmax - r1 * fp_Tmin) / (T_max - T_min)
        #      * (temperatures - T_min))           # (N,)

        fc = (r1 * fp_Tmin
            + (r2 * fp_Tmax - r1 * fp_Tmin) / (inv_T_max - inv_T_min)
            * (inv_T - inv_T_min))
        # Closed-form solve
        zeta_r = A_pinv @ fc                      # (m,)

        # LS residual — what the right panel tracks
        residual = fc - A_mat @ zeta_r            # (N,)
        obj      = float(np.dot(residual, residual))

        # ── update viz anchors and ramp overlay for this attempt ──────
        viz.mark_restart()
        idx_min        = int(np.argmin(np.abs(temperatures - T_min)))
        idx_max        = int(np.argmin(np.abs(temperatures - T_max)))
        viz.anchor_min = (1.0 / T_min, QtPo[idx_min] + r1 * fp_Tmin)
        viz.anchor_max = (1.0 / T_max, QtPo[idx_max] + r2 * fp_Tmax)
        viz.ramp_line  = (1.0 / temperatures, fc)

        # Single record per attempt (no inner optimisation loop)
        viz.record(zeta_r, obj, L_r, indices)

        # ── accept / reject ───────────────────────────────────────────
        dk = delta_kappa(temperatures, L_r, zeta_r, indices)
        if not _has_sign_change(dk):
            continue

        zeta_r = _enforce_dn_constraint(zeta_r, L_r, indices)
        zeta_list.append(zeta_r)
        ramp_targets.append(fc.copy())
        # ── save per-sample animation ─────────────────────────────────
        saved_path = viz.build_and_save(
            filename  = f"classC_ls_sample_{sample_idx:03d}.mp4",
            sample_id = sample_idx,
            folder    = str(sub_dir)
        )
        if saved_path:
            speed_up_animation(saved_path, speedup=10)
        sample_idx += 1

    return zeta_list, ramp_targets


def generate_class_C_original_full_parameters(
        nominal, temperatures, L_full, uncertainties, indices, sub_dir,
        n_samples, rng):
    """
    Generate Class-C samples using the ORIGINAL published SLSQP method
    for m = 3  (all three Arrhenius parameters A, n, Ea active).

    Algorithm
    ---------
    1.  Compute zeta_unc  (uncorrelated reference direction via Nelder-Mead).
    2.  Evaluate Δκ_unc at T_min and T_max to define the ramp endpoints.
    3.  Build the linear ramp target:
          f_c(T) = r1·Δκ_unc(T_min) + [r2·Δκ_unc(T_max) - r1·Δκ_unc(T_min)]
                                        × (T - T_min)/(T_max - T_min)
    4.  Minimise ‖Θ(T)^T L z - f_c(T)‖² via SLSQP  →  zeta_r.
    5.  Record every objective evaluation via ClassCSearchViz; save animation
        after each accepted sample.

    Parameters
    ----------
    nominal       : (3,) nominal Arrhenius parameters (for viz baseline)
    temperatures  : (N,) temperature array in K
    L_full        : (3, 3) full Cholesky factor
    uncertainties : (N,) uncertainty array
    indices       : should be (0, 1, 2) for this method
    sub_dir       : Path-like, directory for per-sample animation output
    n_samples     : number of samples
    rng           : numpy Generator for r1, r2 draws

    Returns
    -------
    zeta_list : list of zeta_r arrays, each shape (3,)
    """
    T_min    = float(temperatures[0])
    T_max    = float(temperatures[-1])
    inv_T     = 1.0 / temperatures
    inv_T_min = 1.0 / T_max    # note: T_max → smallest 1/T
    inv_T_max = 1.0 / T_min    # note: T_min → largest 1/T
    L        = L_full
    zeta_unc = _compute_uncorrelated_direction(L, temperatures, uncertainties)

    # Evaluate unc curve at endpoints (full space)
    th_min     = theta_full(np.array([T_min]))[:, 0]
    th_max     = theta_full(np.array([T_max]))[:, 0]
    dk_unc_min = float(th_min @ L @ zeta_unc)
    dk_unc_max = float(th_max @ L @ zeta_unc)

    # ── viz setup ─────────────────────────────────────────────────────
    # Nominal curve: θ_full(T)^T p₀  (uses all three parameters)
    QtPo = np.array([theta_full(np.array([t]))[:, 0] @ nominal
                     for t in temperatures]) if nominal is not None \
        else np.zeros_like(temperatures)

    viz = ClassCSearchViz(
        temperatures  = temperatures,
        QtPo          = QtPo,
        uncertainties = uncertainties,
        T_min         = T_min,
        T_max         = T_max,
        ftol          = 1e-9,
    )

    zeta_list  = []
    sample_idx = 0   # counts accepted samples (for filename)
    ramp_targets = []
    for _ in range(n_samples * 5):
        if len(zeta_list) >= n_samples:
            break

        r1 = float(rng.uniform(-1.0, 1.0))
        r2 = float(rng.uniform(-1.0, 1.0))
        # Enforce opposite signs so the curve must cross zero
        if np.sign(r1) == np.sign(r2):
            r2 = -r2

        # Build linear ramp target
        FT_min      = r1 * dk_unc_min
        FT_max      = r2 * dk_unc_max
        slope_inv   = (FT_min - FT_max) / (inv_T_max - inv_T_min)
        ramp_target = FT_max + slope_inv * (inv_T - inv_T_min)

        # ── update viz anchors and ramp overlay for this attempt ──────
        viz.mark_restart()
        idx_min        = int(np.argmin(np.abs(temperatures - T_min)))
        idx_max        = int(np.argmin(np.abs(temperatures - T_max)))
        viz.anchor_min = (1.0 / T_min, QtPo[idx_min] + FT_min)
        viz.anchor_max = (1.0 / T_max, QtPo[idx_max] + FT_max)
        viz.ramp_line  = (1.0 / temperatures, ramp_target)   # (inv_T, f_c values)

        # ── SLSQP minimisation with viz recording ─────────────────────
        def mismatch_objective(z, _rt=ramp_target):
            Theta = np.array([temperatures / temperatures,
                              np.log(temperatures),
                              -1.0 / temperatures])
            QtLZ  = np.array([th @ L @ z for th in Theta.T])
            f     = _rt - QtLZ
            obj   = float(np.dot(f, f))
            viz.record(z, obj, L, (0, 1, 2))   # L_full acts as L_r for m=3
            return obj

        try:
            result = minimize(mismatch_objective, np.zeros(3),
                              method='SLSQP',
                              options={'maxiter': 2000, 'ftol': 1e-9})
            zr = result.x
        except Exception as exc:
            print(f"      [!] SLSQP C_original_m3 failed: {exc}")
            continue

        zeta_list.append(_enforce_dn_constraint(np.asarray(zr), L, (0, 1, 2)))
        ramp_targets.append(ramp_target.copy())
        # ── save per-sample animation ──────────────────────────────────
        saved_path = viz.build_and_save(
            filename  = f"classC_orig_m3_sample_{sample_idx:03d}.mp4",
            sample_id = sample_idx,
            folder    = str(sub_dir)
        )
        if saved_path:
            speed_up_animation(saved_path, speedup=10)
        sample_idx += 1

    return zeta_list[:n_samples], ramp_targets[:n_samples]


def generate_class_C_original_reduced_parameters(
        nominal, temperatures, L_full, uncertainties, indices, sub_dir,
        n_samples, rng):
    """
    Generate Class-C samples using the ORIGINAL published SLSQP method
    adapted for m = 2  (any 2-parameter subset via reduced Cholesky L_r).

    Algorithm
    ---------
    Same ramp construction as the m=3 original, but:
    - The reference direction and ramp endpoints are computed in the
      REDUCED L_r space (only the active parameter subset).
    - SLSQP minimises ‖Θ_S(T)^T L_r z - f_c(T)‖² for z ∈ ℝ^m.
    - Every objective evaluation is recorded via ClassCSearchViz and an
      animation is saved after each accepted sample.

    Parameters
    ----------
    nominal       : (3,) nominal Arrhenius parameters (for viz baseline)
    temperatures  : (N,) temperature array
    L_full        : (3, 3) full Cholesky factor
    uncertainties : (N,) uncertainty array
    indices       : 2-element tuple, e.g. (0, 1), (0, 2), or (1, 2)
    sub_dir       : Path-like, directory for per-sample animation output
    n_samples     : number of samples
    rng           : numpy Generator

    Returns
    -------
    zeta_list : list of zeta_r arrays, each shape (2,)
    """
    _, _, L_r     = get_reduced_L(L_full, indices)
    zeta_unc_full = _compute_uncorrelated_direction(L_full, temperatures, uncertainties)
    zeta_unc_S    = zeta_unc_full[list(indices)]   # project to active subset
    T_min  = float(temperatures[0])
    T_max  = float(temperatures[-1])
    inv_T     = 1.0 / temperatures
    inv_T_min = 1.0 / T_max    # note: T_max → smallest 1/T
    inv_T_max = 1.0 / T_min    # note: T_min → largest 1/T

    # Evaluate unc curve at endpoints in the REDUCED space
    thS_min    = theta_S(np.array([T_min]), indices)[:, 0]
    thS_max    = theta_S(np.array([T_max]), indices)[:, 0]
    dk_unc_min = float(thS_min @ L_r @ zeta_unc_S)
    dk_unc_max = float(thS_max @ L_r @ zeta_unc_S)

    # ── viz setup ─────────────────────────────────────────────────────
    # Nominal curve uses full theta (same baseline as Class-B m=2)
    thS_full = theta_S(temperatures, (0, 1, 2))
    QtPo     = np.array([th @ nominal for th in thS_full.T]) \
        if nominal is not None else np.zeros_like(temperatures)

    viz = ClassCSearchViz(
        temperatures  = temperatures,
        QtPo          = QtPo,
        uncertainties = uncertainties,
        T_min         = T_min,
        T_max         = T_max,
        ftol          = 1e-9,
    )

    zeta_list  = []
    sample_idx = 0   # counts accepted samples
    ramp_targets = []
    for _ in range(n_samples * 5):
        if len(zeta_list) >= n_samples:
            break

        r1 = float(rng.uniform(-1.0, 1.0))
        r2 = float(rng.uniform(-1.0, 1.0))
        if np.sign(r1) == np.sign(r2):
            r2 = -r2

        FT_min      = r1 * dk_unc_min
        FT_max      = r2 * dk_unc_max
        slope       = (FT_max - FT_min) / (T_max - T_min)
        ramp_target = FT_min + slope * (temperatures - T_min)   # (N,)

        # ── update viz anchors and ramp overlay ───────────────────────
        viz.mark_restart()
        idx_min        = int(np.argmin(np.abs(temperatures - T_min)))
        idx_max        = int(np.argmin(np.abs(temperatures - T_max)))
        viz.anchor_min = (1.0 / T_min, QtPo[idx_min] + FT_min)
        viz.anchor_max = (1.0 / T_max, QtPo[idx_max] + FT_max)
        viz.ramp_line  = (1.0 / temperatures, ramp_target)

        # ── SLSQP minimisation in the REDUCED L_r space ───────────────
        def mismatch_objective_m2(z, _rt=ramp_target):
            thS_all = theta_S(temperatures, indices)
            QtLZ    = np.array([thS @ L_r @ z for thS in thS_all.T])
            f       = _rt - QtLZ
            obj     = float(np.dot(f, f))
            viz.record(z, obj, L_r, indices)   # reduced L_r and indices
            return obj

        try:
            result = minimize(mismatch_objective_m2,
                              np.zeros(len(indices)),
                              method='SLSQP',
                              options={'maxiter': 2000, 'ftol': 1e-9})
            zr = result.x
        except Exception as exc:
            print(f"      [!] SLSQP C_original_m2 failed: {exc}")
            continue

        zeta_list.append(_enforce_dn_constraint(np.asarray(zr), L_r, indices))
        ramp_targets.append(ramp_target.copy())
        # ── save per-sample animation ──────────────────────────────────
        saved_path = viz.build_and_save(
            filename  = f"classC_orig_m2_sample_{sample_idx:03d}.mp4",
            sample_id = sample_idx,
            folder    = str(sub_dir)
        )
        if saved_path:
            speed_up_animation(saved_path, speedup=10)
        sample_idx += 1

    return zeta_list[:n_samples], ramp_targets[:n_samples]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ METHOD REGISTRY  (add new methods here)
# ─────────────────────────────────────────────────────────────────────────────
#
# To register a new algorithm:
#   1. Implement:
#        def generate_class_C_my_method(
#                nominal, temperatures, L_full, uncertainties, indices,
#                sub_dir, n_samples, rng) -> list[np.ndarray]
#   2. Add to CLASS_C_METHOD_REGISTRY.
#   3. Set METHOD = "my_method" in main(). Done.
#
# ─────────────────────────────────────────────────────────────────────────────

CLASS_C_METHOD_REGISTRY = {

    # ── Published methods ──────────────────────────────────────────────────
    "original_m3": generate_class_C_original_full_parameters,
    "original_m2": generate_class_C_original_reduced_parameters,
    "least_squares": generate_class_C_curves,

    # ── Add faster / analytical methods below as they are developed ───────
    # "least_squares_m2": generate_class_C_least_squares_m2,
    # "least_squares_m3": generate_class_C_least_squares_m3,
    # ──────────────────────────────────────────────────────────────────────

}


def generate_class_C_samples(nominal, temperatures, L_full, uncertainties,
                              indices, sub_dir, n_samples, method, rng):
    """
    Public entry point: generate n_samples Class-C zeta_r vectors.

    Parameters
    ----------
    nominal       : (3,) nominal Arrhenius parameters (for viz baseline;
                    pass None to suppress the nominal curve overlay)
    temperatures  : (N,) temperature array in K
    L_full        : (3, 3) Cholesky factor from MUQ
    uncertainties : (N,) uncertainty array
    indices       : tuple of active parameter indices
    sub_dir       : Path-like, directory for per-sample animation output
    n_samples     : number of samples to generate
    method        : key in CLASS_C_METHOD_REGISTRY
    rng           : numpy Generator

    Returns
    -------
    zeta_list : list of zeta_r arrays, each shape (m,)

    Raises
    ------
    ValueError if method is not in CLASS_C_METHOD_REGISTRY
    """
    if method not in CLASS_C_METHOD_REGISTRY:
        available = list(CLASS_C_METHOD_REGISTRY.keys())
        raise ValueError(
            f"Unknown Class-C method '{method}'. "
            f"Available: {available}"
        )
    sampler_fn = CLASS_C_METHOD_REGISTRY[method]
    return sampler_fn(nominal, temperatures, L_full, uncertainties,
                      indices, sub_dir, n_samples, rng)


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
    """Convert reaction equation to a valid directory name (≤80 chars)."""
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
# PLOTTING HELPERS  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

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
    fig.suptitle(
        f"Class-{curve_class} | m={m_level} [{param_label}]  (n={len(zeta_list)})\n{rxn_short}",
        fontsize=9
    )

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
    ax.set_title(r"Perturbation curves $\Delta\kappa_S(T)$", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    pdf_path = sub_dir / "delta_kappa.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Plot saved → {pdf_path}")


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
    """
    import matplotlib.cm as cm

    if not zeta_list:
        return

    inv_T   = 1000.0 / T
    fp_S    = f_prior_S(T, L_r, indices)
    dk_all  = np.vstack([delta_kappa(T, L_r, zr, indices)
                         for zr in zeta_list])

    kappa_path = sub_dir / "kappa_curves.npy"
    T_path     = sub_dir / "temperatures.npy"
    has_kappa  = kappa_path.exists() and T_path.exists()

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

    colours = cm.plasma(np.linspace(0.15, 0.85, len(zeta_list)))

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

    if has_kappa:
        kappa_all_arr = np.load(kappa_path)
        T_saved       = np.load(T_path)
        inv_T_k       = 1000.0 / T_saved
        kappa_nom     = kappa_all_arr.mean(axis=0)
        for i, krow in enumerate(kappa_all_arr):
            ax_k.plot(inv_T_k, krow, color=colours[i], alpha=0.45, linewidth=0.7)
        ax_k.plot(inv_T_k, kappa_nom, color="black", lw=1.5, ls="-",
                  label=r"mean $\kappa(T)$")
        ax_k.set_xlabel(r"$1000/T$  (K$^{-1}$)", fontsize=10)
        ax_k.set_ylabel(r"$\kappa(T) = \ln k(T)$", fontsize=10)
        ax_k.set_title(r"Full log-rate curves $\kappa(T)$", fontsize=9)
        ax_k.legend(fontsize=8)
        ax_k.grid(True, alpha=0.25)

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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 ─ PER-REACTION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def process_reaction(rxn_tag, rxn_data, output_root, method,
                     yaml_rate_db=None, n_samples=N_SAMPLES, seed=RANDOM_SEED):
    """
    Full Class-C generation pipeline for one reaction.

    Steps
    -----
    1.  Compute L_full via MUQ optimisation.
    2.  For every sub-config:
          a.  Get L_r.
          b.  Select method key (original_m3 for m=3, original_m2 for m=2
              unless a custom method is specified).
          c.  Run generate_class_C_samples (now passes nominal + sub_dir).
          d.  Save zeta_samples.npy (and kappa_curves.npy if YAML provided).
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
        # Auto-select method based on m unless caller overrides
        if method in ("original_m3", "original_m2"):
            active_method = "original_m3" if m_level == 3 else "original_m2"
        else:
            active_method = method   # custom method handles all m

        print(f"    m={m_level}  [{', '.join(PARAM_NAMES[i] for i in indices)}]  "
              f"method={active_method}", end="  ...\n    ", flush=True)

        _, Sigma_r, L_r = get_reduced_L(L_full, indices)

        sub_dir = rxn_folder / f"m{m_level}" / folder
        sub_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.perf_counter()
        try:
            # collect ramp targets alongside zeta_list
            zeta_list, ramp_targets = generate_class_C_samples(
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

            # ── Plot all curves for this sub-config ───────────────────
            plot_curves_for_subfolder(
                sub_dir, T, L_r, indices, zeta_list,
                rxn_eq=rxn_eq, m_level=m_level,
                folder=folder, curve_class="C"
            )
            # pass the new arguments to both plot helpers
            plot_curves_plain_blue(
                sub_dir, T, L_r, indices, zeta_list,
                rxn_eq=rxn_eq, m_level=m_level, folder=folder, curve_class="C",
                uncertainties=uncertainties, nominal=nominal, ramp_targets=ramp_targets
            )
            plot_delta_kappa_standalone(
                sub_dir, T, L_r, indices, zeta_list,
                rxn_eq=rxn_eq, m_level=m_level, folder=folder, curve_class="C",
                ramp_targets=ramp_targets
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
    OUTPUT_DIR = Path("output_C_least_squares_100")  # root for all reaction subfolders
    METHOD     = "least_squares"     # dispatcher auto-selects original_m2/m3
                                    # or pass any registered key
                                    # least_squares
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
    print(f"[+] Class-C method : {METHOD}")
    print(f"[+] Available methods: {list(CLASS_C_METHOD_REGISTRY.keys())}\n")

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
        raise AssertionError("Debug stop after first reaction")  # remove to process all

    print("\n[✓] Class-C generation complete.")
    print(f"    Output written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()