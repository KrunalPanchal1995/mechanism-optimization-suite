"""
Figures for Appendix: Properties of the Principal Submatrix and Sampling Validity

Three figures:
  Fig 1 – Ellipsoid containment E_cond ⊆ E_Σr (correlated vs uncorrelated)
  Fig 2 – 3-D projection  π_I(E_Σp) = E_Σcond, then containment in E_Σr
  Fig 3 – Sampling validity: fresh chol(Σr) vs sub-block of L


  Sample reaction : H + O2 <=> O + OH
  P_o             : [  32.22016354    0.         7705.08303976]
  L               : [[-8.26701580e-01  0.00000000e+00  0.00000000e+00]
 		     [ 1.33985127e-01 -2.04188933e-02  0.00000000e+00]
                     [ 1.98390252e+02  1.08794583e+01  5.28168288e+00]]

  L_r_A  (1×1)    : [[0.82670158]]   zr_A  = 0.029384  →  delta_A  = 0.024292
  L_r_n  (1×1)    : [[0.13553208]]   zr_n  = 0.024162  →  delta_n  = 0.003275
  L_r_Ea (1×1)    : [[198.75852388]]  zr_Ea = -0.167013  →  delta_Ea = -33.195216



[delta_dict] Deltas extracted for 9 reactions.
  Sample reaction : O + H2 <=> H + OH:A
  P_o             : [  28.97857465    0.         4001.00654253]
  L               : [[ 2.16336596e+00  0.00000000e+00  0.00000000e+00]
 [-2.70585145e-01 -1.34026483e-02  0.00000000e+00]
 [ 1.95942750e+02  1.59914357e+01  5.05207579e+00]]
  L_r_A  (1×1)    : [[2.16336596]]   zr_A  = 0.007566  →  delta_A  = 0.016369
  L_r_n  (1×1)    : [[0.27091687]]   zr_n  = 0.008194  →  delta_n  = 0.002220
  L_r_Ea (1×1)    : [[196.65912295]]  zr_Ea = -0.106418  →  delta_Ea = -20.928077


"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3-D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings("ignore")

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "mathtext.fontset" : "cm",          # Computer Modern for math
    "font.size"        : 11,
    "axes.labelsize"   : 12,
    "axes.titlesize"   : 12,
    "legend.fontsize"  : 10,
    "figure.dpi"       : 150,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

RED    = "#d62728"
BLUE   = "#1f77b4"
PURPLE = "#9467bd"
GREY   = "#7f7f7f"
GREEN  = "#2ca02c"
ORANGE = "#ff7f0e"

RNG = np.random.default_rng(42)
os.makedirs("outputs", exist_ok=True)
# ── numerical example ─────────────────────────────────────────────────────────
# Build a 3×3 PD covariance via L L^T so PD is guaranteed
L_full = np.array([[ 2.16336596e+00,  0.00000000e+00,  0.00000000e+00],
                   [-2.70585145e-01, -1.34026483e-02,  0.00000000e+00],
                   [ 1.95942750e+02,  1.59914357e+01,  5.05207579e+00]])
"""
L_full = np.array([[-8.26701580e-01,  0.00000000e+00,  0.00000000e+00],
 		           [ 1.33985127e-01, -2.04188933e-02,  0.00000000e+00],
                   [ 1.98390252e+02, 1.08794583e+01,  5.28168288e+00]])

L_full = np.array([[1.40, 0.00, 0.00],
                   [0.50, 1.10, 0.00],
                   [0.80, 0.40, 0.90]])
"""
Sigma_p = L_full @ L_full.T

# --------------- Partition for Figs 1 & 2:  I = {A,n} = {0,1} ---------------
S11         = Sigma_p[:2, :2]
S12         = Sigma_p[:2, 2:3]
S22         = Sigma_p[2:, 2:]
Sigma_r     = S11
Sigma_cond  = S11 - S12 @ np.linalg.inv(S22) @ S12.T
 
# Uncorrelated variant (zero cross-block) for Fig 1 right panel
Sp_unc        = Sigma_p.copy()
Sp_unc[:2, 2] = 0.0
Sp_unc[2, :2] = 0.0
S11_unc       = Sp_unc[:2, :2]
S12_unc       = Sp_unc[:2, 2:3]
S22_unc       = Sp_unc[2:, 2:]
Sc_unc        = S11_unc - S12_unc @ np.linalg.inv(S22_unc) @ S12_unc.T
 
# --------------- Normalisation ------------------------------------------------
def norm_D(cov):
    """Return D_inv = diag(1/sigma_i) from the diagonal of cov."""
    return np.diag(1.0 / np.sqrt(np.diag(cov)))
 
def norm_cov(cov, D):
    return D @ cov @ D
 
# Figs 1 & 2: normalise by Sigma_r diagonal (A-n plane)
D_r   = norm_D(Sigma_r)
Sr_n  = norm_cov(Sigma_r,    D_r)
Sc_n  = norm_cov(Sigma_cond, D_r)
 
D_unc    = norm_D(S11_unc)
Sr_unc_n = norm_cov(S11_unc, D_unc)
Sc_unc_n = norm_cov(Sc_unc,  D_unc)
 
# Fig 2: normalise full Sigma_p by its diagonal for 3-D plot
D_p3    = norm_D(Sigma_p)
Sp_n    = norm_cov(Sigma_p, D_p3)
D_An    = np.diag(np.diag(D_p3)[:2])          # (A,n) part of 3-D normaliser
Sc_n3   = norm_cov(Sigma_cond, D_An)
Sr_n3   = norm_cov(Sigma_r,    D_An)
 
# --------------- Geometry helpers ---------------------------------------------
def ellipse_pts(cov2, n_std=1.0, n_pts=300):
    ev, evec = np.linalg.eigh(cov2)
    t  = np.linspace(0, 2 * np.pi, n_pts)
    xy = evec @ (n_std * np.sqrt(np.abs(ev[:, None]))
                 * np.array([np.cos(t), np.sin(t)]))
    return xy[0], xy[1]
 
def ellipsoid_surf(cov3, n_pts=50):
    u = np.linspace(0, 2 * np.pi, n_pts)
    v = np.linspace(0, np.pi,     n_pts)
    sphere = np.array([
        np.outer(np.cos(u), np.sin(v)).ravel(),
        np.outer(np.sin(u), np.sin(v)).ravel(),
        np.outer(np.ones_like(u), np.cos(v)).ravel(),
    ])
    ev, evec = np.linalg.eigh(cov3)
    T   = evec @ np.diag(np.sqrt(np.abs(ev)))
    pts = T @ sphere
    return [pts[i].reshape(n_pts, n_pts) for i in range(3)]
 
# ==============================================================================
# FIGURE 1 - Ellipsoid containment  (retained: A, n  |  removed: Ea)
# ==============================================================================
fig1, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(9, 4.2))
 
for ax, Sr_p, Sc_p, title in [
    (ax_c, Sr_n,     Sc_n,     r"Correlated  ($\Sigma_{12}\neq\mathbf{0}$)"),
    (ax_u, Sr_unc_n, Sc_unc_n, r"Uncorrelated  ($\Sigma_{12}=\mathbf{0}$)"),
]:
    xr, yr = ellipse_pts(Sr_p)
    xc, yc = ellipse_pts(Sc_p)
 
    ax.fill(xr, yr, color=RED,  alpha=0.18, zorder=1)
    ax.plot(xr, yr, color=RED,  lw=2,       zorder=3,
            label=r"$\mathcal{E}_{\Sigma_r}$")
    ax.fill(xc, yc, color=BLUE, alpha=0.30, zorder=2)
    ax.plot(xc, yc, color=BLUE, lw=2, ls="--", zorder=4,
            label=r"$\mathcal{E}_{\Sigma_{\rm cond}}$")
 
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta A\,/\,\sigma_A$")
    ax.set_ylabel(r"$\Delta n\,/\,\sigma_n$")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(loc="lower right")
 
fig1.suptitle(r"$\mathcal{E}_{\Sigma_{\rm cond}}\;\subseteq\;\mathcal{E}_{\Sigma_r}$"
              r"  (retained: $A$, $n$;  removed: $E_a$)",
              fontsize=12)
fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig("outputs/fig1_ellipsoid_containment.pdf", bbox_inches="tight")
fig1.savefig("outputs/fig1_ellipsoid_containment.png", bbox_inches="tight", dpi=200)
print("Figure 1 saved.")
 
 
# ==============================================================================
# FIGURE 2 - 3-D projection and 2-D containment  (all three Arrhenius params)
# ==============================================================================
fig2 = plt.figure(figsize=(11, 4.8))
gs   = fig2.add_gridspec(1, 2, wspace=0.30)
 
# LEFT: 3-D ellipsoid + projection onto (A, n) floor
ax3d = fig2.add_subplot(gs[0], projection="3d")
 
X, Y, Z = ellipsoid_surf(Sp_n, n_pts=50)
ax3d.plot_surface(X, Y, Z, alpha=0.15, color=PURPLE,
                  rstride=2, cstride=2, linewidth=0)
 
# Drop lines from equatorial ring to floor
ev3, evec3 = np.linalg.eigh(Sp_n)
T3  = evec3 @ np.diag(np.sqrt(np.abs(ev3)))
th  = np.linspace(0, 2 * np.pi, 20, endpoint=False)
ring = T3 @ np.array([np.cos(th), np.sin(th), np.zeros(20)])
for i in range(20):
    ax3d.plot([ring[0,i]]*2, [ring[1,i]]*2, [ring[2,i], 0],
              color=GREY, lw=0.4, alpha=0.4)
 
# Floor ellipses
xc3, yc3 = ellipse_pts(Sc_n3)
xr3, yr3 = ellipse_pts(Sr_n3)
ax3d.plot(xc3, yc3, np.zeros_like(xc3), color=BLUE, lw=2,
          label=r"$\mathcal{E}_{\Sigma_{\rm cond}}$")
ax3d.plot(xr3, yr3, np.zeros_like(xr3), color=RED,  lw=2, ls="--",
          label=r"$\mathcal{E}_{\Sigma_r}$")
 
ax3d.set_xlabel(r"$\Delta A/\sigma_A$",     labelpad=6)
ax3d.set_ylabel(r"$\Delta n/\sigma_n$",     labelpad=6)
ax3d.set_zlabel(r"$\Delta E_a/\sigma_{E_a}$", labelpad=6)
ax3d.legend(loc="upper left", fontsize=9)
ax3d.view_init(elev=22, azim=-52)
 
# RIGHT: 2-D floor view
ax2d = fig2.add_subplot(gs[1])
 
ax2d.fill(xr3, yr3, color=RED,  alpha=0.18, zorder=1)
ax2d.plot(xr3, yr3, color=RED,  lw=2,       zorder=3,
          label=r"$\mathcal{E}_{\Sigma_r}$")
ax2d.fill(xc3, yc3, color=BLUE, alpha=0.30, zorder=2)
ax2d.plot(xc3, yc3, color=BLUE, lw=2, ls="--", zorder=4,
          label=r"$\mathcal{E}_{\Sigma_{\rm cond}}$")
 
# Annotate the annular gap
angle_ann = np.radians(60)
ev2, evec2 = np.linalg.eigh(Sr_n3)
T2  = evec2 @ np.diag(np.sqrt(np.abs(ev2)))
pt  = T2 @ np.array([np.cos(angle_ann), np.sin(angle_ann)])
ax2d.annotate(
    r"$\mathcal{E}_{\Sigma_r}\smallsetminus\mathcal{E}_{\Sigma_{\rm cond}}$",
    xy=(pt[0] * 0.82, pt[1] * 0.82),
    xytext=(pt[0] * 1.6, pt[1] * 1.45),
    fontsize=10, color=RED,
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    ha="center",
)
 
ax2d.set_aspect("equal")
ax2d.set_xlabel(r"$\Delta A\,/\,\sigma_A$")
ax2d.set_ylabel(r"$\Delta n\,/\,\sigma_n$")
ax2d.axhline(0, color="k", lw=0.5)
ax2d.axvline(0, color="k", lw=0.5)
ax2d.grid(True, alpha=0.25, lw=0.6)
ax2d.legend(loc="lower right")
 
fig2.suptitle(r"$\pi_\mathcal{I}(\mathcal{E}_{\Sigma_p})=\mathcal{E}_{\Sigma_{\rm cond}}"
              r"\subseteq\mathcal{E}_{\Sigma_r}$"
              r"  (retained: $A$, $n$;  normalised axes)",
              fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig("outputs/fig2_projection.pdf", bbox_inches="tight")
fig2.savefig("outputs/fig2_projection.png", bbox_inches="tight", dpi=200)
print("Figure 2 saved.")
 
 
# ==============================================================================
# FIGURE 3 - Sampling validity  (I = {A, Ea} = {0,2},  removed = {n} = {1})
# ==============================================================================
idx_I   = np.array([0, 2])
idx_rem = np.array([1])
 
Sigma_r_f3  = Sigma_p[np.ix_(idx_I, idx_I)]
S12_f3      = Sigma_p[np.ix_(idx_I, idx_rem)]
S22_f3      = Sigma_p[np.ix_(idx_rem, idx_rem)]
Sc_f3       = Sigma_r_f3 - S12_f3 @ np.linalg.inv(S22_f3) @ S12_f3.T
 
L_sub       = L_full[np.ix_(idx_I, idx_I)]
Sigma_wrong = L_sub @ L_sub.T     # misses L[2,1]^2 term
 
# Normalise everything by sigma_A, sigma_Ea from the true Sigma_r_f3
D_f3    = norm_D(Sigma_r_f3)
Sr_f3_n = norm_cov(Sigma_r_f3,  D_f3)   # diagonal = 1  (target)
Sw_f3_n = norm_cov(Sigma_wrong, D_f3)   # smaller in E_a direction
 
# Samples in normalised space
Lr_n   = np.linalg.cholesky(Sr_f3_n)
Lw_n   = D_f3 @ L_sub
z2     = RNG.standard_normal((2, 4000))
s_ok   = Lr_n  @ z2
s_bad  = Lw_n  @ z2
 
print(f"\nFig 3 - Sigma_r (A,Ea):\n{Sigma_r_f3}")
print(f"L_sub @ L_sub^T:\n{Sigma_wrong}")
print(f"Missing term L[2,1]^2 = {L_full[2,1]**2:.4f}")
 
fig3, (ax_ok, ax_bad) = plt.subplots(1, 2, figsize=(9, 4.2))
 
for ax, samp, cov_n, col, title, slabel in [
    (ax_ok,  s_ok,  Sr_f3_n, GREEN,
     r"Correct:  $L_r = \mathrm{chol}(\Sigma_r)$",
     r"$\mathcal{E}_{L_rL_r^T}$  (= $\mathcal{E}_{\Sigma_r}$)"),
    (ax_bad, s_bad, Sw_f3_n, ORANGE,
     r"Incorrect:  $L_\mathcal{I}$ = sub-block of $L$",
     r"$\mathcal{E}_{L_\mathcal{I}L_\mathcal{I}^T}$"),
]:
    ax.scatter(samp[0], samp[1], color=col, alpha=0.06, s=3, rasterized=True)
 
    xT, yT = ellipse_pts(Sr_f3_n, n_std=2)
    xS, yS = ellipse_pts(cov_n,   n_std=2)
    ax.plot(xT, yT, color=RED, lw=2,       label=r"$2\sigma$: $\mathcal{E}_{\Sigma_r}$")
    ax.plot(xS, yS, color=col, lw=2, ls="--", label=rf"$2\sigma$: {slabel}")
 
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta A\,/\,\sigma_A$")
    ax.set_ylabel(r"$\Delta E_a\,/\,\sigma_{E_a}$")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(loc="lower right")
 
fig3.suptitle(r"Sampling validity  ($\mathcal{I}=\{A,\,E_a\}$, removed: $n$)",
              fontsize=12)
fig3.tight_layout(rect=[0, 0, 1, 0.95])
fig3.savefig("outputs/fig3_sampling_validity.pdf", bbox_inches="tight")
fig3.savefig("outputs/fig3_sampling_validity.png", bbox_inches="tight", dpi=200)
print("Figure 3 saved.")
 
print("\nAll figures saved to outputs/")
