"""
Figures for Appendix: Properties of the Principal Submatrix and Sampling Validity

Three figures:
  Fig 1 – Ellipsoid containment  E_cond ⊆ E_Sr  (correlated vs uncorrelated)
  Fig 2 – 3-D: projection shadow = E_Sr (MARGINAL), cross-section at Ea=0 = E_Scond
  Fig 3 – Sampling validity:  L_I = chol(Sigma_I)  vs sub-block  L[I,I]

KEY CORRECTIONS vs original:
  • L_full replaced with synthetic non-degenerate example (rho_An~0.41)
  • Fig 2: the PROJECTION SHADOW of E_Sp is E_Sr (not E_Scond).
    E_Scond is the CROSS-SECTION at z=0 (fixing Ea at nominal).
    Proof: min_z Q(y,z) = y^T Sr^{-1} y  (marginal quadratic form)
           Q(y,0)       = y^T Scond^{-1} y  (since [Sp^{-1}]_{11} = Scond^{-1})
    The original text had these backwards.
  • Fig 3 partition: I={n,Ea}={1,2}, removed={A}={0}  (was {A,Ea} which gave only 0.7% error)
  • Notation: correct factor = L_I = chol(Sigma_I); wrong factor = L[I,I] = sub-block of L
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import warnings
warnings.filterwarnings("ignore")

# ── global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "mathtext.fontset" : "cm",
    "font.size"        : 11,
    "axes.labelsize"   : 12,
    "axes.titlesize"   : 11,
    "legend.fontsize"  : 9,
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

# ── Numerical example ─────────────────────────────────────────────────────────
# Synthetic lower-triangular factor giving moderate correlations
# rho_An=0.41,  rho_AEa=0.63,  rho_nEa=0.55  (non-degenerate, suitable for figures)
L_full = np.array([[1.40, 0.00, 0.00],
                   [0.50, 1.10, 0.00],
                   [0.80, 0.40, 0.90]])

Sigma_p = L_full @ L_full.T   # [[1.96,0.70,1.12],[0.70,1.46,0.84],[1.12,0.84,1.61]]

# ── Partition for Figs 1 & 2:  I = {A,n} = {0,1},  removed = {Ea} = {2} ──────
S11         = Sigma_p[:2, :2]
S12         = Sigma_p[:2, 2:3]
S22         = Sigma_p[2:,  2:]
Sigma_r     = S11                                # marginal  (= Sigma_11)
Sigma_cond  = S11 - S12 @ np.linalg.inv(S22) @ S12.T  # conditional (Schur complement)

# Uncorrelated variant for Fig 1 right panel
Sp_unc        = Sigma_p.copy(); Sp_unc[:2,2] = 0.0; Sp_unc[2,:2] = 0.0
S11_unc       = Sp_unc[:2,:2]; S12_unc = Sp_unc[:2,2:3]; S22_unc = Sp_unc[2:,2:]
Sc_unc        = S11_unc - S12_unc @ np.linalg.inv(S22_unc) @ S12_unc.T

# ── Normalisation helpers ─────────────────────────────────────────────────────
def norm_D(cov):
    return np.diag(1.0 / np.sqrt(np.diag(cov)))

def norm_cov(cov, D):
    return D @ cov @ D

# Figs 1 & 2: normalise by Sigma_r diagonal
D_r   = norm_D(Sigma_r)
Sr_n  = norm_cov(Sigma_r,    D_r)
Sc_n  = norm_cov(Sigma_cond, D_r)

D_unc    = norm_D(S11_unc)
Sr_unc_n = norm_cov(S11_unc, D_unc)
Sc_unc_n = norm_cov(Sc_unc,  D_unc)

# Fig 2: normalise full Sigma_p by its own diagonal
D_p3  = norm_D(Sigma_p)
Sp_n  = norm_cov(Sigma_p, D_p3)
D_An  = np.diag(np.diag(D_p3)[:2])
Sc_n3 = norm_cov(Sigma_cond, D_An)   # cross-section at Ea=0
Sr_n3 = norm_cov(Sigma_r,    D_An)   # projection shadow (marginal)

# ── Geometry helpers ──────────────────────────────────────────────────────────
def ellipse_pts(cov2, n_std=1.0, n_pts=400):
    ev, evec = np.linalg.eigh(cov2)
    t  = np.linspace(0, 2*np.pi, n_pts)
    xy = evec @ (n_std * np.sqrt(np.abs(ev[:,None])) * np.array([np.cos(t), np.sin(t)]))
    return xy[0], xy[1]

def ellipsoid_surf(cov3, n_pts=60):
    u = np.linspace(0, 2*np.pi, n_pts); v = np.linspace(0, np.pi, n_pts)
    sphere = np.array([np.outer(np.cos(u), np.sin(v)).ravel(),
                       np.outer(np.sin(u), np.sin(v)).ravel(),
                       np.outer(np.ones_like(u), np.cos(v)).ravel()])
    ev, evec = np.linalg.eigh(cov3)
    pts = (evec @ np.diag(np.sqrt(np.abs(ev)))) @ sphere
    return [pts[i].reshape(n_pts,n_pts) for i in range(3)]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Ellipsoid containment  E_Scond ⊆ E_Sr
# ══════════════════════════════════════════════════════════════════════════════
fig1, (ax_c, ax_u) = plt.subplots(1, 2, figsize=(9, 4.2))

for ax, Sr_p, Sc_p, title in [
    (ax_c, Sr_n,     Sc_n,
     r"Correlated  ($\Sigma_{12}\neq\mathbf{0}$)"),
    (ax_u, Sr_unc_n, Sc_unc_n,
     r"Uncorrelated  ($\Sigma_{12}=\mathbf{0}$)"),
]:
    xr, yr = ellipse_pts(Sr_p)
    xc, yc = ellipse_pts(Sc_p)

    ax.fill(xr, yr, color=RED,  alpha=0.18, zorder=1)
    ax.plot(xr, yr, color=RED,  lw=2,       zorder=3,
            label=r"$\mathcal{E}_{\Sigma_r}$  (marginal)")
    ax.fill(xc, yc, color=BLUE, alpha=0.35, zorder=2)
    ax.plot(xc, yc, color=BLUE, lw=2, ls="--", zorder=4,
            label=r"$\mathcal{E}_{\Sigma_{\mathrm{cond}}}$  (conditional)")

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta A\,/\,\sigma_A$")
    ax.set_ylabel(r"$\Delta n\,/\,\sigma_n$")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(loc="lower right")

fig1.suptitle(
    r"$\mathcal{E}_{\Sigma_{\mathrm{cond}}}\;\subseteq\;\mathcal{E}_{\Sigma_r}$"
    r"  (retained: $A$, $n$;  removed: $E_a$)",
    fontsize=12)
fig1.tight_layout(rect=[0,0,1,0.95])
fig1.savefig("outputs/fig1_ellipsoid_containment.pdf", bbox_inches="tight")
fig1.savefig("outputs/fig1_ellipsoid_containment.png", bbox_inches="tight", dpi=200)
print("Figure 1 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – 3-D projection and 2-D view
#
#  CORRECTED geometry:
#   • Projection shadow  π_I(E_Sp) = E_Sr      [RED, outer, SOLID]
#   • Cross-section at Ea=0         = E_Scond   [BLUE, inner, DASHED]
#   • Proof: min_z Q(y,z) = y^T Sr^{-1} y   => shadow = E_Sr
#            Q(y,0) = y^T Scond^{-1} y        => slice  = E_Scond
#   Both ellipses shown on the floor of the 3-D plot.
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(11.5, 5.0))
gs   = fig2.add_gridspec(1, 2, wspace=0.55)

# ── Left: 3-D ellipsoid ───────────────────────────────────────────────────────
ax3d = fig2.add_subplot(gs[0], projection="3d")

X, Y, Z = ellipsoid_surf(Sp_n, n_pts=60)

# Floor z-level from ellipsoid extent (before drawing)
ev3, evec3 = np.linalg.eigh(Sp_n)
zfloor = -np.sqrt(np.max(ev3)) * 1.45

ax3d.plot_surface(X, Y, Z, alpha=0.18, color=PURPLE,
                  rstride=2, cstride=2, linewidth=0)

# Drop lines from equatorial ring to floor
T3   = evec3 @ np.diag(np.sqrt(np.abs(ev3)))
th   = np.linspace(0, 2*np.pi, 24, endpoint=False)
ring = T3 @ np.array([np.cos(th), np.sin(th), np.zeros(24)])
for i in range(24):
    ax3d.plot([ring[0,i]]*2, [ring[1,i]]*2, [ring[2,i], zfloor],
              color=GREY, lw=0.5, alpha=0.28)

# Floor ellipses  ── CORRECTED LABELS ──
xr3, yr3 = ellipse_pts(Sr_n3)   # E_Sr  = projection shadow  (OUTER, RED)
xc3, yc3 = ellipse_pts(Sc_n3)   # E_Scond = cross-section    (INNER, BLUE)

ax3d.plot(xr3, yr3, zfloor*np.ones_like(xr3), color=RED,  lw=2.5, zorder=5,
          label=r"$\mathcal{E}_{\Sigma_r}$ (shadow)")
ax3d.plot(xc3, yc3, zfloor*np.ones_like(xc3), color=BLUE, lw=2.5, ls="--", zorder=5,
          label=r"$\mathcal{E}_{\Sigma_{\mathrm{cond}}}$ (slice)")

ax3d.set_xlabel(r"$\Delta A/\sigma_A$",       labelpad=5)
ax3d.set_ylabel(r"$\Delta n/\sigma_n$",       labelpad=5)
ax3d.set_zlabel(r"$\Delta E_a/\sigma_{E_a}$", labelpad=5)
ax3d.legend(loc="upper left", fontsize=8.5)
ax3d.view_init(elev=22, azim=-50)

# ── Right: 2-D floor view ─────────────────────────────────────────────────────
ax2d = fig2.add_subplot(gs[1])

ax2d.fill(xr3, yr3, color=RED,  alpha=0.18, zorder=1)
ax2d.plot(xr3, yr3, color=RED,  lw=2,       zorder=3,
          label=r"$\mathcal{E}_{\Sigma_r}$: projection shadow $\pi_\mathcal{I}(\mathcal{E}_{\Sigma_p})$")
ax2d.fill(xc3, yc3, color=BLUE, alpha=0.35, zorder=2)
ax2d.plot(xc3, yc3, color=BLUE, lw=2, ls="--", zorder=4,
          label=r"$\mathcal{E}_{\Sigma_{\mathrm{cond}}}$: cross-section at $E_a\!=\!E_{a,o}$")

# Annotate the annular gap
angle_ann = np.radians(38)
ev2, evec2 = np.linalg.eigh(Sr_n3)
T2 = evec2 @ np.diag(np.sqrt(np.abs(ev2)))
pt = T2 @ np.array([np.cos(angle_ann), np.sin(angle_ann)])
ax2d.annotate(
    r"$\mathcal{E}_{\Sigma_r} \backslash \mathcal{E}_{\Sigma_{\mathrm{cond}}}$",
    xy=(pt[0]*0.87, pt[1]*0.87),
    xytext=(pt[0]*1.68, pt[1]*1.52),
    fontsize=10, color=RED,
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
    ha="center",
)

ax2d.set_aspect("equal")
ax2d.set_xlabel(r"$\Delta A\,/\,\sigma_A$")
ax2d.set_ylabel(r"$\Delta n\,/\,\sigma_n$")
ax2d.axhline(0, color="k", lw=0.5); ax2d.axvline(0, color="k", lw=0.5)
ax2d.grid(True, alpha=0.25, lw=0.6)
ax2d.legend(loc="lower right", fontsize=8.5)

fig2.suptitle(
    r"$\pi_\mathcal{I}(\mathcal{E}_{\Sigma_p})=\mathcal{E}_{\Sigma_r}"
    r"\supseteq\mathcal{E}_{\Sigma_{\mathrm{cond}}}$"
    r"  (retained: $A$, $n$;  normalised axes)",
    fontsize=12)
fig2.tight_layout(rect=[0,0,1,0.94])
fig2.savefig("outputs/fig2_projection.pdf", bbox_inches="tight")
fig2.savefig("outputs/fig2_projection.png", bbox_inches="tight", dpi=200)
print("Figure 2 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Sampling validity
#   I = {n, Ea} = {1,2},  removed = {A} = {0}
#
#   Correct:   L_I    = chol(Sigma_I)       -- fresh Cholesky (valid factor)
#   Incorrect: L[I,I] = sub-block of L      -- index-extracted (NOT valid)
#
#   With I={n,Ea}: missing L[1,0]^2=0.25 and L[2,0]^2=0.64
#     Diagonal error (Ea) : ~40%
#     Off-diagonal error  : ~48%
# ══════════════════════════════════════════════════════════════════════════════
idx_I   = np.array([1, 2])   # retained: n, Ea
idx_rem = np.array([0])       # removed:  A

Sigma_I       = Sigma_p[np.ix_(idx_I, idx_I)]
L_sub_wrong   = L_full[np.ix_(idx_I, idx_I)]   # L[[1,2],:][:,[1,2]]
Sigma_wrong   = L_sub_wrong @ L_sub_wrong.T    # missing first-column contributions

print(f"\nFig 3: I={{n,Ea}}={{1,2}}, removed={{A}}={{0}}")
print(f"Sigma_I (true):\n{Sigma_I}")
print(f"L[I,I]@L[I,I]^T (wrong):\n{Sigma_wrong}")
print(f"Diagonal error Ea: {100*(Sigma_I[1,1]-Sigma_wrong[1,1])/Sigma_I[1,1]:.1f}%")
print(f"Off-diagonal error: {100*(Sigma_I[0,1]-Sigma_wrong[0,1])/Sigma_I[0,1]:.1f}%")

D_f3    = norm_D(Sigma_I)
SI_n    = norm_cov(Sigma_I,   D_f3)   # normalised target (diagonal = 1)
Sw_n    = norm_cov(Sigma_wrong, D_f3)  # normalised wrong covariance

L_I_correct = np.linalg.cholesky(SI_n)   # correct factor (fresh Cholesky)
L_I_wrong   = D_f3 @ L_sub_wrong         # wrong factor   (sub-block, normalised)

z2    = RNG.standard_normal((2, 5000))
s_ok  = L_I_correct @ z2
s_bad = L_I_wrong   @ z2

fig3, (ax_ok, ax_bad) = plt.subplots(1, 2, figsize=(9, 4.4))

panels = [
    (ax_ok,  s_ok,  SI_n, GREEN,
     r"Correct: $L_\mathcal{I} = \mathrm{chol}(\Sigma_\mathcal{I})$",
     r"$\mathcal{E}_{L_\mathcal{I}L_\mathcal{I}^T}$ ($=\mathcal{E}_{\Sigma_\mathcal{I}}$)"),
    (ax_bad, s_bad, Sw_n, ORANGE,
     r"Incorrect: $L[\mathcal{I},\mathcal{I}]$ = sub-block of $L$",
     r"$\mathcal{E}_{L[\mathcal{I},\mathcal{I}]\,L[\mathcal{I},\mathcal{I}]^T}$"),
]

for ax, samp, cov_n, col, title, ellipse_lbl in panels:
    ax.scatter(samp[0], samp[1], color=col, alpha=0.07, s=3, rasterized=True)

    xT, yT = ellipse_pts(SI_n,  n_std=2)
    xS, yS = ellipse_pts(cov_n, n_std=2)
    ax.plot(xT, yT, color=RED, lw=2,
            label=r"$2\sigma$: $\mathcal{E}_{\Sigma_\mathcal{I}}$ (target)")
    ax.plot(xS, yS, color=col, lw=2, ls="--",
            label=r"$2\sigma$: " + ellipse_lbl)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$\Delta n\,/\,\sigma_n$")
    ax.set_ylabel(r"$\Delta E_a\,/\,\sigma_{E_a}$")
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=8.5)

fig3.suptitle(
    r"Sampling validity  ($\mathcal{I}=\{n,\,E_a\}$, removed: $A$)"
    r"  — correct $L_\mathcal{I}$ vs.\ sub-block $L[\mathcal{I},\mathcal{I}]$",
    fontsize=12)
fig3.tight_layout(rect=[0,0,1,0.95])
fig3.savefig("outputs/fig3_sampling_validity.pdf", bbox_inches="tight")
fig3.savefig("outputs/fig3_sampling_validity.png", bbox_inches="tight", dpi=200)
print("Figure 3 saved.")

print("\nAll figures saved to outputs/")
