#!/usr/bin/env python3
"""
demo_run.py -- synthetic reproduction of the full-PRS vs p-PRS pathology,
used only to validate that prs_diagnostics.py runs and that its verdicts make
sense. Replace the synthetic data with your real arrays to use it for real.
"""
import numpy as np
from prs_diagnostics import run_all, study_subset_ceiling

rng = np.random.default_rng(7)

# A deterministic, mildly stiff "response" eta(x) with strong nonlinearity that
# becomes severe along a few high-gain directions (mimics ignition-limit tail).
def eta(P, active, gains):
    z = P[:, active] * gains                       # scaled perturbations
    lin  = z.sum(1)
    quad = 0.5 * (z[:, :3] ** 2).sum(1) + 0.8 * (z[:, 0] * z[:, 1])
    cube = 1.6 * (z[:, 0] ** 3) + 1.2 * (z[:, 1] ** 2) * z[:, 2]   # >quadratic
    return 6.8 + 0.18 * lin + 0.10 * quad + 0.05 * cube

n_react = 30
gains = np.linspace(1.0, 0.05, n_react)            # a few dominant reactions
gains = gains[np.argsort(rng.random(n_react))]
subset = np.argsort(gains)[::-1][:15]              # "sparsity-selected" top 15

# ---- FULL-PRS: perturb ALL 30, small sigma -> narrow, quasi-quadratic --------
sig_full = 0.25
Xf = rng.normal(0, sig_full, size=(11780, n_react))
yf = eta(Xf, np.arange(n_react), gains)
Xf_tr, yf_tr, Xf_te, yf_te = Xf[:9424], yf[:9424], Xf[9424:], yf[9424:]

# ---- p-PRS: perturb ONLY the 15-subset, but the SAME sigma -> variance is now
#      concentrated on high-gain dirs -> wide range + cubic region reached ------
sig_p = 0.55
Xp = rng.normal(0, sig_p, size=(1947, len(subset)))
yp = eta(Xp, np.arange(len(subset)), gains[subset])
Xp_tr, yp_tr, Xp_te, yp_te = Xp[:1557], yp[:1557], Xp[1557:], yp[1557:]

print("\n############### FULL-PRS (expect: clean, low train MRE) ###############")
run_all(Xf_tr, yf_tr, Xf_te, yf_te, degree=2, lam=0.0, outdir="/home/claude/prs_diag")

print("\n\n############### p-PRS (expect: BIAS verdict, plateau) ###############")
run_all(Xp_tr, yp_tr, Xp_te, yp_te, degree=2, lam=0.0, outdir="/home/claude/prs_diag")

print("\n\n############### Subset-ceiling test on the FULL data #################")
study_subset_ceiling(Xf, yf, list(subset), degree=2)
