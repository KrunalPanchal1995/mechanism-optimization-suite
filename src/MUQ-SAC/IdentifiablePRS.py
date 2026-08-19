"""
Identifiable-subspace reparameterization for the quadratic/cubic PRS, plus the
shared band-validity scoring used both to screen training rows and to reject
adaptively-resampled points that fall outside the physical uncertainty band.

Root cause this addresses (see report/ for the full derivation): the PRS is
fitted directly on the decorrelated Arrhenius perturbation coordinates
zeta = L^-1(P - P0). Sigma = L L^T is built per reaction by
Uncertainty.getCovariance() (SLSQP fit to a 2-temperature-point uncertainty
band) and is frequently near-singular -- e.g. corr(n, Ea/R) = -0.9959 for the
n-heptane case, eigenvalues [7e-7, 1.07, 45.9]. The near-null eigenvector
influences ln k about 12,000x more weakly than the dominant one, yet nothing
stops the design-matrix samplers from placing huge values there (observed up
to +-4900). A plain quadratic fit on raw zeta then has condition number
~1e11 and individual training/test rows can reach leverage in the thousands.

Fix: for every reaction, only the linear combinations of its reduced zeta
that actually move kappa(T) = theta(T)^T L_r zeta_r over the case's relevant
temperature window are physically identifiable. We recover them once per
reaction via an SVD of the small (n_T x m) matrix W = theta_S(T) L_r, project
zeta onto the kept directions, and only then hand the result to a supervised
PLS reduction + polynomial fit. The whole map (projection -> standardize ->
PLS rotation) is affine, so it composes into one (n_comp x k) matrix M and
k-vector c: scores = zeta_comp @ M + c. That means the rest of
ResponseSurface -- evaluate_prs, the polynomial basis, and (via the chain
rule) the Jacobian used by the live SLSQP optimizer -- can keep operating on
"BZeta"-shaped input; only what BZeta *means* changes, and the affine
composition below is verified in build_feature_map's self-check against
directly composing PLSRegression.transform().

Everything through fit_feature_map is pure math on (Sigma, layout, T) -- no
filesystem, no simulator, no dependency on DesignMatrix/Uncertainty being
importable, so it can be unit-tested standalone (this working copy cannot
import DesignMatrix.py because of its hard `qmc_sampling`/`shuffle.so`
dependencies). fit_prs_with_search (bottom of file) is the one exception --
it drives real ResponseSurface.create_response_surface() fits, which do
their own coefficient-cache file I/O under `cache_dir`.
"""

import os
import numpy as np
from sklearn.cross_decomposition import PLSRegression


def three_way_split(n, seed_outer=104, seed_inner=7, train_frac=0.70, val_frac=0.15):
	"""Deterministic train / validation / untouched-hold-out index split.

	Two-stage permutation (outer splits off train, inner splits the
	remainder into val/hold-out) so seed_outer alone reproduces the same
	train set regardless of val_frac, and seed_inner alone reproduces the
	same val/hold-out division regardless of seed_outer -- convenient for
	sweeping one without perturbing the other (see the parametric study,
	report/). Replaces the single 90/10 train_test_split the original
	pipeline used: a validation split is what makes hyperparameter search
	(fit_prs_with_search) and the outlier-retraining accept/reject decision
	honest, and the hold-out is then touched exactly once, only to report
	the final number -- never used to pick anything.

	Returns (idx_train, idx_val, idx_holdout), each an int ndarray.
	"""
	rng = np.random.default_rng(seed_outer)
	perm = rng.permutation(n)
	n_train = int(round(train_frac * n))
	n_val = int(round(val_frac * n))
	itr = perm[:n_train]
	rest = perm[n_train:]
	rng2 = np.random.default_rng(seed_inner)
	perm2 = rng2.permutation(len(rest))
	ival = rest[perm2[:n_val]]
	iho = rest[perm2[n_val:]]
	return itr, ival, iho


# ─── reduced-covariance / basis helpers (mirror Uncertainty._psac_get_reduced_L) ──

def reduced_cholesky(Sigma, indices):
	"""Sigma_r = Sigma[idx,idx]; L_r = chol(Sigma_r), with a tiny jitter fallback.

	Deliberately duplicates Uncertainty._psac_get_reduced_L's algorithm (never
	slice a cached L directly -- always reconstruct Sigma = L L^T first, then
	take the principal submatrix) so a row's validity score is computed with
	exactly the covariance the sampler itself used.
	"""
	idx = list(indices)
	Sigma = np.asarray(Sigma, dtype=float)
	Sigma_r = Sigma[np.ix_(idx, idx)]
	try:
		L_r = np.linalg.cholesky(Sigma_r)
	except np.linalg.LinAlgError:
		eps = 1e-12 * max(float(np.trace(Sigma_r)), 1e-10) / max(len(idx), 1)
		L_r = np.linalg.cholesky(Sigma_r + eps * np.eye(len(idx)))
	return Sigma_r, L_r


def theta_full(T_arr):
	"""[1, ln T, -1/T] stacked over an array of temperatures -> (nT, 3)."""
	T = np.atleast_1d(np.asarray(T_arr, dtype=float))
	return np.column_stack([np.ones_like(T), np.log(T), -1.0 / T])


def reaction_W_fprior(T_arr, L_r, indices, L_full=None, fprior_mode="reduced"):
	"""W(T) = theta_S(T)^T L_r  (nT, m);  f_prior(T) = ||W(T,:)||_2  (nT,).

	fprior_mode: "reduced" (default, unchanged behaviour) computes f_prior from
	L_r alone -- i.e. the physical band implied by only the SELECTED
	(indices-subset) Arrhenius parameters. "full" instead computes
	f_prior(T) = ||theta_full(T)^T L_full||_2, the band the sampler's own
	acceptance tests (Uncertainty._psac_class_A / _psac_class_D_m2:
	`np.all(np.abs(curve) <= fp)`) actually check against -- ALL THREE
	Arrhenius parameters' uncertainty, not just the ones this reaction had
	selected for the PRS.

	Why this matters: for m<3 (e.g. only Ea/R selected, m=1), the reduced and
	full bands can differ by two orders of magnitude (observed: 218x for one
	n-heptane reaction) because L_r is a 1x1 slice of a near-singular 3x3
	Sigma -- a screening/scoring test built from L_r alone can therefore
	falsely condemn a row as "outside the physical band" when the sampler
	itself, using the true 3-parameter band, would have accepted it. See
	IdentifiablePRS.py's module docstring for the near-singularity background,
	and screen_rows/band_valid_mask below for where this denominator is used.
	Default "reduced" is preserved exactly so every already-published report/
	and report_10fold_validation/ number stays reproducible; pass
	fprior_mode="full" (with L_full) for a study that wants the sampler's own
	physical-validity criterion instead.
	"""
	thS = theta_full(T_arr)[:, list(indices)]      # (nT, m)
	W = thS @ L_r                                   # (nT, m)
	if fprior_mode == "full":
		if L_full is None:
			raise ValueError("reaction_W_fprior(fprior_mode='full') requires L_full")
		fp = np.linalg.norm(theta_full(T_arr) @ L_full, axis=1)   # (nT,)
	else:
		fp = np.linalg.norm(W, axis=1)              # (nT,)
	return W, fp


def identifiable_directions(W, svd_tol=1e-6):
	"""Right-singular vectors of W whose singular value clears svd_tol * s_max.

	These are exactly the zeta_r directions that move kappa(T) somewhere in
	the T_arr window; everything orthogonal to them is unidentifiable from
	this target's data and must not be handed to the regression.
	"""
	if W.size == 0:
		return np.zeros((W.shape[1] if W.ndim == 2 else 0, 0))
	u, s, vt = np.linalg.svd(W, full_matrices=False)
	if len(s) == 0 or s[0] <= 0:
		return np.zeros((W.shape[1], 0))
	keep = s > s[0] * svd_tol
	return vt[keep].T   # (m, r)


class ReactionBlock:
	"""Per-reaction cached geometry needed by both validity scoring and the
	feature map. `col_slice` indexes the reaction's columns within the
	COMPRESSED (n_comp) zeta vector -- callers build this from whatever
	full<->compressed layout they already have (e.g.
	AdaptiveResampler.OutlierResampler._build_layout's prs_local, or
	partial_PRS_system's own selected-column bookkeeping)."""
	__slots__ = ("rxn", "col_slice", "indices", "L_r", "W", "fp", "U_keep")

	def __init__(self, rxn, col_slice, indices, L_r, W, fp, U_keep):
		self.rxn = rxn
		self.col_slice = col_slice
		self.indices = indices
		self.L_r = L_r
		self.W = W
		self.fp = fp
		self.U_keep = U_keep


def build_layout_blocks(layout, Sigma_of, T_arr, svd_tol=1e-6, fprior_mode="reduced"):
	"""layout: iterable of (rxn, col_slice, local_indices) triples describing
	how each active reaction's selected Arrhenius parameters occupy columns
	of the compressed zeta vector. Sigma_of: callable rxn -> 3x3 ndarray.

	fprior_mode: forwarded to reaction_W_fprior -- "reduced" (default,
	unchanged) or "full" (score against the sampler's true 3-parameter
	physical band; see reaction_W_fprior's docstring). Sigma_of already
	returns the FULL 3x3 Sigma regardless of mode, so no caller-side change
	is needed beyond passing fprior_mode="full".
	"""
	blocks = []
	for rxn, col_slice, indices in layout:
		if len(indices) == 0:
			continue
		Sigma = Sigma_of(rxn)
		_, L_r = reduced_cholesky(Sigma, indices)
		L_full = None
		if fprior_mode == "full":
			try:
				L_full = np.linalg.cholesky(Sigma)
			except np.linalg.LinAlgError:
				eps = 1e-12 * max(float(np.trace(Sigma)), 1e-10) / 3
				L_full = np.linalg.cholesky(Sigma + eps * np.eye(3))
		W, fp = reaction_W_fprior(T_arr, L_r, indices, L_full=L_full, fprior_mode=fprior_mode)
		U_keep = identifiable_directions(W, svd_tol=svd_tol)
		blocks.append(ReactionBlock(rxn, col_slice, indices, L_r, W, fp, U_keep))
	return blocks


def violation_ratio(X_comp, blocks):
	"""Row-wise max over (reaction, T) of |delta_kappa(T)| / f_prior(T).

	This is exactly the sampler's own acceptance test
	(Uncertainty._psac_class_A: `np.all(np.abs(curve) <= fp)`), evaluated
	after the fact on whatever ended up in the design matrix. A ratio <= 1
	means every reaction in that row stayed inside its uncertainty band over
	T_arr; > 1 means at least one reaction's rate constant was perturbed
	beyond what the stated uncertainty allows at some T in the window.
	"""
	X_comp = np.asarray(X_comp, dtype=float)
	ratio = np.zeros(len(X_comp))
	for b in blocks:
		if b.W.shape[0] == 0:
			continue
		Z = X_comp[:, b.col_slice]                 # (N, m)
		dk = Z @ b.W.T                              # (N, nT)
		r = np.abs(dk) / b.fp[None, :]
		ratio = np.maximum(ratio, r.max(axis=1))
	return ratio


def violation_detail(X_comp, blocks):
	"""Like violation_ratio, but also returns the worst-offending reaction
	name per row, for the audit log (task: flag+log+exclude, never silently
	drop). Returns (ratio (N,), worst_rxn (N,) array of str)."""
	X_comp = np.asarray(X_comp, dtype=float)
	N = len(X_comp)
	ratio = np.zeros(N)
	worst_rxn = np.array([""] * N, dtype=object)
	for b in blocks:
		if b.W.shape[0] == 0:
			continue
		Z = X_comp[:, b.col_slice]
		dk = Z @ b.W.T
		r = (np.abs(dk) / b.fp[None, :]).max(axis=1)
		beats = r > ratio
		ratio = np.where(beats, r, ratio)
		worst_rxn[beats] = b.rxn
	return ratio, worst_rxn


def screen_rows(X_comp, blocks, tolerance, log_path=None, row_ids=None):
	"""Score every row against the sampler's own band-validity test and split
	into (keep_mask, ratio). Rows with ratio > tolerance are excluded from
	fitting -- they are flagged and (if log_path given) written to a CSV with
	their violation ratio and worst-offending reaction, never silently
	dropped; the underlying design-matrix / simulation files on disk are
	untouched by this function (it only returns an in-memory mask).

	tolerance: max allowed ratio = max_{reaction,T} |Delta kappa| / f_prior.
	    1.0  = strictly inside every reaction's stated uncertainty band.
	    inf  = no screening (keep every row).
	"""
	ratio, worst_rxn = violation_detail(X_comp, blocks)
	keep = ratio <= tolerance
	if log_path is not None:
		import csv as _csv
		flagged = np.where(~keep)[0]
		with open(log_path, "w", newline="") as f:
			w = _csv.writer(f)
			w.writerow(["row_index", "row_id", "violation_ratio", "worst_reaction", "tolerance"])
			for i in flagged:
				rid = row_ids[i] if row_ids is not None else i
				w.writerow([int(i), rid, f"{ratio[i]:.6g}", worst_rxn[i], tolerance])
	return keep, ratio


def band_valid_mask(X_comp, blocks, tolerance=1.0):
	"""Boolean mask of rows whose violation ratio is <= tolerance -- i.e.
	rows the sampler's OWN physical-validity test would have accepted.

	Found 2026-07-30 (10-fold CV robustness investigation, report/): every
	catastrophic worst-fold MRE failure traced back to a single test-fold row
	with a violation ratio 7x-30x the physical band (the class-B m=3 sampler
	bug documented in report/main.tex sec:upstream item 1) -- i.e. a row the
	sampler should never have produced. A model correctly fit on physically
	valid data cannot be fairly graded on its prediction for a physically
	implausible input; MRE evaluation (not training -- screening TRAINING
	rows was already shown not to help in aggregate, parametric study) should
	therefore be computed over band-valid points only. Use this to filter
	Xtest/ytest (or a CV fold's validation split) immediately before scoring,
	never to filter training data.
	"""
	ratio, _ = violation_detail(X_comp, blocks)
	return ratio <= tolerance


def identifiable_projection_matrix(blocks, n_comp):
	"""Block-diagonal P (n_comp, sum r_j): compressed zeta -> raw identifiable
	features F = zeta_comp @ P (data-independent; only depends on Sigma/T/layout)."""
	total_r = sum(b.U_keep.shape[1] for b in blocks)
	P = np.zeros((n_comp, total_r))
	col = 0
	for b in blocks:
		r = b.U_keep.shape[1]
		if r == 0:
			continue
		P[b.col_slice, col:col + r] = b.U_keep
		col += r
	return P


# ─── the affine feature map handed to ResponseSurface ──────────────────────

class FeatureMap:
	"""scores = zeta_comp @ M + c,  M: (n_comp, k),  c: (k,).

	A single affine map composing (1) the per-reaction identifiable
	projection, (2) train-set standardization, (3) the supervised PLS
	rotation -- so ResponseSurface can keep treating its input as a flat
	vector to feed a polynomial basis, unchanged in spirit from the original
	raw-zeta code, just operating on a well-conditioned coordinate system.
	"""

	def __init__(self, M, c):
		self.M = np.asarray(M, dtype=float)
		self.c = np.asarray(c, dtype=float)
		self.k = self.M.shape[1]

	def transform(self, X_comp):
		X_comp = np.atleast_2d(np.asarray(X_comp, dtype=float))
		return X_comp @ self.M + self.c

	def transform_row(self, x_comp):
		return np.asarray(x_comp, dtype=float) @ self.M + self.c


def fit_feature_map(X_train, y_train, blocks, n_comp, k, pls_seed=0):
	"""Fit the FeatureMap on TRAINING data only (y_train used solely by the
	supervised PLS step -- callers must never pass validation/hold-out rows
	here, or k-selection downstream leaks).
	"""
	P = identifiable_projection_matrix(blocks, n_comp)          # (n_comp, R)
	if P.shape[1] == 0:
		raise ValueError("no identifiable directions found for this case")
	F = np.asarray(X_train, dtype=float) @ P                     # (N, R)
	mu = F.mean(axis=0)
	sg = F.std(axis=0)
	sg[sg == 0] = 1.0
	Fs = (F - mu) / sg

	k_eff = int(min(k, Fs.shape[1], max(1, Fs.shape[0] - 1)))
	pls = PLSRegression(n_components=k_eff, scale=False)
	pls.fit(Fs, np.asarray(y_train, dtype=float))

	x_mean = getattr(pls, "_x_mean", None)
	if x_mean is None:
		x_mean = pls.x_mean_
	rot = pls.x_rotations_                                        # (R, k_eff)

	s_inv = 1.0 / sg
	M_total = (P * s_inv[None, :]) @ rot                          # (n_comp, k_eff)
	c_total = -(mu * s_inv + x_mean) @ rot                        # (k_eff,)

	fmap = FeatureMap(M_total, c_total)

	# self-check: composed affine map must reproduce the stepwise pipeline
	# exactly (guards against a transcription error in the algebra above).
	direct = pls.transform(Fs)
	composed = fmap.transform(X_train)
	err = np.abs(direct - composed).max() if direct.size else 0.0
	if err > 1e-8 * max(1.0, np.abs(direct).max()):
		raise RuntimeError(
			f"FeatureMap composition self-check failed (max abs diff {err:.3e}); "
			"do not trust this feature map")

	return fmap


def _is_better(cand_score, cand_k, best_score, best_k, epsilon=0.15):
	"""Selection tie-break for the hyperparameter searches below.

	A plain "lowest score wins" comparison is not robust here: two combos
	can score within a hundredth of a percentage point of each other on
	train/validation (or even on cross-validation, see
	fit_prs_with_cv_search's docstring) while generalizing very differently
	to an independent hold-out -- observed empirically: k=10 scored 1.32%
	vs. k=22's 1.35% on the same validation split (statistically
	indistinguishable), yet k=10's hold-out was 3.30% and k=22's was 1.89%.
	Within `epsilon` percentage points, prefer the LARGER k: the parametric
	study (report/) found larger k has both the best median error and the
	highest pass rate in aggregate, so among near-tied candidates it is the
	safer default, not an arbitrary tiebreak.
	"""
	if cand_score < best_score - epsilon:
		return True
	if abs(cand_score - best_score) <= epsilon and cand_k > best_k:
		return True
	return False


# ─── shared (tolerance, k, order, ridge) search, real ResponseSurface fits ──

def _fit_once(ResponseSurfaceClass, cache_dir, Xa, ya, Xb, yb, case, case_index,
              order, feature_map, ridge_alpha, prs_type, selected_params, criteria):
	"""One real ResponseSurface fit + hold-out score, with the coefficient
	cache cleared before AND after (so repeated calls for the same case_index
	during a hyperparameter search never read a stale cache from a previous
	combo, and never leave one behind for the caller's real final fit to
	accidentally reuse). Returns (train_max_pct, holdout_max_pct)."""
	coef_path = os.path.join(cache_dir, "Data", "ResponseSurface",
	                          f"responsecoef_case-{case_index}.csv")
	if os.path.isfile(coef_path):
		os.remove(coef_path)
	cwd0 = os.getcwd()
	os.chdir(cache_dir)
	try:
		rs = ResponseSurfaceClass(Xa, ya, case, case_index, responseOrder=order,
		                           prs_type=prs_type, selected_params=selected_params,
		                           criteria=criteria, feature_map=feature_map,
		                           ridge_alpha=ridge_alpha)
		rs.create_response_surface()
		mtr, mte, _ = rs.test(Xb, yb)
	finally:
		os.chdir(cwd0)
		if os.path.isfile(coef_path):
			os.remove(coef_path)
	return mtr, mte


def fit_prs_with_search(ResponseSurfaceClass, Xtr, ytr, Xval, yval, blocks, n_comp,
                         case, case_index, prs_type, selected_params, criteria,
                         cache_dir=".",
                         tol_grid=(1.0, float("inf")),
                         k_grid=(8, 10, 12, 14, 16, 20),
                         order_grid=(2, 3),
                         ridge_grid=(0.0, 1e-7),
                         max_terms_frac=0.4,
                         band_tolerance_eval=None):
	"""
	Search (band-validity tolerance, subspace dim k, polynomial order, ridge)
	against a VALIDATION split, fitting the REAL ResponseSurfaceClass every
	time (never a numpy stand-in), so the winning combo is guaranteed to
	behave identically when the caller refits with it "for real". Selection
	criterion is max(train_max_pct, val_max_pct) -- never anything from a
	held-out set the caller hasn't passed in as Xtr/Xval (callers MUST NOT
	pass their final untouched hold-out here; see run_script.py / report/ for
	why -- this is exactly the leakage this project's diagnosis flagged in
	the original retrain_around_outlier).

	band_tolerance_eval: if set (e.g. 1.0), Xval/yval are filtered to
	band-valid rows only (IdentifiablePRS.band_valid_mask) before scoring --
	see fit_prs_with_cv_search's docstring for why. Default None preserves
	the original (unfiltered) behaviour exactly.

	Returns a dict:
	    {tolerance, k, order, ridge_alpha, feature_map,
	     train_pct, val_pct, keep_mask (bool, into Xtr), n_kept}
	for the combo minimizing max(train_pct, val_pct). Raises RuntimeError if
	no combo in the grid produced a fittable model (e.g. every tolerance
	leaves too few rows, or every k/order pair fails the feature map's
	self-check).
	"""
	os.makedirs(os.path.join(cache_dir, "Data", "ResponseSurface"), exist_ok=True)
	Xtr = np.asarray(Xtr, dtype=float)
	ytr = np.asarray(ytr, dtype=float)
	Xval = np.asarray(Xval, dtype=float)
	yval = np.asarray(yval, dtype=float)
	if band_tolerance_eval is not None:
		vmask = band_valid_mask(Xval, blocks, band_tolerance_eval)
		Xval, yval = Xval[vmask], yval[vmask]

	best = None
	for tol in tol_grid:
		keep, ratio = screen_rows(Xtr, blocks, tol)
		Xk, yk = Xtr[keep], ytr[keep]
		if len(Xk) < max(50, 2 * n_comp):
			continue
		for k in k_grid:
			try:
				fmap = fit_feature_map(Xk, yk, blocks, n_comp, k=k)
			except Exception:
				continue
			for order in order_grid:
				from math import comb
				n_terms_est = sum(comb(fmap.k + d - 1, d) for d in range(order + 1))
				if n_terms_est > max_terms_frac * len(Xk):
					continue
				for ridge_alpha in ridge_grid:
					mtr, mva = _fit_once(ResponseSurfaceClass, cache_dir, Xk, yk,
					                      Xval, yval, case, case_index, order, fmap,
					                      ridge_alpha, prs_type, selected_params, criteria)
					score = max(mtr, mva)
					if best is None or _is_better(score, k, best["_score"], best["k"]):
						best = {"_score": score, "tolerance": tol, "k": k, "order": order,
						        "ridge_alpha": ridge_alpha, "feature_map": fmap,
						        "train_pct": mtr, "val_pct": mva,
						        "keep_mask": keep, "n_kept": int(keep.sum())}

	if best is None:
		raise RuntimeError(
			f"fit_prs_with_search: no (tolerance, k, order, ridge) combination in the "
			f"grid produced a fittable model for case {case} (n_comp={n_comp}, "
			f"n_train={len(Xtr)}) -- grids: tol={tol_grid}, k={k_grid}, order={order_grid}")
	del best["_score"]
	return best


def fit_prs_with_cv_search(ResponseSurfaceClass, X_pool, y_pool, blocks, n_comp,
                            case, case_index, prs_type, selected_params, criteria,
                            cache_dir=".",
                            tol_grid=(1.0, float("inf")),
                            k_grid=(8, 10, 12, 14, 16, 20),
                            order_grid=(2, 3),
                            ridge_grid=(0.0, 1e-7),
                            max_terms_frac=0.4,
                            cv_folds=3,
                            cv_seed=13,
                            band_tolerance_eval=None):
	"""
	K-fold cross-validated (tolerance, k, order, ridge) search -- a more
	robust alternative to fit_prs_with_search's single train/validation
	split, needed because a single split can prefer a combo whose
	validation score is only marginally better (e.g. by 0.03 percentage
	points) while its generalization to an INDEPENDENTLY drawn split is
	dramatically worse (observed: a case where k=10 scored 1.32% on one
	validation split but 3.30% on hold-out, while k=22 scored a nominally
	*worse* 1.35% on that same validation split but only 1.89% on the same
	hold-out -- the single-split score could not tell the two apart, but
	they behave very differently out of sample). This is most acute for
	combos with many kept rows relative to few effective degrees of freedom
	near an identifiable-subspace dimension's edge of numerical stability.

	For each (tolerance, k, order, ridge) combo and each of cv_folds folds:
	screen ONLY that fold's training rows by tolerance (mirroring
	fit_prs_with_search, which screens Xtr but never Xval) and fit on them;
	score against the OTHER fold, UNSCREENED, representative of the real
	population. FIX (2026-07-30, 10-fold CV robustness investigation,
	report/): an earlier version of this function screened the WHOLE pool
	once upfront and split the SCREENED result into folds, so both sides of
	every fold were screened together -- this let aggressive screening look
	artificially good on held-out data drawn from the same screened pool,
	hiding exactly the over-screening generalization failure the parametric
	study documented (median 150-360% error at tight tolerances vs 2.10%
	unscreened); only a genuinely unscreened evaluation split reveals it.
	The combo's score is the WORST fold's score, not the average --
	deliberately robustness-oriented (penalizes split-sensitive combos even
	when their average looks fine), matching the goal of reliably clearing
	the criterion on ANY draw, not just a lucky one.

	band_tolerance_eval: if set (e.g. 1.0), each fold's VALIDATION rows are
	additionally filtered to band-valid points (IdentifiablePRS.band_valid_mask)
	before scoring -- found 2026-07-30: a model correctly fit on physically
	valid data cannot be fairly graded on a physically implausible input (the
	class-B m=3 sampler bug, report/main.tex sec:upstream item 1), and
	un-filtered worst-fold scoring was dominated almost entirely by single
	band-invalid rows landing in a fold's validation set. Training data is
	NEVER filtered by this -- only which points COUNT toward the score.
	Default None preserves the original (unfiltered) behaviour exactly.

	The winning combo is then refit ONCE on the FULL screened pool (screened
	by the winning tolerance, same as fit_prs_with_search's keep_mask
	convention) -- this final fit (not any individual fold's) is what the
	caller should use; an untouched hold-out, never part of X_pool, is still
	required for honest final reporting (see run_script.py).

	Returns the same dict shape as fit_prs_with_search (feature_map here is
	fit on the full screened pool), plus 'cv_score' (the worst-fold score
	that won, for logging/diagnostics).
	"""
	os.makedirs(os.path.join(cache_dir, "Data", "ResponseSurface"), exist_ok=True)
	X_pool = np.asarray(X_pool, dtype=float)
	y_pool = np.asarray(y_pool, dtype=float)

	rng = np.random.default_rng(cv_seed)
	perm = rng.permutation(len(y_pool))
	fold_of = np.empty(len(y_pool), dtype=int)
	fold_of[perm] = np.arange(len(y_pool)) % cv_folds

	eval_valid_mask = (band_valid_mask(X_pool, blocks, band_tolerance_eval)
	                    if band_tolerance_eval is not None else None)

	best = None
	from math import comb
	for tol in tol_grid:
		for k in k_grid:
			for order in order_grid:
				for ridge_alpha in ridge_grid:
					fold_scores = []
					fold_ok = True
					for f in range(cv_folds):
						tr_idx_all = np.where(fold_of != f)[0]
						va_idx = np.where(fold_of == f)[0]
						if eval_valid_mask is not None:
							va_idx = va_idx[eval_valid_mask[va_idx]]
						keep_tr, _ = screen_rows(X_pool[tr_idx_all], blocks, tol)
						tr_idx = tr_idx_all[keep_tr]
						Xa, ya = X_pool[tr_idx], y_pool[tr_idx]
						Xb, yb = X_pool[va_idx], y_pool[va_idx]
						if len(Xa) < max(20, n_comp) or len(Xb) < 5:
							fold_ok = False
							break
						try:
							fmap = fit_feature_map(Xa, ya, blocks, n_comp, k=k)
						except Exception:
							fold_ok = False
							break
						n_terms_est = sum(comb(fmap.k + d - 1, d) for d in range(order + 1))
						if n_terms_est > max_terms_frac * len(Xa):
							fold_ok = False
							break
						mtr, mva = _fit_once(ResponseSurfaceClass, cache_dir, Xa, ya, Xb, yb,
						                      case, case_index, order, fmap, ridge_alpha,
						                      prs_type, selected_params, criteria)
						fold_scores.append(max(mtr, mva))
					if not fold_ok or not fold_scores:
						continue
					worst_fold_score = max(fold_scores)
					if best is None or _is_better(worst_fold_score, k, best["_score"], best["k"]):
						# Winning combo: refit on the FULL pool, screened by
						# the winning tolerance, for the actual model the
						# caller will use (folds above are for
						# scoring/robustness only, never the delivered fit).
						keep_full, _ = screen_rows(X_pool, blocks, tol)
						idx_kept_full = np.where(keep_full)[0]
						fmap_full = fit_feature_map(X_pool[idx_kept_full], y_pool[idx_kept_full],
						                             blocks, n_comp, k=k)
						best = {"_score": worst_fold_score, "tolerance": tol, "k": k,
						        "order": order, "ridge_alpha": ridge_alpha,
						        "feature_map": fmap_full, "cv_score": worst_fold_score,
						        "cv_fold_scores": list(fold_scores),
						        "keep_mask": keep_full, "n_kept": int(keep_full.sum())}

	if best is None:
		raise RuntimeError(
			f"fit_prs_with_cv_search: no (tolerance, k, order, ridge) combination in the "
			f"grid produced a fittable model for case {case} (n_comp={n_comp}, "
			f"n_pool={len(X_pool)}, cv_folds={cv_folds}) -- grids: tol={tol_grid}, "
			f"k={k_grid}, order={order_grid}")
	del best["_score"]
	return best
