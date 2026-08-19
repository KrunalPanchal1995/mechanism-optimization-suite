"""
Synthetic unit test for AdaptiveResampler.DOptimalResampler (Task #6).

DesignMatrix.py / the real Uncertainty.py extractor aren't importable in
this working copy (hard qmc_sampling / shuffle.so dependencies -- see
AdaptiveResampler.py's own docstring on _get_dm_instance). So, matching this
repo's existing convention for OutlierResampler's other modes (README:
"verified with synthetic unit tests"), this builds minimal mock uncertainty
extractors that implement exactly the two methods DOptimalResampler actually
calls (_psac_get_reduced_L, cholskyDeCorrelateMat) with the REAL formula
(copied verbatim from Uncertainty.py so the physics is genuine, not faked).

NOTE (2026-07-31, ported from test_build/ to tests/ as part of the Part 1
reorganization): the "not importable" rationale above was inherited
unchanged from test_build/test_doptimal.py, where it is true (test_build/
has neither DesignMatrix.py nor a working Uncertainty.py extractor). It is
NOT necessarily true of src/MUQ-SAC/, which has both files. This has NOT
been re-checked here (this task deliberately does not touch src/MUQ-SAC/ --
see the plan's Part 1). Once this tests/ directory is mirrored into
src/MUQ-SAC/tests/, re-run `import Uncertainty, DesignMatrix` from that
location; if it now succeeds, add a second, slower
tests/test_doptimal_real_extractor.py exercising the same contracts against
the real extractor (per the plan's §1.3), and keep this fast mock-based test
regardless -- it has independent value (fast + hermetic).

Run: python3.9 test_doptimal.py
"""
import os
import sys
import numpy as np

# Same MUQ-SAC-root resolution rule as conftest.py (dirname(dirname(__file__)))
# -- kept inline (not just relying on conftest.py) so this file also works
# when run directly (`python3.9 test_doptimal.py`), not only under pytest,
# since conftest.py is only auto-loaded by the pytest collector.
_MUQSAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MUQSAC_ROOT not in sys.path:
    sys.path.append(_MUQSAC_ROOT)

import AdaptiveResampler as AR
from sklearn.preprocessing import PolynomialFeatures


class MockExtractor:
	"""Minimal stand-in for Uncertainty.UncertaintyExtractor exposing only
	what DOptimalResampler.band_valid_mask and OutlierResampler need."""
	def __init__(self, Sigma):
		self.cholskyDeCorrelateMat = np.linalg.cholesky(Sigma)
		self.activeParameters = [0, 1, 2]

	def _psac_get_reduced_L(self, L_full, indices):
		# verbatim copy of Uncertainty.UncertaintyExtractor._psac_get_reduced_L
		idx = list(indices)
		Sigma = L_full @ L_full.T
		Sigma_r = Sigma[np.ix_(idx, idx)]
		try:
			L_r = np.linalg.cholesky(Sigma_r)
		except np.linalg.LinAlgError:
			eps = 1e-12 * max(float(np.trace(Sigma_r)), 1e-10) / max(len(idx), 1)
			L_r = np.linalg.cholesky(Sigma_r + eps * np.eye(len(idx)))
		return Sigma_r, L_r


def make_resampler(n_rxn=5, seed=0):
	Sigma = np.array([[1.22, -0.0112, 2.623],
	                   [-0.0112, 0.001445, -0.2560],
	                   [2.623, -0.2560, 45.73]])
	rxns = [f"rxn{i}" for i in range(n_rxn)]
	unsrt = {r: MockExtractor(Sigma) for r in rxns}
	selected_params = np.ones(3 * n_rxn, dtype=int)   # Full PRS, everything selected
	# temps_for_validity deliberately spans the GLOBAL mechanism-validity
	# window (300-2500K, matching what the real class-A/B/C samplers
	# themselves check -- Uncertainty._psac_class_A etc.), not a single
	# case's narrow local temperature window (that narrower window is
	# Stage 1's identifiable-subspace concern, see IdentifiablePRS.py): a
	# newly resampled point feeds a full perturbed mechanism.yaml, which
	# must be physically plausible everywhere it might be evaluated, not
	# just at the one case's nominal T,P,phi.
	res = AR.DOptimalResampler(
		unsrt, selected_params, design="A1+B1+C1", prs_type="Full",
		base_dir="/tmp/nonexistent_no_class_pool", case_index=0,
		N_new=200, delta_fraction=0.2, seed=seed,
		band_tolerance=1.0, temps_for_validity={"__default__": np.linspace(300., 2500., 12)},
		acquisition_alpha=0.6, candidate_oversample=8,
	)
	assert res.class_pool is None, "test assumes box-fallback (no cached class pkls)"
	return res, Sigma


def build_basis_quadratic(rows):
	rows = np.atleast_2d(np.asarray(rows, dtype=float))
	return PolynomialFeatures(degree=2, include_bias=True).fit_transform(rows)


def test_band_valid_mask_rejects_extreme_rows():
	res, Sigma = make_resampler()
	n_full = res.n_full
	rng = np.random.default_rng(1)
	# valid rows: |zeta| small (well inside a unit-ball-ish region)
	valid_rows = rng.normal(scale=0.1, size=(50, n_full))
	# invalid rows: one reaction's block set to a huge value (mirrors the
	# real class-B m=3 defect: ~4900 in a near-null direction)
	invalid_rows = valid_rows.copy()
	invalid_rows[:, 2] = 20000.0   # rxn0's zeta_3 (near-null direction of Sigma,
	                                # see report/ -- real samples reach ~5000 here)

	mask_valid = res.band_valid_mask(valid_rows)
	mask_invalid = res.band_valid_mask(invalid_rows)
	print(f"valid rows accepted: {mask_valid.mean()*100:.0f}%  "
	      f"(expect high, most/all inside the band)")
	print(f"invalid rows accepted: {mask_invalid.mean()*100:.0f}%  (expect 0%)")
	assert mask_valid.mean() > 0.8, "well-scaled rows should mostly pass the band test"
	assert mask_invalid.sum() == 0, "rows with a 3000-sigma perturbation must ALL be rejected"
	print("PASS: band_valid_mask correctly gates extreme rows\n")


def test_generate_candidates_are_all_band_valid():
	res, Sigma = make_resampler()
	n_prs = res.n_prs
	rng = np.random.default_rng(2)
	X_ref = rng.normal(scale=0.3, size=(300, n_prs))
	outlier = X_ref[0:1] * 2.0

	prs_cand, full_cand = res.generate_candidates(outlier, X_ref, n_target=100, local_fraction=0.5)
	print(f"generated {len(prs_cand)} band-valid candidates (requested budget 100, "
	      f"oversample x{res.candidate_oversample})")
	assert len(prs_cand) > 0, "should find SOME band-valid candidates from a modest local/global draw"
	# Re-verify EVERY returned candidate independently satisfies the band test
	# (the whole point of gating -- must hold with zero exceptions)
	remask = res.band_valid_mask(full_cand)
	assert remask.all(), "every candidate generate_candidates returns must itself be band-valid"
	print("PASS: every generated candidate is independently re-verified band-valid\n")
	return res, X_ref, outlier, prs_cand, full_cand


def test_d_optimal_prefers_isolated_points():
	res, Sigma = make_resampler()
	n_prs = res.n_prs
	rng = np.random.default_rng(3)

	X_train = rng.normal(scale=0.2, size=(80, n_prs))
	B_train = build_basis_quadratic(X_train)

	# candidates: half are near-duplicates of existing training rows
	# (low information gain), half are genuinely novel (far from training
	# cloud) -- a correct D-optimal criterion must prefer the novel half.
	dup = X_train[:40] + rng.normal(scale=0.01, size=(40, n_prs))
	novel = rng.normal(loc=1.5, scale=0.1, size=(40, n_prs))  # shifted cluster
	prs_cand = np.vstack([dup, novel])

	picked = res.select_d_optimal(prs_cand, B_train, build_basis_quadratic, n_select=10)
	frac_novel = np.mean(picked >= 40)
	print(f"pure D-optimal (alpha={res.acquisition_alpha}) picked {len(picked)} rows, "
	      f"{frac_novel*100:.0f}% from the novel (far-from-training) cluster")
	assert frac_novel > 0.8, "D-optimality must strongly prefer novel/isolated points over near-duplicates"
	print("PASS: greedy D-optimal selection prefers isolated candidates\n")


def test_det_increases_monotonically():
	res, Sigma = make_resampler()
	n_prs = res.n_prs
	rng = np.random.default_rng(4)
	X_train = rng.normal(scale=0.3, size=(60, n_prs))
	B_train = build_basis_quadratic(X_train)
	prs_cand = rng.normal(scale=0.6, size=(60, n_prs))

	picked = res.select_d_optimal(prs_cand, B_train, build_basis_quadratic, n_select=8)
	Bc = build_basis_quadratic(prs_cand)
	p = Bc.shape[1]
	G = B_train.T @ B_train + (1e-10 * np.trace(B_train.T @ B_train) / p) * np.eye(p)
	dets = [np.linalg.slogdet(G)[1]]
	for i in picked:
		G = G + np.outer(Bc[i], Bc[i])
		dets.append(np.linalg.slogdet(G)[1])
	diffs = np.diff(dets)
	print(f"log-det trajectory increments: {np.round(diffs, 3)}")
	assert np.all(diffs > -1e-8), "det(G) must be non-decreasing as D-optimal picks are added"
	assert np.all(np.diff(diffs) <= 1e-6), "greedy picks should show diminishing (or equal) marginal gain"
	print("PASS: det(Gram) increases monotonically with diminishing returns\n")


def test_targeting_alpha_extremes():
	res, Sigma = make_resampler()
	n_prs = res.n_prs
	rng = np.random.default_rng(5)
	X_train = rng.normal(scale=0.3, size=(60, n_prs))
	B_train = build_basis_quadratic(X_train)

	outlier = np.zeros((1, n_prs))
	outlier[0, 0] = 5.0    # a specific point far out along axis 0
	near_outlier = np.tile(outlier, (20, 1)) + rng.normal(scale=0.05, size=(20, n_prs))
	far_away = rng.normal(loc=-5.0, scale=0.3, size=(20, n_prs))  # informative but irrelevant to outlier
	prs_cand = np.vstack([near_outlier, far_away])

	res.acquisition_alpha = 0.0   # pure targeting
	picked_tgt = res.select_d_optimal(prs_cand, B_train, build_basis_quadratic,
	                                   n_select=10, target_points_prs=outlier)
	frac_near_tgt = np.mean(picked_tgt < 20)
	print(f"alpha=0.0 (pure targeting) picked {frac_near_tgt*100:.0f}% from the near-outlier cluster (expect high)")
	assert frac_near_tgt > 0.8

	res.acquisition_alpha = 1.0   # pure D-optimal, ignores targeting
	picked_dopt = res.select_d_optimal(prs_cand, B_train, build_basis_quadratic,
	                                    n_select=10, target_points_prs=outlier)
	frac_near_dopt = np.mean(picked_dopt < 20)
	print(f"alpha=1.0 (pure D-optimal) picked {frac_near_dopt*100:.0f}% from the near-outlier cluster "
	      f"(expect lower than alpha=0 -- far_away is equally/more informative and unweighted by target)")
	assert frac_near_dopt <= frac_near_tgt
	print("PASS: acquisition_alpha correctly interpolates targeting <-> pure D-optimality\n")


def test_resample_d_optimal_contract():
	res, Sigma = make_resampler()
	n_prs = res.n_prs
	rng = np.random.default_rng(6)
	X_ref = rng.normal(scale=0.3, size=(200, n_prs))
	X_train = X_ref[:150]
	outlier = X_ref[0:2] * 1.8

	rs_mock = build_basis_quadratic  # stand-in for ResponseSurface.build_basis
	prs_rows, full_rows, sel_mask, row_class, info = res.resample_d_optimal(
		outlier, X_ref, X_train, rs_mock)

	print(f"resample_d_optimal: {len(prs_rows)} rows returned, row_class counts: "
	      f"{ {c: row_class.count(c) for c in set(row_class)} }, info={info}")
	assert prs_rows.shape[1] == n_prs
	assert full_rows.shape[1] == res.n_full
	assert len(sel_mask) == res.n_full
	assert len(row_class) == len(prs_rows) == len(full_rows)
	assert row_class[:2] == ["OUT", "OUT"], "outlier rows must be prepended verbatim"
	assert info["n_selected"] <= info["n_budget"]
	# every DOPT row must itself be band-valid (re-verify end to end)
	dopt_idx = [i for i, c in enumerate(row_class) if c == "DOPT"]
	if dopt_idx:
		remask = res.band_valid_mask(full_rows[dopt_idx])
		assert remask.all(), "every selected DOPT row must be band-valid"
	print("PASS: resample_d_optimal contract matches other resample* methods\n")


def test_regression_existing_modes_still_work():
	"""Sanity: DOptimalResampler subclassing didn't break inherited methods."""
	res, Sigma = make_resampler()
	n_prs = res.n_prs
	rng = np.random.default_rng(7)
	X_ref = rng.normal(scale=0.3, size=(100, n_prs))
	outlier = X_ref[0:1]

	for mode, fn in [("resample", res.resample), ("resample_hybrid", res.resample_hybrid)]:
		prs_rows, full_rows, sel_mask, row_class = fn(outlier, X_ref)
		assert len(prs_rows) == len(full_rows) == len(row_class)
		print(f"PASS: inherited {mode}() still works ({len(prs_rows)} rows)")
	print()


if __name__ == "__main__":
	test_band_valid_mask_rejects_extreme_rows()
	test_generate_candidates_are_all_band_valid()
	test_d_optimal_prefers_isolated_points()
	test_det_increases_monotonically()
	test_targeting_alpha_extremes()
	test_resample_d_optimal_contract()
	test_regression_existing_modes_still_work()
	print("ALL TESTS PASSED")
