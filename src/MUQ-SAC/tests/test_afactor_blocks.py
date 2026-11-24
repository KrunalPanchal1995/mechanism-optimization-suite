"""
Gate test for the A-factor-only ("perturbation_type == 'factor'") crash fixed
in IdentifiablePRS.sigma_from_cholesky (see bug.md / CHANGELOG).

Root cause: for a factor reaction, Uncertainty.py sets
    cholskyDeCorrelateMat = np.array([min(self.uncertainties)])   # shape (1,)
    activeParameters       = [rIndex + '_A']                       # length 1
A plain `L @ L.T` on that 1-D array broadcasts to a 0-d numpy SCALAR (not a
1x1 matrix), and IdentifiablePRS.reduced_cholesky's
`Sigma[np.ix_(idx, idx)]` then raises "too many indices for array: array is
0-dimensional, but 2 were indexed" -- exactly the traceback in bug.md.

This test never imports DesignMatrix/Uncertainty (same importability
constraint documented in IdentifiablePRS.py's module docstring and
tests/test_doptimal.py) -- it exercises IdentifiablePRS directly, which is
pure numpy + sklearn.

Run: python3.9 tests/test_afactor_blocks.py
"""
import os
import sys
import numpy as np

_MUQSAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MUQSAC_ROOT not in sys.path:
    sys.path.append(_MUQSAC_ROOT)

import IdentifiablePRS as IPRS


def test_sigma_from_cholesky_factor_reaction():
    """m=1 (factor) reaction: scalar L -> zero-padded 3x3 Sigma, Sigma[0,0] = sigma^2."""
    sigma_lnA = 1.3863
    for L in (np.array([sigma_lnA]), np.array(sigma_lnA), np.array([[sigma_lnA]])):
        Sigma = IPRS.sigma_from_cholesky(L, n_active=1)
        assert Sigma.shape == (3, 3)
        assert np.isclose(Sigma[0, 0], sigma_lnA ** 2)
        assert np.allclose(Sigma[1:, :], 0.0) and np.allclose(Sigma[:, 1:], 0.0)
    print("PASS: sigma_from_cholesky handles scalar/0-d/1-d/1x1 factor-reaction L")


def test_sigma_from_cholesky_full_reaction_unchanged():
    """m=3 reaction: behaviour matches the old `L @ L.T` exactly."""
    Sigma_true = np.array([[1.22, -0.0112, 2.623],
                            [-0.0112, 0.001445, -0.2560],
                            [2.623, -0.2560, 45.73]])
    L = np.linalg.cholesky(Sigma_true)
    Sigma = IPRS.sigma_from_cholesky(L, n_active=3)
    assert np.allclose(Sigma, L @ L.T)
    assert np.allclose(Sigma, Sigma_true, atol=1e-8)
    print("PASS: sigma_from_cholesky reproduces L @ L.T for a real 3x3 reaction")


def test_sigma_from_cholesky_rejects_diverged_shapes():
    """n_active disagreeing with L's size must fail loudly, not silently."""
    try:
        IPRS.sigma_from_cholesky(np.array([1.0]), n_active=3)
        assert False, "expected ValueError for scalar L with n_active=3"
    except ValueError:
        pass
    try:
        IPRS.sigma_from_cholesky(np.eye(2), n_active=3)
        assert False, "expected ValueError for 2x2 L with n_active=3"
    except ValueError:
        pass
    print("PASS: sigma_from_cholesky raises on activeParameters/L divergence")


def test_reduced_cholesky_rejects_0d_sigma():
    """Guards the exact crash: a 0-d Sigma must fail with a clear message,
    never the bare numpy IndexError from bug.md."""
    try:
        IPRS.reduced_cholesky(np.array(1.9218120557), [0])
        assert False, "expected ValueError for a 0-d Sigma"
    except ValueError as e:
        assert "0-dimensional" not in str(type(e))  # our message, not numpy's
        assert "square 2-D matrix" in str(e)
    print("PASS: reduced_cholesky rejects a 0-d Sigma with a clear message")


def _factor_reaction(sigma_lnA):
    """A single A-factor-only reaction's (col_slice, indices) triple, mirroring
    _build_prs_layout's/phase10_prs.py's layout convention."""
    L = np.array([sigma_lnA])
    return L


def test_build_layout_blocks_factor_reaction():
    """End-to-end: a factor-reaction layout must build a valid ReactionBlock
    (this is exactly Case-52's crash site, phase10_prs.py's _sigma_of ->
    build_layout_blocks -> reduced_cholesky)."""
    sigma_lnA = 0.5
    unsrt = {"rxnA": {"L": _factor_reaction(sigma_lnA), "n_active": 1}}
    layout = [("rxnA", slice(0, 1), [0])]

    def sigma_of(rxn):
        return IPRS.sigma_from_cholesky(unsrt[rxn]["L"], n_active=unsrt[rxn]["n_active"])

    T_arr = np.array([750.0, 1000.0, 1250.0])
    blocks = IPRS.build_layout_blocks(layout, sigma_of, T_arr, svd_tol=1e-6)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.L_r.shape == (1, 1)
    assert np.isclose(b.L_r[0, 0], sigma_lnA)
    # W(T) = theta_S(T) @ L_r = 1 * sigma_lnA, constant over T (theta_S's
    # column for index 0 is all-ones)
    assert np.allclose(b.W, sigma_lnA)
    assert np.allclose(b.fp, sigma_lnA)
    assert b.U_keep.shape == (1, 1)
    assert np.isclose(abs(b.U_keep[0, 0]), 1.0)
    print("PASS: build_layout_blocks handles a single A-factor-only reaction "
          "(the exact Case-52 crash site)")


def test_violation_ratio_matches_class_A_test_for_factor_reaction():
    """ratio = |Delta kappa|/f_prior must collapse to exactly |zeta| for m=1,
    matching the sampler's own class-A acceptance test."""
    sigma_lnA = 0.7
    unsrt = {"rxnA": {"L": _factor_reaction(sigma_lnA), "n_active": 1}}
    layout = [("rxnA", slice(0, 1), [0])]

    def sigma_of(rxn):
        return IPRS.sigma_from_cholesky(unsrt[rxn]["L"], n_active=unsrt[rxn]["n_active"])

    T_arr = np.array([750.0, 1000.0, 1250.0])
    blocks = IPRS.build_layout_blocks(layout, sigma_of, T_arr, svd_tol=1e-6)

    zetas = np.array([[0.3], [1.0], [-2.5]])
    ratio = IPRS.violation_ratio(zetas, blocks)
    assert np.allclose(ratio, np.abs(zetas[:, 0]))
    print("PASS: violation_ratio == |zeta| for a factor reaction, matching class-A")


def test_fit_feature_map_on_factor_reaction_pool():
    """fit_feature_map must succeed (not raise) on a small synthetic pool of
    factor-only reactions -- the STEP 6 code path Case-52 never reached."""
    n_rxn = 4
    sigmas = [0.4, 0.6, 0.9, 1.1]
    unsrt = {f"rxn{i}": {"L": _factor_reaction(s), "n_active": 1}
             for i, s in enumerate(sigmas)}
    layout = [(f"rxn{i}", slice(i, i + 1), [0]) for i in range(n_rxn)]

    def sigma_of(rxn):
        return IPRS.sigma_from_cholesky(unsrt[rxn]["L"], n_active=unsrt[rxn]["n_active"])

    T_arr = np.array([900.0, 1200.0, 1500.0])
    blocks = IPRS.build_layout_blocks(layout, sigma_of, T_arr, svd_tol=1e-6)

    rng = np.random.default_rng(0)
    n_rows = 200
    X = rng.normal(scale=1.0, size=(n_rows, n_rxn))
    y = X @ rng.normal(size=n_rxn) + 0.1 * rng.normal(size=n_rows)

    fmap = IPRS.fit_feature_map(X, y, blocks, n_comp=n_rxn, k=3)
    scores = fmap.transform(X)
    assert scores.shape == (n_rows, fmap.k)
    print("PASS: fit_feature_map succeeds on an all-factor-reaction pool")


if __name__ == "__main__":
    test_sigma_from_cholesky_factor_reaction()
    test_sigma_from_cholesky_full_reaction_unchanged()
    test_sigma_from_cholesky_rejects_diverged_shapes()
    test_reduced_cholesky_rejects_0d_sigma()
    test_build_layout_blocks_factor_reaction()
    test_violation_ratio_matches_class_A_test_for_factor_reaction()
    test_fit_feature_map_on_factor_reaction_pool()
    print("ALL TESTS PASSED")
