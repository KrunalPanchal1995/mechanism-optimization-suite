"""
Standalone validation of the new Sobol/LHS partial-sampling path, exercising
the REAL hooked code in Uncertainty.py (not a re-implemented toy version).
Builds a bare UncertaintyExtractor via object.__new__ (no mechanism files
needed -- get_full_sigma_and_L/get_reduced_cholesky only need self.cov and
self.temperatures, confirmed via static analysis of self.* usage).
"""
import numpy as np
import Uncertainty
import qmc_sampling as qs

np.random.seed(0)


def make_extractor(L_full, temperatures):
    A = object.__new__(Uncertainty.UncertaintyExtractor)
    A.cov = L_full
    A.temperatures = temperatures
    return A


# A reasonably well-conditioned synthetic 3x3 lower-triangular Cholesky
L_full = np.array([
    [1.2, 0.0, 0.0],
    [0.3, 0.9, 0.0],
    [-0.2, 0.4, 0.7],
])
T_arr = np.linspace(300.0, 2500.0, 12)

print("=" * 70)
print("TEST 1: getClassA_partial via a per-reaction QuasiRandomGenerator")
print("=" * 70)
for indices, m_label in [((0,), "m=1"), ((0, 1), "m=2"), ((0, 1, 2), "m=3")]:
    A = make_extractor(L_full, T_arr)
    n = 40
    shadow, n_actual = qs.joint_uniform_block("sobol", n, 2, seed=123)
    rng = qs.QuasiRandomGenerator(shadow, fallback_seed=77, label=f"A-{m_label}")
    out = A.getClassA_partial(indices, n_actual, rng)
    diag = rng.diagnostics()
    print(f"  {m_label}: requested {n} -> actual {n_actual}, got {len(out)} samples, "
          f"overflow_rate={diag['overflow_rate']:.3f}, "
          f"fallback_draw_fraction={diag['fallback_draw_fraction']:.3f}")
    assert len(out) == n_actual
    assert all(len(z) == 3 for z in out)

print()
print("=" * 70)
print("TEST 2: getClassB_partial (m=2, Class-D) and getClassC_partial (m=2,3)")
print("=" * 70)
for indices in [(0, 1), (1, 2)]:
    # NOTE: getClassB_partial's underlying _psac_class_D_m2 is a 2x2 solve
    # specific to m==2; in the real pipeline m==3 reactions are routed to
    # getB2Zeta() directly inside run_sampling_b_partial (see the m==3
    # branch there), never through getClassB_partial. So only m==2 subsets
    # are exercised here -- this matches actual production usage.
    A = make_extractor(L_full, T_arr)
    n = 32
    shadow, n_actual = qs.joint_uniform_block("sobol", n, 2, seed=55)
    rng = qs.QuasiRandomGenerator(shadow, fallback_seed=11, label="B")
    out_b = A.getClassB_partial({}, indices, n_actual, rng)
    diag_b = rng.diagnostics()

    A2 = make_extractor(L_full, T_arr)
    shadow2, n_actual2 = qs.joint_uniform_block("sobol", n, 2, seed=56)
    rng2 = qs.QuasiRandomGenerator(shadow2, fallback_seed=12, label="C")
    out_c = A2.getClassC_partial(indices, n_actual2, rng2)
    diag_c = rng2.diagnostics()

    print(f"  indices={indices}: B got {len(out_b)} (overflow_rate="
          f"{diag_b['overflow_rate']:.3f}), C got {len(out_c)} (overflow_rate="
          f"{diag_c['overflow_rate']:.3f})")

print()
print("=" * 70)
print("TEST 3: get_fsac_partial (m=1,2,3) -- the explicitly requested treatment")
print("=" * 70)
for indices, m_label in [((0,), "m=1"), ((0, 2), "m=2"), ((0, 1, 2), "m=3")]:
    A = make_extractor(L_full, T_arr)
    n = 40
    shadow, n_actual = qs.joint_uniform_block("sobol", n, 3, seed=99)
    rng = qs.QuasiRandomGenerator(shadow, fallback_seed=33, label=f"fSAC-{m_label}")
    out = A.get_fsac_partial(indices, n_actual, rng)
    diag = rng.diagnostics()
    print(f"  {m_label}: requested {n} -> actual {n_actual}, got {len(out)} samples, "
          f"overflow_rate={diag['overflow_rate']:.3f}, "
          f"fallback_draw_fraction={diag['fallback_draw_fraction']:.3f}")
    assert all(len(z) == 3 for z in out)

print()
print("All end-to-end tests against the REAL Uncertainty.py code passed.")
