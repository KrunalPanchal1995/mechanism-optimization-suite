"""
Single source of truth for paths, seeds, and hyperparameter grids shared
across validation/*.py, validation/search_history/*.py, and
validation/report_figures/*.py.

IMPORTANT -- BLAS thread pinning ordering: `set_single_threaded_blas()` only
works if it runs BEFORE numpy (and anything that transitively imports numpy,
e.g. ResponseSurface) is first imported -- numpy/OpenBLAS reads
OMP_NUM_THREADS etc. once, at its own import time, and ignores later changes
to os.environ. This module deliberately does NOT import numpy itself (so
merely `import config` never triggers numpy's early binding), but every
script that needs single-threaded BLAS must still put

    import config
    config.set_single_threaded_blas()

as the very first lines, ahead of `import numpy` (directly or via
`import ResponseSurface` / `_bootstrap` / anything else that pulls numpy in).
Do not rely on config.py doing this automatically at import time -- by the
time *config* is imported, some earlier import in the same file may already
have pulled numpy in first.
"""
import os

# ---------------------------------------------------------------------------
# REPO_ROOT resolution
# ---------------------------------------------------------------------------
# This file lives at <MUQ-SAC-root>/validation/config.py. <MUQ-SAC-root> is
# `test_build/` today and will become `src/MUQ-SAC/` once this tree is
# mirrored there (see the plan's Part 1 final step) -- those two locations
# sit at DIFFERENT depths under the actual repo root
# (MULTI-STAGE_OPTIMIZATION/test_build/validation/ vs.
#  MULTI-STAGE_OPTIMIZATION/src/MUQ-SAC/validation/), so a fixed number of
# `os.path.dirname()` hops is not portable between them. Instead, walk
# upward from this file looking for the sentinel directory
# `TEST_CORRELATIONS_CORRECTED/` that only ever exists at the true repo
# root, in either location.
_SENTINEL_DIRNAME = "TEST_CORRELATIONS_CORRECTED"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir, sentinel=_SENTINEL_DIRNAME, max_hops=8):
    d = start_dir
    for _ in range(max_hops):
        if os.path.isdir(os.path.join(d, sentinel)):
            return d
        parent = os.path.dirname(d)
        if parent == d:  # reached filesystem root
            break
        d = parent
    raise RuntimeError(
        f"Could not locate repo root (a directory containing {sentinel!r}) "
        f"walking up from {start_dir!r}. Set MUQSAC_REPO_ROOT explicitly."
    )


REPO_ROOT = os.environ.get("MUQSAC_REPO_ROOT") or _find_repo_root(_THIS_DIR)
REPO_ROOT = os.path.abspath(REPO_ROOT)

DATA_DIR = os.environ.get("MUQSAC_DATA_DIR") or os.path.join(REPO_ROOT, "TEST_CORRELATIONS_CORRECTED") + os.sep

# ---------------------------------------------------------------------------
# Results / figures / tables / scratch directories
# ---------------------------------------------------------------------------
_STUDY_DIRNAME = {
    "tenfold": "report_10fold_validation",
    "report": "report",
}


def _study_root(study):
    if study not in _STUDY_DIRNAME:
        raise ValueError(f"Unknown study {study!r}; expected one of {list(_STUDY_DIRNAME)}")
    return os.path.join(REPO_ROOT, _STUDY_DIRNAME[study])


def results_dir(study):
    """study: 'tenfold' -> report_10fold_validation/results, 'report' -> report/results."""
    return os.path.join(_study_root(study), "results")


def figures_dir(study):
    return os.path.join(_study_root(study), "figures")


def tables_dir(study):
    return os.path.join(_study_root(study), "tables")


def scratch_dir(name):
    """Ephemeral per-fold working directory -- never a cited result. Honors
    $MUQSAC_SCRATCH if set, else /tmp/muqsac_validation/<name>."""
    base = os.environ.get("MUQSAC_SCRATCH") or "/tmp/muqsac_validation"
    return os.path.join(base, name)


# ---------------------------------------------------------------------------
# CV protocol constants
# ---------------------------------------------------------------------------
N_FOLDS = 10
CV_SPLIT_SEED = 2026        # kfold_cv.py's / kfold_cv_bandvalid.py's own 10-fold split
SEARCH_SPLIT_SEED = 555     # kfold_robust_search{,2,3}.py -- independent of CV_SPLIT_SEED,
                             # so selecting hyperparameters on this split and evaluating on
                             # CV_SPLIT_SEED's is a genuine out-of-selection check.
# kfold_cv_verify.py does NOT use a third seed -- it deliberately re-runs
# kfold_cv.run_case (CV_SPLIT_SEED=2026, the SAME split as the baseline
# 10-fold evaluation) on the SEARCH_SPLIT_SEED=555-selected hyperparameters,
# i.e. "independent of SELECTION" means independent of the search split, not
# a third split. VERIFY_SPLIT_SEED is therefore an alias, not a new value --
# keep it as its own name for readability at call sites and in case a future
# script legitimately wants a third split.
VERIFY_SPLIT_SEED = CV_SPLIT_SEED

BAND_TOLERANCE_EVAL = 1.0   # a test/val point counts only if violation_ratio <= this
CRITERION = 2.0             # pass/fail MRE threshold used throughout the 10-fold study

# ---------------------------------------------------------------------------
# Hyperparameter search grids
# ---------------------------------------------------------------------------
# Shared across all 3 search rounds (kfold_robust_search{,2,3}.py) and the
# report/ studies (offline_replay.py, offline_cv_search.py, parametric_study.py
# each historically had their own, narrower ad hoc grids -- those stay
# script-local since they are NOT identical to the search_history grids and
# changing them would change already-published report/ numbers).
TOLERANCES = [1.0, 1.1, 1.25, 1.5, 2.0, 5.0, 10.0, float("inf")]
K_GRID = [8, 10, 12, 14, 16, 20]        # capped at 20 -- established finding, do not widen
ORDER_GRID = [2, 3]
MAX_TERMS_FRAC = 0.4

# Round 1 (kfold_robust_search.py): narrow ridge grid, all 8 cases.
RIDGE_GRID_R1 = [0.0, 1e-7, 1e-5, 1e-3, 1e-2]

# Round 2 (kfold_robust_search2.py): cases 0,3,4,5 marginal after round 1;
# widen ridge substantially (up to 0.5) to trade bias for variance
# reduction -- the direct lever for a worst-fold criterion.
RIDGE_GRID_R2 = [0.0, 1e-5, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 5e-1]
CASES_TO_SEARCH_R2 = [0, 3, 4, 5, 7]

# Round 3 (kfold_robust_search3.py): same grid as round 2, restricted to the
# 2 cases (3, 4) still not passing, now scored on BAND-VALID points only
# (BAND_TOLERANCE_EVAL above) -- see search_history/README.md.
RIDGE_GRID_R3 = RIDGE_GRID_R2
CASES_TO_SEARCH_R3 = [3, 4]


# ---------------------------------------------------------------------------
# BLAS thread pinning
# ---------------------------------------------------------------------------
_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def set_single_threaded_blas():
    """Force single-threaded BLAS. Must be called before `import numpy`
    (directly or transitively) -- see module docstring."""
    for _v in _BLAS_ENV_VARS:
        os.environ[_v] = "1"
