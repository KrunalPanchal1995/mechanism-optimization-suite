"""
qmc_sampling.py
----------------
Shared low-discrepancy sampling utilities for DesignMatrix.py / Uncertainty.py.

Provides three things:

1. `ceil_pow2`            - smallest power of two >= n.
2. `joint_uniform_block`  - generate one (n_actual, dim) array of points in
                            [0,1) using 'numpy' (legacy pseudo-random),
                            'sobol' (scrambled Sobol sequence, rounded up to
                            the next power of two), or 'lhs' (Latin
                            Hypercube, optionally discrepancy-optimized).
                            This is what the *full* (non-partial) A1+B1+C1
                            path uses directly as a "generator" array -- one
                            joint point per row, one reaction's slot per
                            column-block -- with no further machinery needed,
                            since that path never rejects a draw.
3. `QuasiRandomGenerator` - a drop-in stand-in for `np.random.Generator`
                            (implements .uniform/.choice/.permutation/.random)
                            used by the *partial*-parameter path, where the
                            underlying algorithms in Uncertainty.py use
                            rejection sampling (an unknown number of draws
                            per accepted output sample). It reserves exactly
                            one row of a pre-generated low-discrepancy block
                            per accepted sample (consumed on that sample's
                            first attempt only) and transparently falls back
                            to an independent, deterministically-seeded
                            pseudo-random sub-stream for any retry beyond the
                            first attempt, so a rare rejection never shifts
                            the row<->sample correspondence for every sample
                            that follows. It also tracks how often that
                            fallback gets used, which is reported as a
                            diagnostic (see `diagnostics()`).

Design note on why a shadow block instead of swapping rng.uniform's source
directly: the rejection-sampling routines call `rng.uniform()`/`rng.choice()`
a variable number of times per accepted sample. A flat low-discrepancy
sequence consumed call-by-call would have every rejection shift all later
samples' coordinates, destroying exactly the row<->sample joint structure
the whole point of switching to Sobol/LHS was to get. Reserving one row per
*sample* (not per *call*) and shunting only the rare retry overflow to a
separate fallback stream keeps that structure intact for the (typically
large) majority of samples that accept on the first try.
"""

from __future__ import annotations
import numpy as np

try:
    from scipy.stats import qmc
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "qmc_sampling requires scipy>=1.7 (scipy.stats.qmc). "
        "Install/upgrade scipy to use 'sobol' or 'lhs' methods."
    ) from exc


SOBOL_MAX_DIM = 21201  # hard limit imposed by scipy's direction-number table


# --------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------

def ceil_pow2(n: int) -> int:
    """Smallest power of two >= n (n=0,1 -> 1)."""
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def derive_seed(master_seed, *parts) -> int:
    """
    Deterministically derive an integer seed from a master seed and an
    arbitrary tuple of hashable 'address' parts (reaction name, class tag,
    sample index, ...), so different (reaction, class, sample) combinations
    get independent-looking but fully reproducible sub-streams given the
    same master_seed. If master_seed is None, returns None (caller should
    then fall back to non-reproducible behaviour, matching legacy numpy
    semantics of "no fixed seed unless asked for one").
    """
    if master_seed is None:
        return None
    ss = np.random.SeedSequence([abs(hash((master_seed,) + parts)) % (2**32)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


# --------------------------------------------------------------------------
# One-shot joint point blocks (used directly by the FULL, non-partial path)
# --------------------------------------------------------------------------

def joint_uniform_block(method: str, n_required: int, dim: int,
                         seed=None, optimization="random-cd"):
    """
    Generate one (n_actual, dim) array of points in [0,1).

    method : 'numpy' | 'sobol' | 'lhs'
    n_required : the number of rows (samples) actually needed
    dim    : number of columns (joint dimensionality, e.g. one reaction's
             active parameters concatenated across all reactions)

    Returns (points, n_actual). For 'sobol', n_actual = ceil_pow2(n_required)
    (Sobol's balance/uniformity guarantees only hold for a full power-of-two
    block, so the batch is grown rather than truncated -- per direction, the
    extra rows are kept, not discarded). For 'lhs'/'numpy', n_actual ==
    n_required exactly.
    """
    method = method.lower()
    if dim <= 0:
        raise ValueError("dim must be >= 1")

    if method == "sobol":
        if dim > SOBOL_MAX_DIM:
            raise ValueError(
                f"Requested joint Sobol dimension {dim} exceeds scipy's "
                f"supported maximum of {SOBOL_MAX_DIM}. Use method='lhs' "
                f"instead, or reduce the number of jointly-sampled reactions."
            )
        n_actual = ceil_pow2(n_required)
        m = n_actual.bit_length() - 1
        engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
        pts = engine.random_base2(m) if n_actual > 0 else np.zeros((0, dim))
        return pts, n_actual

    if method == "lhs":
        n_actual = int(n_required)
        engine = qmc.LatinHypercube(d=dim, seed=seed, optimization=optimization)
        pts = engine.random(n=n_actual)
        return pts, n_actual

    if method == "numpy":
        n_actual = int(n_required)
        rng = np.random.default_rng(seed)
        pts = rng.random((n_actual, dim))
        return pts, n_actual

    raise ValueError(f"Unknown sampling method '{method}'. Use 'numpy', 'sobol', or 'lhs'.")


def discrepancy_report(points) -> dict:
    """Thin wrapper around scipy.stats.qmc.discrepancy for diagnostics/logging."""
    points = np.asarray(points)
    return {
        "n_points": points.shape[0],
        "dim": points.shape[1] if points.ndim > 1 else 1,
        "centered_l2_discrepancy": float(qmc.discrepancy(points, method="CD")),
        "wrap_around_discrepancy": float(qmc.discrepancy(points, method="WD")),
    }


# --------------------------------------------------------------------------
# Adapter used by the PARTIAL (rejection-sampling) path
# --------------------------------------------------------------------------

class QuasiRandomGenerator:
    """
    Drop-in stand-in for np.random.Generator for the small subset of its
    API actually used inside Uncertainty.py's rejection-sampling routines:
    .uniform(low, high), .choice(seq), .permutation(seq), .random()/
    .random_sample(). Backed by a pre-generated low-discrepancy 'shadow'
    block (n_rows, d): row i is reserved for accepted output sample i's
    *first* attempt only. Calling `advance_sample()` moves the reservation
    on to the next row; any draw requested beyond the current row's `d`
    values (i.e. a retry past the first attempt) is transparently routed to
    an independent fallback RNG instead of stealing the next sample's row.

    This object does not need to be told in advance how many retries an
    attempt will take -- it only needs `advance_sample()` called once per
    *desired* accepted output, at the point where the algorithm starts (or
    restarts bookkeeping for) that next output. See the call sites added in
    Uncertainty.py's _psac_class_A / _psac_class_D_m2 / _psac_class_C /
    _psac_fsac for the two equivalent conventions used (once-per-outer-loop-
    iteration vs. once-before-the-shared-loop-plus-once-per-success).
    """

    def __init__(self, shadow_block, fallback_rng=None, fallback_seed=None, label=""):
        self.shadow = np.asarray(shadow_block, dtype=float)
        if self.shadow.ndim != 2:
            raise ValueError("shadow_block must be 2-D (n_rows, d)")
        self.n_rows, self.d = self.shadow.shape
        self.fallback = fallback_rng if fallback_rng is not None else np.random.default_rng(fallback_seed)
        self.label = label

        self.row = -1
        self.col = 0
        self._row_in_fallback = False

        # diagnostics
        self.draws_from_shadow = 0
        self.draws_from_fallback = 0
        self.samples_overflowed = 0
        self._rows_touched = 0

    # -- bookkeeping -------------------------------------------------------

    def advance_sample(self):
        """Reserve the next shadow row for the next accepted output sample."""
        self.row += 1
        self.col = 0
        self._row_in_fallback = False
        self._rows_touched += 1

    def _next_raw(self) -> float:
        if self.row < 0:
            # Defensive: a call site forgot to advance_sample() first.
            self.advance_sample()
        if (not self._row_in_fallback) and self.row < self.n_rows and self.col < self.d:
            v = self.shadow[self.row, self.col]
            self.col += 1
            self.draws_from_shadow += 1
            return float(v)
        if not self._row_in_fallback:
            self._row_in_fallback = True
            self.samples_overflowed += 1
        self.draws_from_fallback += 1
        return float(self.fallback.random())

    # -- numpy.random.Generator-compatible surface --------------------------

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return low + (high - low) * self._next_raw()
        n = int(np.prod(size))
        vals = np.fromiter((low + (high - low) * self._next_raw() for _ in range(n)),
                            dtype=float, count=n)
        return vals.reshape(size)

    def choice(self, a):
        seq = list(a)
        u = self._next_raw()
        idx = min(int(u * len(seq)), len(seq) - 1)
        return seq[idx]

    def permutation(self, x):
        # Fisher-Yates using the same sequential draw source. Kept for
        # completeness; the one call site in Uncertainty.py that used
        # rng.permutation (_psac_class_B_m3_) is not on the live call path.
        arr = np.array(x, copy=True)
        for i in range(len(arr) - 1, 0, -1):
            j = min(int(self._next_raw() * (i + 1)), i)
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    def random(self, size=None):
        return self.uniform(0.0, 1.0, size)

    def random_sample(self, size=None):
        return self.random(size)

    # -- diagnostics ---------------------------------------------------------

    def diagnostics(self) -> dict:
        total = self.draws_from_shadow + self.draws_from_fallback
        return {
            "label": self.label,
            "rows_reserved": self.n_rows,
            "rows_used": self._rows_touched,
            "draws_from_shadow": self.draws_from_shadow,
            "draws_from_fallback": self.draws_from_fallback,
            "fallback_draw_fraction": (self.draws_from_fallback / total) if total else 0.0,
            "samples_that_overflowed": self.samples_overflowed,
            "overflow_rate": (self.samples_overflowed / self._rows_touched) if self._rows_touched else 0.0,
        }


def build_joint_shadow(method: str, n_required: int, n_reactions: int, dims_per_reaction: int,
                        seed=None, optimization="random-cd"):
    """
    Convenience wrapper: build the joint (n_actual, n_reactions*dims_per_reaction)
    low-discrepancy block for a given class, and return it together with
    n_actual. Callers slice out reaction j's columns via
    block[:, j*dims_per_reaction:(j+1)*dims_per_reaction].
    """
    dim = n_reactions * dims_per_reaction
    return joint_uniform_block(method, n_required, dim, seed=seed, optimization=optimization)


def make_reaction_generator(shadow_block, reaction_index, dims_per_reaction,
                             fallback_seed=None, label=""):
    """
    Slice out one reaction's column-block from a joint shadow block and wrap
    it in a QuasiRandomGenerator ready to be passed as `rng` into the
    existing partial-parameter sampling methods.
    """
    j = reaction_index
    sub = shadow_block[:, j * dims_per_reaction:(j + 1) * dims_per_reaction]
    return QuasiRandomGenerator(sub, fallback_seed=fallback_seed, label=label)
