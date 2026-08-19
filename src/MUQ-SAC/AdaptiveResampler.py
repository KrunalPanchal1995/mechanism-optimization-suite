"""
AdaptiveResampler.py
====================

Curve-class-aware local resampling around a PRS outlier.

The MUQ-SAC pipeline (run_script.py) builds a design matrix (DM) whose columns
are decorrelated Arrhenius perturbation coordinates (zeta), fits a Polynomial
Response Surface (PRS) per case, and tests it on a hold-out set. When a hold-out
point has a large PRS-vs-simulation error (an "outlier"), the fix is to add more
training points *in the local region around that point* and refit.

Because the DM is already in decorrelated zeta / coded space, an "outlier region"
sample is simply a local perturbation of the outlier's DM row -- no Cholesky
inversion is needed. This module goes one step further, as suggested: it reads the
cached SAC curve-class samples that DesignMatrix.py writes
(``a_type_samples[_partial]_<method>.pkl`` etc.), figures out which curve class
(A / B / C) each reaction's block of the outlier most resembles, and resamples the
*same kind* of curve locally. If those caches are unavailable (e.g. the 'A-facto'
design, which has no class pkls), it falls back to a local Latin-Hypercube box in
DM-column space.

The module is agnostic to:
  * PRS type    -- 'Full' (PRS space == full DM) or 'Partial' (PRS space is the
                   COMPRESSED / selected-columns matrix ``p_design_matrix``).
  * design type -- 'A-facto' (1 col/rxn, no class pkls -> LHS-box fallback) or
                   'A1+B1+C1' (per-rxn active_len cols, class pkls available).

Two coordinate spaces are used:
  * FULL space   -- length = sum of len(unsrt[rxn].activeParameters). This is what
                    the simulator consumes (one zeta value per active parameter,
                    zeros at inactive positions for the partial case). The cached
                    per-reaction samples live here too.
  * PRS space    -- what ResponseSurface was fit on. For 'Full' it equals FULL
                    space; for 'Partial' it is the compressed (selected columns
                    only) representation.

Public API
----------
find_outliers(rel_errors, criteria)              -> ndarray of indices, worst first
OutlierResampler(...).resample()                 -> (prs_rows, full_rows, sel_mask, row_class)
OutlierResampler(...).resample_global()          -> (prs_rows, full_rows, sel_mask, row_class)
OutlierResampler(...).resample_hybrid()          -> (prs_rows, full_rows, sel_mask, row_class)
OutlierResampler(...).resample_gap_fill()        -> (prs_rows, full_rows, sel_mask, row_class)

``resample_gap_fill`` is the only mode that calls DesignMatrix.py's REAL
class-A/B/C (and Partial-PRS ``_partial``) curve generators to synthesize
genuinely NEW Arrhenius curves, instead of recombining/perturbing the
already-cached ones the way ``resample``/``resample_global``/``resample_hybrid``
do. New candidates are drawn oversampled via Sobol/LHS, then filtered by a
kNN isolation test against a caller-supplied, pass-growing reference set (the
original DM plus every row added by earlier passes this run) so gap_fill
never piles new points on top of already-simulated ones. See its docstring
for the unshuffled(20%)/shuffled(80%) DM-mirroring structure.

``resample_global`` is the DM-faithful alternative to ``resample``: instead of
clustering new points locally around the outlier (which can distort the global
fit if overused), it fills the JOINT gaps across the whole design matrix --
mirroring how DesignMatrix.getSample_partial itself builds the pPRS DM: an
"unshuffled" low-discrepancy (Sobol/LHS/numpy) block that recombines the
cached per-reaction class-A/B/C curves into new joint (cross-reaction) rows,
plus a "shuffled" block (each reaction's chosen curves independently
re-permuted, `shuffle_factor`x), with the outlier row(s) always included and
the whole thing re-shuffled at the end. See run_script.retrain_around_outlier.

``resample_hybrid`` is the DEFAULT mode (see README_AdaptiveResampler.md for
the full diagnosis): ``resample_global`` was found to silently ignore the
configured sample budget (Sobol power-of-2 rounding + the shuffled-block
multiplier injected ~4x the requested rows per pass) and never concentrated
points near the actual outlier, which made a real fit worse. ``resample_hybrid``
hard-caps the total new rows per pass at ``HARD_CAP_DEFAULT`` (500) and splits
that budget between a block concentrated near the outlier(s) (on-manifold,
class-aware, like ``resample``) and a sparse, low-discrepancy block spread
evenly across the whole domain (like ``resample_global``'s unshuffled block,
but with no shuffled/duplicate block, so it stays sparse instead of dense).
"""

import os
import glob
import pickle

import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree

# IdentifiablePRS has no heavy dependencies (numpy + sklearn only), unlike
# Uncertainty.py (which unconditionally imports Input_file_reader,
# MechanismParser, qmc_sampling at module load -- see is_band_valid /
# batch_violation_ratio's docstrings there). DOptimalResampler's
# band-validity gate needs exactly the same reduced-Cholesky / f_prior math
# those provide, so it reuses IdentifiablePRS's dependency-light,
# cross-validated (see report/) implementation instead of importing
# Uncertainty.py wholesale just to reach one pure function.
from IdentifiablePRS import reduced_cholesky as _reduced_cholesky
from IdentifiablePRS import reaction_W_fprior as _reaction_W_fprior

CLASS_LABELS = ("A", "B", "C")

# User-mandated ceiling on new rows per resampling pass (5 passes * 500 = 2500
# total, matching the stated budget). resample_hybrid clamps to this no matter
# what N_new is configured to -- see run_script.retrain_around_outlier.
HARD_CAP_DEFAULT = 500


def _ceil_pow2(n):
    """Smallest power of 2 >= n (Sobol wants a power-of-2 batch for its
    balance property -- mirrors DesignMatrix's qmc_sampling.ceil_pow2)."""
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def _isolation_filter(candidates, reference, k=3, radius_frac=0.05):
    """Rank candidate rows by how far they are from an existing reference set.

    Both arrays are (n, d) rows in the same coordinate space (PRS space, here).
    Columns are normalized by the reference set's own observed per-column
    range first, so a single scalar radius is meaningful across columns of
    very different physical scale.

    For each candidate, the "isolation score" is its distance (in the
    normalized space) to its k-th nearest reference-set neighbour -- this one
    statistic does double duty as both a reject test ("are >= k existing
    points already closer than radius_frac?") and a ranking key (bigger
    distance = sitting in a bigger gap).

    Returns
    -------
    order : ndarray of candidate row indices, most-isolated first. Rows whose
            isolation score is below ``radius_frac`` are dropped UNLESS that
            would drop everything, in which case nothing is hard-rejected and
            ``order`` simply ranks all candidates by isolation (soft
            fallback -- a tight radius/k should never fully starve a pass).
    d_k    : (n,) raw isolation score per candidate, in the candidates'
             ORIGINAL (unsorted) order.
    """
    candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    if candidates.shape[0] == 0:
        return np.array([], dtype=int), np.array([])
    if reference.shape[0] == 0:
        # Nothing to avoid -- every candidate is maximally isolated.
        d_k = np.full(candidates.shape[0], np.inf)
        return np.argsort(-d_k), d_k

    lo = reference.min(axis=0)
    span = np.maximum(reference.max(axis=0) - lo, 1e-12)
    ref_n = (reference - lo) / span
    cand_n = (candidates - lo) / span

    tree = cKDTree(ref_n)
    kk = min(int(k), ref_n.shape[0])
    dists, _ = tree.query(cand_n, k=kk)
    d_k = dists if kk == 1 else dists[:, -1]

    order = np.argsort(-d_k)
    survivors = order[d_k[order] >= radius_frac]
    if survivors.size == 0:
        survivors = order  # soft fallback: keep ranking instead of failing.
    return survivors, d_k


# --------------------------------------------------------------------------- #
# Outlier detection                                                           #
# --------------------------------------------------------------------------- #
def find_outliers(rel_errors, criteria, force_worst=True):
    """Return hold-out indices whose |relative error %| exceeds ``criteria``.

    ``rel_errors`` is ``ResponseSurface.error_testing_relative`` (signed %),
    aligned row-for-row with the hold-out set ``xTest``/``yTest``.
    Indices are returned worst-first. If nothing exceeds the criterion and
    ``force_worst`` is set, the single worst point is returned so the caller can
    still choose to densify there.
    """
    rel = np.abs(np.asarray(rel_errors, dtype=float))
    idx = np.where(rel > float(criteria))[0]
    if idx.size == 0:
        if force_worst and rel.size:
            idx = np.array([int(np.argmax(rel))])
        else:
            return np.array([], dtype=int)
    return idx[np.argsort(-rel[idx])]


# --------------------------------------------------------------------------- #
# Resampler                                                                   #
# --------------------------------------------------------------------------- #
class OutlierResampler:
    """Generate a small, focused mini-DM around one or more outlier rows.

    Parameters
    ----------
    unsrt_data : dict
        rxn_key -> uncertainty object. Only ``.activeParameters`` is read, and
        iteration order must match how DesignMatrix built the rows.
    selected_params : sequence of int
        Full-length (one entry per active parameter) 1/0 selection vector.
        For 'Full' PRS this is all 1s.
    design : str
        'A-facto' or 'A1+B1+C1'.
    prs_type : str
        'Full' or 'Partial'.
    base_dir : str
        Directory where the class-sample pkl caches live (the dir run_script
        was launched from -- i.e. before it chdir's into ``Opt``).
    case_index : str
        Case tag, used to locate the partial cache dir.
    N_new : int
        Approximate total number of new samples (split across outliers).
    delta_fraction : float
        Local neighbourhood size, as a fraction of the toward-neighbour step
        (class-aware path) or of the per-column range (LHS-box fallback).
    seed : int
        RNG seed for reproducibility (the base pipeline's per-reaction shuffling
        is unseeded; this resampler is deterministic).
    knn : int
        Number of nearest same-class neighbours used to define the local
        manifold direction per reaction (used by ``resample``, the local mode).
    uns_cap : int
        Cap on the "unshuffled" low-discrepancy block size for
        ``resample_global`` (default 256; rounded up to a power of 2 for
        Sobol, same convention as DesignMatrix.getSample_partial).
    shuffle_factor : int
        How many independent per-reaction re-permutations of the unshuffled
        block make up the "shuffled" block in ``resample_global`` (default 2;
        DesignMatrix itself uses 4 for the main DM -- kept configurable here
        since a resampling pass is meant to be lighter).
    method : str
        'sobol' | 'lhs' | 'numpy' -- the low-discrepancy generator used by
        ``resample_global`` for both the joint per-reaction pool recombination
        and the no-pkl box fallback.
    gapfill_oversample : int
        Candidate-pool multiplier for ``resample_gap_fill`` -- since it calls
        DesignMatrix's REAL (expensive, class-B/C use multiprocessing) curve
        generators rather than recombining cached ones, candidates are drawn
        ``gapfill_oversample`` x the needed count so the isolation filter has
        something to reject/rank (default 2; kept modest for cost).
    gapfill_knn_radius_frac : float
        ``resample_gap_fill``'s isolation-filter radius (fraction of each
        column's observed range in the reference set).
    gapfill_knn_k : int
        ``resample_gap_fill``'s isolation-filter neighbour count (a candidate
        is rejected if its k-th nearest reference neighbour is already closer
        than ``gapfill_knn_radius_frac``).
    gapfill_shuffle_factor : int
        ``resample_gap_fill``'s shuffled-block multiplier -- default 4 matches
        DesignMatrix's own real shuffle factor (unlike ``shuffle_factor``
        above, which is legacy-``dm_global``-only and defaults to 2).
    """

    def __init__(self, unsrt_data, selected_params, design, prs_type,
                 base_dir, case_index, N_new=200, delta_fraction=0.15,
                 seed=42, knn=8, uns_cap=256, shuffle_factor=2, method="sobol",
                 gapfill_oversample=2, gapfill_knn_radius_frac=0.05,
                 gapfill_knn_k=3, gapfill_shuffle_factor=4):
        self.unsrt = unsrt_data
        self.selected_params = np.asarray(selected_params, dtype=int)
        self.design = design
        self.prs_type = prs_type
        self.base_dir = base_dir
        self.case_index = str(case_index)
        self.N_new = int(N_new)
        self.delta_fraction = float(delta_fraction)
        self.rng = np.random.default_rng(int(seed))
        self.knn = int(knn)
        self.uns_cap = int(uns_cap)
        self.shuffle_factor = int(shuffle_factor)
        self.method = str(method).lower()
        self.gapfill_oversample = max(1, int(gapfill_oversample))
        self.gapfill_knn_radius_frac = float(gapfill_knn_radius_frac)
        self.gapfill_knn_k = int(gapfill_knn_k)
        self.gapfill_shuffle_factor = max(0, int(gapfill_shuffle_factor))

        self.layout = self._build_layout()
        self.n_full = self.layout[-1]["full_offset"] + self.layout[-1]["active_len"]
        self.class_pool = self._load_class_pool()  # {rxn: [(vec, label), ...]} or None

    # ---- layout: PRS-space <-> FULL-space per reaction -------------------- #
    def _build_layout(self):
        layout = []
        full_offset = 0
        comp_offset = 0
        for rxn in self.unsrt:                       # same order as DesignMatrix
            alen = len(self.unsrt[rxn].activeParameters)
            chunk = self.selected_params[full_offset: full_offset + alen]
            # NOTE: no "or [0]" fallback here. The compressed p_design_matrix
            # (DesignMatrix.getSample_partial) appends a column ONLY where
            # selected_params == 1, so a fully-unselected reaction contributes
            # ZERO columns. Forcing [0] would over-count n_prs and index past the
            # end of the real PRS row.
            sel_local = [i for i, s in enumerate(chunk) if s == 1]
            if self.prs_type == "Full":
                n_cols = alen                        # PRS == full for this rxn
                prs_local = list(range(alen))        # full-space indices in PRS row
            else:
                n_cols = len(sel_local)              # compressed (may be 0)
                prs_local = list(sel_local)          # which full idx each PRS col is
            layout.append({
                "rxn": rxn, "active_len": alen,
                "full_offset": full_offset, "sel_local": sel_local,
                "comp_offset": comp_offset, "n_cols": n_cols,
                "prs_local": prs_local,
            })
            full_offset += alen
            comp_offset += n_cols
        return layout

    @property
    def n_prs(self):
        return self.layout[-1]["comp_offset"] + self.layout[-1]["n_cols"]

    # ---- load cached SAC class samples ------------------------------------ #
    def _load_class_pool(self):
        """Return {rxn: [(full_len_vec, class_label), ...]} or None.

        The class pkls only exist for the 'A1+B1+C1' design. Filenames are
        method-tagged, so we glob for whichever method actually ran.
        """
        if self.design != "A1+B1+C1":
            return None

        if self.prs_type == "Full":
            search_dir = self.base_dir
            patterns = {c: os.path.join(search_dir, f"{c.lower()}_type_samples_*.pkl")
                        for c in CLASS_LABELS}
        else:
            search_dir = os.path.join(self.base_dir, "DM_FOR_PARTIAL_PRS", self.case_index)
            patterns = {c: os.path.join(search_dir, f"{c.lower()}_type_samples_partial_*.pkl")
                        for c in CLASS_LABELS}

        per_class = {}
        for label, pat in patterns.items():
            hits = sorted(glob.glob(pat))
            if not hits:
                continue
            try:
                with open(hits[0], "rb") as fh:
                    per_class[label] = pickle.load(fh)   # {rxn: [vec, ...]}
            except Exception:
                continue

        if not per_class:
            return None

        pool = {}
        for lay in self.layout:
            rxn = lay["rxn"]
            samples = []
            for label in CLASS_LABELS:
                cd = per_class.get(label)
                if cd is None or rxn not in cd:
                    continue
                for vec in cd[rxn]:
                    v = np.asarray(vec, dtype=float).ravel()
                    if v.size == lay["active_len"]:
                        samples.append((v, label))
            if samples:
                pool[rxn] = samples
        return pool or None

    # ---- PRS-space row  ->  FULL-space row -------------------------------- #
    def prs_to_full(self, prs_row):
        prs_row = np.asarray(prs_row, dtype=float).ravel()
        full = np.zeros(self.n_full)
        for lay in self.layout:
            for k, full_local in enumerate(lay["prs_local"]):
                full[lay["full_offset"] + full_local] = prs_row[lay["comp_offset"] + k]
        return full

    def full_to_prs(self, full_row):
        full_row = np.asarray(full_row, dtype=float).ravel()
        prs = np.zeros(self.n_prs)
        for lay in self.layout:
            for k, full_local in enumerate(lay["prs_local"]):
                prs[lay["comp_offset"] + k] = full_row[lay["full_offset"] + full_local]
        return prs

    def selection_mask_row(self):
        """Full-length 1.0/0.0 mask row expected by SM.getYAML_List (partial)."""
        return self.selected_params.astype(float)

    # ---- per-reaction local sampling -------------------------------------- #
    def available_classes(self):
        """Curve-class labels present anywhere in the loaded pool (ordered
        A, B, C). Empty if there is no class pool (LHS-box fallback regime)."""
        if self.class_pool is None:
            return []
        present = set()
        for samples in self.class_pool.values():
            present.update(lab for _, lab in samples)
        return [c for c in CLASS_LABELS if c in present]

    def _sample_block_classaware(self, rxn, outlier_sub, n, force_label=None):
        """Resample ``n`` per-reaction sub-vectors of a single curve class,
        local to ``outlier_sub``, using the cached class pool.

        If ``force_label`` is given, that class's samples are used (so the whole
        mini-DM block stays class-homogeneous, matching how the original DM is
        generated per class); otherwise the class nearest the outlier is used.
        Falls back to the whole pool if the requested class is too sparse."""
        samples = self.class_pool[rxn]
        vecs = np.array([v for v, _ in samples])
        labels = np.array([lab for _, lab in samples])

        if force_label is not None and np.any(labels == force_label):
            chosen = force_label
        else:
            chosen = labels[int(np.argmin(np.linalg.norm(vecs - outlier_sub[None, :], axis=1)))]

        same = vecs[labels == chosen]
        if len(same) < 2:
            same = vecs  # not enough of that class; use whole pool
        ds = np.linalg.norm(same - outlier_sub[None, :], axis=1)
        k = min(self.knn, len(same))
        knn_vecs = same[np.argsort(ds)[:k]]
        deltas = knn_vecs - outlier_sub[None, :]           # (k, d) local directions

        out = np.empty((n, outlier_sub.size))
        for i in range(n):
            # random convex weights over the k nearest same-class neighbours,
            # scaled by a small local step so we stay near the outlier while
            # tracking the class manifold direction.
            w = self.rng.random(k)
            w /= w.sum()
            step = self.delta_fraction * self.rng.random()
            out[i] = outlier_sub + step * (w @ deltas)
        return out

    def _sample_block_box(self, col_lo, col_hi, outlier_sub, n):
        """LHS-box fallback for one contiguous PRS-column block."""
        d = outlier_sub.size
        rng_span = np.maximum(col_hi - col_lo, 1e-12)
        delta = self.delta_fraction * rng_span
        lo = np.clip(outlier_sub - delta, col_lo, col_hi)
        hi = np.clip(outlier_sub + delta, col_lo, col_hi)
        # Latin-Hypercube in [0,1]^d then scale to the local box.
        u = (np.argsort(self.rng.random((n, d)), axis=0) + self.rng.random((n, d))) / n
        return lo[None, :] + u * (hi - lo)[None, :]

    # ---- top-level resample ----------------------------------------------- #
    def _build_block(self, o_prs, o_full, n, col_lo, col_hi, force_label):
        """Build ``n`` new rows around one outlier, all of curve class
        ``force_label`` (or LHS-box if there is no class pool)."""
        new_full = np.tile(o_full, (n, 1))               # start from the outlier
        for lay in self.layout:
            rxn = lay["rxn"]
            f0 = lay["full_offset"]
            sub = o_full[f0: f0 + lay["active_len"]]
            if self.class_pool is not None and rxn in self.class_pool:
                blk = self._sample_block_classaware(rxn, sub, n, force_label=force_label)
                new_full[:, f0: f0 + lay["active_len"]] = blk
            else:
                c0 = lay["comp_offset"]
                o_sub_prs = o_prs[c0: c0 + lay["n_cols"]]
                blk = self._sample_block_box(col_lo[c0: c0 + lay["n_cols"]],
                                             col_hi[c0: c0 + lay["n_cols"]],
                                             o_sub_prs, n)
                for j in range(n):
                    for kk, full_local in enumerate(lay["prs_local"]):
                        new_full[j, f0 + full_local] = blk[j, kk]
        return np.asarray([self.full_to_prs(new_full[j]) for j in range(n)])

    def resample(self, outlier_prs_rows, X_ref_prs):
        """Build the mini-DM around the given outlier rows, organised by curve
        class (A/B/C) just like the original DesignMatrix generation.

        Parameters
        ----------
        outlier_prs_rows : (k, n_prs) array -- outlier rows in PRS space.
        X_ref_prs        : (N, n_prs) array -- full case DM (PRS space), used for
                           per-column bounds in the LHS-box fallback.

        Returns
        -------
        prs_rows   : (M, n_prs)  new rows in PRS space  (augment training set)
        full_rows  : (M, n_full) new rows in FULL space (feed the simulator)
        sel_mask   : (n_full,)   1.0/0.0 selection mask for SM.getYAML_List
        row_class  : list[str]   curve-class label per row ('A'/'B'/'C', or
                     'MIX' for the LHS-box fallback), length M -- lets the caller
                     write one file per Arrhenius class.
        """
        outlier_prs_rows = np.atleast_2d(np.asarray(outlier_prs_rows, dtype=float))
        X_ref_prs = np.asarray(X_ref_prs, dtype=float)
        col_lo = X_ref_prs.min(axis=0)
        col_hi = X_ref_prs.max(axis=0)

        classes = self.available_classes()               # [] -> box fallback
        k = len(outlier_prs_rows)
        per = max(1, self.N_new // k)

        prs_out, row_class = [], []
        for o_prs in outlier_prs_rows:
            o_full = self.prs_to_full(o_prs)
            if classes:
                # split this outlier's budget across the available classes so
                # each Arrhenius class gets its own homogeneous block.
                counts = [per // len(classes)] * len(classes)
                for extra in range(per - sum(counts)):
                    counts[extra] += 1
                for label, n_c in zip(classes, counts):
                    if n_c == 0:
                        continue
                    block = self._build_block(o_prs, o_full, n_c, col_lo, col_hi, label)
                    prs_out.append(block)
                    row_class.extend([label] * n_c)
            else:
                block = self._build_block(o_prs, o_full, per, col_lo, col_hi, None)
                prs_out.append(block)
                row_class.extend(["MIX"] * per)

        prs_rows = np.vstack(prs_out)
        full_rows = np.asarray([self.prs_to_full(r) for r in prs_rows])
        return prs_rows, full_rows, self.selection_mask_row(), row_class

    # ---- global (DM-gap-filling) resample ---------------------------------- #
    def _pool_arrays(self, rxn):
        """(vecs, labels) full-length-per-reaction pool arrays, or None if no
        cached class samples exist for this reaction."""
        if self.class_pool is None or rxn not in self.class_pool:
            return None
        samples = self.class_pool[rxn]
        vecs = np.array([v for v, _ in samples])
        labels = np.array([lab for _, lab in samples])
        return vecs, labels

    def _qmc_draw(self, n, d, seed):
        """(n, d) array in [0,1) from the configured low-discrepancy method."""
        if n <= 0:
            return np.zeros((0, max(d, 0)))
        if d <= 0:
            return np.zeros((n, 0))
        if self.method == "sobol":
            sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
            return sampler.random(n)
        elif self.method == "lhs":
            sampler = qmc.LatinHypercube(d=d, seed=seed)
            return sampler.random(n)
        else:  # 'numpy' -- plain (non-low-discrepancy) uniform draw
            return np.random.default_rng(seed).random((n, d))

    def _build_global_unshuffled(self, n, seed):
        """Joint low-discrepancy recombination of the cached per-reaction
        class pools: column j of the (n, J) draw picks, for reaction j, WHICH
        cached curve to reuse -- so rows are new joint (cross-reaction)
        combinations that stay on the same A/B/C manifolds, filling the gaps
        the original DM's per-reaction-marginal sampling left in the joint
        space. Reactions with no cached pool fall back to an independent
        coded-space [-1, 1] box draw (matches the A-facto `2*pts - 1`
        convention used elsewhere in DesignMatrix.py).

        Returns
        -------
        full_rows : (n, n_full) array
        per_rxn   : {rxn: (n, active_len) array} -- the per-reaction sub-vectors
                    chosen, needed by ``_build_global_shuffled`` to build the
                    shuffled block from the SAME chosen values (mirrors how
                    DesignMatrix shuffles V_ into V_s).
        """
        J = len(self.layout)
        U = self._qmc_draw(n, J, seed)
        full_rows = np.zeros((n, self.n_full))
        per_rxn = {}
        for j, lay in enumerate(self.layout):
            rxn = lay["rxn"]
            f0, alen = lay["full_offset"], lay["active_len"]
            if alen == 0:
                per_rxn[rxn] = np.zeros((n, 0))
                continue
            pool = self._pool_arrays(rxn)
            if pool is not None:
                vecs, _ = pool
                idx = np.clip((U[:, j] * len(vecs)).astype(int), 0, len(vecs) - 1)
                block = vecs[idx]
            else:
                block = self.rng.uniform(-1.0, 1.0, size=(n, alen))
            full_rows[:, f0: f0 + alen] = block
            per_rxn[rxn] = block
        return full_rows, per_rxn

    def _build_global_shuffled(self, per_rxn_uns, factor, seed):
        """Shuffled block: each reaction's chosen unshuffled sub-vectors are
        independently re-permuted `factor` times and concatenated -- mirrors
        DesignMatrix.getSample_partial's V_ -> V_s construction (there with
        factor=4; configurable here via ``shuffle_factor``)."""
        rng_local = np.random.default_rng(seed)
        n_uns = next(iter(per_rxn_uns.values())).shape[0] if per_rxn_uns else 0
        if n_uns == 0 or factor <= 0:
            return np.zeros((0, self.n_full))
        full_rows = np.zeros((factor * n_uns, self.n_full))
        for lay in self.layout:
            rxn = lay["rxn"]
            f0, alen = lay["full_offset"], lay["active_len"]
            if alen == 0:
                continue
            block = per_rxn_uns[rxn]
            cols = [block[rng_local.permutation(n_uns)] for _ in range(factor)]
            full_rows[:, f0: f0 + alen] = np.concatenate(cols, axis=0)
        return full_rows

    def _sample_block_box_global(self, col_lo, col_hi, n, seed):
        """Low-discrepancy box fill spanning the FULL observed PRS-space DM
        range (not local to any one outlier) -- the no-pkl / A-facto
        equivalent of ``_build_global_unshuffled``."""
        d = len(col_lo)
        if n <= 0 or d == 0:
            return np.zeros((0, d))
        U = self._qmc_draw(n, d, seed)
        span = np.maximum(col_hi - col_lo, 1e-12)
        return col_lo[None, :] + U * span[None, :]

    def resample_global(self, outlier_prs_rows, X_ref_prs):
        """DM-gap-filling alternative to ``resample``: instead of clustering
        new points locally around the outlier, fill the JOINT gaps across the
        whole design matrix the same way DesignMatrix.getSample_partial builds
        the pPRS DM -- an "unshuffled" low-discrepancy block (recombining the
        cached per-reaction class pools, or a global box fill when no pool
        exists) plus a "shuffled" block (``shuffle_factor``x), with the
        outlier row(s) always included, and the combined block row-shuffled.

        Parameters
        ----------
        outlier_prs_rows : (k, n_prs) array -- outlier rows in PRS space,
                           always included verbatim in the output.
        X_ref_prs        : (N, n_prs) array -- full case DM (PRS space); used
                           for the global box bounds in the no-pkl fallback.

        Returns
        -------
        prs_rows, full_rows, sel_mask, row_class  -- same contract as
        ``resample``; row_class in {'OUT', 'UNS', 'SHF'}.
        """
        outlier_prs_rows = np.atleast_2d(np.asarray(outlier_prs_rows, dtype=float))
        X_ref_prs = np.asarray(X_ref_prs, dtype=float)
        col_lo = X_ref_prs.min(axis=0)
        col_hi = X_ref_prs.max(axis=0)

        n_uns = (_ceil_pow2(min(self.uns_cap, self.N_new)) if self.method == "sobol"
                 else max(1, min(self.uns_cap, self.N_new)))

        if self.class_pool is not None:
            seed_uns = int(self.rng.integers(0, 2**31 - 1))
            full_uns, per_rxn_uns = self._build_global_unshuffled(n_uns, seed_uns)
            seed_shf = int(self.rng.integers(0, 2**31 - 1))
            full_shf = self._build_global_shuffled(per_rxn_uns, self.shuffle_factor, seed_shf)
        else:
            # A-facto / no cached pkls: fill the gaps of the observed PRS-space
            # DM box directly (still converted to full-space via prs_to_full).
            seed_uns = int(self.rng.integers(0, 2**31 - 1))
            prs_uns = self._sample_block_box_global(col_lo, col_hi, n_uns, seed_uns)
            seed_shf = int(self.rng.integers(0, 2**31 - 1))
            prs_shf = self._sample_block_box_global(
                col_lo, col_hi, self.shuffle_factor * n_uns, seed_shf)
            full_uns = np.asarray([self.prs_to_full(r) for r in prs_uns]) \
                if len(prs_uns) else np.zeros((0, self.n_full))
            full_shf = np.asarray([self.prs_to_full(r) for r in prs_shf]) \
                if len(prs_shf) else np.zeros((0, self.n_full))

        labels = ["UNS"] * len(full_uns) + ["SHF"] * len(full_shf)
        full_new = (np.vstack([full_uns, full_shf]) if len(full_shf)
                    else np.asarray(full_uns))

        # The outlier row(s) are always present, verbatim, in the augmented set.
        full_outliers = np.asarray([self.prs_to_full(o) for o in outlier_prs_rows])
        full_rows = np.vstack([full_outliers, full_new])
        row_class = ["OUT"] * len(full_outliers) + labels

        prs_rows = np.asarray([self.full_to_prs(r) for r in full_rows])

        # Final shuffle of row order ("just in case").
        perm = self.rng.permutation(len(prs_rows))
        prs_rows = prs_rows[perm]
        full_rows = full_rows[perm]
        row_class = [row_class[i] for i in perm]

        return prs_rows, full_rows, self.selection_mask_row(), row_class

    # ---- hybrid (capped local + sparse global) resample -------------------- #
    def _build_local_budgeted(self, outlier_prs_rows, col_lo, col_hi, n_local):
        """Local, on-manifold densification near each outlier row, budgeted to
        ``n_local`` total new rows (split across outliers, then across
        available curve classes) -- the same per-outlier/per-class logic
        ``resample`` uses (``_build_block``), just re-parented under a shared
        hybrid budget instead of ``self.N_new``. Returns a (M, n_full) array
        (M ~= n_local, subject to the same per-outlier/per-class rounding
        ``resample`` already tolerates)."""
        classes = self.available_classes()
        k = len(outlier_prs_rows)
        per = max(1, n_local // k)
        full_out = []
        for o_prs in outlier_prs_rows:
            o_full = self.prs_to_full(o_prs)
            if classes:
                counts = [per // len(classes)] * len(classes)
                for extra in range(per - sum(counts)):
                    counts[extra] += 1
                for label, n_c in zip(classes, counts):
                    if n_c == 0:
                        continue
                    prs_block = self._build_block(o_prs, o_full, n_c, col_lo, col_hi, label)
                    full_out.append(np.asarray([self.prs_to_full(r) for r in prs_block]))
            else:
                prs_block = self._build_block(o_prs, o_full, per, col_lo, col_hi, None)
                full_out.append(np.asarray([self.prs_to_full(r) for r in prs_block]))
        return np.vstack(full_out) if full_out else np.zeros((0, self.n_full))

    def resample_hybrid(self, outlier_prs_rows, X_ref_prs, local_fraction=0.4):
        """Default resampling mode: a hard-capped mix of (a) a block concentrated
        near the outlier(s), on-manifold via the same class-aware interpolation
        ``resample`` uses, and (b) a sparse, low-discrepancy block spread evenly
        across the *whole* design space (reusing ``resample_global``'s
        recombination/box-fill machinery, but sized to a small budget with NO
        shuffled/duplicate block) -- so the fit gets fixed locally without being
        swamped by redundant global recombinations.

        This replaces ``resample_global`` as the default because that mode
        silently ignored ``N_new`` (it rounded up to a Sobol power-of-2 and then
        multiplied by ``shuffle_factor``, injecting ~4x the configured budget
        per pass) and never concentrated points near the actual outlier. See
        README_AdaptiveResampler.md for the full diagnosis.

        Parameters
        ----------
        outlier_prs_rows : (k, n_prs) array -- outlier rows in PRS space, always
                           included verbatim in the output.
        X_ref_prs        : (N, n_prs) array -- full case DM (PRS space); used for
                           per-column bounds (local box fallback + global spread).
        local_fraction   : float in [0, 1] -- share of the (post-outlier) budget
                           spent densifying near the outlier(s); the rest fills
                           the rest of the domain sparsely.

        Returns
        -------
        prs_rows, full_rows, sel_mask, row_class -- same contract as ``resample``
        / ``resample_global``; row_class in {'OUT', 'LOC', 'GLB'}.
        """
        outlier_prs_rows = np.atleast_2d(np.asarray(outlier_prs_rows, dtype=float))
        X_ref_prs = np.asarray(X_ref_prs, dtype=float)
        col_lo = X_ref_prs.min(axis=0)
        col_hi = X_ref_prs.max(axis=0)

        # Hard cap: always <= HARD_CAP_DEFAULT regardless of the configured N_new.
        n_total = max(1, min(int(self.N_new), HARD_CAP_DEFAULT))
        k = len(outlier_prs_rows)
        n_budget = max(0, n_total - k)          # remaining after outlier verbatim rows
        n_local = int(round(local_fraction * n_budget))
        n_global = n_budget - n_local

        full_outliers = np.asarray([self.prs_to_full(o) for o in outlier_prs_rows])
        row_class = ["OUT"] * len(full_outliers)
        full_blocks = [full_outliers]

        if n_local > 0:
            full_loc = self._build_local_budgeted(outlier_prs_rows, col_lo, col_hi, n_local)
            full_blocks.append(full_loc)
            row_class += ["LOC"] * len(full_loc)

        if n_global > 0:
            seed_glb = int(self.rng.integers(0, 2**31 - 1))
            if self.class_pool is not None:
                full_glb, _ = self._build_global_unshuffled(n_global, seed_glb)
            else:
                prs_glb = self._sample_block_box_global(col_lo, col_hi, n_global, seed_glb)
                full_glb = (np.asarray([self.prs_to_full(r) for r in prs_glb])
                            if len(prs_glb) else np.zeros((0, self.n_full)))
            full_blocks.append(full_glb)
            row_class += ["GLB"] * len(full_glb)

        full_rows = np.vstack(full_blocks)
        prs_rows = np.asarray([self.full_to_prs(r) for r in full_rows])

        # Final shuffle of row order ("just in case", matches resample_global).
        perm = self.rng.permutation(len(prs_rows))
        prs_rows = prs_rows[perm]
        full_rows = full_rows[perm]
        row_class = [row_class[i] for i in perm]

        return prs_rows, full_rows, self.selection_mask_row(), row_class

    # ---- gap-fill (real new curves, DM-faithful) resample ------------------ #
    def _get_dm_instance(self, seed):
        """Lazily construct a live DesignMatrix so ``resample_gap_fill`` can
        call its REAL class-A/B/C curve generators (as opposed to every other
        mode here, which only recombines/perturbs the already-cached ones).

        Imported here, not at module top, so ``local``/``dm_global``/``hybrid``
        stay usable in environments where DesignMatrix.py's own hard
        dependencies (``qmc_sampling.py``, ``/shuffle.so``) aren't importable
        -- this is only ever needed if ``gap_fill`` is actually selected.
        """
        try:
            import DesignMatrix as DM_mod
            import qmc_sampling
        except ImportError as exc:
            raise RuntimeError(
                "sample_mode='gap_fill' calls DesignMatrix's real class-A/B/C "
                "curve generators, which need DesignMatrix.py's own "
                "dependencies (qmc_sampling.py, /shuffle.so) importable. "
                f"Import failed: {exc}"
            ) from exc
        # sample_length/ind are unused by the getClass*_Curves[_partial] call
        # path (verified by reading them: they only touch self.unsrt,
        # self.allowed_count, and the joint_block/shadow argument), so
        # placeholder values are safe here.
        dm = DM_mod.DesignMatrix(self.unsrt, self.design, 1, 1,
                                  method=self.method, seed=seed)
        return dm, qmc_sampling

    def _rxn_to_full_matrix(self, per_rxn, n):
        """Assemble an (n, n_full) array from a {rxn: (n, active_len)} dict,
        in ``self.layout`` order."""
        full = np.zeros((n, self.n_full))
        for lay in self.layout:
            rxn, f0, alen = lay["rxn"], lay["full_offset"], lay["active_len"]
            if alen == 0 or rxn not in per_rxn:
                continue
            full[:, f0: f0 + alen] = np.asarray(per_rxn[rxn])[:n]
        return full

    def _generate_class_candidates(self, dm, qmc_mod, label, n_gen, seed,
                                    rxn_param_indices):
        """Call DesignMatrix's REAL class-``label`` generator for every
        reaction at once (a joint low-discrepancy shadow block, exactly like
        DesignMatrix.getSamples()/getSample_partial() build the real DM).

        ``rxn_param_indices`` is None for Full PRS (use the non-partial
        generators); a {rxn: selected-index-tuple} dict for Partial PRS (use
        the ``_partial`` generators, from ``DesignMatrix._get_rxn_param_indices``).

        Returns
        -------
        per_rxn  : {rxn: (n_actual, active_len) array}
        n_actual : int (>= n_gen; Sobol rounds up to a power of 2)
        """
        J = len(self.layout)
        if rxn_param_indices is None:
            dims = {"A": 1, "B": 2, "C": 2}[label]
            shadow, n_actual = qmc_mod.build_joint_shadow(self.method, n_gen, J, dims, seed=seed)
            if label == "A":
                curves, _ = dm.getClassA_Curves(n_actual, joint_block=shadow)
            elif label == "B":
                curves, _ = dm.getClassB_Curves(n_actual, joint_block=shadow)
            else:
                curves, _ = dm.getClassC_Curves(n_actual, joint_block=shadow)
        else:
            dims = {"A": 2, "B": 4, "C": 2}[label]
            shadow, n_actual = qmc_mod.build_joint_shadow(self.method, n_gen, J, dims, seed=seed)
            if label == "A":
                curves = dm.getClassA_Curves_partial(n_actual, rxn_param_indices, shadow, base_seed=seed)
            elif label == "B":
                curves = dm.getClassB_Curves_partial(n_actual, rxn_param_indices, shadow, base_seed=seed)
            else:
                curves = dm.getClassC_Curves_partial(n_actual, rxn_param_indices, shadow, base_seed=seed)

        per_rxn = {}
        for lay in self.layout:
            rxn, alen = lay["rxn"], lay["active_len"]
            if alen == 0:
                continue
            vecs = list(curves.get(rxn, []))
            if len(vecs) < n_actual:
                # Mirrors DesignMatrix.getSample_partial's V_ backfill: a
                # reaction with < 2 selected params can't have a real class-B/C
                # curve (infeasible), so pad with extra class-A draws instead.
                n_pad = n_actual - len(vecs)
                pad_seed = dm._reaction_fallback_seed(seed, rxn, f"{label}_pad")
                pad_rng = np.random.default_rng(pad_seed)
                if rxn_param_indices is not None:
                    idx = rxn_param_indices[rxn]
                    pad = dm.unsrt[rxn].getClassA_partial(idx, n_pad, pad_rng)
                else:
                    pad = list(2.0 * pad_rng.random((n_pad, alen)) - 1.0)
                vecs = vecs + list(pad)
            per_rxn[rxn] = np.asarray(vecs[:n_actual], dtype=float).reshape(n_actual, alen)
        return per_rxn, n_actual

    def _select_isolated(self, per_rxn, n_actual, n_keep, ref_prs, seed):
        """From an oversampled {rxn: (n_actual, active_len)} candidate pool,
        assemble full-space rows, isolation-filter them (PRS space) against
        ``ref_prs``, and keep the ``n_keep`` most-isolated ones -- returning
        both the kept full-space rows and the matching per-reaction subset
        (needed to build the shuffled block from the SAME chosen curves)."""
        full_cand = self._rxn_to_full_matrix(per_rxn, n_actual)
        prs_cand = np.asarray([self.full_to_prs(r) for r in full_cand])
        order, _ = _isolation_filter(prs_cand, ref_prs, k=self.gapfill_knn_k,
                                      radius_frac=self.gapfill_knn_radius_frac)
        keep = order[:n_keep]
        full_keep = full_cand[keep]
        per_rxn_keep = {rxn: np.asarray(arr)[keep] for rxn, arr in per_rxn.items()}
        return full_keep, per_rxn_keep

    def resample_gap_fill(self, outlier_prs_rows, ref_prs_rows):
        """DM-faithful gap-filling resample: calls DesignMatrix's REAL
        class-A/B/C (or the A-facto low-discrepancy box) generators to
        synthesize genuinely NEW curves -- unlike every other mode here, which
        only recombines/perturbs already-cached curves -- then keeps only the
        candidates that land far from any existing sample. Mirrors
        DesignMatrix's own unshuffled (20%) + shuffled (80%,
        ``gapfill_shuffle_factor``x) structure for 'A1+B1+C1'; 'A-facto' has
        no class/shuffle structure in the real DM either, so this mirrors
        that too with a single flat low-discrepancy block.

        Parameters
        ----------
        outlier_prs_rows : (k, n_prs) array -- outlier rows in PRS space,
                           always included verbatim in the output.
        ref_prs_rows      : (N, n_prs) array -- the gap-avoidance reference
                           set: the original case DM unioned with every row
                           added by prior passes this run (grown by the
                           caller -- see run_script.retrain_around_outlier).

        Returns
        -------
        prs_rows, full_rows, sel_mask, row_class -- same contract as the
        other ``resample*`` methods; row_class in {'OUT','A','B','C','SHF'}
        for 'A1+B1+C1', {'OUT','GAP'} for 'A-facto'.
        """
        outlier_prs_rows = np.atleast_2d(np.asarray(outlier_prs_rows, dtype=float))
        ref_prs_rows = np.asarray(ref_prs_rows, dtype=float)

        n_total = max(1, min(int(self.N_new), HARD_CAP_DEFAULT))
        k_out = len(outlier_prs_rows)
        n_budget = max(0, n_total - k_out)

        full_outliers = np.asarray([self.prs_to_full(o) for o in outlier_prs_rows])
        row_class = ["OUT"] * len(full_outliers)
        full_blocks = [full_outliers]
        ref_running = ref_prs_rows
        seed_base = int(self.rng.integers(0, 2**31 - 1))

        if self.design != "A1+B1+C1":
            # A-facto: the real DM has no class/shuffle structure here either
            # (DesignMatrix.getSamples()'s A-facto branch is a single box
            # draw) -- mirror that with one flat, isolation-filtered block.
            if n_budget > 0:
                dm, qmc_mod = self._get_dm_instance(seed_base)
                n_gen = self.gapfill_oversample * n_budget
                pts, n_actual = qmc_mod.joint_uniform_block(
                    self.method, n_gen, self.n_full, seed=seed_base)
                full_cand = 2.0 * np.asarray(pts) - 1.0
                prs_cand = np.asarray([self.full_to_prs(r) for r in full_cand])
                order, _ = _isolation_filter(prs_cand, ref_running, k=self.gapfill_knn_k,
                                              radius_frac=self.gapfill_knn_radius_frac)
                keep = order[:n_budget]
                full_gap = full_cand[keep]
                full_blocks.append(full_gap)
                row_class += ["GAP"] * len(full_gap)
        elif n_budget > 0:
            dm, qmc_mod = self._get_dm_instance(seed_base)
            rxn_param_indices = (dm._get_rxn_param_indices(self.selected_params)
                                  if self.prs_type != "Full" else None)

            # Unshuffled block sized so unshuffled*(1+shuffle_factor) ~= budget
            # (matches DesignMatrix's real 20:80 split when shuffle_factor=4).
            n_uns_total = max(1, n_budget // (1 + max(1, self.gapfill_shuffle_factor)))
            n_a_req = max(1, int(round(0.1 * n_uns_total)))
            n_b_req = max(1, int(round(0.45 * n_uns_total)))
            n_c_req = max(1, n_uns_total - n_a_req - n_b_req)

            per_rxn_uns = {}
            for label, n_req in (("A", n_a_req), ("B", n_b_req), ("C", n_c_req)):
                n_gen = self.gapfill_oversample * n_req
                seed_c = int(self.rng.integers(0, 2**31 - 1))
                per_rxn_c, n_actual = self._generate_class_candidates(
                    dm, qmc_mod, label, n_gen, seed_c, rxn_param_indices)
                full_c, per_rxn_keep = self._select_isolated(
                    per_rxn_c, n_actual, n_req, ref_running, seed_c)
                full_blocks.append(full_c)
                row_class += [label] * len(full_c)
                for rxn, arr in per_rxn_keep.items():
                    per_rxn_uns.setdefault(rxn, []).append(arr)
                # Grow the running reference set so later classes (B, C) don't
                # re-fill the same gap A just used.
                if len(full_c):
                    ref_running = np.vstack(
                        [ref_running, np.asarray([self.full_to_prs(r) for r in full_c])])

            per_rxn_uns = {rxn: np.concatenate(arrs, axis=0) for rxn, arrs in per_rxn_uns.items()}
            n_uns_actual = next(iter(per_rxn_uns.values())).shape[0] if per_rxn_uns else 0

            if n_uns_actual > 0 and self.gapfill_shuffle_factor > 0:
                seed_shf = int(self.rng.integers(0, 2**31 - 1))
                # Oversample the shuffle so the isolation filter has survivors
                # to pick from, then keep exactly shuffle_factor*n_uns_actual.
                full_shf_cand = self._build_global_shuffled(
                    per_rxn_uns, self.gapfill_shuffle_factor * self.gapfill_oversample, seed_shf)
                prs_shf_cand = np.asarray([self.full_to_prs(r) for r in full_shf_cand])
                order, _ = _isolation_filter(prs_shf_cand, ref_running, k=self.gapfill_knn_k,
                                              radius_frac=self.gapfill_knn_radius_frac)
                n_shf_keep = self.gapfill_shuffle_factor * n_uns_actual
                keep = order[:n_shf_keep]
                full_shf = full_shf_cand[keep]
                full_blocks.append(full_shf)
                row_class += ["SHF"] * len(full_shf)

        full_rows = np.vstack(full_blocks)
        prs_rows = np.asarray([self.full_to_prs(r) for r in full_rows])
        # Re-expand from PRS space so unselected columns are exactly zero for
        # Partial PRS (matches every other resample* method's contract).
        full_rows = np.asarray([self.prs_to_full(r) for r in prs_rows])

        # Final shuffle of row order ("just in case", matches every other mode).
        perm = self.rng.permutation(len(prs_rows))
        prs_rows = prs_rows[perm]
        full_rows = full_rows[perm]
        row_class = [row_class[i] for i in perm]

        return prs_rows, full_rows, self.selection_mask_row(), row_class


# --------------------------------------------------------------------------- #
# D-optimal resampler (Stage 3 of PAAR -- see report/)                        #
# --------------------------------------------------------------------------- #
class DOptimalResampler(OutlierResampler):
    """Classical D-optimal design augmentation, band-validity-gated.

    Replaces the geometric heuristics above (``resample``/``resample_global``/
    ``resample_hybrid``/``resample_gap_fill``) as the recommended mode once the
    PRS is fitted in the identifiable subspace (IdentifiablePRS.py): after
    that fix the surface is LINEAR in its fitted coefficients (a polynomial
    of the score-space features), so the information-theoretic gold standard
    for "which candidate point reduces coefficient-estimate variance the
    most" is exact and cheap. For a candidate's basis-feature row b (built
    the SAME way ResponseSurface.build_basis does) and current Gram matrix
    G = B_train^T B_train, adding b increases det(G) by a factor
    (1 + b @ G^-1 @ b^T); greedily picking the candidate that maximizes this,
    then Sherman-Morrison-updating G^-1, is the standard cheap approximation
    to the (NP-hard) exact D-optimal subset -- "greedy D-optimal
    augmentation", the augmentation-only special case of Fedorov exchange
    (no swap/removal step, since existing training rows are never discarded
    here). This directly minimizes how many new simulations are needed for a
    given error reduction -- the plan's stated cost objective.

    Every candidate is rejection-sampled against
    UncertaintyExtractor.is_band_valid (via the batched
    Uncertainty.batch_violation_ratio) before it is even eligible for
    selection -- per instruction, new (simulated) points must be governed by
    strict band validity, independent of whatever tolerance policy governs
    the historical training rows (see IdentifiablePRS.screen_rows). This
    means a DOptimalResampler pass, unlike every mode above, can never spend
    a simulation on a mechanism no expert would call physically plausible.

    The acquisition score mixes two normalized-to-[0,1] terms:
        score = acquisition_alpha * (b G^-1 b^T, normalized)      -- D-optimal
              + (1 - acquisition_alpha) * (RBF proximity to the      -- targeting
                    failing hold-out point(s), residual-weighted)
    so a pass still concentrates near the region that is actually failing
    (like ``resample_hybrid``'s local block) while never wasting budget on a
    candidate that is redundant with the existing design (unlike every
    geometric mode, which has no such check at all).
    """

    def __init__(self, *args, band_tolerance=1.0, temps_for_validity=None,
                 acquisition_alpha=0.6, candidate_oversample=6, **kwargs):
        """
        band_tolerance : float
            Max allowed max_{reaction,T} |Δκ|/f_prior for a NEW point to be
            eligible at all (see IdentifiablePRS.screen_rows's tolerance for
            the historical-row analogue -- these are independent knobs by
            design, per instruction).
        temps_for_validity : {rxn: array_like of T} or None
            Temperatures each reaction's band-validity is checked over.
            None disables validity gating entirely (every candidate passes) --
            useful for controlled comparisons, but not the default use.
        acquisition_alpha : float in [0, 1]
            D-optimal vs targeting mix (see class docstring). 1.0 = pure
            D-optimal (ignores where the outliers are); 0.0 = pure
            proximity-to-outlier targeting (ignores design information gain).
        candidate_oversample : int
            Candidates generated (before band-validity rejection and
            D-optimal selection) per selected row, so both filters have
            survivors to choose from/rank.
        """
        super().__init__(*args, **kwargs)
        self.band_tolerance = float(band_tolerance)
        self.temps_for_validity = temps_for_validity
        self.acquisition_alpha = float(acquisition_alpha)
        self.candidate_oversample = max(1, int(candidate_oversample))

    # ---- band-validity gate ------------------------------------------------ #
    def _reaction_indices(self, lay):
        return lay["sel_local"] if self.prs_type != "Full" else list(range(lay["active_len"]))

    def band_valid_mask(self, full_rows):
        """Per-row AND over every active reaction's is_band_valid, using the
        LIVE self.unsrt[rxn] extractor objects -- the canonical acceptance
        test (Uncertainty.UncertaintyExtractor.is_band_valid /
        batch_violation_ratio), not a re-derivation. Rows for a reaction with
        no configured T_arr (temps_for_validity missing that key and no
        '__default__') are not checked for that reaction (pass by default) --
        callers that want a hard global gate should always supply
        '__default__'.

        Returns an all-True mask (no gating) if temps_for_validity is None.
        """
        full_rows = np.asarray(full_rows, dtype=float)
        if self.temps_for_validity is None or len(full_rows) == 0:
            return np.ones(len(full_rows), dtype=bool)

        mask = np.ones(len(full_rows), dtype=bool)
        for lay in self.layout:
            rxn, f0, alen = lay["rxn"], lay["full_offset"], lay["active_len"]
            idx = self._reaction_indices(lay)
            if alen == 0 or not idx or rxn not in self.unsrt:
                continue
            T_arr = self.temps_for_validity.get(rxn, self.temps_for_validity.get("__default__"))
            if T_arr is None:
                continue
            ext = self.unsrt[rxn]
            # Sigma = L_full @ L_full.T is reconstructed from the live
            # extractor's own cholskyDeCorrelateMat -- same source of truth
            # UncertaintyExtractor._psac_get_reduced_L /
            # Uncertainty.batch_violation_ratio use, just reached via
            # IdentifiablePRS's dependency-light reimplementation (the two
            # are cross-validated to agree bit-for-bit; see report/).
            Sigma = ext.cholskyDeCorrelateMat @ ext.cholskyDeCorrelateMat.T
            _, L_r = _reduced_cholesky(Sigma, idx)
            W, fp = _reaction_W_fprior(T_arr, L_r, idx)
            Z = full_rows[:, f0:f0 + alen][:, idx]
            dk = Z @ W.T
            ratio = np.max(np.abs(dk) / np.maximum(fp[None, :], 1e-300), axis=1)
            mask &= (ratio <= self.band_tolerance)
        return mask

    # ---- candidate generation ---------------------------------------------- #
    def generate_candidates(self, outlier_prs_rows, X_ref_prs, n_target, local_fraction=0.5):
        """Oversample local (near-outlier, class-aware) + global (sparse,
        low-discrepancy) candidates -- the same machinery ``resample_hybrid``
        uses for its LOC/GLB blocks -- then keep only the band-valid ones.

        Returns (prs_cand, full_cand); may have fewer than
        candidate_oversample * n_target rows if many candidates are rejected
        (best-effort -- the caller's D-optimal selection then picks up to
        n_target of whatever survives).
        """
        outlier_prs_rows = np.atleast_2d(np.asarray(outlier_prs_rows, dtype=float))
        X_ref_prs = np.asarray(X_ref_prs, dtype=float)
        col_lo, col_hi = X_ref_prs.min(axis=0), X_ref_prs.max(axis=0)

        n_gen = max(1, int(self.candidate_oversample * max(n_target, 1)))
        n_local = int(round(local_fraction * n_gen))
        n_global = n_gen - n_local

        blocks = []
        if n_local > 0:
            blocks.append(self._build_local_budgeted(outlier_prs_rows, col_lo, col_hi, n_local))
        if n_global > 0:
            seed_glb = int(self.rng.integers(0, 2**31 - 1))
            if self.class_pool is not None:
                full_glb, _ = self._build_global_unshuffled(n_global, seed_glb)
            else:
                prs_glb = self._sample_block_box_global(col_lo, col_hi, n_global, seed_glb)
                full_glb = (np.asarray([self.prs_to_full(r) for r in prs_glb])
                            if len(prs_glb) else np.zeros((0, self.n_full)))
            blocks.append(full_glb)

        full_cand = np.vstack(blocks) if blocks else np.zeros((0, self.n_full))
        if len(full_cand) == 0:
            return np.zeros((0, self.n_prs)), full_cand

        valid = self.band_valid_mask(full_cand)
        full_cand = full_cand[valid]
        prs_cand = (np.asarray([self.full_to_prs(r) for r in full_cand])
                    if len(full_cand) else np.zeros((0, self.n_prs)))
        return prs_cand, full_cand

    # ---- greedy D-optimal augmentation -------------------------------------- #
    def select_d_optimal(self, prs_cand, B_train, build_basis, n_select,
                          target_points_prs=None, target_weights=None,
                          ridge_alpha=0.0):
        """Greedy D-optimal augmentation with residual-weighted targeting.

        B_train      : (N, p) CURRENT training design's basis matrix (e.g.
                       ResponseSurface.build_basis(X_train_prs)), used to
                       seed G = B_train^T B_train (+ ridge).
        build_basis  : callable, PRS-space rows (n, n_prs) -> (n, p) basis --
                       pass the CURRENT fit's ResponseSurface.build_basis
                       (bound to its feature_map/order) so candidates are
                       scored in the exact basis being refit on.
        n_select     : how many rows to pick (<= len(prs_cand)).
        target_points_prs, target_weights : failing hold-out points (PRS
                       space) and nonnegative weights (e.g. |residual|) for
                       the targeting term; None/empty -> pure D-optimal.

        Returns the selected row indices INTO prs_cand, in pick order
        (best/most-informative first).
        """
        if len(prs_cand) == 0 or n_select <= 0:
            return np.array([], dtype=int)

        Bc = np.asarray(build_basis(prs_cand), dtype=float)
        p = Bc.shape[1]
        Bt = np.asarray(B_train, dtype=float)
        G = Bt.T @ Bt
        if ridge_alpha > 0:
            I = np.eye(p); I[0, 0] = 0.0
            G = G + ridge_alpha * len(Bt) * I
        else:
            # Tiny singular-safe jitter (not a modeling choice -- create_response_surface
            # itself uses plain QR/ridge for the real fit; this is only to keep
            # G^-1 well-defined if B_train is rank-deficient during scoring).
            G = G + (1e-10 * np.trace(G) / max(p, 1)) * np.eye(p)
        Ginv = np.linalg.pinv(G)

        prs_cand = np.asarray(prs_cand, dtype=float)
        if target_points_prs is not None and len(target_points_prs):
            tgt = np.atleast_2d(np.asarray(target_points_prs, dtype=float))
            w = (np.ones(len(tgt)) if target_weights is None
                 else np.abs(np.asarray(target_weights, dtype=float)))
            w = w / (w.sum() + 1e-30)
            span = np.maximum(prs_cand.max(axis=0) - prs_cand.min(axis=0), 1e-12)
            target_score_all = np.zeros(len(prs_cand))
            for t, wt in zip(tgt, w):
                dn = (prs_cand - t[None, :]) / span[None, :]
                target_score_all += wt * np.exp(-0.5 * np.sum(dn ** 2, axis=1))
        else:
            target_score_all = np.zeros(len(prs_cand))

        remaining = list(range(len(prs_cand)))
        chosen = []
        for _ in range(min(int(n_select), len(remaining))):
            Bv = Bc[remaining]
            dvar = np.einsum('ij,jk,ik->i', Bv, Ginv, Bv)          # b G^-1 b^T
            dvar_n = dvar / (dvar.max() + 1e-30)
            tsub = target_score_all[remaining]
            tgt_n = tsub / tsub.max() if tsub.max() > 0 else tsub
            score = self.acquisition_alpha * dvar_n + (1.0 - self.acquisition_alpha) * tgt_n

            best_local = int(np.argmax(score))
            best_global = remaining[best_local]
            chosen.append(best_global)

            b = Bc[best_global]
            Gb = Ginv @ b
            denom = 1.0 + float(b @ Gb)
            if denom > 1e-12:
                Ginv = Ginv - np.outer(Gb, Gb) / denom              # Sherman-Morrison
            remaining.pop(best_local)

        return np.asarray(chosen, dtype=int)

    # ---- top-level entry point ---------------------------------------------- #
    def resample_d_optimal(self, outlier_prs_rows, X_ref_prs, X_train_prs,
                            build_basis, target_weights=None,
                            local_fraction=0.5, ridge_alpha=0.0):
        """D-optimal counterpart to resample_hybrid: band-valid candidates,
        greedily selected to maximize design information gain while
        favouring the region near the failing outlier(s).

        Parameters
        ----------
        outlier_prs_rows : (k, n_prs) array -- outlier rows in PRS space,
                           always included verbatim in the output (row_class
                           'OUT'), matching every other resample* method.
        X_ref_prs        : (N, n_prs) array -- full case DM, for local/global
                           candidate-generation bounds (same role as in
                           resample_hybrid).
        X_train_prs      : (M, n_prs) array -- the CURRENT training rows
                           (post band-validity screening if the caller
                           applies IdentifiablePRS.screen_rows to them --
                           this method doesn't screen X_train_prs itself).
        build_basis      : callable, see select_d_optimal.
        target_weights   : optional per-outlier weight (e.g. |residual %|);
                           None -> uniform.

        Returns
        -------
        prs_rows, full_rows, sel_mask, row_class, info -- same contract as
        every other resample* method plus a diagnostics dict; row_class in
        {'OUT', 'DOPT'}.

        Defense-in-depth (2026-07-31): D-optimal resampling is A1+B1+C1-only
        -- unlike resample_gap_fill/resample_hybrid, which have a defined
        A-facto fallback (a flat box-draw, since DesignMatrix's own A-facto
        branch is a single box draw too), D-optimal candidate SELECTION is
        meaningless without the class-A/B/C curve pools this design mode
        alone provides, so under A-facto this returns the outliers UNCHANGED
        (zero new points picked) rather than falling back to an uninformed
        draw -- the run_script.py call site is expected to skip calling this
        entirely for A-facto (see its own guard), this is a second,
        independent check in case a future caller doesn't.
        """
        outlier_prs_rows = np.atleast_2d(np.asarray(outlier_prs_rows, dtype=float))
        full_outliers = np.asarray([self.prs_to_full(o) for o in outlier_prs_rows])
        if self.design != "A1+B1+C1":
            info = {"n_candidates_generated": 0, "n_candidates_band_valid": 0,
                     "n_selected": 0, "n_budget": 0, "reject_rate": 0.0,
                     "skipped_reason": f"design={self.design!r} (A1+B1+C1-only)"}
            row_class = ["OUT"] * len(full_outliers)
            return outlier_prs_rows, full_outliers, self.selection_mask_row(), row_class, info
        n_budget = max(0, min(int(self.N_new), HARD_CAP_DEFAULT) - len(full_outliers))

        prs_cand, full_cand = self.generate_candidates(
            outlier_prs_rows, X_ref_prs, n_budget, local_fraction=local_fraction)

        B_train = build_basis(np.asarray(X_train_prs, dtype=float))
        picked = self.select_d_optimal(
            prs_cand, B_train, build_basis, n_budget,
            target_points_prs=outlier_prs_rows, target_weights=target_weights,
            ridge_alpha=ridge_alpha)

        prs_new = prs_cand[picked] if len(picked) else np.zeros((0, self.n_prs))
        full_new = full_cand[picked] if len(picked) else np.zeros((0, self.n_full))

        prs_rows = np.vstack([outlier_prs_rows, prs_new]) if len(prs_new) else outlier_prs_rows
        full_rows = np.vstack([full_outliers, full_new]) if len(full_new) else full_outliers
        row_class = ["OUT"] * len(full_outliers) + ["DOPT"] * len(full_new)

        info = {
            "n_candidates_generated": int(self.candidate_oversample * max(n_budget, 1)),
            "n_candidates_band_valid": int(len(prs_cand)),
            "n_selected": int(len(full_new)),
            "n_budget": int(n_budget),
            "reject_rate": (1.0 - len(prs_cand) / max(1, self.candidate_oversample * max(n_budget, 1))),
        }
        return prs_rows, full_rows, self.selection_mask_row(), row_class, info
