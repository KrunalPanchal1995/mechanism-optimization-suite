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

``resample_global`` is the DM-faithful alternative to ``resample``: instead of
clustering new points locally around the outlier (which can distort the global
fit if overused), it fills the JOINT gaps across the whole design matrix --
mirroring how DesignMatrix.getSample_partial itself builds the pPRS DM: an
"unshuffled" low-discrepancy (Sobol/LHS/numpy) block that recombines the
cached per-reaction class-A/B/C curves into new joint (cross-reaction) rows,
plus a "shuffled" block (each reaction's chosen curves independently
re-permuted, `shuffle_factor`x), with the outlier row(s) always included and
the whole thing re-shuffled at the end. See run_script.retrain_around_outlier.
"""

import os
import glob
import pickle

import numpy as np
from scipy.stats import qmc

CLASS_LABELS = ("A", "B", "C")


def _ceil_pow2(n):
    """Smallest power of 2 >= n (Sobol wants a power-of-2 batch for its
    balance property -- mirrors DesignMatrix's qmc_sampling.ceil_pow2)."""
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


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
    """

    def __init__(self, unsrt_data, selected_params, design, prs_type,
                 base_dir, case_index, N_new=200, delta_fraction=0.15,
                 seed=42, knn=8, uns_cap=256, shuffle_factor=2, method="sobol"):
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
