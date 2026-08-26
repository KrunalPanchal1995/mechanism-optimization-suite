"""
target_sampling.py
-------------------
Diversity sampling over experimental T-P-phi condition space, used by the
Mechanism Reduction section of generate_target_opt.py to pick a representative
subset of loaded experimental targets to reduce a mechanism against.

Tk-free by design — self-contained numerical logic so it can be used/tested
independently of the GUI.
"""

from __future__ import annotations

from typing import List

import numpy as np


def _normalize(points: np.ndarray) -> np.ndarray:
    """Min-max scale each column to [0, 1]. Raw T (100s-1000s K), P (fractions
    to 100s atm), and phi (roughly 0.1-4) are on wildly different scales —
    without this, T would dominate any Euclidean distance.
    """
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    span = hi - lo
    span[span == 0] = 1.0  # constant column — avoid divide-by-zero, contributes 0 to distance
    return (points - lo) / span


def farthest_point_sample(points: np.ndarray, n: int, seed: int = 0) -> List[int]:
    """Weighted (k-means++-style) farthest-point diversity sampling.

    Parameters
    ----------
    points : (N, 3) array of [T, P, phi] (or any 3 comparable axes)
    n : number of points to select (2 <= n <= N)
    seed : RNG seed — deterministic and reproducible per seed value, so a
        given seed always returns the same subset ("redo" with the same seed
        reproduces it), but different seeds explore genuinely different
        diverse subsets (unlike a strict greedy argmax farthest-point walk,
        which rarely ties on real-valued data and would otherwise return the
        same subset for almost every seed).

    Returns
    -------
    list[int] of length n — selected row indices into `points`.

    Algorithm (k-means++ seeding, adapted for a fixed sample size)
    ---------------------------------------------------------------
    1. Normalize each column to [0, 1] (see _normalize) — raw T/P/phi are on
       wildly different scales, so unscaled distance would be dominated by T.
    2. First point: sampled with probability proportional to squared distance
       from the domain centroid (points far from the "middle" of the domain
       are more likely to be picked, but it's not forced to be the single
       farthest one).
    3. Maintain min_dist[i] = distance from point i to the nearest already-
       selected point.
    4. Repeat n-1 times: sample the next point with probability proportional
       to min_dist**2 (never an already-selected point, since its min_dist is
       0), then update min_dist via an elementwise min with the new point's
       distances. This still strongly favors spreading points across the
       domain (same intuition as strict farthest-point) while remaining
       genuinely seed-dependent.
    """
    n_points = points.shape[0]
    if n < 2 or n > n_points:
        raise ValueError(f"n must be in [2, {n_points}], got {n}")

    rng = np.random.default_rng(seed)
    norm = _normalize(points.astype(float))

    def _weighted_choice(weights: np.ndarray) -> int:
        total = weights.sum()
        if total <= 0:
            # All remaining candidates coincide with already-selected points
            # (e.g. duplicate rows) — fall back to uniform choice among them.
            return int(rng.choice(n_points))
        return int(rng.choice(n_points, p=weights / total))

    centroid = norm.mean(axis=0)
    dist_to_centroid = np.linalg.norm(norm - centroid, axis=1)
    first = _weighted_choice(dist_to_centroid ** 2)

    selected = [first]
    min_dist = np.linalg.norm(norm - norm[first], axis=1)
    min_dist[first] = 0.0

    for _ in range(n - 1):
        nxt = _weighted_choice(min_dist ** 2)
        selected.append(nxt)
        dist_to_nxt = np.linalg.norm(norm - norm[nxt], axis=1)
        min_dist = np.minimum(min_dist, dist_to_nxt)
        min_dist[nxt] = 0.0

    return selected


def coverage_ratio(points: np.ndarray, subset_idx: List[int]) -> float:
    """Per-axis (normalized) range ratio of the subset vs. the full dataset,
    averaged over all axes — a cheap "did we cover the domain" heuristic.

    Avoids scipy.spatial.ConvexHull, which degenerates for small/coplanar
    subsets (e.g. n=2, or any axis with near-zero spread). Returns a value in
    [0, 1]; callers should suggest increasing N when this falls below ~0.6.
    """
    norm = _normalize(points.astype(float))
    full_range = norm.max(axis=0) - norm.min(axis=0)
    subset = norm[subset_idx]
    subset_range = subset.max(axis=0) - subset.min(axis=0)
    ratios = np.divide(subset_range, full_range, out=np.ones_like(full_range), where=full_range > 0)
    return float(np.mean(np.clip(ratios, 0.0, 1.0)))
