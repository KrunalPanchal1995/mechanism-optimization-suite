"""
rmip/analysis/experiment_design.py
==================================
FEATURE_ROADMAP #8 — optimal experiment design / Design of Experiments.

Which experiment should you run next to pin down the rate parameters the most?
This scores candidate experiments (each represented by a stored gPC surrogate of
its QoI over the uncertain reaction A-factors) by the Fisher information they add
to the calibration, and greedily selects the most informative set.

Method
------
Linearise each candidate's surrogate model M_k(θ) about the prior mean θ₀ = 0
(log-multiplier space).  Its information contribution is the rank-1
   F_k = g_k g_kᵀ / σ_k² ,   g_k = ∇M_k(θ₀)   (PCEPoly.grad),
with σ_k the measurement noise.  A Gaussian prior contributes precision
Λ₀ = diag(1/σ_prior²).  For a selected set S the posterior precision is
Λ_S = Λ₀ + Σ_{k∈S} F_k and the posterior covariance Λ_S⁻¹.  Candidates are added
greedily to maximise

  * **D-optimality**: log det Λ_S  (shrink the joint uncertainty ellipsoid), or
  * **A-optimality**: −trace Λ_S⁻¹ (shrink the average parameter variance).

This reuses the exact posterior-covariance machinery of the MUM-PCE calibrator
(``cov = (JᵀJ)⁻¹``); here each ``g_k g_kᵀ/σ_k²`` is one row of that ``JᵀJ``.

Honest scope
------------
Local (linearised-at-prior) A-/D-optimal *selection* over existing surrogates —
the standard, cheap first cut.  Fully Bayesian (posterior-averaged) or nonlinear
sequential design is a follow-on.

Usage
-----
>>> from rmip.analysis.experiment_design import ExperimentDesigner
>>> res = ExperimentDesigner("project.db").design(
...     candidate_run_ids=[1, 2, 3, 4], n_select=2, criterion="D")
>>> [s["name"] for s in res.selected]
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from rmip.types import DoEResult

log = logging.getLogger(__name__)


def _criterion_gain(Lambda: np.ndarray, F: np.ndarray, criterion: str) -> float:
    """Increase in the design criterion from adding information matrix ``F``."""
    if criterion == "A":
        return float(np.trace(np.linalg.inv(Lambda))
                     - np.trace(np.linalg.inv(Lambda + F)))
    # D-optimality (default): increase in log-det of the precision
    s0, ld0 = np.linalg.slogdet(Lambda)
    s1, ld1 = np.linalg.slogdet(Lambda + F)
    return float(ld1 - ld0)


class ExperimentDesigner:
    """Fisher-information experiment selection over stored gPC surrogates."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def design(
        self,
        candidate_run_ids: Sequence[int],
        sigmas:            Optional[Sequence[float]] = None,
        prior_sigma:       Optional[float] = None,
        prior_factor:      Optional[float] = None,
        n_select:          Optional[int] = None,
        criterion:         str = "D",
        save_db:           bool = True,
    ) -> DoEResult:
        """
        Rank/select candidate experiments by information gain.

        Parameters
        ----------
        candidate_run_ids : sequence[int]
            ``surrogate_runs`` ids — each a candidate experiment's QoI surrogate.
        sigmas : sequence[float], optional
            Per-candidate measurement noise (surrogate output units). Default 1.0.
        prior_sigma / prior_factor : float, optional
            Gaussian prior width per parameter (σ or A-factor f→σ=ln f/2).
            Default: from the first surrogate's stored factor_bound.
        n_select : int, optional
            How many experiments to greedily select (default: all).
        criterion : "D" | "A"
        """
        from rmip.analysis.bayesian_calibration import (
            load_surrogate_target, surrogate_parameter_names,
        )
        import math
        import sqlite3

        def prior_sigma_from_factor(factor: float, dim: int) -> np.ndarray:
            # A-factor uncertainty f → Gaussian σ = ln(f)/2 per parameter
            return np.full(dim, math.log(factor) / 2.0)

        criterion = criterion.upper()
        if criterion not in ("D", "A"):
            raise ValueError("criterion must be 'D' or 'A'")
        run_ids = list(candidate_run_ids)
        if not run_ids:
            raise ValueError("no candidate surrogate runs given")
        if sigmas is None:
            sigmas = [1.0] * len(run_ids)

        param_names = surrogate_parameter_names(self._db_path, run_ids[0])
        d = len(param_names)

        # prior precision
        if prior_sigma is not None:
            psig = np.full(d, float(prior_sigma))
        elif prior_factor is not None:
            psig = prior_sigma_from_factor(prior_factor, d)
        else:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute("SELECT factor_bound, mechanism_id FROM surrogate_runs "
                               "WHERE id=?", (run_ids[0],)).fetchone()
            conn.close()
            factor = (row[0] if row and row[0] else 2.0)
            mech_id = row[1] if row else None
            psig = prior_sigma_from_factor(factor, d)
        if prior_sigma is not None or prior_factor is not None:
            conn = sqlite3.connect(self._db_path)
            r = conn.execute("SELECT mechanism_id FROM surrogate_runs WHERE id=?",
                             (run_ids[0],)).fetchone()
            conn.close()
            mech_id = r[0] if r else None

        Lambda0 = np.diag(1.0 / psig ** 2)

        # information matrices F_k = g_k g_kᵀ / σ_k²
        names: List[str] = []
        F: List[np.ndarray] = []
        x0 = np.zeros(d)
        for rid, sig in zip(run_ids, sigmas):
            tgt = load_surrogate_target(self._db_path, rid, 0.0, float(sig))
            g = np.asarray(tgt.surrogate.grad(x0), dtype=float)
            F.append(np.outer(g, g) / float(sig) ** 2)
            names.append(tgt.name)

        K = len(run_ids)
        n_sel = K if n_select is None else min(n_select, K)

        # greedy selection
        Lambda = Lambda0.copy()
        remaining = set(range(K))
        selected: List[dict] = []
        for _ in range(n_sel):
            best_k, best_gain = None, -np.inf
            for k in remaining:
                gain = _criterion_gain(Lambda, F[k], criterion)
                if gain > best_gain:
                    best_gain, best_k = gain, k
            if best_k is None:
                break
            Lambda = Lambda + F[best_k]
            remaining.discard(best_k)
            selected.append({"rank": len(selected) + 1, "run_id": run_ids[best_k],
                             "name": names[best_k], "gain": float(best_gain),
                             "sigma": float(sigmas[best_k])})

        # first-step marginal gains for all candidates (a full ranking)
        ranked = sorted(
            ({"run_id": run_ids[k], "name": names[k],
              "first_step_gain": _criterion_gain(Lambda0, F[k], criterion)}
             for k in range(K)),
            key=lambda r: -r["first_step_gain"])

        cov_prior = np.linalg.inv(Lambda0)
        cov_post = np.linalg.inv(Lambda)
        post_sigma = np.sqrt(np.diag(cov_post))
        reduction = 1.0 - post_sigma / psig

        result = DoEResult(
            mechanism_id     = mech_id,
            parameter_names  = param_names,
            criterion        = criterion,
            prior_sigma      = psig.tolist(),
            n_candidates     = K,
            n_selected       = len(selected),
            selected         = selected,
            ranked_candidates = ranked,
            posterior_sigma  = post_sigma.tolist(),
            uncertainty_reduction = reduction.tolist(),
            logdet_prior     = float(np.linalg.slogdet(Lambda0)[1]),
            logdet_post      = float(np.linalg.slogdet(Lambda)[1]),
            trace_cov_prior  = float(np.trace(cov_prior)),
            trace_cov_post   = float(np.trace(cov_post)),
            candidate_run_ids = run_ids,
        )
        result.notes.append(
            "Local (linearised-at-prior) A/D-optimal selection over stored gPC "
            "surrogates; fully-Bayesian/nonlinear sequential design is a follow-on.")

        if save_db:
            result.run_id = _insert_doe_run(self._db_path, result)

        log.info(
            "DoE (%s-optimal): selected %d/%d experiments; trace(cov) %.3g→%.3g, "
            "logdet(prec) %.3g→%.3g",
            criterion, len(selected), K, result.trace_cov_prior,
            result.trace_cov_post, result.logdet_prior, result.logdet_post)
        return result


def _insert_doe_run(db_path: str, result: DoEResult) -> Optional[int]:
    import json
    import sqlite3
    from datetime import datetime, timezone
    from rmip.database.migrations import run_doe_migrations, _table_exists

    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "doe_runs"):
            run_doe_migrations(conn)
        cur = conn.execute(
            """
            INSERT INTO doe_runs
                (mechanism_id, criterion, n_candidates, n_selected,
                 logdet_prior, logdet_post, trace_cov_prior, trace_cov_post,
                 parameter_names_json, prior_sigma_json, candidate_run_ids_json,
                 selected_json, ranked_candidates_json, posterior_sigma_json,
                 uncertainty_reduction_json, notes_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result.mechanism_id, result.criterion, result.n_candidates,
                result.n_selected, result.logdet_prior, result.logdet_post,
                result.trace_cov_prior, result.trace_cov_post,
                json.dumps(result.parameter_names), json.dumps(result.prior_sigma),
                json.dumps(result.candidate_run_ids), json.dumps(result.selected),
                json.dumps(result.ranked_candidates), json.dumps(result.posterior_sigma),
                json.dumps(result.uncertainty_reduction), json.dumps(result.notes),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid
