"""
rmip/analysis/bayesian_calibration.py
=====================================
Feature #6 — Bayesian / MUM-PCE calibration.

Given a set of experimental targets and a Polynomial-Chaos (gPC) surrogate of
the corresponding QoI for each one (built in Phase 5, ``analysis/surrogate.py``),
infer the posterior distribution of the uncertain reaction A-factors that best
reconciles the model with the data — the MUM-PCE method (Sheen & Wang, *Method
of Uncertainty Minimization using Polynomial Chaos Expansions*).

Why the PCE surrogate matters here: Bayesian calibration needs *many* forward
evaluations (an MCMC chain is millions; even a Gauss-Newton MAP needs a Jacobian
every iteration). Running a stiff IDT simulation each time is infeasible; the PCE
is a cheap, differentiable polynomial standing in for the simulator. Crucially,
the surrogate is carried here in **portable monomial form** (coefficients +
exponents extracted from chaospy in Phase 5), so this module — and the whole
calibration — runs in the main rmip environment with *pure numpy/scipy*: no
chaospy, no venv, no subprocess. (The optional full-posterior emcee sampler does
use the Phase-5 venv, since emcee is installed there; see :meth:`run_mcmc`.)

Model
-----
Parameters ``x`` are reaction log-A multipliers (natural log; the same space the
surrogate was fit in — ``x_i = 0`` is the nominal mechanism). For target ``k``:

    y_k  ~  N( PCE_k(x), σ_k² )            (likelihood: obs y_k, obs-error σ_k)
    x    ~  N( 0, diag(σ_prior²) )         (prior: nominal within the uncertainty)

MAP (= posterior mode) minimises

    Φ(x) = Σ_k ((PCE_k(x) − y_k)/σ_k)²  +  Σ_i (x_i/σ_prior_i)²

solved as a bounded non-linear least-squares (``scipy.optimize.least_squares``,
analytic Jacobian from the surrogates). The Laplace posterior covariance is
``(JᵀJ)⁻¹`` at the optimum — which, because the prior and likelihood are Gaussian,
is the exact posterior precision ``Σ_k gₖgₖᵀ/σ_k² + diag(1/σ_prior²)``. Per
parameter this yields a *reduced* uncertainty ``σ_post ≤ σ_prior``: the "minimized
uncertainty" MUM-PCE is named for.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from rmip.types import CalibrationResult

log = logging.getLogger("rmip.analysis.bayesian_calibration")


# ─────────────────────────────────────────────────────────────────────────────
# Portable monomial PCE (pure numpy — no chaospy)
# ─────────────────────────────────────────────────────────────────────────────

class PCEPoly:
    """
    A multivariate polynomial in monomial form, evaluable without chaospy::

        f(x) = Σ_k coeff[k] · Π_i x_i^exponents[k][i]

    Built from the ``pce_exponents`` / ``pce_coefficients`` a Phase-5
    :class:`~rmip.types.SurrogateResult` carries (or a ``surrogate_runs`` row).
    """

    def __init__(self, exponents: Sequence[Sequence[int]],
                 coefficients: Sequence[float]) -> None:
        self.exp = np.asarray(exponents, dtype=float)      # (K, d)
        self.coef = np.asarray(coefficients, dtype=float)  # (K,)
        if self.exp.ndim != 2 or self.exp.shape[0] != self.coef.shape[0]:
            raise ValueError("exponents (K,d) and coefficients (K,) mismatch")
        self.dim = self.exp.shape[1]

    def eval(self, x: Sequence[float]) -> float:
        x = np.asarray(x, dtype=float)
        # (K,) monomial values = Π_i x_i^exp[k,i]
        monomials = np.prod(x[np.newaxis, :] ** self.exp, axis=1)
        return float(self.coef @ monomials)

    def grad(self, x: Sequence[float]) -> np.ndarray:
        """Gradient ∇f(x), shape (d,). Safe at x_i = 0."""
        x = np.asarray(x, dtype=float)
        base = x[np.newaxis, :] ** self.exp            # (K, d): x_i^e_ki
        g = np.zeros(self.dim)
        for j in range(self.dim):
            e_j = self.exp[:, j]
            # ∂/∂x_j of x_j^e = e · x_j^(e−1); zero where e == 0.
            pow_jm1 = np.where(e_j >= 1.0,
                               x[j] ** np.maximum(e_j - 1.0, 0.0), 0.0)
            dfac = e_j * pow_jm1                        # (K,)
            # product over i ≠ j of x_i^e_ki  (mask column j to 1)
            other = base.copy()
            other[:, j] = 1.0
            other = np.prod(other, axis=1)
            g[j] = np.sum(self.coef * dfac * other)
        return g

    def is_quadratic(self) -> bool:
        """True if every monomial has total degree ≤ 2 (an exact quadratic form)."""
        return bool(np.all(self.exp.sum(axis=1) <= 2 + 1e-9))

    def to_quadratic(self) -> "tuple":
        """
        Fold the polynomial into the quadratic form ``f(x) = xᵀA x + bᵀx + c``,
        returning ``(A, b, c)`` with ``A`` symmetric (d, d), ``b`` (d,), ``c``
        scalar.  Requires ``is_quadratic()`` — this is the form the S-procedure
        SDP outer bound (B2BDC) consumes.
        """
        if not self.is_quadratic():
            raise ValueError(
                "PCEPoly is not quadratic (has a monomial of total degree > 2); "
                "rebuild the surrogate at order 2 for the SDP outer bound")
        d = self.dim
        A = np.zeros((d, d))
        b = np.zeros(d)
        c = 0.0
        for k in range(self.exp.shape[0]):
            e = self.exp[k]
            coeff = float(self.coef[k])
            nz = np.nonzero(e)[0]
            deg = int(round(e.sum()))
            if deg == 0:
                c += coeff
            elif deg == 1:
                b[nz[0]] += coeff
            elif len(nz) == 1:                 # x_i^2
                A[nz[0], nz[0]] += coeff
            else:                              # x_i x_j (i≠j): split symmetrically
                i, j = nz[0], nz[1]
                A[i, j] += coeff / 2.0
                A[j, i] += coeff / 2.0
        return A, b, c

    @classmethod
    def from_surrogate_result(cls, result) -> "PCEPoly":
        if not result.pce_exponents or not result.pce_coefficients:
            raise ValueError(
                "SurrogateResult carries no portable PCE (pce_exponents/"
                "pce_coefficients empty) — rebuild the surrogate with the "
                "current backend")
        return cls(result.pce_exponents, result.pce_coefficients)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration target
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationTarget:
    """One experimental datum + its surrogate forward model.

    ``observed`` and ``sigma`` are in the surrogate's output space — i.e. if the
    Phase-5 surrogate was fit on log10(IDT) (the default), ``observed`` is
    log10 of the measured IDT and ``sigma`` its 1σ error in log10 units.
    """
    name: str
    surrogate: PCEPoly
    observed: float
    sigma: float


# ─────────────────────────────────────────────────────────────────────────────
# MUM-PCE calibrator
# ─────────────────────────────────────────────────────────────────────────────

class MUMPCECalibrator:
    """
    Bayesian MAP + Laplace-covariance calibration over shared uncertain
    parameters, using PCE surrogates as the forward model.

    Parameters
    ----------
    targets
        One :class:`CalibrationTarget` per experimental datum. Every surrogate
        must share the same parameter dimension/ordering.
    parameter_names
        Names of the ``d`` parameters (reaction equations), length = surrogate dim.
    prior_sigma
        Prior 1σ per parameter, in log-multiplier units. Scalar or length-``d``.
        A convenient default from an "uncertainty factor" ``f`` is
        ``ln(f)/2`` (so ±2σ spans the factor-``f`` band) — see
        :meth:`prior_sigma_from_factor`.
    bounds_sigma
        Optimisation box half-width as a multiple of ``prior_sigma`` (default 2).
        Keeps the MAP inside the region where the PCE is trustworthy.
    """

    def __init__(
        self,
        targets: List[CalibrationTarget],
        parameter_names: Sequence[str],
        prior_sigma,
        bounds_sigma: float = 2.0,
    ) -> None:
        if not targets:
            raise ValueError("no calibration targets given")
        self.targets = targets
        self.names = list(parameter_names)
        self.d = len(self.names)
        for t in targets:
            if t.surrogate.dim != self.d:
                raise ValueError(
                    f"target {t.name!r} surrogate dim {t.surrogate.dim} != "
                    f"parameter count {self.d}")
            if t.sigma <= 0:
                raise ValueError(f"target {t.name!r} has non-positive sigma")
        ps = np.atleast_1d(np.asarray(prior_sigma, dtype=float))
        if ps.size == 1:
            ps = np.full(self.d, float(ps))
        if ps.size != self.d or np.any(ps <= 0):
            raise ValueError("prior_sigma must be positive, scalar or length d")
        self.prior_sigma = ps
        self.bounds_sigma = bounds_sigma

    # ------------------------------------------------------------------
    @staticmethod
    def prior_sigma_from_factor(factor_bound: float, d: int) -> np.ndarray:
        """``ln(f)/2`` per parameter — ±2σ spans the factor-``f`` band."""
        return np.full(d, math.log(factor_bound) / 2.0)

    # ------------------------------------------------------------------
    def _residuals(self, x: np.ndarray) -> np.ndarray:
        data = np.array([(t.surrogate.eval(x) - t.observed) / t.sigma
                         for t in self.targets])
        prior = x / self.prior_sigma
        return np.concatenate([data, prior])

    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        rows = [t.surrogate.grad(x) / t.sigma for t in self.targets]
        data_J = np.array(rows) if rows else np.zeros((0, self.d))
        prior_J = np.diag(1.0 / self.prior_sigma)
        return np.vstack([data_J, prior_J])

    # ------------------------------------------------------------------
    def calibrate(self, x0: Optional[Sequence[float]] = None) -> CalibrationResult:
        from scipy.optimize import least_squares

        x_start = np.zeros(self.d) if x0 is None else np.asarray(x0, dtype=float)
        half = self.bounds_sigma * self.prior_sigma
        lb, ub = -half, half
        x_start = np.clip(x_start, lb, ub)

        sol = least_squares(
            self._residuals, x_start, jac=self._jacobian,
            bounds=(lb, ub), method="trf",
        )
        x_map = sol.x

        # Posterior precision = JᵀJ at the optimum (data + prior rows).
        J = self._jacobian(x_map)
        precision = J.T @ J
        try:
            cov = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(precision)
        post_sigma = np.sqrt(np.clip(np.diag(cov), 0.0, None))

        # Fit diagnostics (data residuals only)
        data_res = np.array([(t.surrogate.eval(x_map) - t.observed) / t.sigma
                             for t in self.targets])
        chi2 = float(np.sum(data_res ** 2))
        dof = max(1, len(self.targets) - self.d)
        reduced_chi2 = chi2 / dof

        prior_factor = np.exp(self.bounds_sigma * self.prior_sigma)
        posterior_factor = np.exp(self.bounds_sigma * post_sigma)
        reduction = 1.0 - post_sigma / self.prior_sigma

        return CalibrationResult(
            parameter_names=self.names,
            n_targets=len(self.targets),
            prior_sigma=self.prior_sigma.tolist(),
            posterior_mean=x_map.tolist(),
            posterior_sigma=post_sigma.tolist(),
            posterior_cov=cov.tolist(),
            prior_factor=prior_factor.tolist(),
            posterior_factor=posterior_factor.tolist(),
            uncertainty_reduction=reduction.tolist(),
            chi2=chi2,
            reduced_chi2=reduced_chi2,
            target_names=[t.name for t in self.targets],
            observed=[t.observed for t in self.targets],
            predicted_prior=[t.surrogate.eval(np.zeros(self.d)) for t in self.targets],
            predicted_map=[t.surrogate.eval(x_map) for t in self.targets],
            obs_sigma=[t.sigma for t in self.targets],
            converged=bool(sol.success),
        )

    # ------------------------------------------------------------------
    # Optional full-posterior MCMC (emcee, in the Phase-5 chaospy venv)
    # ------------------------------------------------------------------

    def run_mcmc(
        self,
        work_dir: str,
        n_walkers: int = 0,
        n_steps: int = 4000,
        n_burn: int = 1000,
        seed: Optional[int] = None,
        emcee_python: Optional[str] = None,
    ) -> dict:
        """
        Sample the full posterior with emcee (run in the dedicated venv via
        subprocess, same isolation as the chaospy backend). Returns
        ``{"mean": [...], "sigma": [...], "n_samples": int}``. Use to attach
        ``mcmc_*`` fields to a :class:`CalibrationResult` and to check the
        Gaussian Laplace approximation.
        """
        import subprocess

        py = emcee_python or "/home/krithika/.venvs/chaospy_gpc/bin/python3"
        backend = str(Path(__file__).resolve().parent.parent
                      / "simulation" / "_chaospy_backend.py")
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

        request = {
            "targets": [
                {"exponents": t.surrogate.exp.astype(int).tolist(),
                 "coefficients": t.surrogate.coef.tolist(),
                 "observed": t.observed, "sigma": t.sigma}
                for t in self.targets
            ],
            "prior_sigma": self.prior_sigma.tolist(),
            "bounds_sigma": self.bounds_sigma,
            "n_walkers": n_walkers,
            "n_steps": n_steps,
            "n_burn": n_burn,
            "seed": seed,
        }
        req_path = work / "mcmc_request.json"
        resp_path = work / "mcmc_response.json"
        with open(req_path, "w") as fh:
            json.dump(request, fh)
        proc = subprocess.run(
            [py, backend, "mcmc", str(req_path), str(resp_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"emcee backend failed (rc={proc.returncode}). Is emcee "
                f"installed in {py}?\n{proc.stderr[-4000:]}")
        with open(resp_path) as fh:
            return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers: load a surrogate as a forward model, persist a calibration
# ─────────────────────────────────────────────────────────────────────────────

def load_surrogate_target(
    db_path: str, surrogate_run_id: int, observed: float, sigma: float,
    name: Optional[str] = None,
) -> "CalibrationTarget":
    """Build a :class:`CalibrationTarget` from a stored ``surrogate_runs`` row."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM surrogate_runs WHERE id=?", (surrogate_run_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"no surrogate_runs row id={surrogate_run_id}")
    exps = json.loads(row["pce_exponents_json"] or "[]")
    coefs = json.loads(row["pce_coefficients_json"] or "[]")
    if not exps or not coefs:
        raise ValueError(
            f"surrogate_runs id={surrogate_run_id} has no stored PCE "
            f"(built before Phase 6) — rebuild it")
    label = name or f"surrogate_{surrogate_run_id}"
    return CalibrationTarget(label, PCEPoly(exps, coefs), observed, sigma)


def surrogate_parameter_names(db_path: str, surrogate_run_id: int) -> List[str]:
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT parameter_names_json, factor_bound FROM surrogate_runs WHERE id=?",
            (surrogate_run_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"no surrogate_runs row id={surrogate_run_id}")
    return json.loads(row["parameter_names_json"])


def insert_calibration_run(
    db_path: str, result: CalibrationResult, mechanism_id: Optional[int],
    surrogate_run_ids: Sequence[int],
) -> Optional[int]:
    import sqlite3
    from datetime import datetime, timezone
    from rmip.database.migrations import run_calibration_migrations, _table_exists

    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "calibration_runs"):
            run_calibration_migrations(conn)
        cur = conn.execute(
            """
            INSERT INTO calibration_runs
                (mechanism_id, method, n_targets, n_parameters, chi2,
                 reduced_chi2, converged, surrogate_run_ids_json,
                 parameter_names_json, prior_sigma_json, posterior_mean_json,
                 posterior_sigma_json, posterior_cov_json,
                 uncertainty_reduction_json, mcmc_mean_json, mcmc_sigma_json,
                 created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                mechanism_id, "mumpce", result.n_targets,
                len(result.parameter_names), result.chi2, result.reduced_chi2,
                1 if result.converged else 0, json.dumps(list(surrogate_run_ids)),
                json.dumps(result.parameter_names), json.dumps(result.prior_sigma),
                json.dumps(result.posterior_mean), json.dumps(result.posterior_sigma),
                json.dumps(result.posterior_cov),
                json.dumps(result.uncertainty_reduction),
                json.dumps(result.mcmc_mean), json.dumps(result.mcmc_sigma),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
