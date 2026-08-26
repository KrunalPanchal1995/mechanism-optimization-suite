"""
rmip/analysis/b2bdc.py
======================
Feature #5 — Bound-to-Bound Data Collaboration (B2BDC).

B2BDC (Frenklach, Packard, Seiler, Feeley, Russi) treats a combustion model +
its supporting experiments as one *dataset*: each experiment ``e`` contributes a
surrogate model ``M_e(x)`` of a QOI (here the Phase-5 PCE over uncertain reaction
A-factors ``x``) together with measured bounds ``[L_e, U_e]``. Two questions
follow:

  1. **Consistency** — can the model, within the prior parameter box ``H``,
     reproduce *all* the experiments simultaneously? The scalar consistency
     measure

         C = max_{x ∈ H}  min_e ( 1 − |r_e(x)| ),   r_e(x) = (M_e(x) − m_e)/h_e

     (m_e, h_e the midpoint/half-width of ``[L_e, U_e]``) answers it: C > 0 means
     the feasible set F = { x ∈ H : L_e ≤ M_e(x) ≤ U_e ∀e } is non-empty with
     margin C; C < 0 means the dataset is inconsistent by |C| (some experiment
     and the model cannot both hold) — the flag B2BDC is prized for.

  2. **Posterior bounds** — over that feasible set, how tightly is each parameter
     constrained (``parameter_bounds``, the collaborative uncertainty reduction),
     and what range can a new/target QOI take (``predict`` — the model prediction
     interval consistent with *all* the data at once)?

Scope / rigour
--------------
This module computes **inner bounds** by local optimisation (``scipy.optimize``,
multi-start SLSQP with analytic Jacobians) — pure numpy/scipy, no extra deps,
and valid for PCE surrogates of any order. The guarantees are the honest ones of
inner bounds:

  * a positive consistency measure *with* a returned feasible point is a rigorous
    certificate that the dataset **is** consistent;
  * every reported parameter/prediction interval lies **inside** the true
    feasible-set range (never over-claims tightness);
  * a negative measure is strong evidence of inconsistency but, being from local
    optimisation, is not a proof (a better basin could exist) — reported as such.

The complementary rigorous **outer** bounds (the SDP / S-procedure relaxation of
the quadratic-surrogate QCQP) are a documented follow-on; they require a
quadratic refit and an SDP solver (cvxpy) in a dedicated venv.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from rmip.analysis.bayesian_calibration import PCEPoly
from rmip.types import B2BDCResult

log = logging.getLogger("rmip.analysis.b2bdc")

# Dedicated cvxpy venv for the SDP outer bounds (D7 convention — module default,
# overridable per call / CLI flag). Built alongside the chaospy/salib venvs.
_CVXPY_PYTHON_DEFAULT = "/home/krithika/.venvs/cvxpy_sdp/bin/python3"


# ─────────────────────────────────────────────────────────────────────────────
# Constraint
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class B2BDCConstraint:
    """One experiment: a surrogate QOI model plus its measured bounds.

    ``lower``/``upper`` are in the surrogate's output space (e.g. log10(IDT)).
    """
    name: str
    surrogate: PCEPoly
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.upper <= self.lower:
            raise ValueError(f"constraint {self.name!r}: upper must exceed lower")
        self.mid = 0.5 * (self.upper + self.lower)
        self.half = 0.5 * (self.upper - self.lower)

    def residual(self, x: np.ndarray) -> float:
        """Normalised model−data residual r_e(x) ∈ [−1, 1] on the feasible set."""
        return (self.surrogate.eval(x) - self.mid) / self.half

    def margin(self, x: np.ndarray) -> float:
        """1 − |r_e(x)|: ≥ 0 iff the constraint holds at ``x``."""
        return 1.0 - abs(self.residual(x))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class B2BDCDataset:
    """
    A B2BDC dataset: constraints over a shared prior parameter box ``H``.

    Parameters
    ----------
    constraints
        One :class:`B2BDCConstraint` per experiment; all surrogates must share
        the parameter dimension/ordering.
    parameter_names
        Names of the ``d`` parameters (reaction equations).
    prior_bounds
        ``(lo, hi)`` per parameter — the box ``H`` the PCE is trusted on. From an
        uncertainty factor ``f`` this is ``(−ln f, +ln f)`` in log-multiplier
        units (see :meth:`box_from_factor`).
    """

    def __init__(
        self,
        constraints: List[B2BDCConstraint],
        parameter_names: Sequence[str],
        prior_bounds: Sequence[Tuple[float, float]],
    ) -> None:
        if not constraints:
            raise ValueError("no constraints given")
        self.constraints = constraints
        self.names = list(parameter_names)
        self.d = len(self.names)
        if len(prior_bounds) != self.d:
            raise ValueError("prior_bounds length must match parameter count")
        for c in constraints:
            if c.surrogate.dim != self.d:
                raise ValueError(
                    f"constraint {c.name!r} surrogate dim {c.surrogate.dim} != "
                    f"parameter count {self.d}")
        self.box = [(float(lo), float(hi)) for lo, hi in prior_bounds]

    # ------------------------------------------------------------------
    @staticmethod
    def box_from_factor(factor_bound: float, d: int) -> List[Tuple[float, float]]:
        import math
        ln = math.log(factor_bound)
        return [(-ln, ln)] * d

    # ------------------------------------------------------------------
    # scipy constraint dicts: L_e ≤ M_e(x) ≤ U_e  (feasibility)
    # ------------------------------------------------------------------

    def _feasibility_constraints(self, x_slice=slice(None)):
        cons = []
        for c in self.constraints:
            cons.append({
                "type": "ineq",
                "fun": (lambda z, c=c: c.upper - c.surrogate.eval(z[x_slice])),
                "jac": (lambda z, c=c: self._pad(-c.surrogate.grad(z[x_slice]), z, x_slice)),
            })
            cons.append({
                "type": "ineq",
                "fun": (lambda z, c=c: c.surrogate.eval(z[x_slice]) - c.lower),
                "jac": (lambda z, c=c: self._pad(c.surrogate.grad(z[x_slice]), z, x_slice)),
            })
        return cons

    def _pad(self, gx: np.ndarray, z: np.ndarray, x_slice) -> np.ndarray:
        """Embed a d-vector gradient into the full optimisation vector length."""
        if x_slice == slice(None) or len(z) == self.d:
            return gx
        out = np.zeros(len(z))
        out[:self.d] = gx
        return out

    def _starts(self, n_starts: int, rng: np.random.Generator) -> List[np.ndarray]:
        pts = [np.zeros(self.d)]     # nominal / box centre
        for _ in range(max(0, n_starts - 1)):
            pts.append(np.array([rng.uniform(lo, hi) for lo, hi in self.box]))
        return pts

    # ------------------------------------------------------------------
    # Consistency measure:  max_{x,γ} γ  s.t.  γ ≤ 1 − |r_e(x)| ∀e,  x ∈ H
    # ------------------------------------------------------------------

    def consistency(self, n_starts: int = 12, seed: Optional[int] = None):
        """
        Consistency measure C = max_{x∈H} min_e (1 − |r_e(x)|), via the epigraph
        program ``max γ s.t. γ ≤ (U_e−M_e)/h_e and γ ≤ (M_e−L_e)/h_e ∀e``.

        Solved with ``trust-constr`` (robust on this non-smooth minimax where
        SLSQP stalls at the constraint corner), multi-start. The measure
        returned is the **actual** minimum margin at the best point found —
        ``min_e (1−|r_e(x*)|)`` — which is ≥ the optimiser's γ (a valid, tighter
        inner bound).
        """
        import warnings
        from scipy.optimize import minimize, NonlinearConstraint

        rng = np.random.default_rng(seed)
        d = self.d
        bounds = list(self.box) + [(-10.0, 1.0)]      # x ∈ H, γ ∈ [−10, 1]

        # γ ≤ (U_e − M_e)/h_e  and  γ ≤ (M_e − L_e)/h_e  ⇒  each fun ≥ 0
        def con_vec(z):
            x = z[:d]
            out = []
            for c in self.constraints:
                out.append((c.upper - c.surrogate.eval(x)) / c.half - z[d])
                out.append((c.surrogate.eval(x) - c.lower) / c.half - z[d])
            return np.array(out)

        def con_jac(z):
            x = z[:d]
            rows = []
            for c in self.constraints:
                g = c.surrogate.grad(x) / c.half
                rows.append(np.concatenate([-g, [-1.0]]))
                rows.append(np.concatenate([g, [-1.0]]))
            return np.array(rows)

        nlc = NonlinearConstraint(con_vec, 0.0, np.inf, jac=con_jac)

        def neg_gamma(z):
            return -z[d]

        def neg_gamma_jac(z):
            g = np.zeros(d + 1)
            g[d] = -1.0
            return g

        def actual_margin(x):
            return min(c.margin(x) for c in self.constraints)

        best_val = -np.inf
        best_x = np.zeros(d)
        for x0 in self._starts(n_starts, rng):
            gamma0 = min(1.0, actual_margin(x0))
            z0 = np.concatenate([x0, [max(-9.0, gamma0)]])
            try:
                with warnings.catch_warnings():
                    # trust-constr suggests a zero Hessian for linear surrogates;
                    # benign and irrelevant for the quadratic+ PCE case.
                    warnings.filterwarnings(
                        "ignore", message="delta_grad == 0.0")
                    res = minimize(neg_gamma, z0, jac=neg_gamma_jac,
                                   method="trust-constr", bounds=bounds,
                                   constraints=[nlc],
                                   options={"maxiter": 500, "gtol": 1e-9,
                                            "xtol": 1e-10, "verbose": 0})
                x_sol = res.x[:d]
            except Exception as exc:      # pragma: no cover — optimiser hiccup
                log.debug("trust-constr start failed: %s", exc)
                x_sol = x0
            # report the true achieved margin (≥ optimiser γ), an inner bound
            for cand in (x_sol, x0):
                m = actual_margin(cand)
                if m > best_val:
                    best_val = m
                    best_x = np.asarray(cand, dtype=float)

        margins = [float(c.margin(best_x)) for c in self.constraints]
        return float(best_val), best_x, margins

    # ------------------------------------------------------------------
    # Feasible-set extremum of a scalar objective  g(x)
    # ------------------------------------------------------------------

    def _feasible_extremum(
        self, obj, obj_grad, sense: str, n_starts: int,
        rng: np.random.Generator,
    ) -> Optional[Tuple[float, np.ndarray]]:
        from scipy.optimize import minimize

        sign = 1.0 if sense == "min" else -1.0
        cons = self._feasibility_constraints()
        best_val = np.inf
        best_x = None
        for x0 in self._starts(n_starts, rng):
            res = minimize(
                (lambda x: sign * obj(x)), x0,
                jac=(lambda x: sign * obj_grad(x)),
                method="SLSQP", bounds=self.box, constraints=cons,
                options={"maxiter": 200, "ftol": 1e-9},
            )
            # accept only genuinely feasible optima
            if res.success and all(cd["fun"](res.x) >= -1e-6 for cd in cons):
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
        if best_x is None:
            return None
        return sign * best_val, best_x

    def parameter_bounds(self, n_starts: int = 8, seed: Optional[int] = None):
        """``{name: [min, max]}`` of each parameter over the feasible set F."""
        rng = np.random.default_rng(seed)
        out = {}
        for i, name in enumerate(self.names):
            def obj(x, i=i):
                return x[i]

            def grad(x, i=i):
                g = np.zeros(self.d); g[i] = 1.0; return g

            lo = self._feasible_extremum(obj, grad, "min", n_starts, rng)
            hi = self._feasible_extremum(obj, grad, "max", n_starts, rng)
            if lo is None or hi is None:
                return None                    # infeasible dataset
            out[name] = [float(lo[0]), float(hi[0])]
        return out

    def predict(self, surrogate: PCEPoly, name: str = "prediction",
                n_starts: int = 8, seed: Optional[int] = None) -> Optional[dict]:
        """Model prediction interval ``[min, max]`` of ``surrogate`` over F."""
        if surrogate.dim != self.d:
            raise ValueError("prediction surrogate dim mismatch")
        rng = np.random.default_rng(seed)
        lo = self._feasible_extremum(surrogate.eval, surrogate.grad, "min", n_starts, rng)
        hi = self._feasible_extremum(surrogate.eval, surrogate.grad, "max", n_starts, rng)
        if lo is None or hi is None:
            return None
        return {"name": name, "lower": float(lo[0]), "upper": float(hi[0])}

    # ------------------------------------------------------------------
    # One-shot analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        predictions: Optional[List[Tuple[str, PCEPoly]]] = None,
        n_starts: int = 12,
        seed: Optional[int] = None,
        mechanism_id: Optional[int] = None,
    ) -> B2BDCResult:
        measure, x_star, margins = self.consistency(n_starts=n_starts, seed=seed)
        consistent = measure > 0.0
        notes: List[str] = []

        prior_bounds = {n: [lo, hi] for n, (lo, hi) in zip(self.names, self.box)}
        param_bounds: dict = {}
        param_reduction: dict = {}
        preds: List[dict] = []

        if consistent:
            pb = self.parameter_bounds(n_starts=max(6, n_starts // 2), seed=seed)
            if pb is not None:
                param_bounds = pb
                for n, (lo, hi) in zip(self.names, self.box):
                    prior_w = hi - lo
                    post_w = param_bounds[n][1] - param_bounds[n][0]
                    param_reduction[n] = (
                        float(1.0 - post_w / prior_w) if prior_w > 0 else 0.0)
            for pname, psurr in (predictions or []):
                pr = self.predict(psurr, name=pname,
                                  n_starts=max(6, n_starts // 2), seed=seed)
                if pr is not None:
                    preds.append(pr)
        else:
            notes.append(
                f"dataset inconsistent (C={measure:.3f}); parameter/prediction "
                f"bounds skipped. NOTE: inner-bound local search — a negative "
                f"measure is strong evidence, not proof, of inconsistency.")

        return B2BDCResult(
            n_constraints=len(self.constraints),
            parameter_names=self.names,
            consistency_measure=measure,
            consistent=consistent,
            feasible_point=[float(v) for v in x_star],
            constraint_names=[c.name for c in self.constraints],
            constraint_margins=margins,
            prior_bounds=prior_bounds,
            parameter_bounds=param_bounds,
            parameter_reduction=param_reduction,
            predictions=preds,
            n_starts=n_starts,
            bound_type="inner",
            mechanism_id=mechanism_id,
            notes=notes,
        )

    # ── SDP outer bounds (rigorous, via the cvxpy venv) ──────────────────────

    def analyze_outer(
        self,
        predictions:  Optional[List[Tuple[str, PCEPoly]]] = None,
        mechanism_id: Optional[int] = None,
        cvxpy_python: Optional[str] = None,
        work_dir:     Optional[str] = None,
    ) -> B2BDCResult:
        """
        Rigorous **outer** bounds via the S-procedure SDP relaxation (needs the
        cvxpy venv).  Requires every constraint (and prediction) surrogate to be
        quadratic (order ≤ 2) — rebuild those surrogates at order 2 otherwise.

        Outer bounds *contain* the true feasible ranges and give a rigorous
        consistency certificate: ``consistency_measure < 0`` ⇒ provably
        inconsistent; every parameter/prediction interval over-approximates the
        true one.  Complements the local inner bounds from :meth:`analyze`.
        """
        models = []
        for c in self.constraints:
            if not c.surrogate.is_quadratic():
                raise ValueError(
                    f"constraint '{c.name}' surrogate is not quadratic — the SDP "
                    f"outer bound needs order-2 surrogates (rebuild it)")
            A, b, cc = c.surrogate.to_quadratic()
            models.append({"A": A.tolist(), "b": b.tolist(), "c": cc,
                           "L": c.lower, "U": c.upper})

        preds_req = []
        for pname, psurr in (predictions or []):
            if not psurr.is_quadratic():
                raise ValueError(f"prediction '{pname}' surrogate is not quadratic")
            A, b, cc = psurr.to_quadratic()
            preds_req.append({"A": A.tolist(), "b": b.tolist(), "c": cc, "name": pname})

        lb = [lo for lo, _ in self.box]
        ub = [hi for _, hi in self.box]
        request = {"d": self.d, "box": {"lb": lb, "ub": ub},
                   "models": models, "predictions": preds_req}
        resp = self._cvxpy_call("solve", request, cvxpy_python, work_dir)

        measure = resp.get("consistency_measure")
        consistent = bool(resp.get("consistent", False))
        prior_bounds = {n: [lo, hi] for n, (lo, hi) in zip(self.names, self.box)}
        param_bounds: dict = {}
        param_reduction: dict = {}
        for i, name in enumerate(self.names):
            lohi = resp["parameter_bounds"].get(str(i))
            if lohi and lohi[0] is not None and lohi[1] is not None:
                param_bounds[name] = [float(lohi[0]), float(lohi[1])]
                prior_w = self.box[i][1] - self.box[i][0]
                post_w = param_bounds[name][1] - param_bounds[name][0]
                param_reduction[name] = (float(1.0 - post_w / prior_w)
                                         if prior_w > 0 else 0.0)
        preds = [{"name": p["name"], "lower": p["lower"], "upper": p["upper"]}
                 for p in resp.get("predictions", [])]

        notes = ["Rigorous S-procedure SDP outer bounds (cvxpy venv): intervals "
                 "contain the true feasible ranges; C<0 is a rigorous "
                 "inconsistency certificate."]
        return B2BDCResult(
            n_constraints=len(self.constraints),
            parameter_names=self.names,
            consistency_measure=(float(measure) if measure is not None else float("nan")),
            consistent=consistent,
            feasible_point=[],                 # a relaxation yields no single point
            constraint_names=[c.name for c in self.constraints],
            constraint_margins=[],
            prior_bounds=prior_bounds,
            parameter_bounds=param_bounds,
            parameter_reduction=param_reduction,
            predictions=preds,
            n_starts=0,
            bound_type="outer",
            mechanism_id=mechanism_id,
            notes=notes,
        )

    @staticmethod
    def _cvxpy_call(cmd: str, request: dict, cvxpy_python: Optional[str],
                    work_dir: Optional[str]) -> dict:
        """Subprocess+JSON bridge to the cvxpy venv (mirrors surrogate._backend_call)."""
        import json
        import subprocess
        from pathlib import Path

        py = cvxpy_python or _CVXPY_PYTHON_DEFAULT
        backend = Path(__file__).resolve().parent.parent / "simulation" / "_cvxpy_backend.py"
        wd = Path(work_dir or "b2bdc_sdp")
        wd.mkdir(parents=True, exist_ok=True)
        req_path = wd / f"cvxpy_{cmd}_request.json"
        resp_path = wd / f"cvxpy_{cmd}_response.json"
        with open(req_path, "w") as fh:
            json.dump(request, fh)
        proc = subprocess.run([py, str(backend), cmd, str(req_path), str(resp_path)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"cvxpy backend {cmd!r} failed (rc={proc.returncode}).\n"
                f"Is the venv present at {py}?\n{proc.stderr[-4000:]}")
        with open(resp_path) as fh:
            return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_constraint(
    db_path: str, surrogate_run_id: int, lower: float, upper: float,
    name: Optional[str] = None,
) -> B2BDCConstraint:
    """Build a :class:`B2BDCConstraint` from a stored ``surrogate_runs`` row."""
    from rmip.analysis.bayesian_calibration import load_surrogate_target
    t = load_surrogate_target(db_path, surrogate_run_id,
                              observed=0.0, sigma=1.0, name=name)
    label = name or f"surrogate_{surrogate_run_id}"
    return B2BDCConstraint(label, t.surrogate, lower, upper)


def insert_b2bdc_run(
    db_path: str, result: B2BDCResult, surrogate_run_ids: Sequence[int],
) -> Optional[int]:
    import json
    import sqlite3
    from datetime import datetime, timezone
    from rmip.database.migrations import run_b2bdc_migrations, _table_exists

    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "b2bdc_runs"):
            run_b2bdc_migrations(conn)
        cur = conn.execute(
            """
            INSERT INTO b2bdc_runs
                (mechanism_id, n_constraints, n_parameters, consistency_measure,
                 consistent, bound_type, surrogate_run_ids_json,
                 parameter_names_json, prior_bounds_json, parameter_bounds_json,
                 parameter_reduction_json, predictions_json, notes_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result.mechanism_id, result.n_constraints,
                len(result.parameter_names), result.consistency_measure,
                1 if result.consistent else 0, result.bound_type,
                json.dumps(list(surrogate_run_ids)),
                json.dumps(result.parameter_names), json.dumps(result.prior_bounds),
                json.dumps(result.parameter_bounds),
                json.dumps(result.parameter_reduction),
                json.dumps(result.predictions), json.dumps(result.notes),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
