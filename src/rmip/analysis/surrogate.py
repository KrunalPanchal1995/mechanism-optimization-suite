"""
rmip/analysis/surrogate.py
==========================
Feature #7 — generalized Polynomial-Chaos (gPC) surrogate + active subspace.

Builds a cheap polynomial surrogate of a scalar QoI (ignition delay time, by
default) as a function of a handful of uncertain reaction A-factors, and from
that surrogate reads off:

  * the **analytic Sobol indices** (first-order + total) — a global sensitivity
    ranking obtained directly from the PCE coefficients, no extra Monte-Carlo;
  * the **active subspace** — the eigendecomposition of C = E[∇f ∇fᵀ] in
    standardised input space, revealing the few linear combinations of
    parameters along which the QoI actually varies (dimension reduction for
    downstream optimisation / Bayesian calibration).

Why this is separate from the Zero-RK Sobol GSA (``simulation/zerork_gsa.py``):
that path Monte-Carlo–estimates Sobol indices by brute-force sampling; here we
fit a *surrogate* once (far fewer QoI evaluations for smooth responses) that is
then reusable — cheap to evaluate, differentiable, and the source of both the
Sobol decomposition and the active subspace.

Subprocess boundary (D7 principle, same as ``_salib_backend.py``): chaospy is
never imported into the main rmip environment — it drags in a numpy/numpoly ABI
combination incompatible with the shared, numpy-1.24-pinned anaconda env (see
``simulation/_chaospy_backend.py``'s module docstring for the incident). It runs
in the dedicated venv ``/home/krithika/.venvs/chaospy_gpc`` and is called only
via JSON request/response files.

QoI evaluation is in-process Cantera: for each sample the selected reactions'
A-factors are scaled by ``exp(log-multiplier)`` via ``gas.set_multiplier`` and
one IDT is integrated (reusing ``IDTRunner``'s integrate/detect helpers). The
builder is decoupled from this through the ``qoi_fn`` callable, so it can be
driven by any QoI (or a synthetic function, as the tests do).
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from rmip.types import SimCondition, SurrogateResult

log = logging.getLogger("rmip.analysis.surrogate")

_CHAOSPY_PYTHON_DEFAULT = "/home/krithika/.venvs/chaospy_gpc/bin/python3"
_MAX_FAILED_SAMPLE_FRACTION = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# In-process Cantera IDT QoI with per-reaction A-factor multipliers
# ─────────────────────────────────────────────────────────────────────────────

class CanteraIDTMultiplierQoI:
    """
    Callable ``X → Y`` mapping a matrix of log-A multipliers to log10(IDT).

    Parameters
    ----------
    solution
        A Cantera ``Solution`` (loaded once, reused for every sample).
    reaction_indices
        0-based Cantera reaction indices whose A-factors are perturbed. Their
        equations become the surrogate's parameter names.
    T_K, P_atm, phi, fuel, oxidizer
        Initial condition for every IDT integration.
    t_end_s, n_steps, rtol, atol
        Integrator controls (defaults match ``IDTRunner``).

    Notes
    -----
    ``X`` rows are **log**-multipliers; the actual A-factor scaling applied is
    ``exp(row[j])`` on reaction ``reaction_indices[j]``. A non-igniting sample
    yields ``nan`` (the builder drops it before the fit).
    """

    def __init__(
        self,
        solution,
        reaction_indices: Sequence[int],
        T_K: float,
        P_atm: float,
        phi: float,
        fuel: str,
        oxidizer: str = "O2:1.0, N2:3.76",
        t_end_s: float = 10.0,
        n_steps: int = 2000,
        rtol: float = 1e-9,
        atol: float = 1e-15,
    ) -> None:
        self._sol = solution
        self._idx = list(reaction_indices)
        n_rxn = solution.n_reactions
        bad = [i for i in self._idx if i < 0 or i >= n_rxn]
        if bad:
            raise ValueError(f"reaction indices out of range [0,{n_rxn}): {bad}")
        self._T_K = T_K
        self._P_atm = P_atm
        self._phi = phi
        self._fuel = fuel
        self._oxidizer = oxidizer
        self._t_end_s = t_end_s
        self._n_steps = n_steps
        self._rtol = rtol
        self._atol = atol

    def parameter_names(self) -> List[str]:
        return [self._sol.reaction(i).equation for i in self._idx]

    def _reset_multipliers(self) -> None:
        for i in self._idx:
            self._sol.set_multiplier(1.0, i)

    def _run_idt(self) -> Optional[float]:
        import cantera as ct
        from rmip.simulation.idt_runner import IDTRunner

        gas = self._sol
        gas.set_equivalence_ratio(self._phi, fuel=self._fuel, oxidizer=self._oxidizer)
        gas.TP = self._T_K, self._P_atm * ct.one_atm
        reactor = ct.IdealGasReactor(gas)
        sim = ct.ReactorNet([reactor])
        sim.rtol = self._rtol
        sim.atol = self._atol
        times, temps = IDTRunner._integrate(sim, reactor, self._t_end_s, self._n_steps)
        return IDTRunner._detect_idt(times, temps, self._T_K)

    def __call__(self, X: Sequence[Sequence[float]]) -> List[float]:
        Y: List[float] = []
        for row in X:
            self._reset_multipliers()
            for j, i in enumerate(self._idx):
                self._sol.set_multiplier(math.exp(row[j]), i)
            try:
                idt = self._run_idt()
            except Exception as exc:               # integrator blow-up, etc.
                log.debug("IDT eval failed for a sample: %s", exc)
                idt = None
            Y.append(math.log10(idt) if (idt and idt > 0) else float("nan"))
        self._reset_multipliers()
        return Y


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

class PCESurrogateBuilder:
    """
    Build a gPC surrogate (+ Sobol + active subspace) of a QoI over uncertain
    reaction A-factors, via the chaospy venv backend.

    Parameters
    ----------
    db_path : str, optional
        RMIP database (needed only for :meth:`build_for_mechanism` and DB
        persistence).
    chaospy_python : str, optional
        Path to the dedicated chaospy venv's python3. Defaults to
        ``/home/krithika/.venvs/chaospy_gpc/bin/python3``.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        chaospy_python: Optional[str] = None,
    ) -> None:
        self._db = db_path
        self._python = chaospy_python or _CHAOSPY_PYTHON_DEFAULT
        self._backend = str(
            Path(__file__).resolve().parent.parent
            / "simulation" / "_chaospy_backend.py"
        )

    # ------------------------------------------------------------------
    # Backend subprocess (see _chaospy_backend.py for the JSON contract)
    # ------------------------------------------------------------------

    def _backend_call(self, cmd: str, request: dict, work_dir: Path) -> dict:
        work_dir.mkdir(parents=True, exist_ok=True)
        req_path = work_dir / f"chaospy_{cmd}_request.json"
        resp_path = work_dir / f"chaospy_{cmd}_response.json"
        with open(req_path, "w") as fh:
            json.dump(request, fh)
        proc = subprocess.run(
            [self._python, self._backend, cmd, str(req_path), str(resp_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"chaospy backend {cmd!r} failed (rc={proc.returncode}).\n"
                f"Is the venv present at {self._python}?\n{proc.stderr[-4000:]}"
            )
        with open(resp_path) as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Distributions
    # ------------------------------------------------------------------

    @staticmethod
    def uniform_logA_distributions(n: int, factor_bound: float) -> List[dict]:
        """
        ``n`` i.i.d. uniform log-multiplier marginals on ``[-ln f, +ln f]`` —
        the standard "A-factor uncertain within a factor of ``f``" prior, same
        convention as ``zerork_gsa.py``.
        """
        ln_b = math.log(factor_bound)
        return [{"type": "uniform", "lower": -ln_b, "upper": ln_b}] * n

    # ------------------------------------------------------------------
    # Sampling / fitting primitives
    # ------------------------------------------------------------------

    def sample(
        self,
        distributions: List[dict],
        n_samples: int,
        work_dir: Path,
        rule: str = "latin_hypercube",
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        resp = self._backend_call(
            "sample",
            {"distributions": distributions, "n_samples": n_samples,
             "rule": rule, "seed": seed},
            work_dir,
        )
        return resp["X"]

    def fit(
        self,
        distributions: List[dict],
        X: List[List[float]],
        Y: List[float],
        parameter_names: List[str],
        order: int,
        mechanism_id: int,
        qoi: str,
        work_dir: Path,
        condition: Optional[SimCondition] = None,
        log_transformed: bool = True,
        seed: Optional[int] = None,
        n_failed_samples: int = 0,
    ) -> SurrogateResult:
        """Fit a PCE to a ready (X, Y) dataset and assemble a SurrogateResult."""
        resp = self._backend_call(
            "fit",
            {"distributions": distributions, "X": X, "Y": Y,
             "order": order, "seed": seed},
            work_dir,
        )
        active = resp["active_subspace"]
        return SurrogateResult(
            mechanism_id=mechanism_id,
            qoi=qoi,
            order=order,
            parameter_names=list(parameter_names),
            n_samples=len(X),
            n_failed_samples=n_failed_samples,
            n_terms=resp["n_terms"],
            r2_insample=resp["r2_insample"],
            qoi_mean=resp["mean"],
            qoi_variance=resp["variance"],
            sobol_main=dict(zip(parameter_names, resp["sobol_main"])),
            sobol_total=dict(zip(parameter_names, resp["sobol_total"])),
            as_eigenvalues=active["eigenvalues"],
            as_eigenvectors=active["eigenvectors"],
            as_activity_scores=dict(zip(parameter_names, active["activity_scores"])),
            as_n_active=active["n_active"],
            condition=condition,
            log_transformed=log_transformed,
            pce_exponents=resp.get("pce_exponents", []),
            pce_coefficients=resp.get("pce_coefficients", []),
        )

    # ------------------------------------------------------------------
    # High-level: build from an arbitrary QoI callable
    # ------------------------------------------------------------------

    def build(
        self,
        distributions: List[dict],
        parameter_names: List[str],
        qoi_fn: Callable[[List[List[float]]], List[float]],
        n_samples: int,
        order: int,
        work_dir: str,
        mechanism_id: int = 0,
        qoi: str = "custom",
        condition: Optional[SimCondition] = None,
        log_transformed: bool = True,
        rule: str = "latin_hypercube",
        seed: Optional[int] = None,
    ) -> SurrogateResult:
        """
        Sample → evaluate ``qoi_fn`` → fit. NaN QoI values (e.g. non-ignition)
        are dropped from both X and Y before the fit; the run aborts if more
        than 30 % of samples fail.
        """
        work = Path(work_dir)
        d = len(distributions)
        if len(parameter_names) != d:
            raise ValueError("parameter_names length must match distributions")

        n_terms_guess = math.comb(d + order, order)
        if n_samples < n_terms_guess:
            log.warning(
                "n_samples=%d < PCE term count ~%d (d=%d, order=%d): the "
                "regression is under-determined; increase n_samples or lower order.",
                n_samples, n_terms_guess, d, order,
            )

        X = self.sample(distributions, n_samples, work, rule=rule, seed=seed)
        Y = list(qoi_fn(X))

        X_ok, Y_ok = [], []
        for xrow, yval in zip(X, Y):
            if yval == yval and math.isfinite(yval):     # not NaN/inf
                X_ok.append(xrow)
                Y_ok.append(yval)
        n_failed = len(X) - len(X_ok)
        if n_failed:
            log.warning("%d/%d QoI samples failed (dropped before fit)",
                        n_failed, len(X))
        if not X_ok:
            raise RuntimeError("All QoI samples failed — nothing to fit")
        if n_failed / len(X) > _MAX_FAILED_SAMPLE_FRACTION:
            raise RuntimeError(
                f"{n_failed}/{len(X)} QoI samples failed "
                f"(>{_MAX_FAILED_SAMPLE_FRACTION:.0%}) — aborting rather than "
                f"fitting an unreliable surrogate"
            )

        return self.fit(
            distributions, X_ok, Y_ok, parameter_names, order,
            mechanism_id, qoi, work, condition=condition,
            log_transformed=log_transformed, seed=seed, n_failed_samples=n_failed,
        )

    # ------------------------------------------------------------------
    # High-level: build for a DB mechanism's IDT
    # ------------------------------------------------------------------

    def build_for_mechanism(
        self,
        mechanism_id: int,
        reaction_indices: Sequence[int],
        T_K: float,
        P_atm: float,
        phi: float,
        fuel: str,
        work_dir: str,
        oxidizer: str = "O2:1.0, N2:3.76",
        factor_bound: float = 2.0,
        n_samples: int = 200,
        order: int = 3,
        seed: Optional[int] = None,
        t_end_s: float = 10.0,
        save_db: bool = True,
    ) -> SurrogateResult:
        """
        Build an IDT surrogate over the given Cantera reaction indices' A-factors.
        Loads the mechanism once, perturbs the selected reactions per sample, and
        integrates one IDT per sample in-process.
        """
        if self._db is None:
            raise ValueError("db_path is required for build_for_mechanism")
        from rmip.simulation.base import MechanismLoader

        solution = MechanismLoader(self._db).load(mechanism_id)
        qoi = CanteraIDTMultiplierQoI(
            solution, reaction_indices, T_K, P_atm, phi, fuel,
            oxidizer=oxidizer, t_end_s=t_end_s,
        )
        param_names = qoi.parameter_names()
        distributions = self.uniform_logA_distributions(
            len(reaction_indices), factor_bound)
        condition = SimCondition(T_K=T_K, P_atm=P_atm, phi=phi,
                                 fuel=fuel, oxidizer=oxidizer)

        result = self.build(
            distributions, param_names, qoi, n_samples, order, work_dir,
            mechanism_id=mechanism_id, qoi="idt", condition=condition,
            log_transformed=True, seed=seed,
        )
        if save_db:
            result.run_id = self._insert_run(result, factor_bound, seed)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _insert_run(
        self, result: SurrogateResult, factor_bound: float, seed: Optional[int],
    ) -> Optional[int]:
        import sqlite3
        from rmip.database.migrations import run_surrogate_migrations, _table_exists

        with sqlite3.connect(self._db) as conn:
            if not _table_exists(conn, "surrogate_runs"):
                run_surrogate_migrations(conn)
            cond = result.condition
            active = {
                "eigenvalues": result.as_eigenvalues,
                "eigenvectors": result.as_eigenvectors,
                "activity_scores": result.as_activity_scores,
                "n_active": result.as_n_active,
            }
            cur = conn.execute(
                """
                INSERT INTO surrogate_runs
                    (mechanism_id, qoi, method, poly_order, n_parameters,
                     n_samples, n_failed_samples, n_terms, r2_insample,
                     qoi_mean, qoi_variance, log_transformed, factor_bound,
                     T_K, P_atm, phi, fuel, oxidizer,
                     parameter_names_json, sobol_main_json, sobol_total_json,
                     active_subspace_json, pce_exponents_json,
                     pce_coefficients_json, seed, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    result.mechanism_id, result.qoi, "pce", result.order,
                    len(result.parameter_names), result.n_samples,
                    result.n_failed_samples, result.n_terms, result.r2_insample,
                    result.qoi_mean, result.qoi_variance,
                    1 if result.log_transformed else 0, factor_bound,
                    cond.T_K if cond else None, cond.P_atm if cond else None,
                    cond.phi if cond else None, cond.fuel if cond else None,
                    cond.oxidizer if cond else None,
                    json.dumps(result.parameter_names),
                    json.dumps(result.sobol_main), json.dumps(result.sobol_total),
                    json.dumps(active), json.dumps(result.pce_exponents),
                    json.dumps(result.pce_coefficients), seed,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return cur.lastrowid
