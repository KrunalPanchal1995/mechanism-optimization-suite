"""
rmip/reduction/csp.py
=====================
Phase 8 (Future.md / FEATURE_ROADMAP #10) — Computational Singular Perturbation
(CSP) timescale analysis and QSS-species identification.

Why this exists
---------------
The existing ``reduction/`` package (DRG / DRGEP / DRGPSA / PFA) produces
*skeletal* mechanisms: fewer species, same functional form.  CSP is the
principled *second stage* — it looks at the local chemical Jacobian, separates
the fast (exhausted) modes from the slow (governing) ones by timescale, and
tells you **which species are in quasi-steady state (QSS)** and can therefore be
algebraically eliminated.  This is the physically-motivated way to choose what
to lump (see :mod:`rmip.reduction.lumping`) or eliminate, rather than a bare
graph threshold.

Physics
-------
At a frozen state (T, P, composition) the isothermal chemical Jacobian is

    J_ij = ∂(ω̇_i) / ∂c_j          [1/s]

where ω̇_i is the net molar production rate of species i [kmol/m³/s] and c_j the
molar concentration [kmol/m³].  Its eigenvalues λ_k have units 1/s; the modal
timescales are τ_k = 1/|Re(λ_k)|.  Modes with Re(λ_k) < 0 and τ_k below a
threshold τ* are **exhausted** (fast, equilibrated).  A species whose dynamics
is dominated by an exhausted mode is a QSS species.

Two, cross-checking, identifications are produced:

1. **CSP pointers** — the diagonal of the modal projection, P_{k,i} =
   |A_{i,k}·B_{k,i}| (A = right eigenvectors, B = A⁻¹ = left eigenvectors),
   normalised per mode.  The pointer says which species a mode "points to";
   a species pointed to chiefly by exhausted modes is a QSS candidate.
2. **Lifetime / defect** — a robust, eigenvector-free classic: the chemical
   lifetime τ_i = c_i / D_i (D_i = destruction rate) and the QSS *defect*
   |C_i − D_i| / max(C_i, D_i) (creation vs destruction imbalance).  A genuine
   QSS species is short-lived *and* nearly balanced (small defect).

A species is reported as a QSS candidate when its lifetime is below τ* — the
lifetime test is the primary gate (it needs only rates, never the eigensolve, so
it is always available) and the CSP-pointer test annotates each candidate with
the exhausted mode it belongs to when the eigendecomposition is well-conditioned.

Honest scope
------------
This module *identifies and quantifies* QSS species and exhausted modes and
reports the reduction they would enable.  It does **not** emit a re-derived
analytic reduced mechanism (symbolic QSSA elimination / code generation) — that
is a documented follow-on, exactly as the SDP outer bound is for B2BDC.  What is
returned here is a rigorous, condition-specific diagnostic and a QSS candidate
set ready to feed such a generator.

Usage (Python API)
------------------
>>> from rmip.reduction.csp import CSPAnalyzer
>>> an  = CSPAnalyzer(mechanism_id=1, db_path="project.db")
>>> res = an.analyze(T_K=1200, P_atm=1.0, phi=1.0, fuel="H2:1.0", at="ignition")
>>> [q["species"] for q in res.qss_candidates]
['H2O2', 'HO2', 'O', 'H', 'OH']
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rmip.simulation.base import AbstractSimRunner
from rmip.types import CSPResult, SimCondition

log = logging.getLogger(__name__)

_VALID_AT = ("ignition", "initial", "equilibrium")  # plus "time:<s>"


# ---------------------------------------------------------------------------
# Pure, Cantera-only helpers (no DB) — reusable and unit-testable in isolation
# ---------------------------------------------------------------------------

def chemical_jacobian(gas) -> Tuple[np.ndarray, bool]:
    """
    Isothermal chemical Jacobian J_ij = ∂(ω̇_i)/∂c_j [1/s] at the gas's current
    state, shape (n_species, n_species).

    Prefers Cantera's exact analytic derivative (``net_production_rates_ddCi``,
    Cantera ≥ 2.6); falls back to a central finite-difference over the molar
    concentrations at constant T, P when the analytic form is unavailable.

    Returns
    -------
    (J, used_analytic) : (ndarray, bool)
    """
    if hasattr(gas, "net_production_rates_ddCi"):
        J = gas.net_production_rates_ddCi
        if hasattr(J, "toarray"):          # scipy sparse (Cantera 3.x)
            J = J.toarray()
        return np.asarray(J, dtype=float), True
    return _fd_chemical_jacobian(gas), False


def _fd_chemical_jacobian(gas, rel: float = 1e-6, floor: float = 1e-18) -> np.ndarray:
    """
    Central finite-difference concentration Jacobian at constant T, P.

    The concentration of one species is perturbed while temperature is held
    fixed; net production rates are re-read and differenced.  The gas state is
    restored on exit.
    """
    import cantera as ct

    T, P = gas.TP
    c0 = np.asarray(gas.concentrations, dtype=float)
    n = c0.size
    J = np.zeros((n, n))
    try:
        for j in range(n):
            h = rel * abs(c0[j]) + floor
            for sign, store in ((+1, "p"), (-1, "m")):
                c = c0.copy()
                c[j] = max(c0[j] + sign * h, 0.0)
                tot = c.sum()
                if tot <= 0:
                    continue
                gas.TPX = T, P, c / tot          # mole fractions from concentrations
                if store == "p":
                    wp = np.asarray(gas.net_production_rates, dtype=float)
                else:
                    wm = np.asarray(gas.net_production_rates, dtype=float)
            J[:, j] = (wp - wm) / (2.0 * h)
    finally:
        tot0 = c0.sum()
        gas.TPX = T, P, c0 / tot0 if tot0 > 0 else gas.X
    return J


def mode_decomposition(
    J: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Eigendecomposition of the Jacobian, ordered fastest → slowest mode.

    Returns
    -------
    (eigvals, A, B) : eigenvalues (complex, len n), right eigenvectors ``A``
    (columns), and left eigenvectors ``B`` (rows, ``B = A⁻¹`` via pseudo-inverse
    when ``A`` is ill-conditioned).  ``A[:, k]`` / ``B[k, :]`` are the k-th mode.

    Ordering is by descending ``|Re(eigval)|`` (largest decay rate = fastest =
    smallest timescale first).
    """
    eigvals, A = np.linalg.eig(np.asarray(J, dtype=float))
    order = np.argsort(-np.abs(eigvals.real))
    eigvals = eigvals[order]
    A = A[:, order]
    try:
        B = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        B = np.linalg.pinv(A)
    return eigvals, A, B


def timescales_from_eigs(eigvals: np.ndarray, huge: float = 1e30) -> np.ndarray:
    """
    Modal timescales τ_k = 1/|Re(λ_k)| [s]; conserved / near-zero modes
    (|Re λ| ≈ 0) map to ``huge`` rather than ``inf`` so they stay sortable and
    JSON-serialisable.
    """
    re = np.abs(np.asarray(eigvals).real)
    with np.errstate(divide="ignore"):
        tau = np.where(re > 0, 1.0 / np.maximum(re, 1.0 / huge), huge)
    return np.minimum(tau, huge)


def csp_pointers(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    CSP pointer matrix, shape (n_modes, n_species): P[k, i] = |A_{i,k}·B_{k,i}|,
    normalised so each mode row sums to 1.  P[k, i] is the participation of
    species i in mode k — the mode "points to" its largest-pointer species.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    n = A.shape[0]
    P = np.abs(A.T * B)                       # (n_modes, n_species): A[i,k]*B[k,i]
    row = P.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return np.real(P / row)


def count_exhausted_modes(timescales: Sequence[float], tau_threshold: float) -> int:
    """Number of modes strictly faster than ``tau_threshold``."""
    t = np.asarray(timescales, dtype=float)
    return int(np.sum(t < tau_threshold))


def auto_threshold(timescales: Sequence[float], huge: float = 1e29) -> float:
    """
    Pick a fast/slow timescale split automatically as the geometric centre of the
    **largest spectral gap** among the *active* modal timescales.

    Elemental conservation makes a handful of eigenvalues ≈ 0, i.e. modes with
    astronomically large timescales; those are excluded (any mode more than 1e9×
    slower than the fastest is treated as conserved/inactive) so the split lands
    among the real chemical timescales rather than at the conserved-mode gap.
    Falls back to the median when there are fewer than two usable timescales.
    """
    t = np.sort([x for x in np.asarray(timescales, dtype=float)
                 if 0 < x < huge])
    if t.size < 2:
        return float(t[0]) if t.size == 1 else 1.0
    active = t[t <= t[0] * 1e9]
    if active.size < 2:
        active = t
    log_t = np.log(active)
    gaps = np.diff(log_t)
    g = int(np.argmax(gaps))
    return float(np.exp(0.5 * (log_t[g] + log_t[g + 1])))


def species_qss_metrics(gas) -> Dict[str, Dict[str, float]]:
    """
    Per-species lifetime and QSS defect at the gas's current state (rate-only,
    no eigensolve).

    Returns ``{species_name: {conc, creation, destruction, lifetime, defect}}``
    with lifetime τ_i = c_i / D_i [s] and defect |C_i − D_i| / max(C_i, D_i).
    """
    c = np.asarray(gas.concentrations, dtype=float)
    cr = np.asarray(gas.creation_rates, dtype=float)
    de = np.asarray(gas.destruction_rates, dtype=float)
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(gas.species_names):
        Ci, Di, ci = float(cr[i]), float(de[i]), float(c[i])
        tau = ci / Di if Di > 0 else float("inf")
        denom = max(Ci, Di, 1e-300)
        out[name] = {
            "conc":        ci,
            "creation":    Ci,
            "destruction": Di,
            "lifetime":    tau,
            "defect":      abs(Ci - Di) / denom,
        }
    return out


# ---------------------------------------------------------------------------
# DB-backed analyzer (mirrors simulation/rop_runner.ROPRunner staging)
# ---------------------------------------------------------------------------

class CSPAnalyzer(AbstractSimRunner):
    """
    Native, in-process Cantera CSP timescale / QSS-species diagnostic for one
    mechanism.  Reuses :class:`~rmip.simulation.base.MechanismLoader` (lazy
    ``gas`` property) and the ROP-runner ``_fresh_gas`` / ``_stage`` pattern to
    reach the evaluation state, then eigendecomposes the chemical Jacobian.

    Parameters mirror :class:`~rmip.simulation.rop_runner.ROPRunner`.
    """

    def __init__(
        self,
        mechanism_id: int,
        db_path:      str,
        t_end_s:      float = 10.0,
        n_steps:      int   = 2000,
        rtol:         float = 1e-9,
        atol:         float = 1e-15,
        cache_dir:    Optional[str] = None,
    ) -> None:
        super().__init__(mechanism_id, db_path, cache_dir)
        self._t_end_s = t_end_s
        self._n_steps = n_steps
        self._rtol = rtol
        self._atol = atol

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        T_K:           float,
        P_atm:         float,
        phi:           float,
        fuel:          str,
        oxidizer:      str = "O2:1.0, N2:3.76",
        at:            str = "ignition",
        tau_threshold: Optional[float] = None,
        max_qss:       Optional[int] = None,
        exclude:       Optional[Sequence[str]] = None,
        save_db:       bool = True,
    ) -> CSPResult:
        """
        Run one CSP analysis and return a :class:`~rmip.types.CSPResult`.

        Parameters
        ----------
        tau_threshold : float, optional
            Fast/slow split [s].  Default: auto (largest spectral gap).
        max_qss : int, optional
            Cap the reported QSS candidate list to the ``max_qss`` shortest-lived.
        exclude : sequence[str], optional
            Species never reported as QSS (defaults to the fuel, the oxidiser
            components and any inert/bath species inferred from ``oxidizer``).
        """
        cond = SimCondition(T_K=T_K, P_atm=P_atm, phi=phi,
                            fuel=fuel, oxidizer=oxidizer)
        t0 = time.perf_counter()
        excluded = self._default_excludes(fuel, oxidizer, exclude)

        gas = self._fresh_gas()
        self._stage(gas, T_K, P_atm, phi, fuel, oxidizer, at)
        Tstate = float(gas.T)

        # Jacobian → modes → timescales
        J, analytic = chemical_jacobian(gas)
        eigvals, A, B = mode_decomposition(J)
        taus = timescales_from_eigs(eigvals)
        P = csp_pointers(A, B)

        if tau_threshold is None:
            tau_threshold = auto_threshold(taus)
        n_exhausted = count_exhausted_modes(taus, tau_threshold)

        # For each species, the exhausted mode that most strongly points to it.
        names = list(gas.species_names)
        exhausted_mode_idx = [k for k in range(len(taus)) if taus[k] < tau_threshold]
        pointer_mode: Dict[str, Optional[int]] = {}
        for i, name in enumerate(names):
            if exhausted_mode_idx:
                k = max(exhausted_mode_idx, key=lambda k: P[k, i])
                pointer_mode[name] = k if P[k, i] > 0 else None
            else:
                pointer_mode[name] = None

        metrics = species_qss_metrics(gas)

        # QSS candidates: lifetime below threshold, not excluded, ranked short→long.
        cands: List[Dict[str, Any]] = []
        for name in names:
            m = metrics[name]
            if name in excluded:
                continue
            if not (0.0 < m["lifetime"] < tau_threshold):
                continue
            k = pointer_mode[name]
            cands.append({
                "species":      name,
                "lifetime":     m["lifetime"],
                "defect":       m["defect"],
                "conc":         m["conc"],
                "pointer_mode": k,
                "pointer_tau":  (float(taus[k]) if k is not None else None),
                "pointer_val":  (float(P[k, names.index(name)]) if k is not None else None),
                "mode_exhausted": (k is not None),
            })
        cands.sort(key=lambda d: d["lifetime"])
        if max_qss is not None:
            cands = cands[:max_qss]

        finite = taus[taus < 1e29]
        result = CSPResult(
            mechanism_id      = self.mechanism_id,
            condition         = cond,
            eval_state        = at,
            state_T_K         = Tstate,
            n_species         = gas.n_species,
            n_reactions       = gas.n_reactions,
            jacobian_analytic = analytic,
            tau_threshold     = float(tau_threshold),
            fastest_tau       = float(finite.min()) if finite.size else None,
            slowest_tau       = float(finite.max()) if finite.size else None,
            n_exhausted_modes = n_exhausted,
            mode_timescales   = [float(x) for x in np.sort(taus)],
            qss_candidates    = cands,
            n_qss             = len(cands),
            excluded_species  = sorted(excluded),
            cpu_time_s        = time.perf_counter() - t0,
        )
        result.notes.append(
            "QSS species are identified and quantified (lifetime + CSP pointer); "
            "analytic elimination / reduced-mechanism code-gen is a documented "
            "follow-on."
        )
        if not analytic:
            result.notes.append(
                "Analytic Jacobian unavailable — used a finite-difference "
                "concentration Jacobian."
            )

        if save_db:
            result.run_id = _insert_csp_run(self._loader._db_path, result)  # type: ignore[attr-defined]

        log.info(
            "CSP mech=%d at %s (T=%.1fK): %d/%d modes exhausted below τ*=%.2e s, "
            "%d QSS candidate(s)",
            self.mechanism_id, at, Tstate, n_exhausted, gas.n_species,
            tau_threshold, len(cands),
        )
        return result

    def run_single(self, *args, **kwargs) -> CSPResult:
        """Alias for :meth:`analyze` satisfying the ``AbstractSimRunner`` API."""
        return self.analyze(*args, **kwargs)

    # ── Staging (mirrors ROPRunner) ──────────────────────────────────────────

    def _fresh_gas(self):
        import cantera as ct
        return ct.Solution(
            thermo=self.gas.thermo_model,
            kinetics=self.gas.kinetics_model,
            transport=None,
            species=self.gas.species(),
            reactions=self.gas.reactions(),
        )

    def _stage(self, gas, T_K, P_atm, phi, fuel, oxidizer, at: str) -> None:
        import cantera as ct

        gas.set_equivalence_ratio(phi, fuel=fuel, oxidizer=oxidizer)
        gas.TP = T_K, P_atm * ct.one_atm

        if at == "initial":
            return
        if at == "equilibrium":
            gas.equilibrate("UV")
            return
        if at == "ignition":
            t_target = self._ignition_time(T_K, P_atm, phi, fuel, oxidizer)
        elif at.startswith("time:"):
            try:
                t_target = float(at.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid time spec '{at}' (want 'time:<seconds>')") from exc
        else:
            raise ValueError(
                f"Unknown evaluation state '{at}'. "
                f"Expected one of {_VALID_AT} or 'time:<seconds>'."
            )

        r = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([r])
        net.rtol = self._rtol
        net.atol = self._atol
        net.advance(t_target)

    def _ignition_time(self, T_K, P_atm, phi, fuel, oxidizer) -> float:
        import cantera as ct

        g = self._fresh_gas()
        g.set_equivalence_ratio(phi, fuel=fuel, oxidizer=oxidizer)
        g.TP = T_K, P_atm * ct.one_atm
        r = ct.IdealGasReactor(g)
        net = ct.ReactorNet([r])
        net.rtol = self._rtol
        net.atol = self._atol

        t_arr = [0.0]
        T_arr = [r.T]
        dt_fine = self._t_end_s * 0.001 / max(self._n_steps // 4, 1)
        dt_coarse = self._t_end_s / self._n_steps
        t_switch = self._t_end_s * 0.001
        t = 0.0
        while t < self._t_end_s:
            dt = dt_fine if t < t_switch else dt_coarse
            t = min(t + dt, self._t_end_s)
            net.advance(t)
            t_arr.append(t)
            T_arr.append(r.T)
            if T_arr[-1] - T_arr[0] > 700:
                break

        times = np.asarray(t_arr)
        temps = np.asarray(T_arr)
        if len(times) >= 3:
            dTdt = np.gradient(temps, times)
            peak = int(np.argmax(dTdt))
            if temps[peak] - temps[0] > 100 and dTdt[peak] > 0:
                return float(times[peak])
            idx = np.where(temps >= T_K + 400.0)[0]
            if len(idx) > 0:
                return float(times[idx[0]])
        return float(times[-1])

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _default_excludes(fuel: str, oxidizer: str,
                          extra: Optional[Sequence[str]]) -> set:
        """Fuel + oxidiser components (+ user extras) never reported as QSS."""
        excl = set()
        for spec in (fuel, oxidizer):
            for tok in spec.replace(",", " ").split():
                excl.add(tok.split(":", 1)[0].strip())
        if extra:
            excl.update(extra)
        return {s for s in excl if s}


# ---------------------------------------------------------------------------
# QSSA export (Phase-8 follow-on)
# ---------------------------------------------------------------------------

def export_qssa(
    db_path:   str,
    result:    CSPResult,
    out_yaml:  Optional[str] = None,
    new_name:  Optional[str] = None,
) -> int:
    """
    Materialise the reduced mechanism implied by a CSP analysis: drop the QSS
    candidate species (and every reaction that involves them) from the base
    mechanism, producing a new ``format='reduced'`` mechanism row, and optionally
    emit it as a Cantera YAML.

    Honest scope (documented follow-on): this is the *skeletal* reduction the QSS
    set implies — the QSS species are **removed**, not analytically eliminated.
    True QSSA replaces each QSS species' ODE with its algebraic steady-state
    expression folded into the surviving rates (symbolic code-gen); that remains
    the follow-on.  The removed set + the CSP diagnostics are recorded in the new
    mechanism's metadata so a code-gen step can consume them.

    Returns the new mechanism id.
    """
    from rmip.reduction.materialize import materialize_subset, mechanism_lists

    if result.mechanism_id is None:
        raise ValueError("CSPResult has no mechanism_id — cannot export.")
    names, formulas = mechanism_lists(db_path, result.mechanism_id)
    qss = set(result.qss_species())
    retained = [n for n in names if n not in qss]
    name = new_name or f"qssa_reduced_{result.mechanism_id}"

    new_id = materialize_subset(
        db_path, result.mechanism_id, species=retained, reactions=formulas,
        new_name=name,
        description=f"QSSA skeletal reduction of mechanism {result.mechanism_id} "
                    f"({len(qss)} QSS species removed)",
        metadata={"qssa_removed_species": sorted(qss),
                  "eval_state": result.eval_state,
                  "tau_threshold": result.tau_threshold},
    )
    if out_yaml:
        from rmip.io.cantera_writer import CanteraYAMLWriter
        CanteraYAMLWriter(db_path).write(new_id, out_yaml)
        log.info("QSSA-reduced mechanism written → %s", out_yaml)
    return new_id


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------

def _insert_csp_run(db_path: str, result: CSPResult) -> Optional[int]:
    """Insert a ``csp_runs`` row (lazily creating the table). Returns row id."""
    import json
    import sqlite3
    from datetime import datetime, timezone
    from rmip.database.migrations import run_reduction_analysis_migrations, _table_exists

    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "csp_runs"):
            run_reduction_analysis_migrations(conn)
        cond = result.condition
        cur = conn.execute(
            """
            INSERT INTO csp_runs
                (mechanism_id, eval_state, state_T_K, T_K, P_atm, phi, fuel, oxidizer,
                 n_species, n_reactions, jacobian_analytic, tau_threshold,
                 fastest_tau, slowest_tau, n_exhausted_modes, n_qss,
                 mode_timescales_json, qss_candidates_json, excluded_species_json,
                 notes_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result.mechanism_id, result.eval_state, result.state_T_K,
                cond.T_K if cond else None, cond.P_atm if cond else None,
                cond.phi if cond else None, cond.fuel if cond else None,
                cond.oxidizer if cond else None,
                result.n_species, result.n_reactions,
                1 if result.jacobian_analytic else 0, result.tau_threshold,
                result.fastest_tau, result.slowest_tau,
                result.n_exhausted_modes, result.n_qss,
                json.dumps(result.mode_timescales),
                json.dumps(result.qss_candidates),
                json.dumps(result.excluded_species),
                json.dumps(result.notes),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid
