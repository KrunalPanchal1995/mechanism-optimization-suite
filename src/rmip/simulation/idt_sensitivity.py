"""
rmip/simulation/idt_sensitivity.py
==================================
W6 — Ignition-delay sensitivity as a first-class, documented capability.

Closes community problem **P20** ("there is no built-in ignition-delay
sensitivity in Cantera", a recurring ask 2018→2026) and helps **P36**. See
``Cantera google groups/W6_idt_sensitivity.md``.

Users have re-derived IDT sensitivity from scratch for ~8 years, re-hitting the
same gotchas this module settles once:

* **A clear, selectable IDT definition** — max ``dT/dt`` (default, standard for
  shock-tube IDT), a fixed temperature-rise threshold, or the peak of a chosen
  species (e.g. OH).
* **Correct normalisation** — the reported quantity is the *logarithmic*
  sensitivity ``S_i = d ln(τ) / d ln(k_i)`` (dimensionless), because ignition
  delays span orders of magnitude; a ``raw`` mode (``dτ / d ln k_i``, seconds) is
  also available.
* **Two methods that agree** —
  ``brute``  : perturb each reaction's rate *multiplier* by a factor and
               re-integrate (central difference by default, forward optional).
               This is the gold-standard, definition-agnostic answer.
  ``cvodes`` : Cantera's built-in reactor sensitivity converted to a *τ*
               sensitivity via the implicit-function relation at the ignition
               threshold ``S_τ = −(T_ign·S_T)/(τ·(dT/dt)|_τ)`` — fast, and now a
               true ``d ln τ / d ln k`` rather than a temperature proxy.

The recurring ``set_multiplier`` gotcha (it scales a *runtime* multiplier vector,
never the stored pre-exponential factor) is exactly why the brute-force path is
safe to run reaction-by-reaction on a fresh gas.
"""
from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("rmip.simulation.idt_sensitivity")

_IDT_DEFS = ("dTdt", "threshold", "species")
_METHODS = ("brute", "cvodes")
_NORMS = ("log", "raw")
_DIFFS = ("central", "forward")


# ===========================================================================
# Result
# ===========================================================================
@dataclass
class IdtSensitivityResult:
    conditions: Dict[str, float]
    idt_s: float
    method: str
    idt_definition: str
    normalization: str
    reactor: str
    coefficients: Dict[str, float]           # reaction equation -> S_i
    reaction_index: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def ranked(self) -> List[Tuple[str, float]]:
        return sorted(self.coefficients.items(),
                      key=lambda kv: abs(kv[1]), reverse=True)

    def top(self, n: int) -> List[Tuple[str, float]]:
        return self.ranked()[:n]

    def top_indices(self, n: int) -> List[int]:
        return [self.reaction_index[eq] for eq, _ in self.top(n)
                if eq in self.reaction_index]

    def to_dict(self) -> dict:
        return {
            "observable": "IDT",
            "method": self.method,
            "idt_definition": self.idt_definition,
            "normalization": self.normalization,
            "reactor": self.reactor,
            "idt_s": self.idt_s,
            "conditions": self.conditions,
            "coefficients": dict(self.ranked()),
            "meta": self.meta,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        log.info("IDT sensitivity written to %s", p)

    def render_text(self, n: int = 20) -> str:
        lines = [
            f"IDT sensitivity — {self.method} / {self.idt_definition} / "
            f"{self.normalization}-norm / {self.reactor}",
            f"  conditions: T={self.conditions.get('T_K')} K, "
            f"P={self.conditions.get('P_atm')} atm, phi={self.conditions.get('phi')}, "
            f"fuel={self.conditions.get('fuel')}",
            f"  base IDT = {self.idt_s:.4e} s   "
            f"({len(self.coefficients)} reactions)",
            f"  top {n} reactions by |S|:",
        ]
        for eq, s in self.top(n):
            bar = "#" * int(min(40, abs(s) / (abs(self.top(1)[0][1]) or 1) * 40))
            lines.append(f"    {s:+.3e}  {bar:<40}  {eq}")
        return "\n".join(lines)


# ===========================================================================
# Analyzer
# ===========================================================================
class IdtSensitivityAnalyzer:
    """First-class IDT-sensitivity analyzer for a single (T,P,phi) condition."""

    def __init__(
        self,
        mechanism_path: str,
        fuel: str,
        T_K: float,
        P_atm: float,
        phi: float,
        oxidizer: str = "O2:1.0, N2:3.76",
        reactor: str = "constV",
        idt_definition: str = "dTdt",
        threshold_dT: float = 400.0,
        species: Optional[str] = None,
        rtol: float = 1e-6,
        atol: float = 1e-15,
        max_time: float = 10.0,
        max_steps: int = 100_000,
    ) -> None:
        if idt_definition not in _IDT_DEFS:
            raise ValueError(f"idt_definition must be one of {_IDT_DEFS}")
        if reactor not in ("constV", "constP"):
            raise ValueError("reactor must be 'constV' or 'constP'")
        self.mechanism_path = mechanism_path
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.T_K = T_K
        self.P_atm = P_atm
        self.phi = phi
        self.reactor = reactor
        self.idt_definition = idt_definition
        self.threshold_dT = threshold_dT
        self.species = species
        self.rtol = rtol
        self.atol = atol
        self.max_time = max_time
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    def _make_gas(self):
        import cantera as ct
        gas = ct.Solution(self.mechanism_path)
        gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer)
        gas.TP = self.T_K, self.P_atm * ct.one_atm
        return gas

    def _new_net(self, gas):
        import cantera as ct
        r = (ct.IdealGasReactor(gas) if self.reactor == "constV"
             else ct.IdealGasConstPressureReactor(gas))
        sim = ct.ReactorNet([r])
        sim.rtol = self.rtol
        sim.atol = self.atol
        return r, sim

    # ------------------------------------------------------------------
    # IDT for a given multiplier map (None → base rates)
    # ------------------------------------------------------------------
    def compute_idt(self, multipliers: Optional[Dict[int, float]] = None
                    ) -> Optional[float]:
        import numpy as np
        gas = self._make_gas()
        if multipliers:
            for i, f in multipliers.items():
                gas.set_multiplier(f, i)
        r, sim = self._new_net(gas)
        T0 = gas.T
        want_sp = self.species if self.idt_definition == "species" else None
        ts: List[float] = [0.0]
        Ts: List[float] = [T0]
        xs: List[float] = [float(r.thermo[want_sp].X[0])] if want_sp else []
        n = 0
        while sim.time < self.max_time and n < self.max_steps:
            try:
                sim.step()
            except Exception:                  # integrator gave up — treat as no-ignition
                break
            ts.append(sim.time)
            Ts.append(r.T)
            if want_sp:
                xs.append(float(r.thermo[want_sp].X[0]))
            n += 1
            if r.T - T0 > 700.0:               # clearly ignited — enough profile
                break
        return _detect_idt(np.asarray(ts), np.asarray(Ts),
                           np.asarray(xs) if want_sp else None,
                           self.idt_definition, T0, self.threshold_dT)

    def base_idt(self) -> Optional[float]:
        return self.compute_idt(None)

    # ------------------------------------------------------------------
    # Brute-force sensitivity
    # ------------------------------------------------------------------
    def run_brute(self, delta: float = 0.05, diff: str = "central",
                  norm: str = "log", n_top: Optional[int] = None,
                  n_workers: int = 1) -> IdtSensitivityResult:
        if diff not in _DIFFS:
            raise ValueError(f"diff must be one of {_DIFFS}")
        if norm not in _NORMS:
            raise ValueError(f"norm must be one of {_NORMS}")
        import cantera as ct
        gas = self._make_gas()
        equations = [rxn.equation for rxn in gas.reactions()]
        n_rxn = gas.n_reactions

        tau0 = self.base_idt()
        if tau0 is None or tau0 <= 0:
            raise RuntimeError(
                f"No ignition at base conditions (T={self.T_K} K, P={self.P_atm} "
                f"atm, phi={self.phi}) within {self.max_time}s — cannot compute "
                "IDT sensitivity. Check fuel/mechanism/reactor/time (see §2.2).")

        coeffs: Dict[str, float] = {}
        if n_workers and n_workers > 1:
            coeffs = self._brute_parallel(equations, n_rxn, tau0, delta, diff,
                                          norm, n_workers)
        else:
            for i in range(n_rxn):
                s = self._brute_one(i, tau0, delta, diff, norm)
                if s is not None:
                    coeffs[equations[i]] = s

        rxn_index = {equations[i]: i for i in range(n_rxn)}
        result = IdtSensitivityResult(
            conditions={"T_K": self.T_K, "P_atm": self.P_atm, "phi": self.phi,
                        "fuel": self.fuel},
            idt_s=tau0, method="brute", idt_definition=self.idt_definition,
            normalization=norm, reactor=self.reactor,
            coefficients=coeffs, reaction_index=rxn_index,
            meta={"delta": delta, "diff": diff, "n_reactions": n_rxn})
        if n_top is not None:
            result.coefficients = dict(result.top(n_top))
        return result

    def _brute_one(self, i: int, tau0: float, delta: float, diff: str,
                   norm: str) -> Optional[float]:
        tau_plus = self.compute_idt({i: 1.0 + delta})
        if diff == "central":
            tau_minus = self.compute_idt({i: 1.0 - delta})
            if not tau_plus or not tau_minus:
                return None
            dlnk = math.log(1.0 + delta) - math.log(1.0 - delta)
            if norm == "log":
                return (math.log(tau_plus) - math.log(tau_minus)) / dlnk
            return (tau_plus - tau_minus) / dlnk
        # forward
        if not tau_plus:
            return None
        dlnk = math.log(1.0 + delta)
        if norm == "log":
            return (math.log(tau_plus) - math.log(tau0)) / dlnk
        return (tau_plus - tau0) / dlnk

    def _brute_parallel(self, equations, n_rxn, tau0, delta, diff, norm,
                        n_workers) -> Dict[str, float]:
        args = [(_analyzer_kwargs(self), i, tau0, delta, diff, norm)
                for i in range(n_rxn)]
        out: Dict[str, float] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_brute_worker, a): a[1] for a in args}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    s = fut.result()
                except Exception as exc:       # noqa: BLE001
                    log.warning("reaction %d sensitivity failed: %s", i, exc)
                    s = None
                if s is not None:
                    out[equations[i]] = s
        return out

    # ------------------------------------------------------------------
    # CVODES sensitivity (Cantera reactor SA → τ sensitivity)
    # ------------------------------------------------------------------
    def run_cvodes(self, norm: str = "log", n_top: Optional[int] = None
                   ) -> IdtSensitivityResult:
        if norm not in _NORMS:
            raise ValueError(f"norm must be one of {_NORMS}")
        import cantera as ct
        import numpy as np

        gas = self._make_gas()
        equations = [rxn.equation for rxn in gas.reactions()]
        r, sim = self._new_net(gas)
        for i in range(gas.n_reactions):
            r.add_sensitivity_reaction(i)
        sim.rtol_sensitivity = max(self.rtol, 1e-6)
        sim.atol_sensitivity = max(self.atol, 1e-6)

        T0 = gas.T
        T_ign = T0 + self.threshold_dT
        t_prev, T_prev = 0.0, T0
        tau = dTdt = None
        n = 0
        while sim.time < self.max_time and n < self.max_steps:
            try:
                sim.step()
            except Exception:
                break
            n += 1
            if r.T >= T_ign:
                tau = sim.time
                dt = sim.time - t_prev
                dTdt = (r.T - T_prev) / dt if dt > 0 else float("inf")
                break
            t_prev, T_prev = sim.time, r.T

        if tau is None or dTdt is None or dTdt <= 0:
            raise RuntimeError(
                "cvodes IDT sensitivity: ignition threshold "
                f"(T0+{self.threshold_dT:.0f}K) not reached — use method='brute' "
                "or adjust threshold/time.")

        coeffs: Dict[str, float] = {}
        for i, rxn in enumerate(gas.reactions()):
            s_T = sim.sensitivity("temperature", i)     # d ln T / d ln k_i
            if not math.isfinite(s_T):
                continue
            # implicit-function relation at the threshold crossing:
            #   d ln τ / d ln k_i = − (T_ign · s_T) / (τ · dT/dt|_τ)
            s_log = -(r.T * s_T) / (tau * dTdt)
            coeffs[rxn.equation] = s_log if norm == "log" else tau * s_log

        rxn_index = {equations[i]: i for i in range(gas.n_reactions)}
        result = IdtSensitivityResult(
            conditions={"T_K": self.T_K, "P_atm": self.P_atm, "phi": self.phi,
                        "fuel": self.fuel},
            idt_s=tau, method="cvodes", idt_definition="threshold",
            normalization=norm, reactor=self.reactor,
            coefficients=coeffs, reaction_index=rxn_index,
            meta={"threshold_dT": self.threshold_dT, "dTdt_at_ign": dTdt,
                  "note": "cvodes uses the temperature-threshold IDT definition"})
        if n_top is not None:
            result.coefficients = dict(result.top(n_top))
        return result

    # ------------------------------------------------------------------
    def run(self, method: str = "brute", **kw) -> IdtSensitivityResult:
        if method == "brute":
            return self.run_brute(**kw)
        if method == "cvodes":
            return self.run_cvodes(**{k: v for k, v in kw.items()
                                      if k in ("norm", "n_top")})
        raise ValueError(f"method must be one of {_METHODS}")


# ===========================================================================
# IDT detection
# ===========================================================================
def _detect_idt(t, T, X, definition: str, T0: float, threshold_dT: float
                ) -> Optional[float]:
    import numpy as np
    if len(t) < 3:
        return None
    if definition == "species" and X is not None and len(X) == len(t):
        idx = int(np.argmax(X))
        return float(t[idx]) if idx > 0 else None
    if definition == "threshold":
        idx = np.where(T >= T0 + threshold_dT)[0]
        return float(t[idx[0]]) if len(idx) else None
    # default: max dT/dt
    dTdt = np.gradient(T, t)
    peak = int(np.argmax(dTdt))
    if T[peak] - T[0] > 100.0 and dTdt[peak] > 0 and t[peak] > 0:
        return float(t[peak])
    idx = np.where(T >= T0 + threshold_dT)[0]      # fallback
    return float(t[idx[0]]) if len(idx) else None


# ===========================================================================
# picklable brute-force worker (Cantera thread-safety → separate process)
# ===========================================================================
def _analyzer_kwargs(a: "IdtSensitivityAnalyzer") -> dict:
    return dict(mechanism_path=a.mechanism_path, fuel=a.fuel, T_K=a.T_K,
                P_atm=a.P_atm, phi=a.phi, oxidizer=a.oxidizer,
                reactor=a.reactor, idt_definition=a.idt_definition,
                threshold_dT=a.threshold_dT, species=a.species,
                rtol=a.rtol, atol=a.atol, max_time=a.max_time,
                max_steps=a.max_steps)


def _brute_worker(args: tuple) -> Optional[float]:
    kwargs, i, tau0, delta, diff, norm = args
    analyzer = IdtSensitivityAnalyzer(**kwargs)
    return analyzer._brute_one(i, tau0, delta, diff, norm)
