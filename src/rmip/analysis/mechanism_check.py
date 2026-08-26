"""
rmip/analysis/mechanism_check.py
================================
Mechanism health-check engine — the ``rmip check`` diagnostics.

This is the implementation of the mechanism-checking phase scoped in
``MECHANISM_CHECK_RESEARCH.md``. It re-implements, over Cantera, the battery of
diagnostics that the combustion-community tools (LLNL MechChecker, KAUST
CloudFlame, Northeastern ChemCheck) perform, plus the recurring Cantera-Users
failure modes, and emits a single structured report per mechanism.

Design
------
Everything works **directly on files** through Cantera 3.x — no RMIP database is
required (that keeps the checker usable standalone and trivially testable):

* A ``.yaml``/``.yml`` file is loaded straight into a ``cantera.Solution``.
* A CHEMKIN ``.inp``/``.mech``/``.ck`` file is converted with ``ck2yaml`` first;
  any conversion error is surfaced as an ERROR finding (this *is* the check for
  the whole class of ck2yaml load failures — undeclared species, unmarked
  duplicates, thermo-record mismatches — that block a mechanism before anything
  else can run).

Each individual check is a method returning ``list[Finding]``. ``run()``
aggregates them into a :class:`CheckReport`. A ``Finding`` carries a severity,
a category, a human message, a machine-readable ``details`` dict, and — when the
defect is repairable — a list of :class:`FixOption` (``kind="auto"`` for the
obvious single answer, ``kind="choice"`` for the ambiguous cases where the user
must pick). The companion :mod:`rmip.analysis.mechanism_fixer` consumes those
options to emit a corrected YAML.

Checks implemented (mapped to the backlog in MECHANISM_CHECK_RESEARCH.md §4):

  load               — surface ck2yaml / Solution load errors            (P0/P9)
  rate_sweep         — NaN / Inf / negative fwd&rev k over a (T,P) grid   (P1)
  collision_limit    — bimolecular k vs gas-kinetic limit, both dirs,
                       Arrhenius + falloff + PLOG (evaluated at grid P)   (P2)
  thermo_discontinuity — NASA-7 Cp/H/S jump at Tmid, per species         (P3)
  element_balance    — per-reaction element + charge balance             (P4)
  duplicates         — unmarked duplicate reactions                      (P5)
  pdep_sanity        — PLOG negative/duplicate-P; falloff Troe/limits     (P5/P9)
  completeness       — unused species; missing transport; T-range        (P6)
  unit_sanity        — Ea magnitude plausibility (cal/mol vs J/kmol trap) (P14)
  zero_rates         — dead reactions (k≡0) & decoupled species          (P33, W8a)
  name_consistency   — case-variant / undeclared species references
                       across YAML sections (raw scan, works pre-load)  (P35, W8a)
  sim_setup_sanity   — bounded const-V ignition smoke test at the hottest/
                       lowest-P screening point (generic, composition-
                       independent; not a fuel/target IDT check)         (P7)
  csp_timescales     — CSP chemical-Jacobian eigen-timescales; flags
                       unphysically-fast ("ultrafast") modes at a
                       synthetic screening composition                   (P8)
  syntax_lint        — line/column ChemCheck-style raw text scan (missing
                       exponent, stray tab, truncated field, illegal
                       species-name char); works pre-load                (P9)
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("rmip.analysis.mechanism_check")

# ---------------------------------------------------------------------------
# Physical constants / thresholds
# ---------------------------------------------------------------------------
R_CGS_CAL = 1.987_204_259              # cal / mol / K  (for Ea unit heuristics)
J_PER_CAL = 4.184
# Gas-kinetic bimolecular collision limit in *Cantera* units (kmol, m, s).
# ~1e14 cm^3/mol/s at 298 K == 1e8 m^3/mol/s == 1e11 m^3/kmol/s, scaled by
# sqrt(T/298.15) from hard-sphere theory. See MECHANISM_CHECK_RESEARCH.md.
_COLLISION_LIMIT_298_KMOL = 1.0e11


def collision_limit_kmol(T_K: float) -> float:
    """Bimolecular gas-kinetic collision limit [m^3/kmol/s] at temperature T."""
    return _COLLISION_LIMIT_298_KMOL * math.sqrt(T_K / 298.15)


# Default screening grid — spans typical combustion T and P.
DEFAULT_T_GRID: Tuple[float, ...] = (300.0, 500.0, 800.0, 1000.0, 1400.0,
                                     1800.0, 2200.0, 2500.0)
DEFAULT_P_ATM_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0, 50.0)

ALL_CHECKS: Tuple[str, ...] = (
    "load",
    "schema_deprecation",
    "rate_sweep",
    "collision_limit",
    "thermo_discontinuity",
    "element_balance",
    "duplicates",
    "third_body_duplicates",
    "pdep_sanity",
    "completeness",
    "unit_sanity",
    "thermo_provenance",
    "multiphase_sanity",
    "zero_rates",
    "name_consistency",
    "sim_setup_sanity",
    "csp_timescales",
    "syntax_lint",
)

# Below this, a rate constant / production rate is treated as "identically
# zero" (the P33 dead-reaction floor). Real chemistry at *some* point of the
# screening grid (which reaches 2500 K) sits many orders of magnitude above
# this; only a structurally-dead channel (A=0, zeroed fit, decoupled species)
# lands under it.
_ZERO_RATE_FLOOR = 1.0e-90

# Severity ordering for sorting / rollups.
_SEV_RANK = {"error": 0, "warning": 1, "info": 2}


# ===========================================================================
# Result data model
# ===========================================================================
@dataclass
class FixOption:
    """One repair the user (or ``--fix auto``) can apply to a finding.

    ``kind="auto"``   — the single obvious repair (safe to apply unattended).
    ``kind="choice"`` — one of several mutually-exclusive answers to an
                        ambiguous defect; the user must choose.
    """
    id: str
    label: str
    description: str
    kind: str = "choice"                 # "auto" | "choice" | "manual"
    recommended: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Finding:
    check: str                            # which check produced it
    severity: str                         # "error" | "warning" | "info"
    category: str                         # short kebab slug
    message: str                          # human one-liner
    location: str = ""                    # species name / reaction eq / line
    details: Dict[str, Any] = field(default_factory=dict)
    fix_options: List[FixOption] = field(default_factory=list)

    @property
    def fixable(self) -> bool:
        return bool(self.fix_options)

    @property
    def auto_fix(self) -> Optional[FixOption]:
        """The single auto-applicable repair, if this finding has exactly one."""
        autos = [o for o in self.fix_options if o.kind == "auto"]
        return autos[0] if len(autos) == 1 else None

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["fixable"] = self.fixable
        return d


@dataclass
class CheckReport:
    mechanism: str                        # display name / path
    findings: List[Finding] = field(default_factory=list)
    ran_checks: List[str] = field(default_factory=list)
    loaded: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- rollups ----------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        c = {"error": 0, "warning": 0, "info": 0}
        for f in self.findings:
            c[f.severity] = c.get(f.severity, 0) + 1
        return c

    @property
    def n_errors(self) -> int:
        return self.counts()["error"]

    @property
    def n_warnings(self) -> int:
        return self.counts()["warning"]

    @property
    def ok(self) -> bool:
        return self.n_errors == 0

    def sorted_findings(self) -> List[Finding]:
        return sorted(self.findings,
                      key=lambda f: (_SEV_RANK.get(f.severity, 9), f.check))

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "loaded": self.loaded,
            "ran_checks": self.ran_checks,
            "counts": self.counts(),
            "meta": self.meta,
            "findings": [f.to_dict() for f in self.sorted_findings()],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_text(self, color: bool = False) -> str:
        return render_text_report(self, color=color)


# ===========================================================================
# The checker
# ===========================================================================
class MechanismChecker:
    """Run mechanism health diagnostics over a single mechanism file.

    Parameters
    ----------
    path : str
        Path to a ``.yaml``/``.yml`` (loaded directly) or CHEMKIN
        ``.inp``/``.mech``/``.ck`` file (converted with ``ck2yaml`` first).
    thermo, transport : str, optional
        CHEMKIN thermo/transport files, if separate from the mechanism.
    T_grid_K, P_atm_grid : sequence of float, optional
        Screening grids for the rate/collision checks.
    collision_exceedance : float
        k / collision_limit ratio above which a reaction is flagged (default
        10 — one order of magnitude; CloudFlame flags any exceedance, but a
        small margin avoids noise from the heuristic limit).
    thermo_jump_tol : float
        Relative Cp/H/S jump at Tmid above which a species is flagged (0.01).
    target_T_high : float
        Upper simulation temperature used for the thermo T-range coverage
        check (default 2500 K).
    csp_max_species : int
        Species-count guard for ``check_csp_timescales``: above this the
        O(n^3) eigendecomposition is skipped (an ``info`` finding is emitted
        instead of paying the cost). Default 500.
    """

    def __init__(
        self,
        path: str,
        thermo: Optional[str] = None,
        transport: Optional[str] = None,
        T_grid_K: Optional[Sequence[float]] = None,
        P_atm_grid: Optional[Sequence[float]] = None,
        collision_exceedance: float = 10.0,
        thermo_jump_tol: float = 0.01,
        target_T_high: float = 2500.0,
        csp_max_species: int = 500,
    ) -> None:
        self.path = path
        self.thermo = thermo
        self.transport = transport
        self.T_grid = list(T_grid_K) if T_grid_K else list(DEFAULT_T_GRID)
        self.P_atm_grid = list(P_atm_grid) if P_atm_grid else list(DEFAULT_P_ATM_GRID)
        self.collision_exceedance = collision_exceedance
        self.thermo_jump_tol = thermo_jump_tol
        self.target_T_high = target_T_high
        self.csp_max_species = csp_max_species

        self.gas = None                    # cantera.Solution once loaded
        self.yaml_path: Optional[str] = None   # resolved/converted YAML path
        self._raw_yaml: Optional[Dict[str, Any]] = None  # parsed YAML dict
        self._load_error: Optional[str] = None
        self._tmpdir: Optional[str] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> Optional[str]:
        """Load (converting CHEMKIN if needed). Returns an error string or None.

        Populates ``self._raw_yaml`` (parsed YAML dict) whenever a YAML text is
        available — even if ``cantera.Solution`` subsequently rejects it — so the
        file-level checks (duplicates, pdep sanity) can still run and their fixes
        can turn an unloadable mechanism into a loadable one.
        """
        try:
            import cantera as ct
        except ImportError as exc:                       # pragma: no cover
            self._load_error = f"Cantera is not importable: {exc}"
            return self._load_error

        ext = os.path.splitext(self.path)[1].lower()
        yaml_path = self.path
        if ext not in (".yaml", ".yml"):
            yaml_path, conv_err = self._convert_chemkin()
            if conv_err:
                self._load_error = conv_err
                return conv_err

        self.yaml_path = yaml_path
        self._raw_yaml = _try_parse_yaml(yaml_path)

        try:
            import warnings
            with warnings.catch_warnings():
                # We detect + report thermo discontinuities ourselves; silence
                # Cantera's per-species NasaPoly2::validate warning spam on load.
                warnings.simplefilter("ignore")
                self.gas = ct.Solution(yaml_path)
            return None
        except Exception as exc:                          # noqa: BLE001
            self._load_error = _clean_cantera_error(str(exc))
            return self._load_error

    def _convert_chemkin(self) -> Tuple[Optional[str], Optional[str]]:
        """Convert a CHEMKIN mechanism to YAML via ck2yaml. Returns (path, err)."""
        self._tmpdir = tempfile.mkdtemp(prefix="rmip_check_")
        out = os.path.join(self._tmpdir, "converted.yaml")
        cmd = ["ck2yaml", f"--input={self.path}", f"--output={out}",
               "--permissive"]
        if self.thermo:
            cmd.append(f"--thermo={self.thermo}")
        if self.transport:
            cmd.append(f"--transport={self.transport}")
        exe = shutil.which("ck2yaml")
        if exe is None:
            # Fall back to the module form (works in the same interpreter env).
            cmd = ["python", "-m", "cantera.ck2yaml"] + cmd[1:]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except (subprocess.TimeoutExpired, OSError) as exc:
            return None, f"ck2yaml invocation failed: {exc}"
        if r.returncode != 0 or not os.path.exists(out):
            msg = (r.stderr or r.stdout or "").strip()
            return None, "ck2yaml conversion failed:\n" + _clean_cantera_error(msg)
        return out, None

    def cleanup(self) -> None:
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(self, checks: Optional[Sequence[str]] = None) -> CheckReport:
        """Load and run the requested checks (default: all)."""
        checks = list(checks) if checks else list(ALL_CHECKS)
        report = CheckReport(mechanism=self.path, ran_checks=list(checks))

        err = self.load()
        if err is not None:
            report.loaded = False
            report.findings.append(Finding(
                check="load", severity="error", category="load-failure",
                message="Mechanism failed to load in Cantera — see error detail. "
                        "File-level checks below may still apply.",
                location=os.path.basename(self.path),
                details={"error": err},
            ))
            # Even without a live Solution, run the checks that only need the
            # raw YAML text — their fixes can make the mechanism loadable.
            for name in ("schema_deprecation", "duplicates", "third_body_duplicates",
                         "pdep_sanity", "name_consistency"):
                if self._raw_yaml is not None and name in checks:
                    try:
                        report.findings.extend(getattr(self, f"check_{name}")())
                    except Exception:                      # noqa: BLE001
                        log.exception("raw %s scan failed", name)
            # syntax_lint is a plain text scan of self.path — it needs no
            # parsed YAML at all, so it runs even for a CHEMKIN .inp that
            # ck2yaml itself refused to convert (today's only signal there is
            # ck2yaml's raw stderr passthrough via _clean_cantera_error).
            if "syntax_lint" in checks:
                try:
                    report.findings.extend(self.check_syntax_lint())
                except Exception:                          # noqa: BLE001
                    log.exception("raw syntax_lint scan failed")
            return report

        report.loaded = True
        report.meta = {
            "n_species": self.gas.n_species,
            "n_reactions": self.gas.n_reactions,
            "n_elements": self.gas.n_elements,
            "yaml_path": self.yaml_path,
        }

        dispatch: Dict[str, Callable[[], List[Finding]]] = {
            "load": lambda: [],            # load already succeeded
            "schema_deprecation": self.check_schema_deprecation,
            "rate_sweep": self.check_rate_sweep,
            "collision_limit": self.check_collision_limit,
            "thermo_discontinuity": self.check_thermo_discontinuity,
            "element_balance": self.check_element_balance,
            "duplicates": self.check_duplicates,
            "third_body_duplicates": self.check_third_body_duplicates,
            "pdep_sanity": self.check_pdep_sanity,
            "completeness": self.check_completeness,
            "unit_sanity": self.check_unit_sanity,
            "thermo_provenance": self.check_thermo_provenance,
            "multiphase_sanity": self.check_multiphase_sanity,
            "zero_rates": self.check_zero_rates,
            "name_consistency": self.check_name_consistency,
            "sim_setup_sanity": self.check_sim_setup_sanity,
            "csp_timescales": self.check_csp_timescales,
            "syntax_lint": self.check_syntax_lint,
        }
        for name in checks:
            fn = dispatch.get(name)
            if fn is None:
                log.warning("Unknown check '%s' — skipped.", name)
                continue
            try:
                report.findings.extend(fn())
            except Exception as exc:                       # noqa: BLE001
                log.exception("check '%s' raised", name)
                report.findings.append(Finding(
                    check=name, severity="warning", category="check-error",
                    message=f"Check '{name}' could not complete: {exc}",
                ))
        return report

    # ==================================================================
    # W1 — deprecated / renamed CTI→YAML keyword detection (runs pre-load)
    # ==================================================================
    def check_schema_deprecation(self) -> List[Finding]:
        if self._raw_yaml is None:
            return []
        from rmip.analysis.yaml_validator import YamlValidator
        return YamlValidator(data=self._raw_yaml).validate()

    # ==================================================================
    # P9 — ChemCheck-style line/column syntax linter (raw text scan; works
    # pre-load, on either a CHEMKIN .inp or a Cantera YAML file)
    # ==================================================================
    def check_syntax_lint(self) -> List[Finding]:
        from rmip.analysis.syntax_lint import SyntaxLinter
        issues = SyntaxLinter(self.path).lint()
        return [Finding(check="syntax_lint", severity=i.severity, category="syntax",
                         message=i.message,
                         location=f"line {i.line}" + (f", col {i.column}" if i.column else ""),
                         details={"line": i.line, "column": i.column})
                for i in issues]

    # ==================================================================
    # W4 — thermo provenance cross-check vs reference data (P10/P27)
    # ==================================================================
    def check_thermo_provenance(self) -> List[Finding]:
        if self.gas is None:
            return []
        from rmip.analysis.thermo_provenance import ThermoProvenanceChecker
        return ThermoProvenanceChecker().findings_for_gas(self.gas)

    def check_multiphase_sanity(self) -> List[Finding]:
        """W5c/P25 — condensed-phase composition sanity (warning-only; a
        mechanism with no condensed phases yields no findings)."""
        if self.yaml_path is None:
            return []
        from rmip.analysis.multiphase_sanity import MultiphaseSanityChecker
        return MultiphaseSanityChecker(self.yaml_path).findings()

    # ==================================================================
    # P1 — rate NaN / negative sweep over (T,P)
    # ==================================================================
    def check_rate_sweep(self) -> List[Finding]:
        import numpy as np
        gas = self.gas
        X = self._screen_composition()
        # Track, per reaction, the worst pathology seen across the grid.
        bad_fwd: Dict[int, Dict[str, Any]] = {}
        bad_rev: Dict[int, Dict[str, Any]] = {}
        for T in self.T_grid:
            for P_atm in self.P_atm_grid:
                try:
                    gas.TPX = T, P_atm * 101325.0, X
                    kf = np.asarray(gas.forward_rate_constants, dtype=float)
                    kr = np.asarray(gas.reverse_rate_constants, dtype=float)
                except Exception as exc:                   # noqa: BLE001
                    # A state that itself throws is a serious integrator hazard.
                    return [Finding(
                        check="rate_sweep", severity="error",
                        category="rate-eval-failure",
                        message=("Rate evaluation raised at "
                                 f"T={T:.0f} K, P={P_atm:g} atm: {exc}"),
                        details={"T_K": T, "P_atm": P_atm, "error": str(exc)},
                    )]
                self._collect_bad_rates(kf, T, P_atm, bad_fwd)
                self._collect_bad_rates(kr, T, P_atm, bad_rev)

        findings: List[Finding] = []
        for ridx, info in sorted(bad_fwd.items()):
            findings.append(self._rate_finding(ridx, info, "forward"))
        for ridx, info in sorted(bad_rev.items()):
            findings.append(self._rate_finding(ridx, info, "reverse"))
        return findings

    @staticmethod
    def _collect_bad_rates(k, T, P_atm, store) -> None:
        import numpy as np
        nan = np.isnan(k)
        inf = np.isinf(k)
        neg = k < 0
        bad = nan | inf | neg
        for ridx in np.nonzero(bad)[0]:
            ridx = int(ridx)
            kind = "nan" if nan[ridx] else "inf" if inf[ridx] else "negative"
            prev = store.get(ridx)
            # error kinds (nan/inf) dominate negative; first occurrence wins T,P.
            rank = {"nan": 0, "inf": 0, "negative": 1}
            if prev is None or rank[kind] < rank[prev["kind"]]:
                store[ridx] = {"kind": kind, "T_K": float(T),
                               "P_atm": float(P_atm), "value": float(k[ridx])}

    def _rate_finding(self, ridx: int, info: Dict[str, Any], direction: str) -> Finding:
        rxn = self.gas.reaction(ridx)
        eq = rxn.equation
        kind = info["kind"]
        sev = "error" if kind in ("nan", "inf") else "warning"
        fixes: List[FixOption] = []
        # PLOG-derived NaN is the classic negative-A-interpolation failure.
        if _rate_type(rxn) == "pressure-dependent-Arrhenius":
            fixes = _plog_fix_options(eq, ridx)
        return Finding(
            check="rate_sweep", severity=sev, category=f"rate-{kind}",
            message=(f"{direction.capitalize()} rate constant is {kind} at "
                     f"T={info['T_K']:.0f} K, P={info['P_atm']:g} atm "
                     f"(value={info['value']:.3g})."),
            location=eq,
            details={"reaction_index": ridx, "direction": direction,
                     "rate_type": _rate_type(rxn), **info},
            fix_options=fixes,
        )

    # ==================================================================
    # P2 — collision-limit screen (fwd + rev, all rate classes)
    # ==================================================================
    def check_collision_limit(self) -> List[Finding]:
        import numpy as np
        gas = self.gas
        X = self._screen_composition()
        n_react = np.array([sum(r.reactants.values()) for r in gas.reactions()])
        n_prod = np.array([sum(r.products.values()) for r in gas.reactions()])
        # worst exceedance per reaction/direction
        worst: Dict[Tuple[int, str], Dict[str, Any]] = {}
        # reactions we cannot judge (order != 2) — reported once as info
        multibody: Dict[int, int] = {}

        for T in self.T_grid:
            limit = collision_limit_kmol(T)
            for P_atm in self.P_atm_grid:
                try:
                    gas.TPX = T, P_atm * 101325.0, X
                    kf = np.asarray(gas.forward_rate_constants, dtype=float)
                    kr = np.asarray(gas.reverse_rate_constants, dtype=float)
                except Exception:                          # noqa: BLE001
                    continue
                self._collision_scan(kf, n_react, T, P_atm, limit, "forward", worst, multibody)
                self._collision_scan(kr, n_prod, T, P_atm, limit, "reverse", worst, multibody)

        findings: List[Finding] = []
        for (ridx, direction), info in sorted(worst.items()):
            if info["ratio"] <= self.collision_exceedance:
                continue
            findings.append(self._collision_finding(ridx, direction, info))
        # summarise the >2-body reactions we could not check (info, once)
        if multibody:
            findings.append(Finding(
                check="collision_limit", severity="info",
                category="collision-uncheckable",
                message=(f"{len(multibody)} reaction(s) have >2 reactants/products "
                         "on the evaluated side; the gas-kinetic collision limit "
                         "is undefined for them and they were not screened."),
                details={"reaction_indices": sorted(multibody)[:50]},
            ))
        findings.sort(key=lambda f: -(f.details.get("ratio", 0)))
        return findings

    def _collision_scan(self, k, order, T, P_atm, limit, direction, worst, multibody):
        import numpy as np
        finite = np.isfinite(k) & (k > 0)
        bimolecular = np.isclose(order, 2.0)
        checkable = finite & bimolecular
        ratios = np.where(checkable, k / limit, 0.0)
        for ridx in np.nonzero(ratios > self.collision_exceedance)[0]:
            ridx = int(ridx)
            info = worst.get((ridx, direction))
            if info is None or ratios[ridx] > info["ratio"]:
                worst[(ridx, direction)] = {
                    "ratio": float(ratios[ridx]), "k": float(k[ridx]),
                    "limit": float(limit), "T_K": float(T), "P_atm": float(P_atm),
                }
        # note uncheckable >2-body reactions (evaluate finiteness so we skip zeros)
        for ridx in np.nonzero(finite & (order > 2.0))[0]:
            multibody[int(ridx)] = int(order[int(ridx)])

    def _collision_finding(self, ridx: int, direction: str, info: Dict[str, Any]) -> Finding:
        rxn = self.gas.reaction(ridx)
        eq = rxn.equation
        ratio = info["ratio"]
        orders = math.log10(ratio) if ratio > 0 else 0.0
        sev = "error" if ratio > 1e3 else "warning"
        return Finding(
            check="collision_limit", severity=sev, category="collision-limit",
            message=(f"{direction.capitalize()} rate exceeds the gas-kinetic "
                     f"collision limit by {ratio:.3g}× ({orders:.1f} orders) at "
                     f"T={info['T_K']:.0f} K, P={info['P_atm']:g} atm."),
            location=eq,
            details={"reaction_index": ridx, "direction": direction,
                     "rate_type": _rate_type(rxn), **info},
            fix_options=_collision_fix_options(eq, ridx, direction, ratio),
        )

    # ==================================================================
    # P3 — thermo Tmid discontinuity
    # ==================================================================
    def check_thermo_discontinuity(self) -> List[Finding]:
        from rmip.analysis.thermo_fitter import continuity_jumps
        findings: List[Finding] = []
        for sp in self.gas.species():
            th = sp.thermo
            if type(th).__name__ != "NasaPoly2":
                continue
            c = list(th.coeffs)
            if len(c) < 15:
                continue
            tmid = c[0]
            coeffs_high = c[1:8]
            coeffs_low = c[8:15]
            try:
                jumps = continuity_jumps(coeffs_low, coeffs_high, tmid)
            except Exception:                              # noqa: BLE001
                continue
            worst = max(jumps.values())
            if worst <= self.thermo_jump_tol:
                continue
            sev = "error" if worst > 0.05 else "warning"
            findings.append(Finding(
                check="thermo_discontinuity", severity=sev,
                category="thermo-discontinuity",
                message=(f"{sp.name}: NASA-7 Cp/H/S discontinuous at "
                         f"Tmid={tmid:.0f} K "
                         f"(rel. jump Cp={jumps['cp']:.2%}, H={jumps['h']:.2%}, "
                         f"S={jumps['s']:.2%})."),
                location=sp.name,
                details={"species": sp.name, "tmid_K": tmid, "jumps": jumps,
                         "worst_jump": worst},
                fix_options=[FixOption(
                    id="refit_thermo", kind="auto", recommended=True,
                    label="Auto-refit NASA-7 (restore continuity)",
                    description=("Resample the existing polynomial and refit a "
                                 "C0/C1-continuous NASA-7 that preserves the data "
                                 "(rmip.analysis.thermo_fitter.refit_from_coeffs)."),
                    params={"species": sp.name},
                )],
            ))
        return findings

    # ==================================================================
    # P4 — element & charge balance
    # ==================================================================
    def check_element_balance(self) -> List[Finding]:
        import numpy as np
        gas = self.gas
        findings: List[Finding] = []
        elem = gas.element_names
        for i, rxn in enumerate(gas.reactions()):
            net = {}
            for name, nu in rxn.reactants.items():
                self._accumulate_elements(gas, name, -nu, net)
            for name, nu in rxn.products.items():
                self._accumulate_elements(gas, name, +nu, net)
            imbalance = {e: v for e, v in net.items() if abs(v) > 1e-6}
            if imbalance:
                findings.append(Finding(
                    check="element_balance", severity="error",
                    category="element-imbalance",
                    message=(f"Reaction not element-balanced: net "
                             + ", ".join(f"{e}={v:+.3g}" for e, v in imbalance.items())),
                    location=rxn.equation,
                    details={"reaction_index": i, "imbalance": imbalance},
                ))
            # charge balance
            qnet = 0.0
            for name, nu in rxn.reactants.items():
                qnet -= nu * gas.species(name).charge
            for name, nu in rxn.products.items():
                qnet += nu * gas.species(name).charge
            if abs(qnet) > 1e-6:
                findings.append(Finding(
                    check="element_balance", severity="error",
                    category="charge-imbalance",
                    message=f"Reaction not charge-balanced: net charge {qnet:+.3g}.",
                    location=rxn.equation,
                    details={"reaction_index": i, "net_charge": qnet},
                ))
        return findings

    @staticmethod
    def _accumulate_elements(gas, species_name, mult, net) -> None:
        sp = gas.species(species_name)
        for e, cnt in sp.composition.items():
            net[e] = net.get(e, 0.0) + mult * cnt

    # ==================================================================
    # P5a — unmarked duplicate reactions (raw-YAML scan; works pre-load)
    # ==================================================================
    def check_duplicates(self) -> List[Finding]:
        rxns = _yaml_reactions(self._raw_yaml)
        if rxns is None:
            return self._check_duplicates_gas()
        seen: Dict[Tuple, List[int]] = {}
        for i, rd in enumerate(rxns):
            eq = rd.get("equation") if isinstance(rd, dict) else None
            if not eq:
                continue
            seen.setdefault(_yaml_reaction_signature(eq), []).append(i)
        findings: List[Finding] = []
        for _key, idxs in seen.items():
            if len(idxs) < 2:
                continue
            unmarked = [i for i in idxs
                        if not bool(rxns[i].get("duplicate", False))]
            if not unmarked:
                continue
            eq = rxns[idxs[0]].get("equation", "")
            findings.append(Finding(
                check="duplicates", severity="error",
                category="unmarked-duplicate",
                message=(f"{len(idxs)} reactions share the same equation but "
                         f"{len(unmarked)} are not flagged 'duplicate' — Cantera "
                         "will refuse to load this."),
                location=eq,
                details={"reaction_indices": idxs, "unmarked": unmarked},
                fix_options=[FixOption(
                    id="mark_duplicate", kind="auto", recommended=True,
                    label="Mark all as duplicate",
                    description="Set 'duplicate: true' on the colliding reactions "
                                "(Cantera/CHEMKIN semantics: their rates sum).",
                    params={"reaction_indices": idxs},
                ), FixOption(
                    id="delete_reaction", kind="choice",
                    label="Delete the extra reaction(s)",
                    description="Keep the first, remove the others — use only if "
                                "the duplicate is an accidental paste.",
                    params={"reaction_indices": idxs[1:]},
                )],
            ))
        return findings

    def _check_duplicates_gas(self) -> List[Finding]:
        """Fallback duplicate scan on the loaded Solution (marked dups only)."""
        gas = self.gas
        if gas is None:
            return []
        seen: Dict[Tuple, List[int]] = {}
        for i, rxn in enumerate(gas.reactions()):
            r = tuple(sorted(rxn.reactants.items()))
            p = tuple(sorted(rxn.products.items()))
            key = (frozenset((r, p)) if rxn.reversible else (r, p))
            seen.setdefault(key, []).append(i)
        findings: List[Finding] = []
        for _key, idxs in seen.items():
            if len(idxs) < 2:
                continue
            unmarked = [i for i in idxs
                        if not bool(getattr(gas.reaction(i), "duplicate", False))]
            if unmarked:
                findings.append(Finding(
                    check="duplicates", severity="error",
                    category="unmarked-duplicate",
                    message=(f"{len(idxs)} reactions share reactants/products but "
                             f"{len(unmarked)} are not flagged 'duplicate'."),
                    location=gas.reaction(idxs[0]).equation,
                    details={"reaction_indices": idxs, "unmarked": unmarked},
                ))
        return findings

    # ==================================================================
    # M6 follow-on — third-body-implied duplicate reactions (bug.md)
    # ==================================================================
    def check_third_body_duplicates(self) -> List[Finding]:
        """check_duplicates (above) matches identical equation strings/
        signatures only — it has no concept of third-body expansion, so it
        cannot catch the class of bug that broke this suite's own canonical
        fused mechanism: a bare-M reaction (``H2+M<=>2H+M``) and an explicit
        self-third-body form of the SAME physical reaction
        (``2H+H2<=>2H2``, H2 acting as its own collider) look completely
        different textually, but Cantera expands M at load time and rejects
        them as an "Undeclared duplicate reactions" error with no actionable
        guidance from this suite (found stress-testing fusion on Keromnes/
        Li/Davis, 2026-07-14 — see canonicalize.py::_cancel_self_third_body,
        the same equivalence this check mirrors on raw equation text/species
        names rather than SMILES; single-mechanism scope, so raw names are
        sufficient). Runs pre-load (raw YAML), same as check_duplicates, so
        it can explain — or even preempt — a load failure this suite's own
        "load-failure" Finding otherwise leaves as a cryptic Cantera message.
        """
        rxns = _yaml_reactions(self._raw_yaml)
        if rxns is None:
            return []

        # Two disjoint pools -- comparisons only ever cross pools, never
        # within one. This is what keeps the check from false-firing on
        # real mechanisms (verified against GRI-3.0, which is FULL of
        # per-species-collider reaction clusters -- see the two exclusions
        # below for exactly why each would otherwise false-positive):
        generic_m: List[Tuple[int, Tuple, frozenset]] = []   # (idx, core, efficiency species)
        explicit: List[Tuple[int, Tuple, frozenset]] = []    # (idx, core, cancelled species)
        for i, rd in enumerate(rxns):
            if not isinstance(rd, dict):
                continue
            eq = rd.get("equation")
            if not eq:
                continue
            parsed = _parse_reaction_core(eq)
            if parsed is None:
                continue
            core, cancelled, has_generic_m = parsed
            if has_generic_m and not cancelled:
                # A bare/parenthesized generic-M reaction with nothing else
                # to cancel. Its `efficiencies` map (if any) records species
                # this mechanism's author ALREADY handles via a separate,
                # deliberate override -- e.g. GRI-3.0's "H+O2+M<=>HO2+M"
                # explicitly zeroes O2/H2O/N2/AR because each has its own
                # explicit reaction below; without checking this, every one
                # of those legitimate reactions would false-positive as a
                # "collision" with the generic-M form.
                eff = rd.get("efficiencies") or {}
                eff_keys = frozenset(str(k).strip().upper() for k in eff.keys())
                generic_m.append((i, core, eff_keys))
            elif not has_generic_m and cancelled:
                # An explicit reaction with a genuine self-collider (a
                # species literally present on both sides, cancelled here).
                # Reactions with NOTHING to cancel (e.g. GRI-3.0's plain
                # "NNH<=>N2+H", which happens to share a core with the
                # three-body "NNH+M<=>N2+H+M" purely because stripping M
                # leaves the same species) are excluded by the `cancelled`
                # check -- there is no textual basis for treating them as a
                # third-body-equivalent form.
                explicit.append((i, core, cancelled))

        findings: List[Finding] = []
        reported: set = set()
        for gi, gcore, geffs in generic_m:
            for ei, ecore, ecancelled in explicit:
                if gcore != ecore:
                    continue
                if ecancelled & geffs:
                    continue  # already explicitly handled via an efficiency override
                pair = (min(gi, ei), max(gi, ei))
                if pair in reported:
                    continue
                eq_g, eq_e = rxns[gi].get("equation", ""), rxns[ei].get("equation", "")
                if _yaml_reaction_signature(eq_g) == _yaml_reaction_signature(eq_e):
                    continue  # check_duplicates' job, already correctly handled
                if rxns[gi].get("duplicate") and rxns[ei].get("duplicate"):
                    # Already resolved -- both sides correctly flagged
                    # 'duplicate: true' (Cantera sums their rates and loads
                    # fine; e.g. the canonical-key-rebuild migration's own
                    # output). Nothing actionable left to report.
                    continue
                reported.add(pair)
                idxs = sorted(pair)
                eqs = [rxns[k].get("equation", "") for k in idxs]
                findings.append(Finding(
                    check="third_body_duplicates", severity="error",
                    category="third-body-implied-duplicate",
                    message=(
                        f"{len(idxs)} reactions become the SAME physical "
                        "reaction once the generic third body 'M' is "
                        f"instantiated: {', '.join(repr(e) for e in eqs)}. "
                        "Cantera will reject this as an 'Undeclared duplicate "
                        "reactions' error even though the equations look "
                        "different (a bare-M form vs. an explicit self-"
                        "collider form of the same elementary step) — this "
                        "requires third-body expansion, which neither raw-YAML "
                        "nor loaded-Solution textual duplicate matching "
                        "(check_duplicates, above) performs."
                    ),
                    location=eqs[0],
                    details={"reaction_indices": idxs, "equations": eqs},
                    fix_options=[FixOption(
                        id="mark_duplicate", kind="choice",
                        label="Mark all as duplicate",
                        description="Set 'duplicate: true' on all of them "
                                    "(Cantera will then sum their rates, "
                                    "matching its own third-body semantics) — "
                                    "verify this is physically correct for "
                                    "your mechanism before applying; unlike a "
                                    "plain textual duplicate, no auto-fix is "
                                    "offered here since these were written as "
                                    "genuinely different-looking reactions.",
                        params={"reaction_indices": idxs},
                    )],
                ))
        return findings

    # ==================================================================
    # P5b — pressure-dependent reaction sanity
    # ==================================================================
    def check_pdep_sanity(self) -> List[Finding]:
        findings: List[Finding] = []
        raw = _yaml_reactions(self._raw_yaml)
        if raw is not None:
            # Prefer the raw YAML: it works even when Cantera refused the file
            # (e.g. a negative PLOG piece rejected at PlogRate::validate).
            for i, rd in enumerate(raw):
                if not isinstance(rd, dict):
                    continue
                eq = rd.get("equation", "")
                rtype = str(rd.get("type", ""))
                if "pressure-dependent-Arrhenius" in rtype or "plog" in rtype.lower():
                    findings.extend(self._check_plog(i, eq, rd))
                elif rtype in ("falloff", "Troe", "SRI", "Lindemann", "three-body") \
                        or "Troe" in rd or "low-P-rate-constant" in rd:
                    findings.extend(self._check_falloff(i, eq, rd))
            return findings
        if self.gas is None:
            return findings
        for i, rxn in enumerate(self.gas.reactions()):
            data = _safe_input_data(rxn)
            rtype = _rate_type(rxn)
            if rtype == "pressure-dependent-Arrhenius":
                findings.extend(self._check_plog(i, rxn.equation, data))
            elif rtype in ("falloff", "Troe", "SRI", "Lindemann"):
                findings.extend(self._check_falloff(i, rxn.equation, data))
        return findings

    def _check_plog(self, i, eq, data) -> List[Finding]:
        out: List[Finding] = []
        rates = data.get("rate-constants", []) if isinstance(data, dict) else []
        pressures: List[float] = []
        neg_A = False
        n_pieces = 0
        n_neg = 0
        for entry in rates:
            try:
                A = float(entry.get("A"))
                P = _parse_pressure(entry.get("P"))
            except (TypeError, ValueError):
                continue
            n_pieces += 1
            pressures.append(P)
            if A < 0:
                neg_A = True
                n_neg += 1
        if neg_A:
            # A whole-negative duplicate reaction (every pressure piece has
            # A < 0) is the "subtraction channel" half of a vendor dual-PLOG
            # fit meant to be summed against a companion duplicate reaction.
            # plog_drop_negative can't repair it (dropping every piece would
            # leave an empty rate-constants list) -- delete_reaction is the
            # only viable auto path, at the cost of losing that channel's
            # correction entirely. Found 2026-07-22 on Glarborg 2018's
            # 'C2H3 + CH2O <=> C2H4 + HCO' Reaction 287.
            all_negative = n_pieces > 0 and n_neg == n_pieces
            out.append(Finding(
                check="pdep_sanity", severity="error", category="plog-negative-A",
                message=("PLOG reaction has a negative A-factor on at least one "
                         "pressure — log-P interpolation can produce NaN/negative "
                         "combined rates (Cantera 3.x rejects this at load; older "
                         "versions fail mid-integration in CVODES)."
                         + (" Every pressure piece is negative (a dual-PLOG "
                            "subtraction channel) — dropping negative pieces "
                            "would leave nothing; delete the whole duplicate "
                            "reaction instead." if all_negative else "")),
                location=eq,
                details={"reaction_index": i, "all_negative": all_negative},
                fix_options=_plog_fix_options(eq, i, all_negative=all_negative),
            ))
        # unsorted pressures
        if len(pressures) >= 2:
            if any(pressures[j] > pressures[j + 1] for j in range(len(pressures) - 1)):
                out.append(Finding(
                    check="pdep_sanity", severity="info", category="plog-unsorted",
                    message="PLOG pressure points are not in ascending order.",
                    location=eq, details={"reaction_index": i},
                ))
        return out

    def _check_falloff(self, i, eq, data) -> List[Finding]:
        out: List[Finding] = []
        if not isinstance(data, dict):
            return out
        troe = data.get("Troe")
        if isinstance(troe, dict):
            bad = [k for k in ("T3", "T1") if _is_nonpositive(troe.get(k))]
            if bad:
                out.append(Finding(
                    check="pdep_sanity", severity="warning", category="troe-params",
                    message=(f"Troe falloff parameters {bad} are ≤ 0, which makes "
                             "the broadening factor ill-defined."),
                    location=eq, details={"reaction_index": i, "Troe": troe},
                ))
        rtype = str(data.get("type", ""))
        if rtype == "falloff" or "Troe" in data:
            has_low = "low-P-rate-constant" in data
            has_high = "high-P-rate-constant" in data
            if not (has_low and has_high):
                out.append(Finding(
                    check="pdep_sanity", severity="warning", category="falloff-limits",
                    message="Falloff reaction is missing a high- or low-pressure limit.",
                    location=eq, details={"reaction_index": i,
                                          "has_low": has_low, "has_high": has_high},
                ))
        return out

    # ==================================================================
    # P6 — completeness (unused species, transport, T-range)
    # ==================================================================
    def check_completeness(self) -> List[Finding]:
        gas = self.gas
        findings: List[Finding] = []

        used = set()
        for rxn in gas.reactions():
            used.update(rxn.reactants)
            used.update(rxn.products)
            third = getattr(rxn, "third_body", None)
            if third is not None:
                used.update(getattr(third, "efficiencies", {}) or {})
        unused = [s for s in gas.species_names if s not in used]
        # a bath gas (AR, HE, N2) legitimately appears only as a third body / diluent
        for name in unused:
            diluent = name.upper() in ("AR", "HE", "N2", "HELIUM", "ARGON")
            findings.append(Finding(
                check="completeness",
                severity="info" if diluent else "warning",
                category="unused-species",
                message=(f"Species '{name}' participates in no reaction"
                         + (" (looks like a diluent/bath gas)." if diluent else ".")),
                location=name, details={"species": name, "diluent": diluent},
                fix_options=[] if diluent else [FixOption(
                    id="remove_species", kind="choice",
                    label="Remove unused species",
                    description="Delete the species and its thermo/transport "
                                "entries. Skip if it is an intended diluent.",
                    params={"species": name},
                )],
            ))

        # transport completeness (only if the phase declares a transport model)
        has_transport = self._has_transport_block()
        if has_transport:
            for sp in gas.species():
                tr = getattr(sp, "transport", None)
                if tr is None or getattr(tr, "geometry", None) in (None, ""):
                    findings.append(Finding(
                        check="completeness", severity="warning",
                        category="missing-transport",
                        message=f"Species '{sp.name}' has no/incomplete transport data.",
                        location=sp.name, details={"species": sp.name},
                    ))

        # thermo T-range coverage vs the target simulation temperature
        for sp in gas.species():
            th = sp.thermo
            if th is None:
                continue
            if th.max_temp < self.target_T_high - 1.0:
                findings.append(Finding(
                    check="completeness", severity="warning",
                    category="thermo-trange",
                    message=(f"Species '{sp.name}' thermo tops out at "
                             f"{th.max_temp:.0f} K < target {self.target_T_high:.0f} K "
                             "— high-T extrapolation will be used."),
                    location=sp.name,
                    details={"species": sp.name, "max_temp": th.max_temp,
                             "target": self.target_T_high},
                ))
        return findings

    # ==================================================================
    # P14 — Ea unit-sanity heuristic
    # ==================================================================
    def check_unit_sanity(self) -> List[Finding]:
        findings: List[Finding] = []
        for i, rxn in enumerate(self.gas.reactions()):
            Ea_cal = _reaction_Ea_cal(rxn)
            if Ea_cal is None:
                continue
            # Combustion Ea essentially never exceeds ~150 kcal/mol; flag above
            # 200 kcal/mol. The two classic traps are a J/mol number in a cal/mol
            # field (~4.184x too big) and a J/kmol number in a cal/mol field
            # (~4184x too big — the "silent 1000x" error from §2.1).
            if abs(Ea_cal) <= 2.0e5:           # ≤ 200 kcal/mol → plausible
                continue
            kcal = Ea_cal / 1e3
            huge = abs(Ea_cal) > 1.0e5 * J_PER_CAL   # would a ÷4184 land <100 kcal
            findings.append(Finding(
                check="unit_sanity", severity="warning", category="ea-magnitude",
                message=(f"Activation energy ≈ {kcal:.0f} kcal/mol is implausibly "
                         "large — likely a cal/mol vs J/mol vs J/kmol unit trap."),
                location=rxn.equation,
                details={"reaction_index": i, "Ea_cal_per_mol": Ea_cal},
                fix_options=[
                    FixOption(id="ea_reinterpret_j_kmol", kind="choice",
                              recommended=huge,
                              label="Reinterpret Ea as J/kmol (÷4184)",
                              description="If a J/kmol value was dropped into a "
                                          "cal/mol field (the silent 1000x trap), "
                                          "divide by 4184.",
                              params={"reaction_index": i}),
                    FixOption(id="ea_reinterpret_j", kind="choice",
                              recommended=not huge,
                              label="Reinterpret Ea as J/mol (÷4.184)",
                              description="If a J/mol value was mislabelled cal/mol, "
                                          "divide by 4.184.",
                              params={"reaction_index": i}),
                    FixOption(id="ignore", kind="choice",
                              label="Keep as-is (acknowledge)",
                              description="Leave Ea unchanged.",
                              params={"reaction_index": i}),
                ],
            ))
        return findings

    # ==================================================================
    # P7 — bounded constant-volume ignition smoke test
    # ==================================================================
    def check_sim_setup_sanity(self) -> List[Finding]:
        """Generic, composition-independent integrator smoke test.

        Builds a ``ct.IdealGasReactor`` + ``ct.ReactorNet`` at the existing
        benign screening composition (:meth:`_screen_composition`) and the
        hottest/lowest-pressure point of the existing (T,P) screening grid
        — the point most likely to stress the ODE integrator — and attempts
        a short, step- **and** wall-clock-bounded ``advance``. If CVODES
        raises or the reactor state goes non-finite/non-physical, that is
        reported as a warning; it usually means a rate/thermo defect that
        will make *any* downstream simulation on this mechanism fail the
        same way.

        This is deliberately **not** the target-specific "IDT vs
        integration time" check from the original P7 scope: that needs a
        fuel + reactor-type + target input this DB-free checker does not
        have (``MechanismChecker`` only ever sees a bare ``self.gas`` and a
        composition-independent T/P grid). That check belongs with
        ``MuqsacBridge`` / ``nominal_sim_gui.py``, which do have a target.
        Here we only rule out the generic "the integrator cannot even take
        a step on this mechanism" failure class (§2.1 of
        MECHANISM_CHECK_RESEARCH.md).
        """
        import time
        import cantera as ct

        gas = self.gas
        if gas is None or not self.T_grid or not self.P_atm_grid:
            return []

        T = max(self.T_grid)
        P_atm = min(self.P_atm_grid)
        X = self._screen_composition()

        # Hard caps so one pathological mechanism can never hang `rmip
        # check`: bounded step count AND a wall-clock deadline checked
        # inside the loop (no signal.alarm — must stay safe off the main
        # thread / inside a GUI worker).
        max_steps = 200
        max_wall_s = 5.0
        t_end_s = 1.0    # generously beyond any realistic ignition delay here

        try:
            gas.TPX = T, P_atm * 101325.0, X
            r = ct.IdealGasReactor(gas)
            net = ct.ReactorNet([r])
            deadline = time.monotonic() + max_wall_s
            t = 0.0
            steps = 0
            while t < t_end_s and steps < max_steps:
                if time.monotonic() > deadline:
                    break
                t = net.step()
                steps += 1
            if not math.isfinite(r.T) or r.T <= 0:
                raise RuntimeError(
                    f"reactor temperature non-finite/non-physical "
                    f"({r.T!r}) after {steps} step(s)")
        except Exception as exc:                           # noqa: BLE001
            return [Finding(
                check="sim_setup_sanity", severity="warning",
                category="integrator-failure",
                message=("Constant-volume ignition smoke test failed to "
                         f"advance at T={T:.0f} K, P={P_atm:g} atm: {exc}"),
                location=f"T={T:.0f} K, P={P_atm:g} atm",
                details={"T_K": T, "P_atm": P_atm, "error": str(exc)},
            )]
        return []

    # ==================================================================
    # P8 — CSP ultrafast-timescale model screening
    # ==================================================================
    def check_csp_timescales(self) -> List[Finding]:
        """Flag unphysically-fast ("ultrafast") chemical-Jacobian modes.

        Calls the pure, DB-free helpers in :mod:`rmip.reduction.csp`
        (``chemical_jacobian``/``mode_decomposition``/``timescales_from_eigs``/
        ``csp_pointers``) directly against ``self.gas`` at the benign
        screening composition (:meth:`_screen_composition`) and a small
        reduced subset (first/middle/last) of the configured ``T_grid`` at
        the lowest configured pressure — this is an O(n^3) eigendecomposition
        per point, so the grid is deliberately tiny.

        This uses a **synthetic/generic screening composition**, not a
        physically-anchored ignition state — it is a cheap model-quality
        screen for reactions/species whose implied timescale is absurdly
        fast (e.g. a wildly oversized A-factor), not a substitute for a
        real analysis. For a physically-grounded CSP/QSSA analysis at an
        actual ignition state (needs a fuel + DB), use
        ``rmip.reduction.csp.CSPAnalyzer.analyze(at="ignition")``.
        """
        gas = self.gas
        if gas is None:
            return []
        if gas.n_species > self.csp_max_species:
            return [Finding(
                check="csp_timescales", severity="info",
                category="csp-skipped",
                message=(f"Skipped CSP timescale screening: {gas.n_species} "
                         f"species exceeds csp_max_species="
                         f"{self.csp_max_species} (the eigendecomposition is "
                         "O(n^3) and too costly to run on every `check`)."),
                details={"n_species": gas.n_species,
                         "csp_max_species": self.csp_max_species},
            )]

        import numpy as np
        from rmip.reduction.csp import (
            chemical_jacobian, mode_decomposition, timescales_from_eigs,
            csp_pointers,
        )

        ultrafast_floor = 1.0e-12    # s — well below any real combustion mode
        X = self._screen_composition()
        P_atm = min(self.P_atm_grid) if self.P_atm_grid else 1.0
        grid = self.T_grid
        if len(grid) >= 3:
            points = [grid[0], grid[len(grid) // 2], grid[-1]]
        else:
            points = list(grid) or [1000.0]

        findings: List[Finding] = []
        names = gas.species_names
        seen_T = set()
        for T in points:
            if T in seen_T:
                continue
            seen_T.add(T)
            try:
                gas.TPX = T, P_atm * 101325.0, X
                J, _analytic = chemical_jacobian(gas)
                eigvals, A, B = mode_decomposition(J)
                taus = timescales_from_eigs(eigvals)
            except Exception:                              # noqa: BLE001
                log.debug("csp_timescales: skipped T=%s", T, exc_info=True)
                continue
            try:
                pointers = csp_pointers(A, B)
            except Exception:                              # noqa: BLE001
                pointers = None
            for k, tau in enumerate(taus):
                tau = float(tau)
                if not math.isfinite(tau) or tau >= ultrafast_floor:
                    continue
                details: Dict[str, Any] = {"T_K": T, "P_atm": P_atm,
                                           "mode_index": k, "tau_s": tau}
                species_note = ""
                if pointers is not None:
                    i = int(np.argmax(pointers[k]))
                    species_note = f", dominant species '{names[i]}'"
                    details["species"] = names[i]
                findings.append(Finding(
                    check="csp_timescales", severity="warning",
                    category="ultrafast-timescale",
                    message=(f"Chemical-Jacobian mode {k} has an ultrafast "
                             f"timescale τ={tau:.3g} s at T={T:.0f} K, "
                             f"P={P_atm:g} atm{species_note} — below "
                             f"{ultrafast_floor:.0e} s, well under any "
                             "physically real combustion timescale; check "
                             "the rate/thermo data feeding this mode."),
                    location=f"T={T:.0f} K, mode {k}",
                    details=details,
                ))
        return findings

    # ==================================================================
    # W8a / P33 — silent-failure detectors: dead reactions & decoupled
    # species (the "loads fine but computes nothing" class)
    # ==================================================================
    def check_zero_rates(self) -> List[Finding]:
        import numpy as np
        gas = self.gas
        if gas is None or gas.n_reactions == 0:
            return []
        # Every species present so no rate is zero merely because a reactant
        # has zero concentration (the screen composition uses only 4 species).
        X = {n: 1.0 for n in gas.species_names}
        k_max = np.zeros(gas.n_reactions)
        w_max = np.zeros(gas.n_species)
        n_states = 0
        for T in self.T_grid:
            for P_atm in self.P_atm_grid:
                try:
                    gas.TPX = T, P_atm * 101325.0, X
                    kf = np.abs(np.asarray(gas.forward_rate_constants, dtype=float))
                    kr = np.abs(np.asarray(gas.reverse_rate_constants, dtype=float))
                    cr = np.abs(np.asarray(gas.creation_rates, dtype=float))
                    de = np.abs(np.asarray(gas.destruction_rates, dtype=float))
                except Exception:                          # noqa: BLE001
                    continue                # rate_sweep reports evaluation errors
                n_states += 1
                k_max = np.maximum(k_max, np.nan_to_num(np.maximum(kf, kr)))
                w_max = np.maximum(w_max, np.nan_to_num(cr + de))
        if n_states == 0:
            return []

        findings: List[Finding] = []
        dead = [int(i) for i in np.nonzero(k_max <= _ZERO_RATE_FLOOR)[0]]

        if dead and len(dead) == gas.n_reactions:
            # The P33 headline case: the mechanism "loads fine" but every
            # single rate is zero — an import/unit catastrophe, not chemistry.
            findings.append(Finding(
                check="zero_rates", severity="error", category="all-rates-zero",
                message=(f"ALL {gas.n_reactions} reactions have forward AND "
                         "reverse rate constants identically zero over the whole "
                         f"({len(self.T_grid)}×{len(self.P_atm_grid)}) screening "
                         "grid — the mechanism loads but computes nothing. "
                         "This is the classic silent import failure (all "
                         "A-factors dropped/zeroed during conversion)."),
                location=os.path.basename(self.path),
                details={"n_reactions": gas.n_reactions},
            ))
        else:
            for ridx in dead[:25]:
                rxn = gas.reaction(ridx)
                findings.append(Finding(
                    check="zero_rates", severity="warning",
                    category="dead-reaction",
                    message=("Forward and reverse rate constants are identically "
                             "zero at every screening (T,P) — this reaction can "
                             "never fire (typically A=0 from a botched edit or "
                             "conversion)."),
                    location=rxn.equation,
                    details={"reaction_index": ridx,
                             "rate_type": _rate_type(rxn),
                             "k_max": float(k_max[ridx])},
                    fix_options=[FixOption(
                        id="delete_reaction", kind="choice",
                        label="Delete the dead reaction",
                        description="Remove the reaction — it contributes "
                                    "nothing. Keep it only if you zeroed it "
                                    "deliberately (e.g. for a sensitivity test).",
                        params={"equation": rxn.equation, "reaction_index": ridx},
                    )],
                ))
            if len(dead) > 25:
                findings.append(Finding(
                    check="zero_rates", severity="warning",
                    category="dead-reaction",
                    message=(f"…and {len(dead) - 25} more dead reactions "
                             "(all rate constants identically zero)."),
                    details={"reaction_indices": dead[25:]},
                ))

        # Decoupled species: participates as reactant/product yet is never
        # created nor destroyed anywhere on the grid (with every species
        # present) — it is chemically disconnected despite appearing coupled.
        participates: set = set()
        for rxn in gas.reactions():
            participates.update(rxn.reactants)
            participates.update(rxn.products)
        for sidx, name in enumerate(gas.species_names):
            if name not in participates:
                continue                    # plain unused → completeness check
            if w_max[sidx] > _ZERO_RATE_FLOOR:
                continue
            findings.append(Finding(
                check="zero_rates", severity="warning",
                category="decoupled-species",
                message=(f"Species '{name}' appears in reactions but its "
                         "creation and destruction rates are identically zero "
                         "at every screening (T,P) — it is silently decoupled "
                         "from the chemistry (all its reactions are dead)."),
                location=name,
                details={"species": name, "w_max": float(w_max[sidx])},
            ))
        return findings

    # ==================================================================
    # W8a / P35 — cross-section species-name consistency (raw-YAML scan;
    # works pre-load, so its fix can make an unloadable mechanism load)
    # ==================================================================
    def check_name_consistency(self) -> List[Finding]:
        raw = self._raw_yaml
        if not isinstance(raw, dict):
            return []
        declared = [str(sp["name"]) for sp in (raw.get("species") or [])
                    if isinstance(sp, dict) and sp.get("name") is not None]
        if not declared:
            return []
        exact = set(declared)
        by_fold: Dict[str, List[str]] = {}
        for n in declared:
            by_fold.setdefault(n.upper(), []).append(n)

        findings: List[Finding] = []

        # Two species *declared* as case-variants of each other is almost
        # always one physical species written two ways (AR vs Ar) — each gets
        # its own thermo block and they silently never react with each other.
        for fold, names in sorted(by_fold.items()):
            if len(names) > 1:
                findings.append(Finding(
                    check="name_consistency", severity="warning",
                    category="species-case-twins",
                    message=(f"Species {names} differ only by letter case — "
                             "if they are the same physical species, reactions "
                             "referencing one never see the other's chemistry."),
                    location=", ".join(names),
                    details={"names": names},
                ))

        # Collect every reference site: phase species lists, reaction
        # equations, third-body efficiencies.
        refs: Dict[str, List[str]] = {}   # token -> where seen

        for ph in (raw.get("phases") or []):
            if not isinstance(ph, dict):
                continue
            for entry in (ph.get("species") or []):
                if isinstance(entry, str):
                    refs.setdefault(entry, []).append(
                        f"phase '{ph.get('name', '?')}' species list")

        for key, val in raw.items():
            if key != "reactions" and not str(key).endswith("-reactions"):
                continue
            if not isinstance(val, list):
                continue
            for rd in val:
                if not isinstance(rd, dict):
                    continue
                eq = rd.get("equation", "")
                for tok in _equation_species(eq):
                    refs.setdefault(tok, []).append(f"equation: {eq}")
                eff = rd.get("efficiencies")
                if isinstance(eff, dict):
                    for tok in eff:
                        refs.setdefault(str(tok), []).append(
                            f"efficiencies of: {eq}")

        for tok, sites in sorted(refs.items()):
            if tok in exact:
                continue
            variants = by_fold.get(tok.upper())
            if variants:
                canonical = variants[0]
                findings.append(Finding(
                    check="name_consistency", severity="error",
                    category="species-case-mismatch",
                    message=(f"'{tok}' is referenced but the species is "
                             f"declared as '{canonical}' — Cantera is "
                             "case-sensitive, so this either fails to load or "
                             "silently targets a different species."),
                    location=sites[0],
                    details={"referenced": tok, "declared": canonical,
                             "n_sites": len(sites), "sites": sites[:10]},
                    fix_options=[FixOption(
                        id="rename_species_ref", kind="auto", recommended=True,
                        label=f"Rename references '{tok}' → '{canonical}'",
                        description="Rewrite every reference (equations, "
                                    "efficiencies, phase species lists) to the "
                                    "declared spelling.",
                        params={"from": tok, "to": canonical},
                    )],
                ))
            else:
                findings.append(Finding(
                    check="name_consistency", severity="warning",
                    category="undeclared-species-ref",
                    message=(f"'{tok}' is referenced but never declared in the "
                             "species section (no case-variant match either)."),
                    location=sites[0],
                    details={"referenced": tok, "n_sites": len(sites),
                             "sites": sites[:10]},
                ))
        return findings

    # ------------------------------------------------------------------
    # shared helpers
    # ------------------------------------------------------------------
    def _screen_composition(self) -> Dict[str, float]:
        """A benign, always-valid composition for state-setting on the grid."""
        names = self.gas.species_names
        # equal parts of the first few species keeps every mole fraction > 0
        # so reverse rates are always evaluated; the actual values don't matter
        # for a NaN/limit screen (rate constants are composition-independent).
        pick = names[: min(4, len(names))]
        return {n: 1.0 for n in pick}

    def _has_transport_block(self) -> bool:
        try:
            for sp in self.gas.species():
                tr = getattr(sp, "transport", None)
                if tr is not None and getattr(tr, "geometry", None):
                    return True
        except Exception:                                  # noqa: BLE001
            return False
        return False


# ===========================================================================
# Fix-option factories (shared between checker and fixer)
# ===========================================================================
def _collision_fix_options(eq: str, idx: int, direction: str, ratio: float) -> List[FixOption]:
    return [
        FixOption(id="scale_A_to_limit", kind="choice", recommended=True,
                  label="Scale A-factor down to the collision limit",
                  description=("Reduce the pre-exponential so the peak rate sits "
                               "at the gas-kinetic limit. Preserves n and Ea. "
                               "Only valid for a pressure-independent Arrhenius "
                               "in the offending direction."),
                  params={"equation": eq, "reaction_index": idx,
                          "direction": direction, "ratio": ratio}),
        FixOption(id="delete_reaction", kind="choice",
                  label="Delete this reaction",
                  description="Remove the reaction entirely (use if it is spurious).",
                  params={"equation": eq, "reaction_index": idx}),
        FixOption(id="ignore", kind="choice",
                  label="Keep as-is (acknowledge)",
                  description="Leave the reaction unchanged — e.g. a barrierless "
                              "recombination legitimately near the limit.",
                  params={"equation": eq, "reaction_index": idx}),
    ]


def _plog_fix_options(eq: str, idx: int, all_negative: bool = False) -> List[FixOption]:
    if all_negative:
        # Every pressure piece is negative: plog_drop_negative has nothing
        # left to keep, so deleting the whole duplicate reaction is the only
        # fix that yields a loadable mechanism (recommended). Dropping is
        # still offered (will report "failed" cleanly) so --fix all's log
        # explains why it didn't help, matching the diagnostic message above.
        return [
            FixOption(id="delete_reaction", kind="choice", recommended=True,
                      label="Delete the whole duplicate reaction",
                      description="Every pressure piece has A<0 (a dual-PLOG "
                                  "subtraction channel) — drop the entire "
                                  "duplicate reaction so the mechanism loads; "
                                  "the companion positive-A duplicate for the "
                                  "same equation still carries the rate.",
                      params={"equation": eq, "reaction_indices": [idx]}),
            FixOption(id="plog_drop_negative", kind="choice",
                      label="Drop the negative-A pressure piece(s)",
                      description="Remove PLOG entries with A<0 (will fail: "
                                  "no non-negative piece remains here).",
                      params={"equation": eq, "reaction_index": idx}),
            FixOption(id="ignore", kind="choice",
                      label="Keep as-is (acknowledge)",
                      description="Leave the PLOG untouched (mechanism will "
                                  "not load in Cantera as-is).",
                      params={"equation": eq, "reaction_index": idx}),
        ]
    return [
        FixOption(id="plog_drop_negative", kind="choice", recommended=True,
                  label="Drop the negative-A pressure piece(s)",
                  description="Remove PLOG entries with A<0 so the log-P "
                              "interpolation cannot go negative/NaN.",
                  params={"equation": eq, "reaction_index": idx}),
        FixOption(id="ignore", kind="choice",
                  label="Keep as-is (acknowledge)",
                  description="Leave the PLOG untouched (some fits intentionally "
                              "use a negative piece that stays net-positive).",
                  params={"equation": eq, "reaction_index": idx}),
    ]


# ===========================================================================
# small module-level helpers
# ===========================================================================
def _rate_type(rxn) -> str:
    data = _safe_input_data(rxn)
    if isinstance(data, dict) and data.get("type"):
        return str(data["type"])
    return type(rxn.rate).__name__ if getattr(rxn, "rate", None) is not None else "unknown"


def _safe_input_data(obj) -> Dict[str, Any]:
    try:
        d = obj.input_data
        return dict(d) if d else {}
    except Exception:                                      # noqa: BLE001
        return {}


def _reaction_Ea_cal(rxn) -> Optional[float]:
    """Best-effort activation energy in cal/mol from a simple Arrhenius rate."""
    rate = getattr(rxn, "rate", None)
    Ea_J_per_kmol = None
    try:
        Ea_J_per_kmol = float(rate.activation_energy)     # J/kmol in Cantera
    except Exception:                                      # noqa: BLE001
        data = _safe_input_data(rxn)
        rc = data.get("rate-constant") if isinstance(data, dict) else None
        if isinstance(rc, dict) and "Ea" in rc:
            # input_data Ea is a quantity string or number in the file's units;
            # skip — activation_energy above is the reliable path.
            return None
        return None
    # J/kmol -> J/mol -> cal/mol
    return (Ea_J_per_kmol / 1000.0) / J_PER_CAL


def _parse_pressure(P) -> float:
    """PLOG 'P' may be a number or a '<value> <unit>' quantity string."""
    if isinstance(P, (int, float)):
        return float(P)
    if isinstance(P, str):
        return float(P.split()[0])
    raise ValueError(f"unparseable pressure {P!r}")


def _is_nonpositive(v) -> bool:
    try:
        return float(v) <= 0.0
    except (TypeError, ValueError):
        return False


def _try_parse_yaml(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a Cantera YAML file into a plain dict.

    Uses :mod:`rmip.analysis.yaml_safe` rather than plain ``yaml.safe_load``:
    stock PyYAML resolves bare ``NO`` (nitric oxide — present in every NOx
    mechanism) as a YAML-1.1 boolean, silently corrupting species-name lists.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        from rmip.analysis.yaml_safe import safe_load_path
        data = safe_load_path(path)
        return data if isinstance(data, dict) else None
    except Exception:                                      # noqa: BLE001
        return None


def _yaml_reactions(raw: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Return the top-level 'reactions' list from a parsed Cantera YAML."""
    if not isinstance(raw, dict):
        return None
    rxns = raw.get("reactions")
    if isinstance(rxns, list):
        return rxns
    return None


def _yaml_reaction_signature(eq: str) -> Tuple:
    """Signature of a reaction equation string for duplicate detection.

    Splits on the reversible/irreversible arrow and normalises whitespace and
    within-side species order so 'A + B <=> C' and 'B + A <=> C' collide --
    a genuinely reversible reaction represents one symmetric physical
    process, so which side is written first doesn't matter.

    For an IRREVERSIBLE reaction ('=>' or '<='), direction is preserved
    instead of collapsed: 'A + B => C' and 'C => A + B' are two distinct,
    independently-parameterized one-way reactions (a common authoring
    pattern when a mechanism gives separate forward/backward rate
    expressions rather than relying on Cantera's computed equilibrium
    reverse rate) and must NOT be flagged as duplicates of each other.
    Bug found 2026-08 (Test_run_23_JULY/mechanism_check_gui/SUMMARY.md):
    collapsing both sides into an order-insensitive frozenset regardless of
    reversibility produced false-positive "unmarked duplicate" errors on
    exactly this legitimate pattern.
    """
    is_reversible = "<=>" in eq
    s = eq.replace("<=>", "=").replace("=>", "=").replace("<=", "=")
    parts = [p.strip() for p in s.split("=")]
    sides = []
    for side in parts:
        toks = tuple(sorted(t.strip() for t in side.split("+") if t.strip()))
        sides.append(toks)
    if len(sides) != 2:
        return tuple(sides)
    return frozenset(sides) if is_reversible else tuple(sides)


def _parse_reaction_core(eq: str) -> Optional[Tuple[Tuple, frozenset, bool]]:
    """Parse a reaction equation into ``(core_signature, cancelled_species,
    has_generic_m)`` for third-body-duplicate detection.

    ``core_signature``: order/direction-insensitive signature of the
    reaction after stripping any bare/parenthesized generic 'M' and
    cancelling the shared min count of any species appearing on BOTH sides
    (the explicit self-third-body case, e.g. ``2H+H2<=>2H2`` — H2 acting as
    its own collider) — mirrors
    ``canonicalize.py::_cancel_self_third_body``, on raw equation text/
    species names instead of SMILES (this operates within a single
    mechanism, so raw names are sufficient; no RDKit/glossary needed).

    ``cancelled_species``: the species names actually cancelled (empty for
    a plain generic-M reaction with nothing else to cancel).

    ``has_generic_m``: True if a bare 'M' term or a parenthesized generic
    '(+M)' tag was present anywhere.

    Returns None if the equation doesn't parse into exactly two sides, OR
    if it names a SPECIFIC collider (e.g. "(+AR)") anywhere — a specific-
    collider falloff is a physically distinct reaction (different LOW/TROE
    parameters) from the generic-M/bare-M/self-collider case this check
    targets, and from any OTHER specific-collider reaction of the same base
    equation; conflating them would be a false positive (see M6, bug.md —
    the generic-M-vs-specific-collider distinction is exactly what that fix
    preserves).
    """
    if re.search(r'\(\+([^)]+)\)', eq):
        for m in re.finditer(r'\(\+([^)]+)\)', eq):
            if m.group(1).strip().upper() != "M":
                return None

    s = eq.replace("<=>", "=").replace("=>", "=").replace("<=", "=")
    parts = [p.strip() for p in s.split("=")]
    if len(parts) != 2:
        return None

    def parse_side(side: str) -> Tuple[Dict[str, float], bool]:
        counts: Dict[str, float] = {}
        has_m = False
        for term in side.split(" + "):
            term = term.strip()
            if not term:
                continue
            while "(+" in term:
                # Any parenthesized tag remaining here is guaranteed generic
                # (a specific collider already triggered the early return
                # above), so stripping it also signals has_generic_m.
                has_m = True
                start = term.index("(+")
                end = term.find(")", start)
                term = (term[:start] + (term[end + 1:] if end != -1 else "")).strip()
            if not term:
                continue
            toks = term.split()
            coeff = 1.0
            if len(toks) > 1:
                try:
                    coeff = float(toks[0])
                    toks = toks[1:]
                except ValueError:
                    pass
            name = " ".join(toks)
            if not name:
                continue
            if name.upper() == "M":
                has_m = True
                continue
            counts[name] = counts.get(name, 0.0) + coeff
        return counts, has_m

    r_counts, r_has_m = parse_side(parts[0])
    p_counts, p_has_m = parse_side(parts[1])
    has_generic_m = r_has_m or p_has_m

    cancelled: set = set()
    for name in set(r_counts) & set(p_counts):
        shared = min(r_counts[name], p_counts[name])
        if shared <= 0:
            continue
        cancelled.add(name)
        r_counts[name] -= shared
        p_counts[name] -= shared
        if r_counts[name] <= 0:
            del r_counts[name]
        if p_counts[name] <= 0:
            del p_counts[name]

    def freeze(counts: Dict[str, float]) -> Tuple:
        return tuple(sorted(counts.items()))

    core = frozenset((freeze(r_counts), freeze(p_counts)))
    return core, frozenset(cancelled), has_generic_m


def _equation_species(eq: str) -> List[str]:
    """Species tokens referenced by a Cantera YAML equation string.

    Handles stoichiometric coefficients ('2 H2O'), the falloff third body
    '(+M)' / '(+AR)', and the plain third body 'M'. Cantera writes ' + '
    with surrounding spaces, so splitting on that keeps ionic names intact.
    """
    s = eq.replace("<=>", "=").replace("=>", "=").replace("<=", "=")
    tokens: List[str] = []
    for side in s.split("="):
        for term in side.split(" + "):
            term = term.strip()
            # strip falloff '(+M)' / '(+AR)' — the parenthesised third body
            while "(+" in term:
                start = term.index("(+")
                end = term.find(")", start)
                inner = term[start + 2:end].strip() if end != -1 else ""
                if inner and inner.upper() != "M":
                    tokens.append(inner)
                term = (term[:start] + (term[end + 1:] if end != -1 else "")).strip()
            if not term:
                continue
            parts = term.split()
            # leading numeric stoichiometric coefficient
            if len(parts) > 1:
                try:
                    float(parts[0])
                    parts = parts[1:]
                except ValueError:
                    pass
            name = " ".join(parts)
            if name and name.upper() != "M":
                tokens.append(name)
    return tokens


def _clean_cantera_error(msg: str) -> str:
    """Trim Cantera's noisy multi-line error to the informative core."""
    if not msg:
        return "(no error message)"
    lines = [ln.rstrip() for ln in msg.strip().splitlines() if ln.strip()]
    # keep the most relevant tail (Cantera puts the reason near the end)
    keep = [ln for ln in lines
            if not ln.startswith(("Traceback", "  File ", "    "))]
    return "\n".join(keep[-12:]) if keep else "\n".join(lines[-12:])


# ===========================================================================
# Text rendering
# ===========================================================================
_SEV_ICON = {"error": "✗", "warning": "▲", "info": "•"}


def render_text_report(report: CheckReport, color: bool = False) -> str:
    def c(txt, code):
        return f"\033[{code}m{txt}\033[0m" if color else txt
    sev_color = {"error": "31", "warning": "33", "info": "36"}

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"Mechanism health check: {report.mechanism}")
    lines.append("=" * 72)
    if not report.loaded:
        lines.append(c("LOAD FAILED", "31;1"))
    if report.meta:
        m = report.meta
        lines.append(f"  species={m.get('n_species','?')}  "
                     f"reactions={m.get('n_reactions','?')}  "
                     f"elements={m.get('n_elements','?')}")
    cnt = report.counts()
    summary = (f"{cnt['error']} error(s), {cnt['warning']} warning(s), "
               f"{cnt['info']} info")
    verdict = c("PASS", "32;1") if report.ok else c("FAIL", "31;1")
    lines.append(f"  verdict: {verdict}   ({summary})")
    lines.append("")

    if not report.findings:
        lines.append("  No issues found. ✓")
        return "\n".join(lines)

    # group by check
    by_check: Dict[str, List[Finding]] = {}
    for f in report.sorted_findings():
        by_check.setdefault(f.check, []).append(f)

    for check, items in by_check.items():
        lines.append(c(f"── {check}  ({len(items)}) ".ljust(72, "─"), "1"))
        for f in items:
            icon = _SEV_ICON.get(f.severity, "•")
            head = c(f"  {icon} [{f.severity}] {f.message}", sev_color.get(f.severity, "0"))
            lines.append(head)
            if f.location:
                lines.append(f"        at: {f.location}")
            if f.fix_options:
                auto = f.auto_fix
                if auto:
                    lines.append(f"        fix (auto): {auto.label}")
                else:
                    labels = " | ".join(o.label for o in f.fix_options)
                    lines.append(f"        options: {labels}")
        lines.append("")
    return "\n".join(lines)
