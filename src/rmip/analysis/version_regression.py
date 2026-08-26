"""
rmip/analysis/version_regression.py
===================================
W8b — version-behaviour regression harness (P28, P15).

The hardest version bumps to survive are the ones that change **numbers, not
APIs**: Cantera 2.6→3.x shifting a laminar flame speed by a few percent (P28),
or an RMG upgrade quietly producing a different model (P15). This harness makes
those shifts *deliberate*: a canonical case suite (small mechanisms ×
{IDT, LFS, equilibrium} at fixed conditions) with committed golden outputs;
``run_suite`` executes the suite against a chosen Cantera environment and
``diff_against_golden`` reports every observable that moved beyond tolerance,
with the magnitude.

Environment isolation
---------------------
The actual simulations run in :mod:`rmip.analysis._regression_worker`, a
self-contained script (stdlib + cantera only, **no** rmip imports). Passing
``env_python=/path/to/other/venv/bin/python`` therefore runs the identical
suite under a *different Cantera version* without that env needing rmip
installed — the project's subprocess+JSON isolation pattern.

Golden discipline
-----------------
Goldens live in ``goldens/version_regression_golden.json`` next to this module
and are only ever (re)written through :func:`bless` (CLI ``--bless``) — an
audited, explicit action; a drift is never silently absorbed.

The same diff engine (:func:`diff_result_sets`) is deliberately generic over
two ``{case: {observables}}`` payloads, so pointing it at the outputs of two
RMG builds gives the behavioural half of the W7d reproducibility diff
(``rmip rmg-diff`` covers the structural half).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from rmip.simulation.subprocess_util import sanitized_subprocess_env

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_regression_worker.py")
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "goldens", "version_regression_golden.json")

# Per-task relative tolerances: how far an observable may move between two
# environments before it is reported as drift. IDT is the most integrator-
# sensitive (5%); flame speeds are grid-converged tighter (2%); equilibrium
# is thermo-only and essentially deterministic (0.5%).
DEFAULT_TOLERANCES: Dict[str, float] = {
    "idt": 0.05,
    "lfs": 0.02,
    "equilibrium": 0.005,
}
# Below this absolute magnitude the relative delta is measured against the
# floor instead (avoids 1e-30 mole fractions exploding the relative error).
_ABS_FLOOR = 1.0e-12

# ---------------------------------------------------------------------------
# Canonical case suite — small, deterministic, spans the three observable
# classes users validate against. Mechanisms are the ones bundled with every
# Cantera install, so the suite runs identically in any candidate env.
# ---------------------------------------------------------------------------
CANONICAL_SUITE: List[Dict[str, Any]] = [
    {"id": "h2o2-equilibrium-phi1", "mechanism": "h2o2.yaml",
     "task": "equilibrium", "T_K": 300.0, "P_atm": 1.0, "phi": 1.0,
     "fuel": "H2:1.0", "track_species": ["H2O", "OH", "O2"]},
    {"id": "h2o2-idt-1100K-1atm", "mechanism": "h2o2.yaml",
     "task": "idt", "T_K": 1100.0, "P_atm": 1.0, "phi": 1.0,
     "fuel": "H2:1.0", "t_end_s": 0.05},
    {"id": "h2o2-lfs-phi1-1atm", "mechanism": "h2o2.yaml",
     "task": "lfs", "T_K": 300.0, "P_atm": 1.0, "phi": 1.0,
     "fuel": "H2:1.0", "width_m": 0.03},
    {"id": "gri30-equilibrium-phi1", "mechanism": "gri30.yaml",
     "task": "equilibrium", "T_K": 300.0, "P_atm": 1.0, "phi": 1.0,
     "fuel": "CH4:1.0", "track_species": ["CO2", "H2O", "CO"]},
    {"id": "gri30-idt-1500K-1atm", "mechanism": "gri30.yaml",
     "task": "idt", "T_K": 1500.0, "P_atm": 1.0, "phi": 1.0,
     "fuel": "CH4:1.0", "t_end_s": 0.02},
    {"id": "gri30-lfs-phi1-1atm", "mechanism": "gri30.yaml",
     "task": "lfs", "T_K": 300.0, "P_atm": 1.0, "phi": 1.0,
     "fuel": "CH4:1.0", "width_m": 0.03},
]


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------
@dataclass
class Drift:
    case_id: str
    observable: str                 # "" for case-level statuses
    status: str                     # ok | drift | failed | missing | new
    golden: Optional[float] = None
    current: Optional[float] = None
    rel_delta: Optional[float] = None
    tolerance: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class RegressionReport:
    entries: List[Drift] = field(default_factory=list)
    meta_current: Dict[str, Any] = field(default_factory=dict)
    meta_golden: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(e.status == "ok" for e in self.entries)

    @property
    def n_drift(self) -> int:
        return sum(1 for e in self.entries if e.status != "ok")

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok,
                "meta_current": self.meta_current,
                "meta_golden": self.meta_golden,
                "entries": [e.to_dict() for e in self.entries]}

    def to_text(self) -> str:
        lines = ["=" * 72, "Version-behaviour regression report", "=" * 72]
        cur = self.meta_current or {}
        gold = self.meta_golden or {}
        lines.append(f"  current: Cantera {cur.get('cantera_version', '?')} "
                     f"({cur.get('executable', sys.executable)})")
        lines.append(f"  golden : Cantera {gold.get('cantera_version', '?')} "
                     f"(blessed {gold.get('blessed_at', '?')})")
        lines.append("")
        for e in self.entries:
            if e.status == "ok":
                lines.append(f"  ✓ {e.case_id:32s} {e.observable:12s} "
                             f"{e.current:.6g}  (Δ {e.rel_delta:.2%} "
                             f"≤ tol {e.tolerance:.1%})")
            elif e.status == "drift":
                lines.append(f"  ✗ {e.case_id:32s} {e.observable:12s} "
                             f"golden {e.golden:.6g} → current {e.current:.6g}  "
                             f"(Δ {e.rel_delta:.2%} > tol {e.tolerance:.1%})")
            else:
                lines.append(f"  ✗ {e.case_id:32s} {e.observable or '-':12s} "
                             f"[{e.status}] {e.detail}")
        lines.append("")
        n_bad = self.n_drift
        lines.append("  verdict: " + ("PASS — behaviour matches golden."
                                      if n_bad == 0 else
                                      f"FAIL — {n_bad} drifted/failed entr"
                                      f"{'y' if n_bad == 1 else 'ies'}."))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_suite(
    env_python: Optional[str] = None,
    cases: Optional[Sequence[Dict[str, Any]]] = None,
    timeout_s: float = 3600.0,
) -> Dict[str, Any]:
    """Execute the case suite in a subprocess and return the worker payload.

    Parameters
    ----------
    env_python : str, optional
        Python executable of the Cantera environment to test (default: this
        interpreter). The target env only needs cantera — not rmip.
    cases : sequence of dict, optional
        Case definitions (default: :data:`CANONICAL_SUITE`).
    """
    cases = list(cases) if cases is not None else list(CANONICAL_SUITE)
    py = env_python or sys.executable
    with tempfile.TemporaryDirectory(prefix="rmip_regression_") as tmp:
        cases_path = os.path.join(tmp, "cases.json")
        results_path = os.path.join(tmp, "results.json")
        with open(cases_path, "w") as fh:
            json.dump({"cases": cases}, fh)
        proc = subprocess.run(
            [py, _WORKER, cases_path, results_path],
            capture_output=True, text=True, timeout=timeout_s,
            env=sanitized_subprocess_env(),
        )
        if proc.returncode != 0 or not os.path.exists(results_path):
            raise RuntimeError(
                f"regression worker failed (rc={proc.returncode}) under {py}:\n"
                + (proc.stderr or proc.stdout or "").strip()[-2000:])
        with open(results_path) as fh:
            return json.load(fh)


# ---------------------------------------------------------------------------
# Diffing — generic over any two {case_id: {"observables": {...}}} payloads
# ---------------------------------------------------------------------------
def diff_result_sets(
    current: Dict[str, Any],
    reference: Dict[str, Any],
    tolerances: Optional[Dict[str, float]] = None,
    case_tasks: Optional[Dict[str, str]] = None,
) -> List[Drift]:
    """Compare two result sets; return one :class:`Drift` per observable.

    ``case_tasks`` maps case_id → task name for tolerance lookup; when a case
    is unknown the loosest default tolerance applies.
    """
    tol_map = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol_map.update(tolerances)
    fallback_tol = max(tol_map.values())
    case_tasks = case_tasks or {}
    entries: List[Drift] = []

    for cid in sorted(set(reference) | set(current)):
        ref = reference.get(cid)
        cur = current.get(cid)
        if ref is None:
            entries.append(Drift(cid, "", "new",
                                 detail="case has no golden entry — bless to add"))
            continue
        if cur is None:
            entries.append(Drift(cid, "", "missing",
                                 detail="case missing from current run"))
            continue
        if not cur.get("ok", True):
            entries.append(Drift(cid, "", "failed",
                                 detail=cur.get("error") or "case failed"))
            continue
        if not ref.get("ok", True):
            entries.append(Drift(cid, "", "new",
                                 detail="golden entry is a failure — re-bless"))
            continue
        tol = tol_map.get(case_tasks.get(cid, ""), fallback_tol)
        ref_obs = ref.get("observables", {})
        cur_obs = cur.get("observables", {})
        for name in sorted(set(ref_obs) | set(cur_obs)):
            if name not in cur_obs:
                entries.append(Drift(cid, name, "missing",
                                     detail="observable missing from current run"))
                continue
            if name not in ref_obs:
                entries.append(Drift(cid, name, "new",
                                     detail="observable has no golden value"))
                continue
            g, c = float(ref_obs[name]), float(cur_obs[name])
            rel = abs(c - g) / max(abs(g), _ABS_FLOOR)
            entries.append(Drift(
                cid, name, "ok" if rel <= tol else "drift",
                golden=g, current=c, rel_delta=rel, tolerance=tol))
    return entries


def diff_against_golden(
    payload: Dict[str, Any],
    golden_path: Optional[str] = None,
    tolerances: Optional[Dict[str, float]] = None,
    cases: Optional[Sequence[Dict[str, Any]]] = None,
) -> RegressionReport:
    """Diff a :func:`run_suite` payload against the committed golden file."""
    golden_path = golden_path or GOLDEN_PATH
    if not os.path.exists(golden_path):
        raise FileNotFoundError(
            f"no golden file at {golden_path} — run with --bless once to "
            "create it (audited, explicit)")
    with open(golden_path) as fh:
        golden = json.load(fh)
    case_tasks = {c["id"]: c["task"]
                  for c in (cases if cases is not None else CANONICAL_SUITE)}
    entries = diff_result_sets(payload.get("results", {}),
                               golden.get("results", {}),
                               tolerances=tolerances, case_tasks=case_tasks)
    return RegressionReport(entries=entries,
                            meta_current=payload.get("meta", {}),
                            meta_golden=golden.get("meta", {}))


def bless(payload: Dict[str, Any], golden_path: Optional[str] = None) -> str:
    """Write a suite payload as the new golden (the ONLY writer of goldens)."""
    import datetime
    golden_path = golden_path or GOLDEN_PATH
    failed = [cid for cid, r in payload.get("results", {}).items()
              if not r.get("ok")]
    if failed:
        raise RuntimeError(
            "refusing to bless a payload with failed cases: " + ", ".join(failed))
    os.makedirs(os.path.dirname(golden_path), exist_ok=True)
    doc = {"meta": {**payload.get("meta", {}),
                    "blessed_at": datetime.datetime.now().isoformat(timespec="seconds")},
           "results": payload.get("results", {})}
    with open(golden_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return golden_path
