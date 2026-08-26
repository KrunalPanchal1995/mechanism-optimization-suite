"""
rmip/reduction/lumping.py
=========================
Phase 8 (Future.md / FEATURE_ROADMAP #10) — a-priori chemical / isomer lumping.

Why this exists
---------------
The other half of the "second stage" of reduction (the first being CSP /
QSS identification, :mod:`rmip.reduction.csp`).  Detailed mechanisms carry many
**isomers** — species with identical elemental composition but different
structure (e.g. the C₇H₁₅ heptyl radicals, the C₄H₉ butyls).  When such isomers
are close in thermochemistry they interconvert fast and behave, for the bulk
oxidation, as a single *lumped pseudo-species*.  Merging them is where the
order-of-magnitude species-count reduction needed for CFD comes from.

Method (Matrisciano et al. 2022, a-priori thermodynamic lumping)
----------------------------------------------------------------
1. Group species by exact elemental composition → isomer classes.
2. Within each class, single-linkage cluster on standard-state thermochemistry
   at 298.15 K: two species join the same lump when both
       |ΔH_f° | ≤ dHf_tol   and   |ΔS° | ≤ dS_tol.
   (Thermodynamically similar isomers → one lump; a class can split into several
   lumps if its members are thermochemically far apart.)
3. Each lump of ≥ 2 members proposes a pseudo-species; its representative is the
   most stable member (lowest ΔH_f°).  Reduction = Σ (lump_size − 1) species.

Honest scope
------------
This module *proposes* thermodynamically-consistent lump groups and quantifies
the species reduction they enable.  It does **not** refit lumped Arrhenius rate
expressions or emit a lumped mechanism file (the rate-lumping / re-fitting step,
and the resulting reduced ``.yaml``) — that is a documented follow-on, mirroring
the analytic-QSSA code-gen deferral in :mod:`rmip.reduction.csp`.  The groups
returned here are exactly the input a lumped-rate fitter would consume.

Usage (Python API)
------------------
>>> from rmip.reduction.lumping import ChemicalLumper
>>> lumper = ChemicalLumper("project.db")
>>> res = lumper.propose(mech_id=1, dHf_tol_kJ=10.0, dS_tol=20.0)
>>> [g["members"] for g in res.groups]
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rmip.types import LumpingResult

log = logging.getLogger(__name__)

_R = 8314.462618  # J/kmol/K (Cantera molar units)
_T_REF = 298.15


# ---------------------------------------------------------------------------
# Pure helpers (no Cantera, no DB) — the grouping logic is unit-testable alone
# ---------------------------------------------------------------------------

def composition_key(composition: Dict[str, float]) -> Tuple[Tuple[str, int], ...]:
    """
    Canonical, hashable elemental-composition key: sorted (element, int-count)
    pairs with zero-count elements dropped.  Two species share a key iff they are
    isomers (identical molecular formula).
    """
    return tuple(sorted(
        (el, int(round(n))) for el, n in composition.items() if round(n) != 0
    ))


def total_atoms(composition: Dict[str, float]) -> int:
    """Total heavy+light atom count of a composition."""
    return int(round(sum(composition.values())))


def single_linkage_lumps(
    members: Sequence[str],
    hf: Sequence[float],
    s: Sequence[float],
    dHf_tol: float,
    dS_tol: float,
) -> List[List[str]]:
    """
    Single-linkage clustering of one isomer class on thermochemical similarity.

    Two species are linked when |ΔHf| ≤ ``dHf_tol`` and |ΔS| ≤ ``dS_tol`` (same
    units as ``hf`` / ``s``).  Connected components (union-find) become lumps.
    Members are returned within each lump sorted by ascending ``hf`` (most stable
    first), and the list of lumps is deterministic (sorted by first member).
    """
    n = len(members)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(hf[i] - hf[j]) <= dHf_tol and abs(s[i] - s[j]) <= dS_tol:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    lumps: List[List[str]] = []
    for idxs in groups.values():
        idxs.sort(key=lambda k: hf[k])       # most stable first
        lumps.append([members[k] for k in idxs])
    lumps.sort(key=lambda g: g[0])
    return lumps


# ---------------------------------------------------------------------------
# Cantera-facing extraction
# ---------------------------------------------------------------------------

def species_thermo_298(gas) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard-state ΔH_f° [J/kmol] and S° [J/kmol/K] at 298.15 K, 1 atm for every
    species, as arrays aligned with ``gas.species_names``.

    Evaluated on a throwaway copy so the caller's gas state is untouched.
    """
    import cantera as ct

    g = ct.Solution(thermo=gas.thermo_model, kinetics="none",
                    species=gas.species())
    g.TP = _T_REF, ct.one_atm
    hf = np.asarray(g.standard_enthalpies_RT, dtype=float) * _R * _T_REF
    s = np.asarray(g.standard_entropies_R, dtype=float) * _R
    return hf, s


# ---------------------------------------------------------------------------
# DB-backed lumper
# ---------------------------------------------------------------------------

class ChemicalLumper:
    """
    Propose a-priori thermodynamic lumping groups for a stored mechanism.

    Parameters
    ----------
    db_path : str
        Path to the RMIP SQLite database.
    """

    METHOD = "thermo_lumping"

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def propose(
        self,
        mech_id:    int,
        dHf_tol_kJ: float = 10.0,
        dS_tol:     float = 20.0,
        min_atoms:  int = 4,
        exclude:    Optional[Sequence[str]] = None,
        save_db:    bool = True,
    ) -> LumpingResult:
        """
        Run the lumping proposal.

        Parameters
        ----------
        dHf_tol_kJ : float
            Max ΔH_f° spread within a lump [kJ/mol].
        dS_tol : float
            Max S° spread within a lump [J/mol/K].
        min_atoms : int
            Skip isomer classes with fewer than ``min_atoms`` atoms (small,
            uninteresting species — e.g. never lump diatomics).
        exclude : sequence[str], optional
            Species never lumped.
        """
        t0 = time.perf_counter()
        gas = self._load_gas(mech_id)
        names = list(gas.species_names)
        hf_all, s_all = species_thermo_298(gas)                 # J/kmol, J/kmol/K
        # Work in kJ/mol and J/mol/K to match the tolerances (÷1000 twice / ÷1000).
        hf_kJmol = hf_all / 1e6
        s_Jmol = s_all / 1e3
        excluded = set(exclude or ())

        # 1. isomer classes
        classes: Dict[Tuple, List[int]] = {}
        for i, name in enumerate(names):
            comp = gas.species(i).composition
            if name in excluded or total_atoms(comp) < min_atoms:
                continue
            classes.setdefault(composition_key(comp), []).append(i)

        # 2/3. cluster each multi-member class, keep lumps of size ≥ 2
        groups: List[dict] = []
        for key, idxs in classes.items():
            if len(idxs) < 2:
                continue
            members = [names[i] for i in idxs]
            hf = [hf_kJmol[i] for i in idxs]
            s = [s_Jmol[i] for i in idxs]
            for lump in single_linkage_lumps(members, hf, s, dHf_tol_kJ, dS_tol):
                if len(lump) < 2:
                    continue
                lidx = [names.index(m) for m in lump]
                lhf = [hf_kJmol[i] for i in lidx]
                ls = [s_Jmol[i] for i in lidx]
                groups.append({
                    "formula":       _formula_str(key),
                    "members":       lump,
                    "representative": lump[0],           # lowest ΔHf (most stable)
                    "n_members":     len(lump),
                    "dHf_spread_kJ": float(max(lhf) - min(lhf)),
                    "dS_spread":     float(max(ls) - min(ls)),
                    "Hf_kJmol":      [round(x, 3) for x in lhf],
                })

        groups.sort(key=lambda g: (-g["n_members"], g["formula"]))
        n_reduced = sum(g["n_members"] - 1 for g in groups)

        result = LumpingResult(
            mechanism_id    = mech_id,
            method          = self.METHOD,
            dHf_tol_kJ      = dHf_tol_kJ,
            dS_tol          = dS_tol,
            min_atoms       = min_atoms,
            n_species_full  = gas.n_species,
            n_lumps         = len(groups),
            n_species_removed = n_reduced,
            n_species_lumped  = gas.n_species - n_reduced,
            groups          = groups,
            excluded_species = sorted(excluded),
            cpu_time_s      = time.perf_counter() - t0,
        )
        result.notes.append(
            "Groups are a-priori thermodynamic lumping proposals; lumped-rate "
            "re-fitting and reduced-mechanism export are a documented follow-on."
        )
        if not groups:
            result.notes.append(
                "No isomer classes met the thermochemical tolerances — the "
                "mechanism has no (or only thermochemically dissimilar) isomers."
            )

        if save_db:
            result.run_id = _insert_lumping_run(self._db_path, result)

        log.info(
            "Lumping mech=%d: %d lump(s) over %d species → −%d species "
            "(%d after lumping)",
            mech_id, len(groups), gas.n_species, n_reduced,
            result.n_species_lumped,
        )
        return result

    def _load_gas(self, mech_id: int):
        from rmip.simulation.base import MechanismLoader
        return MechanismLoader(self._db_path).load(mech_id)


# ---------------------------------------------------------------------------
# Lumped-mechanism export (Phase-8 follow-on)
# ---------------------------------------------------------------------------

def export_lumped(
    db_path:   str,
    result:    LumpingResult,
    out_yaml:  Optional[str] = None,
    new_name:  Optional[str] = None,
    dedup:     bool = True,
) -> int:
    """
    Materialise the lumped mechanism implied by a lumping proposal: collapse each
    group's members onto its representative species, producing a new
    ``format='reduced'`` mechanism row, and optionally emit it as a Cantera YAML.

    Honest scope (documented follow-on): this collapses the *species* (members →
    representative, keeping the representative's thermo) and remaps stoichiometry.
    It does **not re-fit lumped Arrhenius rates** — reactions differing only by a
    collapsed isomer become identical and are, by default, de-duplicated (first
    kept); a proper lumped-rate fit (summing member fluxes) is the follow-on.

    Returns the new mechanism id.
    """
    from rmip.reduction.materialize import materialize_subset, mechanism_lists

    if result.mechanism_id is None:
        raise ValueError("LumpingResult has no mechanism_id — cannot export.")
    names, formulas = mechanism_lists(db_path, result.mechanism_id)
    lump_map = {g["representative"]: g["members"] for g in result.groups}
    name = new_name or f"lumped_{result.mechanism_id}"

    new_id = materialize_subset(
        db_path, result.mechanism_id, species=names, reactions=formulas,
        new_name=name, lump_map=lump_map,
        description=f"Thermodynamically lumped mechanism {result.mechanism_id} "
                    f"({result.n_lumps} lumps, −{result.n_species_removed} species)",
        metadata={"lump_groups": [g["members"] for g in result.groups]},
        dedup_reactions=dedup,
    )
    if out_yaml:
        from rmip.io.cantera_writer import CanteraYAMLWriter
        CanteraYAMLWriter(db_path).write(new_id, out_yaml)
        log.info("Lumped mechanism written → %s", out_yaml)
    return new_id


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _formula_str(key: Tuple[Tuple[str, int], ...]) -> str:
    """Hill-ish formula string from a composition key, e.g. C7H15."""
    return "".join(f"{el}{n if n != 1 else ''}" for el, n in key)


def _insert_lumping_run(db_path: str, result: LumpingResult) -> Optional[int]:
    """Insert a ``lumping_runs`` row (lazily creating the table). Returns id."""
    import json
    from datetime import datetime, timezone
    from rmip.database.migrations import run_reduction_analysis_migrations, _table_exists

    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "lumping_runs"):
            run_reduction_analysis_migrations(conn)
        cur = conn.execute(
            """
            INSERT INTO lumping_runs
                (mechanism_id, method, dHf_tol_kJ, dS_tol, min_atoms,
                 n_species_full, n_lumps, n_species_removed, n_species_lumped,
                 groups_json, excluded_species_json, notes_json, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result.mechanism_id, result.method, result.dHf_tol_kJ,
                result.dS_tol, result.min_atoms, result.n_species_full,
                result.n_lumps, result.n_species_removed, result.n_species_lumped,
                json.dumps(result.groups),
                json.dumps(result.excluded_species),
                json.dumps(result.notes),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cur.lastrowid
