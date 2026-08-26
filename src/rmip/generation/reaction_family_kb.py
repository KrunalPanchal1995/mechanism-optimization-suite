"""
rmip/generation/reaction_family_kb.py
=====================================
FEATURE_ROADMAP #2 — targeted RMG-style generation (the scoped, non-full-RMG
angle the user asked for).

Two capabilities over a mechanism already in the DB:

1. **family_scan** — build the reaction-family knowledge base for the mechanism's
   species and diff it against what the mechanism contains:
     * *missing reactions* — reactions the families admit among species already in
       the mechanism, whose canonical key is absent;
     * *missing species* — species the families would produce (edge products) that
       the mechanism does not contain ("missing from kinetic theory").
2. **generate_targeted** — for a chosen seed species set, enumerate one generation
   of their core (among existing species) + edge (creating new species) reactions
   and graft them into the mechanism, reusing the existing
   :meth:`RMGInterface.append_to_mechanism`.

The heavy chemistry (family templates, rate rules) lives in RMG-Py, reached
through :mod:`_rmg_backend` in a dedicated env (subprocess+JSON, D7).  The diff
and DB-insert logic here is pure and testable with an injected backend — the
``backend_fn`` constructor hook returns canned responses so the whole pipeline
runs without RMG installed.

Honest scope
------------
Targeted single-generation enumeration + family-completeness diff + graft — NOT a
full flux-driven RMG model-generation run (that is ``rmg_interface.generate``).
Rate estimates come from RMG's rate rules as-is (no re-fitting).

Activating the live RMG backend
--------------------------------
This module (and ``_rmg_backend.py`` / ``rmg_supervisor.py``) is complete and
exercised end-to-end via an injected stub/fake backend (``backend_fn=`` /
``command_template=``) — no RMG installation is needed to run its test suite.
The **live** path (the real subprocess call into RMG-Py) is currently
non-runnable on this machine, because:

1. ``/data/OPEN_CODE/RMG_py`` exists on disk but is **not Cython-built**
   (36 ``.pyx`` sources, 0 compiled ``.so`` extensions).
2. The dedicated conda env it expects,
   ``/home/krithika/anaconda3/envs/rmg_env/bin/python``, does not exist.
3. No RMG-database checkout exists anywhere under ``/data/OPEN_CODE``.

To activate the live path on a machine that has the disk space (RMG-Py +
its database together run to several GB):

1. ``cd /data/OPEN_CODE/RMG_py && make`` — builds the Cython extensions.
2. ``conda env create -f environment.yml`` (from that same checkout) to
   create ``rmg_env``, then pass its interpreter as ``rmg_python=`` /
   CLI ``--rmg-python``.
3. Obtain the separate RMG-database repository and pass its path as
   ``rmg_database=`` / CLI ``--rmg-database`` (or set the
   ``RMG_DATABASE_PATH`` env var — see ``_rmg_backend.py::_load_database``
   for the exact resolution order).

Until then, ``ReactionFamilyKB`` without a ``backend_fn`` will raise a
``RuntimeError`` naming the missing interpreter path, rather than hang or
fail silently.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# RMG conda env python (D7 convention — module default, overridable per call/CLI).
_RMG_PYTHON_DEFAULT = "/home/krithika/anaconda3/envs/rmg_env/bin/python"


@dataclass
class FamilyScanReport:
    """Result of :meth:`ReactionFamilyKB.family_scan`."""
    mechanism_id:      int
    n_species:         int
    n_enumerated:      int
    missing_reactions: List[Dict[str, Any]] = field(default_factory=list)
    missing_species:   List[Dict[str, Any]] = field(default_factory=list)
    families_seen:     List[str] = field(default_factory=list)
    notes:             List[str] = field(default_factory=list)
    run_id:            Optional[int] = None

    @property
    def n_missing_reactions(self) -> int:
        return len(self.missing_reactions)

    @property
    def n_missing_species(self) -> int:
        return len(self.missing_species)

    def to_dict(self) -> dict:
        return {
            "mechanism_id":      self.mechanism_id,
            "n_species":         self.n_species,
            "n_enumerated":      self.n_enumerated,
            "n_missing_reactions": self.n_missing_reactions,
            "n_missing_species": self.n_missing_species,
            "missing_reactions": self.missing_reactions,
            "missing_species":   self.missing_species,
            "families_seen":     self.families_seen,
            "notes":             self.notes,
            "run_id":            self.run_id,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


class ReactionFamilyKB:
    """Family knowledge base + targeted generation for a stored mechanism."""

    def __init__(
        self,
        db_path:      str,
        rmg_python:   Optional[str] = None,
        rmg_database: Optional[str] = None,
        backend_fn:   Optional[Callable[[str, dict], dict]] = None,
    ) -> None:
        self._db_path = db_path
        self._rmg_python = rmg_python or _RMG_PYTHON_DEFAULT
        self._rmg_database = rmg_database
        self._backend_fn = backend_fn          # injected for tests; else subprocess

    # ── backend bridge ────────────────────────────────────────────────────────

    def _call(self, cmd: str, request: dict, work_dir: Optional[str] = None) -> dict:
        if self._rmg_database and "rmg_database" not in request:
            request = {**request, "rmg_database": self._rmg_database}
        if self._backend_fn is not None:
            return self._backend_fn(cmd, request)
        import subprocess
        from rmip.simulation.subprocess_util import rmg_subprocess_env
        backend = Path(__file__).resolve().parent / "_rmg_backend.py"
        wd = Path(work_dir or "rmg_gen")
        wd.mkdir(parents=True, exist_ok=True)
        req_path = wd / f"rmg_{cmd}_request.json"
        resp_path = wd / f"rmg_{cmd}_response.json"
        with open(req_path, "w") as fh:
            json.dump(request, fh)
        proc = subprocess.run([self._rmg_python, str(backend), cmd,
                               str(req_path), str(resp_path)],
                              capture_output=True, text=True,
                              env=rmg_subprocess_env(self._rmg_python))
        if proc.returncode != 0:
            raise RuntimeError(
                f"RMG backend {cmd!r} failed (rc={proc.returncode}).\n"
                f"Is the RMG env present at {self._rmg_python}?\n{proc.stderr[-4000:]}")
        with open(resp_path) as fh:
            return json.load(fh)

    # ── public API ────────────────────────────────────────────────────────────

    def family_scan(
        self,
        mech_id:       int,
        only_families: Optional[Sequence[str]] = None,
        save_db:       bool = True,
    ) -> FamilyScanReport:
        """Enumerate one generation of family reactions for the mechanism's
        species and diff against what the mechanism already contains."""
        from rmip.core.canonicalize import make_reaction_canonical_key, smiles_to_inchi

        species = self._mech_species(mech_id)
        present_ik = {s["inchi_key"] for s in species if s["inchi_key"]}
        present_ck = self._mech_reaction_keys(mech_id)
        with_smiles = [s for s in species if s["smiles"]]

        req = {"species": [{"label": s["name"], "smiles": s["smiles"]}
                           for s in with_smiles]}
        if only_families:
            req["families"] = list(only_families)
        resp = self._call("react", req)
        reactions = resp.get("reactions", [])

        missing_rxn: List[Dict] = []
        missing_sp: Dict[str, Dict] = {}
        families_seen = set()
        for rxn in reactions:
            families_seen.add(rxn.get("family"))
            r_sm, p_sm = rxn["reactants"], rxn["products"]
            ck = make_reaction_canonical_key(r_sm, p_sm,
                                             rxn.get("reversible", True))
            if ck in present_ck:
                continue
            # inchi keys of all participants
            parts = [(sm, smiles_to_inchi(sm)[1]) for sm in r_sm + p_sm]
            new_parts = [(sm, ik) for sm, ik in parts if ik not in present_ik]
            entry = {"canonical_key": ck, "reactants": r_sm, "products": p_sm,
                     "family": rxn.get("family"), "arrhenius": rxn.get("arrhenius")}
            if not new_parts:
                entry["kind"] = "core"          # among existing species
                missing_rxn.append(entry)
            else:
                entry["kind"] = "edge"          # introduces new species
                missing_rxn.append(entry)
                for sm, ik in new_parts:
                    if ik and ik not in missing_sp:
                        missing_sp[ik] = {"smiles": sm, "inchi_key": ik}

        report = FamilyScanReport(
            mechanism_id     = mech_id,
            n_species        = len(species),
            n_enumerated     = len(reactions),
            missing_reactions = missing_rxn,
            missing_species  = list(missing_sp.values()),
            families_seen    = sorted(f for f in families_seen if f),
        )
        report.notes.append(
            "Targeted one-generation family enumeration + completeness diff; rate "
            "estimates are RMG rate-rule values (not re-fitted).")
        if save_db:
            report.run_id = _insert_family_scan(self._db_path, report)
        log.info("family_scan mech=%d: %d enumerated → %d missing reactions, "
                 "%d missing species", mech_id, len(reactions),
                 report.n_missing_reactions, report.n_missing_species)
        return report

    def generate_targeted(
        self,
        mech_id:         int,
        seed_smiles:     Sequence[str],
        reactions_only:  bool = False,
        new_name:        Optional[str] = None,
        only_families:   Optional[Sequence[str]] = None,
    ) -> int:
        """
        Enumerate one generation of reactions for ``seed_smiles`` (in the context
        of the mechanism's existing species) and graft them in.

        ``reactions_only`` keeps only reactions whose participants are all already
        present (adds no new species) — the "I just need those reactions" mode.
        Returns the grafted mechanism id.
        """
        from rmip.core.canonicalize import make_reaction_canonical_key, smiles_to_inchi
        from rmip.generation.rmg_interface import RMGInterface

        species = self._mech_species(mech_id)
        present_ik = {s["inchi_key"] for s in species if s["inchi_key"]}
        seed_ik = {smiles_to_inchi(sm)[1] for sm in seed_smiles}

        # enumerate in the context of existing species + seeds
        ctx = [{"label": s["name"], "smiles": s["smiles"]}
               for s in species if s["smiles"]]
        ctx += [{"label": sm, "smiles": sm} for sm in seed_smiles]
        req = {"species": ctx}
        if only_families:
            req["families"] = list(only_families)
        resp = self._call("react", req)

        # keep reactions that involve at least one seed
        kept, gen_species = [], {}
        for rxn in resp.get("reactions", []):
            parts = rxn["reactants"] + rxn["products"]
            part_ik = {sm: smiles_to_inchi(sm)[1] for sm in parts}
            if not (seed_ik & set(part_ik.values())):
                continue
            new_ik = {ik for ik in part_ik.values() if ik not in present_ik}
            if reactions_only and (new_ik - seed_ik):
                continue                        # would introduce a non-seed species
            kept.append(rxn)
            for sm, ik in part_ik.items():
                if ik not in present_ik and ik not in gen_species:
                    gen_species[ik] = sm

        if not kept:
            log.info("generate_targeted mech=%d: nothing to add for seeds %s",
                     mech_id, list(seed_smiles))
            return mech_id

        # materialise the generated chemistry as a temporary RMG mechanism, then
        # graft with the existing append_to_mechanism.
        rmg_mech_id, add_ik, add_ck = self._insert_generated(
            f"rmg_targeted_{mech_id}", gen_species, kept)
        target = RMGInterface(self._db_path).append_to_mechanism(
            base_mech_id=mech_id, rmg_mech_id=rmg_mech_id,
            missing_inchi_keys=add_ik, missing_canonical_keys=add_ck,
            new_name=new_name)
        log.info("generate_targeted mech=%d: grafted %d species, %d reactions → mech %d",
                 mech_id, len(add_ik), len(add_ck), target)
        return target

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _mech_species(self, mech_id: int) -> List[Dict]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT name, canonical_smiles AS smiles, inchi_key "
                "FROM species WHERE mechanism_id=?", (mech_id,)).fetchall()
        return [dict(r) for r in rows]

    def _mech_reaction_keys(self, mech_id: int) -> set:
        with sqlite3.connect(self._db_path) as conn:
            return {r[0] for r in conn.execute(
                "SELECT canonical_key FROM reactions WHERE mechanism_id=? "
                "AND canonical_key IS NOT NULL", (mech_id,))}

    def _insert_generated(
        self, name: str, gen_species: Dict[str, str], reactions: List[Dict],
    ) -> Tuple[int, List[str], List[str]]:
        """
        Insert generated species+reactions as a new mechanism and return
        ``(mech_id, added_inchi_keys, added_canonical_keys)`` for the graft.
        Only the *new* species (``gen_species``) are inserted; existing species
        are resolved by inchi_key against the base at graft time.
        """
        from rmip.core.canonicalize import (
            make_reaction_canonical_key, smiles_to_inchi, smiles_to_formula, smiles_to_mw,
        )
        added_ik: List[str] = []
        added_ck: List[str] = []
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                "INSERT INTO mechanisms (name, format, description, n_species, "
                "n_reactions, loaded_at) VALUES (?,?,?,?,?,?)",
                (name, "chemkin", "RMG targeted one-generation enumeration",
                 0, 0, datetime.now(timezone.utc).isoformat()))
            rmg_mech = cur.lastrowid

            # every species referenced by the kept reactions must exist in this
            # RMG mechanism so stoichiometry can resolve; insert both new and
            # (placeholder) existing species by inchi_key.
            sm_to_ik: Dict[str, str] = {}
            all_smiles = set()
            for rxn in reactions:
                all_smiles.update(rxn["reactants"])
                all_smiles.update(rxn["products"])
            sp_id: Dict[str, int] = {}
            for sm in all_smiles:
                inchi, ik = smiles_to_inchi(sm)
                sm_to_ik[sm] = ik
                c = conn.execute(
                    "INSERT INTO species (mechanism_id, name, canonical_smiles, "
                    "inchi, inchi_key, formula, mw_kg_mol, glossary_matched) "
                    "VALUES (?,?,?,?,?,?,?,1)",
                    (rmg_mech, sm, sm, inchi, ik, smiles_to_formula(sm),
                     smiles_to_mw(sm)))
                sp_id[sm] = c.lastrowid
                if ik in gen_species_ik(gen_species):
                    added_ik.append(ik)

            for rxn in reactions:
                r_sm, p_sm = rxn["reactants"], rxn["products"]
                ck = make_reaction_canonical_key(r_sm, p_sm,
                                                 rxn.get("reversible", True))
                arr = rxn.get("arrhenius") or {}
                formula = " + ".join(r_sm) + (" <=> " if rxn.get("reversible", True)
                                              else " => ") + " + ".join(p_sm)
                rc = conn.execute(
                    "INSERT INTO reactions (mechanism_id, formula_string, "
                    "canonical_key, reversible, family, kinetics_type, A_si, n, Ea_si, "
                    "rmg_comment) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (rmg_mech, formula, ck, 1 if rxn.get("reversible", True) else 0,
                     rxn.get("family"), "arrhenius", arr.get("A"), arr.get("n"),
                     arr.get("Ea"), "RMG targeted enumeration"))
                rxn_id = rc.lastrowid
                added_ck.append(ck)
                for sm in r_sm:
                    conn.execute("INSERT INTO stoichiometry (reaction_id, species_id, "
                                 "stoich_coeff, role) VALUES (?,?,?,'reactant')",
                                 (rxn_id, sp_id[sm], 1.0))
                for sm in p_sm:
                    conn.execute("INSERT INTO stoichiometry (reaction_id, species_id, "
                                 "stoich_coeff, role) VALUES (?,?,?,'product')",
                                 (rxn_id, sp_id[sm], 1.0))
            conn.execute("UPDATE mechanisms SET n_species=?, n_reactions=? WHERE id=?",
                         (len(all_smiles), len(reactions), rmg_mech))
            conn.commit()
        # dedupe added keys, preserve order
        return rmg_mech, list(dict.fromkeys(added_ik)), list(dict.fromkeys(added_ck))


def gen_species_ik(gen_species: Dict[str, str]) -> set:
    """The inchi_key set of the newly-generated (non-existing) species."""
    return set(gen_species.keys())


def _insert_family_scan(db_path: str, report: FamilyScanReport) -> Optional[int]:
    from rmip.database.migrations import run_family_scan_migrations, _table_exists
    with sqlite3.connect(db_path) as conn:
        if not _table_exists(conn, "family_scan_runs"):
            run_family_scan_migrations(conn)
        cur = conn.execute(
            """INSERT INTO family_scan_runs
               (mechanism_id, n_species, n_enumerated, n_missing_reactions,
                n_missing_species, missing_reactions_json, missing_species_json,
                families_seen_json, notes_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (report.mechanism_id, report.n_species, report.n_enumerated,
             report.n_missing_reactions, report.n_missing_species,
             json.dumps(report.missing_reactions), json.dumps(report.missing_species),
             json.dumps(report.families_seen), json.dumps(report.notes),
             datetime.now(timezone.utc).isoformat()))
        return cur.lastrowid
