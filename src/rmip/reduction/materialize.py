"""
rmip/reduction/materialize.py
=============================
Shared primitive: turn a *subset* of an existing mechanism (a retained
species/reaction selection, optionally with an isomer-lump collapse map) into a
new first-class mechanism row in the DB, and optionally emit it as a Cantera
YAML.

Why this exists
---------------
Every reducer in the suite (DRG/DRGEP/PFA, plus the Phase-8 CSP/lumping
diagnostics) produces only *lists* — retained species names and retained
reaction identities, or lump groups — but nothing materialises those into a
runnable reduced mechanism.  The skeletal-vs-detailed error map (#11), the
CSP-QSSA export and the lumping export all need exactly the same operation:
"clone this mechanism, keeping only these species/reactions, collapsing these
isomer groups".  This module is that single operation, reused by all three.

It deliberately mirrors the row-copy clone pattern in
``generation/rmg_interface.py::_clone_if_needed`` (copy every column so thermo,
transport and kinetics survive), but filters to the retained subset and remaps
stoichiometry.

Honest scope
------------
This does a *structural* subset + isomer collapse: species rows are copied
verbatim (the representative's thermo/transport is used for a lump), reactions
whose species all survive are copied with remapped stoichiometry.  It does **not
re-fit lumped Arrhenius rates** and does not algebraically eliminate QSS species
— those remain the documented follow-ons in ``lumping.py`` / ``csp.py``.  Lump
collapse can produce duplicate or self-cancelling reactions; these are kept
as-is (rate-lumping / dedup is the follow-on).

Usage
-----
>>> from rmip.reduction.materialize import materialize_subset
>>> new_id = materialize_subset(
...     "project.db", base_mech_id=1,
...     species=step.retained_species, reactions=step.retained_reactions,
...     new_name="h2o2_skeletal")
>>> # optionally emit YAML:
>>> from rmip.io.cantera_writer import CanteraYAMLWriter
>>> CanteraYAMLWriter("project.db").write(new_id, "h2o2_skeletal.yaml")
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


def _rewrite_formula(formula: Optional[str], member_to_rep: Dict[str, str]) -> Optional[str]:
    """
    Replace collapsed isomer member names with their representative in a reaction
    equation string, matching whole species tokens only (so ``CH3O`` inside
    ``CH3OH`` is left alone).  Preserves third-body ``(+M)`` / arrow notation.
    """
    if not formula or not member_to_rep:
        return formula
    out = formula
    for member, rep in member_to_rep.items():
        out = re.sub(rf"(?<![A-Za-z0-9(]){re.escape(member)}(?![A-Za-z0-9)])",
                     rep, out)
    return out


def mechanism_lists(db_path: str, mech_id: int) -> Tuple[List[str], List[str]]:
    """
    Return ``(species_names, reaction_formula_strings)`` for a mechanism — the
    full retained-everything lists that CSP/lumping export pass to
    :func:`materialize_subset` after removing/collapsing a subset.  Raises if the
    mechanism has no parsed species rows (materialise needs the DB rows, not just
    a Cantera-loadable filepath).
    """
    with sqlite3.connect(db_path) as conn:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM species WHERE mechanism_id=?", (mech_id,))]
        formulas = [r[0] for r in conn.execute(
            "SELECT formula_string FROM reactions WHERE mechanism_id=?", (mech_id,))]
    if not names:
        raise ValueError(
            f"Mechanism {mech_id} has no species rows in the DB — export needs the "
            f"mechanism parsed into the DB (rmip parse ...), not only a "
            f"Cantera-loadable filepath.")
    return names, formulas


def materialize_subset(
    db_path:      str,
    base_mech_id: int,
    species:      Iterable[str],
    reactions:    Iterable[str],
    new_name:     str,
    *,
    lump_map:     Optional[Dict[str, List[str]]] = None,
    description:  Optional[str] = None,
    metadata:     Optional[dict] = None,
    dedup_reactions: bool = False,
    write_yaml:   bool = True,
) -> int:
    """
    Create a new ``format='reduced'`` mechanism = ``base_mech_id`` restricted to
    the given species/reactions (with optional isomer-lump collapse).

    Parameters
    ----------
    db_path : str
    base_mech_id : int
        Mechanism to subset.
    species : iterable[str]
        Species **names** to retain (within ``base_mech_id``).
    reactions : iterable[str]
        Reaction identities to retain — each matched against a reaction's
        ``formula_string`` *or* ``canonical_key``.
    new_name : str
        Name for the new mechanism row.
    lump_map : dict[str, list[str]], optional
        ``{representative_name: [member_names, ...]}`` — every member's
        stoichiometry is redirected onto the representative and the non-
        representative members are dropped from the species list.
    description, metadata : optional
        Stored on the new mechanism row (metadata merged into ``metadata_json``).

    Returns
    -------
    int — the new mechanism id.
    """
    from rmip.database.migrations import run_reduction_materialize_migrations

    retained_names: Set[str] = set(species)
    reaction_ids: Set[str] = set(reactions)

    # member -> representative, and ensure representatives are retained
    member_to_rep: Dict[str, str] = {}
    if lump_map:
        for rep, members in lump_map.items():
            retained_names.add(rep)
            for m in members:
                if m != rep:
                    member_to_rep[m] = rep
    # species actually stored = retained minus collapsed (non-representative) members
    stored_names = {n for n in retained_names if n not in member_to_rep}

    def resolve(name: str) -> Optional[str]:
        """Map a base species name to its stored (post-lump) name, or None."""
        mapped = member_to_rep.get(name, name)
        return mapped if mapped in stored_names else None

    with sqlite3.connect(db_path) as conn:
        run_reduction_materialize_migrations(conn)
        conn.row_factory = sqlite3.Row

        base_row = conn.execute(
            "SELECT * FROM mechanisms WHERE id=?", (base_mech_id,)
        ).fetchone()
        if base_row is None:
            raise ValueError(f"Mechanism id={base_mech_id} not found")

        # ── new mechanism row ────────────────────────────────────────────────
        meta = {"materialized_from": base_mech_id,
                "n_species_full": base_row["n_species"],
                "n_reactions_full": base_row["n_reactions"]}
        if lump_map:
            meta["lump_map"] = lump_map
        if metadata:
            meta.update(metadata)
        ts = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            """
            INSERT INTO mechanisms
                (name, format, filepath, description, n_species, n_reactions,
                 loaded_at, metadata_json)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (new_name, "reduced", base_row["filepath"],
             description or f"Materialized subset of mechanism {base_mech_id}",
             0, 0, ts, json.dumps(meta)),
        )
        new_mech_id = cur.lastrowid

        # ── copy retained species (name -> new species id) ───────────────────
        name_to_new_sp: Dict[str, int] = {}
        base_species = conn.execute(
            "SELECT * FROM species WHERE mechanism_id=?", (base_mech_id,)
        ).fetchall()
        base_sp_id_to_name = {r["id"]: r["name"] for r in base_species}
        for sp in base_species:
            if sp["name"] not in stored_names:
                continue
            d = dict(sp)
            d.pop("id")
            d["mechanism_id"] = new_mech_id
            cols = list(d.keys())
            c = conn.execute(
                f"INSERT INTO species ({','.join(cols)}) "
                f"VALUES ({','.join(['?'] * len(cols))})",
                [d[k] for k in cols],
            )
            name_to_new_sp[sp["name"]] = c.lastrowid

        # ── copy retained reactions + remapped stoichiometry ─────────────────
        # species that vanish from the subset (dropped, or lumped-away members);
        # a reaction naming any of them as an explicit third-body collider in its
        # equation (e.g. "H + O2 + AR <=> HO2 + AR") must be dropped too — the
        # collider is not in the stoichiometry so the participant check misses it.
        dropped_names = set(base_sp_id_to_name.values()) - stored_names

        def _names_a_dropped_species(formula: Optional[str]) -> bool:
            if not formula or not dropped_names:
                return False
            return any(
                re.search(rf"(?<![A-Za-z0-9(]){re.escape(dn)}(?![A-Za-z0-9)])", formula)
                for dn in dropped_names)

        n_rxn = 0
        n_skipped = 0
        n_deduped = 0
        n_null = 0
        seen_signatures: Set[tuple] = set()
        base_reactions = conn.execute(
            "SELECT * FROM reactions WHERE mechanism_id=?", (base_mech_id,)
        ).fetchall()
        for rxn in base_reactions:
            if (rxn["formula_string"] not in reaction_ids
                    and rxn["canonical_key"] not in reaction_ids):
                continue
            stoich = conn.execute(
                "SELECT species_id, role, stoich_coeff FROM stoichiometry "
                "WHERE reaction_id=?", (rxn["id"],)
            ).fetchall()
            # resolve every participant to a stored species; skip reaction if any
            # participant did not survive the subset/lump.
            mapped: List[Tuple[str, str, float]] = []
            ok = True
            for srow in stoich:
                base_name = base_sp_id_to_name.get(srow["species_id"])
                tgt = resolve(base_name) if base_name else None
                if tgt is None:
                    ok = False
                    break
                mapped.append((tgt, srow["role"], srow["stoich_coeff"]))
            if not ok:
                n_skipped += 1
                continue

            # explicit third-body collider (named in the equation) was dropped
            new_formula = _rewrite_formula(rxn["formula_string"], member_to_rep)
            if _names_a_dropped_species(new_formula):
                n_skipped += 1
                continue

            # lump collapse can make a reaction null (identical reactant/product
            # multiset after members fold onto a representative) — drop those.
            if member_to_rep:
                reac = sorted((n, c) for n, r, c in mapped if r == "reactant")
                prod = sorted((n, c) for n, r, c in mapped if r == "product")
                if reac == prod:
                    n_null += 1
                    continue

            # de-duplicate reactions that a lump collapse made identical (else
            # Cantera flags undeclared duplicates). Legitimately-marked
            # duplicates (duplicate=1) are kept — only accidental collapses drop.
            if dedup_reactions:
                sig = (tuple(sorted(mapped)),
                       rxn["kinetics_type"], rxn["has_third_body"])
                if sig in seen_signatures and not rxn["duplicate"]:
                    n_deduped += 1
                    continue
                seen_signatures.add(sig)

            d = dict(rxn)
            d.pop("id")
            d["mechanism_id"] = new_mech_id
            # equation string with representative names (unchanged for a plain
            # subset; collapsed-member names rewritten for a lump) so the writer
            # never emits a collapsed member.
            d["formula_string"] = new_formula
            cols = list(d.keys())
            c = conn.execute(
                f"INSERT INTO reactions ({','.join(cols)}) "
                f"VALUES ({','.join(['?'] * len(cols))})",
                [d[k] for k in cols],
            )
            new_rxn_id = c.lastrowid
            n_rxn += 1
            for tgt_name, role, coeff in mapped:
                conn.execute(
                    "INSERT INTO stoichiometry (reaction_id, species_id, stoich_coeff, role) "
                    "VALUES (?,?,?,?)",
                    (new_rxn_id, name_to_new_sp[tgt_name], coeff, role),
                )

            # third-body collider efficiencies (needed so Cantera can tell a
            # generic +M reaction apart from an explicit-collider variant — else
            # it flags undeclared duplicates). Remap collider species onto their
            # retained/lumped names; drop colliders that did not survive.
            for eff in conn.execute(
                "SELECT species_id, species_name, efficiency FROM third_body_efficiencies "
                "WHERE reaction_id=?", (rxn["id"],)
            ).fetchall():
                base_name = (base_sp_id_to_name.get(eff["species_id"])
                             if eff["species_id"] is not None else eff["species_name"])
                tgt = resolve(base_name) if base_name else None
                if tgt is None and eff["species_id"] is not None:
                    continue                    # collider species not retained
                new_sp_id = name_to_new_sp.get(tgt) if tgt else None
                conn.execute(
                    "INSERT INTO third_body_efficiencies "
                    "(reaction_id, species_id, species_name, efficiency) VALUES (?,?,?,?)",
                    (new_rxn_id, new_sp_id, tgt or eff["species_name"], eff["efficiency"]),
                )

        conn.execute(
            "UPDATE mechanisms SET n_species=?, n_reactions=? WHERE id=?",
            (len(name_to_new_sp), n_rxn, new_mech_id),
        )
        conn.commit()

    # Write a companion Cantera YAML and point the mechanism's filepath at it, so
    # the materialised mechanism is directly loadable by MechanismLoader (IDT/LFS/
    # CSP/… all resolve a 'reduced' mechanism through this file).
    if write_yaml:
        try:
            from pathlib import Path
            from rmip.io.cantera_writer import CanteraYAMLWriter
            outdir = Path(db_path).resolve().parent / ".rmip_reduced"
            outdir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", new_name)
            ypath = outdir / f"mech{new_mech_id}_{safe}.yaml"
            CanteraYAMLWriter(db_path).write(new_mech_id, str(ypath))
            with sqlite3.connect(db_path) as conn2:
                conn2.execute("UPDATE mechanisms SET filepath=? WHERE id=?",
                              (str(ypath), new_mech_id))
                conn2.commit()
        except Exception as exc:                       # noqa: BLE001
            log.warning(
                "materialize_subset: could not write companion YAML for mech %d "
                "(%s); rows are stored but the mechanism may not be directly "
                "loadable via filepath.", new_mech_id, exc)

    log.info(
        "materialize_subset: base=%d → new mech %d ('%s'): %d species, %d reactions "
        "(%d skipped, %d null-after-lump, %d de-duplicated)",
        base_mech_id, new_mech_id, new_name, len(name_to_new_sp), n_rxn,
        n_skipped, n_null, n_deduped,
    )
    return new_mech_id
