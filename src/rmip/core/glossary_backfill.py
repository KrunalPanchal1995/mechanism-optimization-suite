"""
rmip/core/glossary_backfill.py
===============================
Retroactive glossary resolution for already-ingested mechanisms.

KRUNAL_JULY_23.md point 6: fusing Davis-2005 + Keromenes-2013 + Li-2015
produced "Fused_Mech (fused, 0 sp / 0 rxn)". Root cause: those three
mechanisms were parsed into the DB (``rmip parse``) at some point WITHOUT
``--glossary`` being passed. ``MechanismParser._insert_species`` only calls
``lookup_smiles`` when a glossary dict was actually loaded
(``rmip/core/parser.py:232``) — with none given, EVERY species is stored with
a null ``canonical_smiles``/``inchi_key``, and ``_build_canonical_key``
(``parser.py:382``) then stores a null ``canonical_key`` for every reaction
too, since it requires ``canonical_smiles`` on every participant
(``parser.py:403``). Verified against the real DB: 100% of species in
mechanisms 2/3/4 have a blank ``inchi_key`` — yet ``species_smiles.csv``
already covers every one of them except the obscure excited-state "OHEX".

``GraphComparator`` already has a *deliberate, reviewed* safety fallback for
unresolved species/reactions (the "B4" fix in ``graph_comparator.py``): a
missing InChI/canonical key never invents a cross-mechanism match — it keys
per-mechanism instead, so an unresolved species classifies UNIQUE rather
than silently vanishing (the pre-B4 bug) or, worse, colliding with an
unrelated species that merely shares a name/formula. That conservative
choice is correct and is NOT changed here. This module instead fixes the
actual root cause: species that genuinely could have been resolved (the
glossary already lists them) but never were, because no glossary was
supplied at parse time. It backfills the SAME fields
``MechanismParser._insert_species``/``_build_canonical_key`` would have
written had ``--glossary`` been passed originally, and only ever fills a
gap — never overwrites an already-resolved row — so it's safe to run
repeatedly, and safe to run on a mechanism that's a mix of resolved and
unresolved species.

Usage::

    from rmip.core.glossary_backfill import backfill_glossary
    summary = backfill_glossary("project.db", [2, 3, 4], "species_smiles.csv")
    # {2: {"species_total": 14, "species_resolved": 14, "species_still_unresolved": 0,
    #      "reactions_total_unresolved_before": 38, "reactions_resolved": 38,
    #      "reactions_still_unresolved": 0}, ...}
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Dict, List, Union

from rmip.core.canonicalize import Canonicalizer, smiles_to_formula, smiles_to_mw
from rmip.io.glossary_reader import load_glossary, lookup_smiles

log = logging.getLogger("rmip.core.glossary_backfill")


def _connect(db: Union[str, sqlite3.Connection]) -> sqlite3.Connection:
    if isinstance(db, sqlite3.Connection):
        return db
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def _backfill_species(
    conn: sqlite3.Connection,
    mech_id: int,
    glossary: Dict[str, str],
    canon: Canonicalizer,
) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT id, name, canonical_smiles, inchi_key, formula, mw_kg_mol "
        "FROM species WHERE mechanism_id = ?", (mech_id,)
    ).fetchall()
    total = len(rows)
    unresolved_before = sum(1 for r in rows if not r["inchi_key"])
    resolved = 0
    for r in rows:
        if r["inchi_key"]:
            continue                              # never clobber an existing resolution
        smiles = lookup_smiles(glossary, r["name"])
        if not smiles:
            continue                              # genuinely not in the glossary
        can_smiles = canon.canonicalize(smiles)
        inchi, inchi_key = canon.to_inchi(can_smiles or smiles)
        if not inchi_key:
            continue                              # glossary SMILES didn't canonicalize
        formula = r["formula"] or smiles_to_formula(smiles)
        mw = r["mw_kg_mol"] if r["mw_kg_mol"] is not None else smiles_to_mw(smiles)
        conn.execute(
            "UPDATE species SET canonical_smiles=?, source_smiles=?, inchi=?, "
            "inchi_key=?, glossary_matched=1, formula=?, mw_kg_mol=? WHERE id=?",
            (can_smiles, smiles, inchi, inchi_key, formula, mw, r["id"]),
        )
        resolved += 1
        log.info("Backfilled species '%s' (mech %d) -> inchi_key=%s",
                  r["name"], mech_id, inchi_key)
    return {
        "species_total": total,
        "species_unresolved_before": unresolved_before,
        "species_resolved": resolved,
        "species_still_unresolved": unresolved_before - resolved,
    }


def _backfill_reactions(
    conn: sqlite3.Connection,
    mech_id: int,
    canon: Canonicalizer,
) -> Dict[str, int]:
    rxns = conn.execute(
        "SELECT id, reversible FROM reactions "
        "WHERE mechanism_id = ? AND (canonical_key IS NULL OR canonical_key = '')",
        (mech_id,),
    ).fetchall()
    total_unresolved_before = len(rxns)
    resolved = 0
    for rx in rxns:
        parts = conn.execute(
            "SELECT sp.canonical_smiles AS smiles, st.stoich_coeff AS coeff, st.role "
            "FROM stoichiometry st JOIN species sp ON sp.id = st.species_id "
            "WHERE st.reaction_id = ? ORDER BY st.id",
            (rx["id"],),
        ).fetchall()
        r_smiles = [p["smiles"] for p in parts if p["role"] == "reactant"]
        p_smiles = [p["smiles"] for p in parts if p["role"] == "product"]
        if not parts or any(s is None for s in r_smiles + p_smiles):
            continue                              # still missing a species resolution
        r_coeffs = [p["coeff"] for p in parts if p["role"] == "reactant"]
        p_coeffs = [p["coeff"] for p in parts if p["role"] == "product"]
        can_key = canon.reaction_key(
            r_smiles, p_smiles, reversible=bool(rx["reversible"]),
            reactant_coeffs=r_coeffs, product_coeffs=p_coeffs,
            cancel_self_third_body=True,
        )
        if not can_key:
            continue
        conn.execute("UPDATE reactions SET canonical_key=? WHERE id=?",
                     (can_key, rx["id"]))
        resolved += 1
    return {
        "reactions_total_unresolved_before": total_unresolved_before,
        "reactions_resolved": resolved,
        "reactions_still_unresolved": total_unresolved_before - resolved,
    }


def backfill_glossary(
    db: Union[str, sqlite3.Connection],
    mechanism_ids: List[int],
    glossary_path: str,
) -> Dict[int, Dict[str, int]]:
    """Fill in missing species inchi_key / reaction canonical_key for each
    mechanism in ``mechanism_ids``, using ``glossary_path``. Only ever fills
    a gap (null/blank inchi_key or canonical_key) — never overwrites an
    already-resolved row, so this is idempotent and safe to re-run, including
    on a mechanism that already has partial glossary coverage.

    Two passes per call (species for ALL requested mechanisms, then
    reactions for ALL requested mechanisms) so that a reaction whose
    participants span... no, reactions only ever reference species within
    their own mechanism, but doing species-then-reactions in that order
    (rather than interleaved per mechanism) keeps the two concerns clearly
    separated and mirrors MechanismParser's own parse-time ordering
    (_insert_species before _insert_reactions).

    Returns ``{mechanism_id: {species_total, species_resolved,
    species_still_unresolved, reactions_total_unresolved_before,
    reactions_resolved, reactions_still_unresolved}}``.
    """
    conn = _connect(db)
    glossary = load_glossary(glossary_path)
    canon = Canonicalizer()
    summary: Dict[int, Dict[str, int]] = {}
    for mech_id in mechanism_ids:
        summary[mech_id] = _backfill_species(conn, mech_id, glossary, canon)
    for mech_id in mechanism_ids:
        summary[mech_id].update(_backfill_reactions(conn, mech_id, canon))
    conn.commit()
    return summary
