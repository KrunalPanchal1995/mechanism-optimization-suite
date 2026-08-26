"""
rmip/tests/test_reaction_family_kb.py
=====================================
Tests for targeted RMG-style generation
(``rmip/generation/reaction_family_kb.py``, FEATURE_ROADMAP #2).

The RMG chemistry lives behind a subprocess backend; here that backend is
*injected* (``backend_fn``) with canned, RMG-shaped responses, so the whole
family-completeness diff + targeted-graft pipeline is exercised end-to-end
without RMG-Py installed. (A real RMG-backed smoke test would guard on the env,
like the cvxpy/emcee tests.)

Run with:
    python -m pytest src/rmip/tests/test_reaction_family_kb.py -v
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

rdkit = pytest.importorskip("rdkit")   # canonicalization needs RDKit (base env has it)

from rmip.core.canonicalize import make_reaction_canonical_key, smiles_to_inchi
from rmip.generation.reaction_family_kb import ReactionFamilyKB, FamilyScanReport


_SPECIES = {"H2": "[H][H]", "O2": "O=O", "H": "[H]", "O": "[O]",
            "OH": "[OH]", "H2O": "O"}


@pytest.fixture()
def kb_db(tmp_path):
    from rmip.database.schema import init_db
    db_path = str(tmp_path / "kb.db")
    conn = init_db(db_path)
    conn.execute("INSERT INTO mechanisms (name, format, loaded_at) VALUES (?,?,?)",
                 ("base", "chemkin", datetime.now(timezone.utc).isoformat()))
    mech_id = 1
    for name, sm in _SPECIES.items():
        inchi, ik = smiles_to_inchi(sm)
        conn.execute("INSERT INTO species (mechanism_id, name, canonical_smiles, "
                     "inchi, inchi_key, glossary_matched) VALUES (?,?,?,?,?,1)",
                     (mech_id, name, sm, inchi, ik))
    # one existing reaction: H2 + O <=> OH + H
    ck = make_reaction_canonical_key(["[H][H]", "[O]"], ["[OH]", "[H]"])
    conn.execute("INSERT INTO reactions (mechanism_id, formula_string, canonical_key, "
                 "reversible, kinetics_type) VALUES (?,?,?,1,'arrhenius')",
                 (mech_id, "H2 + O <=> OH + H", ck))
    conn.commit()
    conn.close()
    return db_path, mech_id


# ---------------------------------------------------------------------------
# family_scan diff
# ---------------------------------------------------------------------------

def _scan_backend(cmd, request):
    assert cmd == "react"
    return {"reactions": [
        # already present (H2 + O <=> OH + H)
        {"reactants": ["[H][H]", "[O]"], "products": ["[OH]", "[H]"],
         "family": "H_Abstraction", "reversible": True,
         "arrhenius": {"A": 1e13, "n": 0.0, "Ea": 0.0}},
        # core-missing: all species exist, key absent (2 OH <=> H2O + O)
        {"reactants": ["[OH]", "[OH]"], "products": ["O", "[O]"],
         "family": "H_Abstraction", "reversible": True, "arrhenius": None},
        # edge-missing: introduces H2O2 (OO)
        {"reactants": ["[OH]", "[OH]"], "products": ["OO"],
         "family": "R_Recombination", "reversible": True,
         "arrhenius": {"A": 1e12, "n": 0.0, "Ea": 0.0}},
    ], "new_species": [{"smiles": "OO", "label": "H2O2"}]}


def test_family_scan_classifies_missing(kb_db):
    db_path, mech_id = kb_db
    kb = ReactionFamilyKB(db_path, backend_fn=_scan_backend)
    rep = kb.family_scan(mech_id, save_db=False)

    assert isinstance(rep, FamilyScanReport)
    assert rep.n_enumerated == 3
    # the present reaction is skipped; two are missing
    assert rep.n_missing_reactions == 2
    kinds = sorted(r["kind"] for r in rep.missing_reactions)
    assert kinds == ["core", "edge"]
    # H2O2 flagged as a missing species (from kinetic theory)
    h2o2_ik = smiles_to_inchi("OO")[1]
    assert any(s["inchi_key"] == h2o2_ik for s in rep.missing_species)
    assert set(rep.families_seen) == {"H_Abstraction", "R_Recombination"}


def test_family_scan_persists(kb_db):
    db_path, mech_id = kb_db
    kb = ReactionFamilyKB(db_path, backend_fn=_scan_backend)
    rep = kb.family_scan(mech_id, save_db=True)
    assert rep.run_id is not None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT n_missing_reactions, n_missing_species "
                           "FROM family_scan_runs WHERE id=?", (rep.run_id,)).fetchone()
    assert row == (2, 1)


# ---------------------------------------------------------------------------
# generate_targeted graft
# ---------------------------------------------------------------------------

def _gen_backend(cmd, request):
    assert cmd == "react"
    # reactions involving the seed H2O2 (OO)
    return {"reactions": [
        {"reactants": ["OO"], "products": ["[OH]", "[OH]"],
         "family": "R_Recombination", "reversible": True,
         "arrhenius": {"A": 2e12, "n": 0.0, "Ea": 1000.0}},
        {"reactants": ["OO", "[H]"], "products": ["[OH]", "O"],
         "family": "H_Abstraction", "reversible": True,
         "arrhenius": {"A": 1e13, "n": 0.0, "Ea": 3000.0}},
    ], "new_species": [{"smiles": "OO", "label": "H2O2"}]}


def test_generate_targeted_grafts_new_species(kb_db):
    db_path, mech_id = kb_db
    kb = ReactionFamilyKB(db_path, backend_fn=_gen_backend)
    new_id = kb.generate_targeted(mech_id, seed_smiles=["OO"], new_name="base+H2O2")
    assert new_id != mech_id

    h2o2_ik = smiles_to_inchi("OO")[1]
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sp = {r["inchi_key"] for r in conn.execute(
            "SELECT inchi_key FROM species WHERE mechanism_id=?", (new_id,))}
        n_rxn = conn.execute("SELECT COUNT(*) FROM reactions WHERE mechanism_id=?",
                             (new_id,)).fetchone()[0]
    # H2O2 grafted in, on top of the base's original species
    assert h2o2_ik in sp
    assert len(sp) == len(_SPECIES) + 1
    # base had 1 reaction; 2 targeted reactions grafted
    assert n_rxn == 1 + 2


def test_generate_targeted_reactions_only_adds_no_species(kb_db):
    db_path, mech_id = kb_db

    def backend(cmd, request):
        # one reaction among existing species only (H + OH <=> O + H2), plus one
        # that would introduce H2O2 — reactions_only must drop the latter.
        return {"reactions": [
            {"reactants": ["[H]", "[OH]"], "products": ["[O]", "[H][H]"],
             "family": "H_Abstraction", "reversible": True, "arrhenius": None},
            {"reactants": ["[OH]", "[OH]"], "products": ["OO"],
             "family": "R_Recombination", "reversible": True, "arrhenius": None},
        ], "new_species": [{"smiles": "OO", "label": "H2O2"}]}

    kb = ReactionFamilyKB(db_path, backend_fn=backend)
    # seed on an existing radical so the core reaction is kept
    new_id = kb.generate_targeted(mech_id, seed_smiles=["[OH]"],
                                  reactions_only=True, new_name="base+rxn")
    h2o2_ik = smiles_to_inchi("OO")[1]
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sp = {r["inchi_key"] for r in conn.execute(
            "SELECT inchi_key FROM species WHERE mechanism_id=?", (new_id,))}
    assert h2o2_ik not in sp                       # no new species added
    assert len(sp) == len(_SPECIES)


def test_generate_targeted_noop_when_nothing_matches(kb_db):
    db_path, mech_id = kb_db
    kb = ReactionFamilyKB(db_path, backend_fn=lambda c, r: {"reactions": [], "new_species": []})
    out = kb.generate_targeted(mech_id, seed_smiles=["OO"])
    assert out == mech_id                          # unchanged
