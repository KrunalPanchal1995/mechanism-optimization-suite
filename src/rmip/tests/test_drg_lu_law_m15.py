"""
rmip/tests/test_drg_lu_law_m15.py
==================================
Regression test for M15 (bug.md / review.md deep review).

DRG (Lu & Law, 2005) must retain a reaction only when *every* species it
involves is retained. The reducer used to keep a reaction whenever it shared
*any* species with the retained set (``retained.intersection(sp_set)``), which
leaves reactions that reference removed species -- an internally inconsistent
skeletal mechanism that then fails to build / materialize. The fix requires the
full subset (``sp_set <= retained``).

The reducer had no test coverage before; this pins the corrected semantics at
the exact locus of the bug (``_reduce_at_epsilon``).
"""
from __future__ import annotations

from rmip.reduction.drg import DRGReducer


def _make_reducer_with_reactions(rxn_map):
    r = DRGReducer(":memory:")
    r._rxn_formula_to_species = dict(rxn_map)
    return r


def test_reaction_kept_only_if_all_species_retained():
    """A → B is strongly coupled; C and D are unreachable and get removed.
    Only the reaction whose species are *all* retained ({A,B}) survives."""
    reducer = _make_reducer_with_reactions({
        "rxn_AB": {"A", "B"},   # all retained  -> keep
        "rxn_AC": {"A", "C"},   # A retained, C removed -> DROP (was wrongly kept)
        "rxn_CD": {"C", "D"},   # none retained -> drop
        "rxn_BD": {"B", "D"},   # B retained, D removed -> DROP (was wrongly kept)
    })

    all_species   = {"A", "B", "C", "D"}
    all_reactions = {"rxn_AB", "rxn_AC", "rxn_CD", "rxn_BD"}
    r_AB = {("A", "B"): 0.9}          # only A→B exceeds eps
    eps  = 0.1

    retained_sp, retained_rx, _levels = reducer._reduce_at_epsilon(
        all_species, all_reactions, stoich={}, r_AB=r_AB,
        target_species=["A"], eps=eps,
    )

    # Propagation keeps only A and B.
    assert retained_sp == {"A", "B"}

    # Lu-Law: only the all-retained reaction survives.
    assert retained_rx == {"rxn_AB"}

    # The core M15 invariant: no retained reaction references a removed species.
    removed = all_species - retained_sp
    for formula in retained_rx:
        assert not (reducer._rxn_formula_to_species[formula] & removed), (
            f"retained reaction {formula} references removed species "
            f"{reducer._rxn_formula_to_species[formula] & removed}"
        )


def test_empty_species_reaction_not_retained():
    """A degenerate reaction with no species must never be retained (guards the
    `set() <= retained` == True corner of the subset test)."""
    reducer = _make_reducer_with_reactions({"empty": set(), "real": {"A"}})
    _sp, retained_rx, _ = reducer._reduce_at_epsilon(
        {"A"}, {"empty", "real"}, stoich={}, r_AB={},
        target_species=["A"], eps=0.1,
    )
    assert "empty" not in retained_rx
    assert retained_rx == {"real"}
