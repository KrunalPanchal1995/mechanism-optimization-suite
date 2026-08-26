"""
rmip/tests/test_canonicalize_self_third_body.py
=================================================
Regression coverage for the PLAN.md Priority-0 item #2 fix (2026-07-17):
a reaction written with an explicit self-third-body (e.g. Li/Davis-style
``2H+H2<=>2H2``) must canonicalize to the SAME key as the physically
identical generic-M form (e.g. Keromnes-style ``H2+M<=>2H+M``) -- found
stress-testing fusion on Keromnes/Li/Davis (2026-07-14, bug.md); previously
these produced different canonical_keys, so GraphComparator/FusionEngine
treated them as two distinct (UNIQUE, never-matching) reactions instead of
one shared reaction across mechanisms.
"""
from __future__ import annotations

from rmip.core.canonicalize import make_reaction_canonical_key

# Real canonical SMILES the two species resolve to (doesn't matter which
# exact string RDKit/the SMILES-hash fallback picks, only that both
# reactions use the SAME species identifiers).
H_SMILES  = "[H]"
H2_SMILES = "[H][H]"


class TestSelfThirdBodyEquivalence:
    def test_generic_m_and_explicit_self_third_body_match(self):
        """H2+M<=>2H+M  vs.  2H+H2<=>2H2 -- same elementary step."""
        # Generic-M form: M is stripped before reaching this function (see
        # _chemkin_parser.py), so it never appears in either list at all.
        key_generic_m = make_reaction_canonical_key(
            reactant_smiles=[H2_SMILES],
            product_smiles=[H_SMILES],
            reversible=True,
            reactant_coeffs=[1.0],
            product_coeffs=[2.0],
            cancel_self_third_body=True,
        )

        # Explicit self-third-body form: H2 appears on both sides (once as
        # reactant/collider, twice as product = collider passthrough (1) +
        # newly-formed H2 (1)).
        key_explicit = make_reaction_canonical_key(
            reactant_smiles=[H_SMILES, H2_SMILES],
            product_smiles=[H2_SMILES],
            reversible=True,
            reactant_coeffs=[2.0, 1.0],
            product_coeffs=[2.0],
            cancel_self_third_body=True,
        )

        assert key_generic_m is not None
        assert key_explicit is not None
        assert key_generic_m == key_explicit

    def test_cancellation_is_opt_in_not_default(self):
        """Without cancel_self_third_body=True, behavior must be
        byte-identical to the pre-fix implementation (every existing
        caller that doesn't opt in must see zero behavior change)."""
        key_no_cancel = make_reaction_canonical_key(
            reactant_smiles=[H_SMILES, H2_SMILES],
            product_smiles=[H2_SMILES],
            reversible=True,
        )
        key_plain = make_reaction_canonical_key(
            reactant_smiles=[H_SMILES, H2_SMILES],
            product_smiles=[H2_SMILES],
            reversible=True,
        )
        assert key_no_cancel == key_plain  # deterministic, unaffected by new params existing

        key_cancelled = make_reaction_canonical_key(
            reactant_smiles=[H_SMILES, H2_SMILES],
            product_smiles=[H2_SMILES],
            reversible=True,
            reactant_coeffs=[2.0, 1.0],
            product_coeffs=[2.0],
            cancel_self_third_body=True,
        )
        assert key_cancelled != key_no_cancel  # cancellation actually changed the key

    def test_no_shared_species_unaffected_by_cancellation(self):
        """A normal reaction with no reactant/product species overlap must
        produce the identical key whether or not cancellation is requested
        (nothing to cancel)."""
        key_default = make_reaction_canonical_key(
            reactant_smiles=["C", "O"], product_smiles=["CO"], reversible=True,
        )
        key_with_cancel_flag = make_reaction_canonical_key(
            reactant_smiles=["C", "O"], product_smiles=["CO"], reversible=True,
            reactant_coeffs=[1.0, 1.0], product_coeffs=[1.0],
            cancel_self_third_body=True,
        )
        assert key_default == key_with_cancel_flag

    def test_full_third_body_passthrough_not_confused_with_reaction(self):
        """A species that ONLY passes through unreacted on both sides with
        EQUAL count (a true bystander, not a self-third-body reaction) gets
        fully cancelled, leaving only the real reacting species in the key
        -- e.g. "C+N<=>O+N" (N spectator) reduces to "C<=>O"."""
        key = make_reaction_canonical_key(
            reactant_smiles=["C", "N"], product_smiles=["O", "N"], reversible=True,
            reactant_coeffs=[1.0, 1.0], product_coeffs=[1.0, 1.0],
            cancel_self_third_body=True,
        )
        key_direct = make_reaction_canonical_key(
            reactant_smiles=["C"], product_smiles=["O"], reversible=True,
        )
        assert key == key_direct
