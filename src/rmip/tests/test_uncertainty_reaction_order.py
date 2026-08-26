"""
rmip/tests/test_uncertainty_reaction_order.py
================================================
Regression coverage for a bug found 2026-07-17 while investigating a
recurring "Rxn ... not in the mechanism" AssertionError seen both in
forward_uq_gui's real-mechanism sweep and in
test_gsa_prs_crosscheck.py::TestFullCrossCheckEndToEnd::test_run_crosscheck_completes_on_real_mechanism
(a previously pre-existing, unrelated-looking failure -- same reaction,
same assertion, same module in both cases).

Root cause: `Uncertainty.reaction.__init__`/`Uncertainty.PLOG.__init__` (both
src/MUQ-SAC/Uncertainty.py and src/Sensitivity_Analysis/Uncertainty.py,
byte-identical for this code path) matched an uncertainty XML's `rxn=`
attribute against `MechanismParser.Parser(mechPath).rxnList` via exact
string equality. `rxnList` reads each reaction's `equation:` field VERBATIM
from the mechanism YAML (preserving the source CHEMKIN file's own reactant
order, e.g. "O + H2 <=> H + OH") -- but Cantera's own
`gas.reaction_equations()`, loading that SAME file, reorders reactant/
product species internally and returns a different string for the
identical physical reaction (e.g. "H2 + O <=> H + OH", confirmed via direct
Cantera inspection of a real fused Keromnes/Li/Davis mechanism). Any
uncertainty XML populated from a live Solution object then fails the
exact-string check even though the reaction is genuinely present.

Fix: fall back to an order-independent canonical form (sort each side's
species tokens) before giving up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


def _fresh_uncertainty_module(subdir: str):
    mod_dir = str(_REPO / "src" / subdir)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    # Both subdirs define a top-level `Uncertainty` module (and it
    # cross-imports same-named siblings like MechanismParser) -- force a
    # fresh import so we don't accidentally test whichever copy a prior
    # test already cached.
    for name in ("Uncertainty", "MechanismParser"):
        sys.modules.pop(name, None)
    import Uncertainty
    return Uncertainty


@pytest.mark.parametrize("subdir", ["MUQ-SAC", "Sensitivity_Analysis"])
class TestCanonRxnEquation:
    def test_reordered_reactants_are_equal(self, subdir):
        mod = _fresh_uncertainty_module(subdir)
        assert mod._canon_rxn_equation("O + H2 <=> H + OH") == \
            mod._canon_rxn_equation("H2 + O <=> H + OH")

    def test_genuinely_different_reactions_are_not_equal(self, subdir):
        mod = _fresh_uncertainty_module(subdir)
        assert mod._canon_rxn_equation("O + H2 <=> H + OH") != \
            mod._canon_rxn_equation("O + H2O <=> 2 OH")


@pytest.mark.parametrize("subdir", ["MUQ-SAC", "Sensitivity_Analysis"])
def test_reaction_class_matches_reordered_equation(subdir, tmp_path):
    """End-to-end: build a minimal mechanism YAML whose stored `equation:`
    uses one reactant order, then an uncertainty XML `<reaction>` element
    whose `rxn=` attribute uses the OTHER order (mimicking a real
    Cantera-derived string) -- Uncertainty.reaction must resolve it to the
    correct index instead of raising."""
    import xml.etree.ElementTree as ET

    mod = _fresh_uncertainty_module(subdir)

    mech_yaml = tmp_path / "mech.yaml"
    mech_yaml.write_text(
        "reactions:\n"
        "- equation: O + H2 <=> H + OH\n"
        "  rate-constant: {A: 5.08e+04, b: 2.67, Ea: 6290.0}\n"
        "- equation: H2 + O2 <=> HO2 + H\n"
        "  rate-constant: {A: 5.176e+05, b: 2.433, Ea: 53502.0}\n"
    )

    element = ET.Element("reaction", {"rxn": "H2 + O <=> H + OH", "no": "1"})
    for tag, text in (
        ("class", "Elementary"),
        ("type", "Elementary"),
        ("perturbation_type", "factor"),
        ("perturbation_factor", "2.0"),
        ("data_type", "constant;end_points"),
        ("temp", "300,2500"),
        ("unsrt", "1.5,1.5"),
    ):
        child = ET.SubElement(element, tag)
        child.text = text

    r = mod.reaction(element, str(mech_yaml), {})

    # Pre-fix: this would raise AssertionError("Rxn H2 + O <=> H + OH not
    # in the mechanism...") since the exact string never appears in
    # rxnList (which stores "O + H2 <=> H + OH").
    assert r.index == 0


@pytest.mark.parametrize("subdir", ["MUQ-SAC", "Sensitivity_Analysis"])
class TestCanonRxnEquationStoichShorthand:
    """Round-2 audit (WORKBENCH_AUDIT_2026-08-10.md, defect B3): duplicated-
    species notation ("OH+OH<=>O+H2O") vs Cantera's stoichiometric shorthand
    ("2 OH <=> O + H2O"), plus reversible reactions where different
    mechanisms disagree on which side is "forward" -- both confirmed
    against the real FFCM1.yaml/GRI_CO.yaml equation strings in this repo's
    own MECHANISM/ directory."""

    def test_duplicated_species_matches_stoich_shorthand(self, subdir):
        mod = _fresh_uncertainty_module(subdir)
        assert mod._canon_rxn_equation("OH+OH<=>O+H2O") == \
            mod._canon_rxn_equation("2 OH <=> O + H2O")

    def test_reversible_direction_disagreement_still_matches(self, subdir):
        """FFCM1.yaml (Reaction 6): 'H2 + M <=> 2 H + M'.
        GRI_CO.yaml (Reaction 18): '2 H + M <=> H2 + M'.
        Same physical (reversible) reaction, opposite formal direction."""
        mod = _fresh_uncertainty_module(subdir)
        assert mod._canon_rxn_equation("H2 + M <=> 2 H + M") == \
            mod._canon_rxn_equation("2 H + M <=> H2 + M")
        # And the old duplicated-species spelling of the same reaction:
        assert mod._canon_rxn_equation("H+H+M<=>H2+M") == \
            mod._canon_rxn_equation("H2 + M <=> 2 H + M")

    def test_irreversible_direction_is_not_conflated(self, subdir):
        """Unlike '<=>', '=>' is one-directional -- swapping sides must
        NOT be treated as the same reaction."""
        mod = _fresh_uncertainty_module(subdir)
        assert mod._canon_rxn_equation("CH2 + O => 2 H + CO") != \
            mod._canon_rxn_equation("2 H + CO => CH2 + O")

    def test_still_distinguishes_genuinely_different_stoichiometry(self, subdir):
        mod = _fresh_uncertainty_module(subdir)
        assert mod._canon_rxn_equation("2 OH <=> O + H2O") != \
            mod._canon_rxn_equation("3 OH <=> O + H2O + OH")

    def test_falloff_parens_notation_matches_plain_third_body(self, subdir):
        """FFCM1.yaml (Reaction 15): 'H + O2 (+M) <=> HO2 (+M)' (Cantera
        falloff notation) must match the same reaction spelled with the
        plain-reactant third-body notation an uncertainty XML commonly
        uses: 'H+O2+M<=>HO2+M'. Found live: the shipped H2_uncertainty.xml
        uses the latter for exactly this <type>pressure_dependent</type>
        reaction and failed "not in the mechanism" against FFCM1.yaml
        before this fix."""
        mod = _fresh_uncertainty_module(subdir)
        assert mod._canon_rxn_equation("H + O2 (+M) <=> HO2 (+M)") == \
            mod._canon_rxn_equation("H+O2+M<=>HO2+M")
