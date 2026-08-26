"""
W5c — multiphase composition sanity (P25) + P4 surface-site balancing tests.

Acceptance (from W5_surface_heterogeneous.md):
- the liquid-composition diagnostic reproduces and explains the P25
  order-of-magnitude gap (fixtures/liquid_p25.yaml has water's density
  wrong by exactly 10x);
- balance_surface_sites turns a Cantera-rejected site-unbalanced surface
  mechanism into one that loads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rmip.analysis.multiphase_sanity import MultiphaseSanityChecker  # noqa: E402
from rmip.analysis.surface_sites import balance_surface_sites        # noqa: E402
from rmip.analysis.mechanism_check import MechanismChecker           # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
LIQUID = str(FIXTURES / "liquid_p25.yaml")
CHAR = str(FIXTURES / "char_oxidation.yaml")


# ===========================================================================
# W5c — multiphase sanity
# ===========================================================================
def test_p25_ten_x_gap_reproduced_and_explained():
    findings = MultiphaseSanityChecker(LIQUID).findings()
    hits = [f for f in findings
            if f.category == "pure-concentration-mismatch"]
    assert len(hits) == 1
    f = hits[0]
    assert "H2O(l)" in f.location
    # The diagnostic quantifies the 10x gap…
    assert f.details["log10_ratio"] == pytest.approx(-1.0, abs=0.01)
    assert f.details["c_ref_mol_L"] == pytest.approx(55.35)
    assert f.details["c_pure_mol_L"] == pytest.approx(5.53, abs=0.05)
    # …and explains the likely cause with the densities side by side.
    assert "density" in f.message
    assert "10.0x" in f.message and "below" in f.message


def test_correct_species_not_flagged():
    findings = MultiphaseSanityChecker(LIQUID).findings()
    assert not any("C2H5OH" in f.location for f in findings)


def test_internal_identity_holds_on_consistent_phase():
    """ideal-condensed is internally consistent — probe 1 must be silent
    (it exists to catch the activity-model/composition disagreement class)."""
    findings = MultiphaseSanityChecker(LIQUID).findings()
    assert not any(f.category == "concentration-identity" for f in findings)


def test_single_species_bulk_phase_checked():
    """char_oxidation.yaml's C(b) bulk: correct graphite density, and not in
    the liquid reference table — no findings at all."""
    findings = MultiphaseSanityChecker(CHAR).findings()
    assert findings == []


def test_pure_gas_mechanism_yields_no_findings():
    import cantera as ct
    h2o2 = None
    for d in ct.get_data_directories():
        cand = Path(d) / "h2o2.yaml"
        if cand.is_file():
            h2o2 = str(cand)
            break
    assert h2o2, "h2o2.yaml missing from Cantera data path"
    assert MultiphaseSanityChecker(h2o2).findings() == []


def test_wired_into_rmip_check_battery():
    checker = MechanismChecker(LIQUID)
    report = checker.run(checks=["multiphase_sanity"])
    cats = [f.category for f in report.findings]
    assert "pure-concentration-mismatch" in cats
    # Warning-only: the finding must never fail the verdict.
    assert all(f.severity == "warning" for f in report.findings)
    assert report.ok


def test_implausible_density_flagged(tmp_path):
    doc = yaml.safe_load(open(LIQUID))
    for sp in doc["species"]:
        if sp["name"] == "H2O(l)":
            sp["equation-of-state"]["density"] = "0.00001 g/cm^3"
    p = tmp_path / "absurd.yaml"
    yaml.safe_dump(doc, open(p, "w"), sort_keys=False)
    findings = MultiphaseSanityChecker(str(p)).findings()
    assert any(f.category == "implausible-density" for f in findings)


# ===========================================================================
# P4 — balance_surface_sites
# ===========================================================================
def _load_char_doc():
    return yaml.safe_load(open(CHAR))


def _make_site_unbalanced_doc():
    """The classic P4 authoring mistake: the freed site is dropped from the
    product side.  (s) is made an empty-composition site and O(s) plain
    adsorbed O so element balance still holds — only sites are broken."""
    doc = _load_char_doc()
    for sp in doc["species"]:
        if sp["name"] == "(s)":
            sp["composition"] = {}
        if sp["name"] == "O(s)":
            sp["composition"] = {"O": 1}
    doc["surface-reactions"][1]["equation"] = "O(s) + C(b) => CO"
    return doc


def _loads_in_cantera(doc, tmp_path, tag) -> bool:
    import cantera as ct
    p = tmp_path / f"{tag}.yaml"
    yaml.safe_dump(doc, open(p, "w"), sort_keys=False)
    try:
        gas = ct.Solution(str(p), "gas")
        bulk = ct.Solution(str(p), "char")
        ct.Interface(str(p), "char_surface", [gas, bulk])
        return True
    except Exception:
        return False


def test_unbalanced_mechanism_rejected_then_repaired(tmp_path):
    doc = _make_site_unbalanced_doc()
    assert not _loads_in_cantera(doc, tmp_path, "broken"), \
        "fixture must actually be Cantera-rejected for the test to mean anything"

    fixed, notes = balance_surface_sites(doc)
    assert len(notes) == 1
    assert "'O(s) + C(b) => CO' -> 'O(s) + C(b) => CO + (s)'" in notes[0]
    assert fixed["surface-reactions"][1]["equation"] == \
        "O(s) + C(b) => CO + (s)"
    assert _loads_in_cantera(fixed, tmp_path, "fixed")


def test_balanced_mechanism_untouched():
    doc = _load_char_doc()
    fixed, notes = balance_surface_sites(doc)
    assert notes == []
    assert fixed == doc


def test_input_doc_not_mutated():
    doc = _make_site_unbalanced_doc()
    before = yaml.safe_dump(doc)
    balance_surface_sites(doc)
    assert yaml.safe_dump(doc) == before


def test_multi_site_species_balancing(tmp_path):
    """O(s) occupying 2 sites: both reactions become unbalanced and both
    get empty sites appended on the correct (deficient) side."""
    doc = _make_site_unbalanced_doc()
    for sp in doc["species"]:
        if sp["name"] == "O(s)":
            sp["sites"] = 2
    doc["surface-reactions"][1]["equation"] = "O(s) + C(b) => CO + (s)"

    fixed, notes = balance_surface_sites(doc)
    assert len(notes) == 2
    # R1: O2 + 2 (s) => 2 O(s) is 2 vs 4 sites -> 2 empty sites on the left.
    assert fixed["surface-reactions"][0]["equation"] == \
        "O2 + 2 (s) + 2 (s) => 2 O(s)"
    # R2: 2 vs 1 -> one more empty site on the right.
    assert fixed["surface-reactions"][1]["equation"] == \
        "O(s) + C(b) => CO + (s) + (s)"
    assert _loads_in_cantera(fixed, tmp_path, "multisite")


def test_synthesizes_placeholder_when_no_empty_site(tmp_path):
    """No empty-composition species in the phase at all -> a '(v)' vacancy
    is synthesized and the mechanism loads."""
    doc = _make_site_unbalanced_doc()
    # Remove (s) entirely; adsorption becomes site-free chemistry except R2.
    doc["phases"][2]["species"] = ["O(s)"]
    doc["phases"][2]["state"] = {"T": 1400.0, "coverages": {"O(s)": 1.0}}
    doc["species"] = [s for s in doc["species"] if s["name"] != "(s)"]
    doc["surface-reactions"] = [
        {"equation": "O2 + C(b) => O(s) + CO",       # 0 vs 1 sites
         "rate-constant": {"A": 1.0e+8, "b": 0.0, "Ea": 30000.0}},
    ]
    fixed, notes = balance_surface_sites(doc)
    assert any("synthesized" in n for n in notes)
    assert fixed["surface-reactions"][0]["equation"] == \
        "O2 + C(b) + (v) => O(s) + CO"
    assert "(v)" in fixed["phases"][2]["species"]
    assert _loads_in_cantera(fixed, tmp_path, "vacancy")
