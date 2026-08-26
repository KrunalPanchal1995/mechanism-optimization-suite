"""
Generator for the W2 CHEMKIN-preflight test fixtures — deliberately-broken
minimal CHEMKIN ``.inp`` files, one fault class each.

Run:  python src/rmip/tests/fixtures/chemkin/_make_chemkin_fixtures.py
(kept as a script so the fixtures are inspectable/committed; the test module
also calls FIXTURES directly.)
"""
from __future__ import annotations

import os

_BASE_HEAD = """ELEMENTS
H O AR
END
SPECIES
H2 O2 H2O H O OH AR
END
REACTIONS
"""

# (filename, fault description, file body)
FIXTURES = {
    "case_mismatch.inp": (
        "P35: SPECIES declares AR, a reaction uses 'Ar'",
        _BASE_HEAD
        + "H+O2=O+OH         2.0E14 0.0 16800\n"
          "O+H2=H+OH         5.0E04 2.67 6290\n"
          "H+H+Ar=H2+Ar      6.4E17 -1.0 0.0\n"     # 'Ar' vs declared 'AR'
          "END\n"),
    "fortran_exp.inp": (
        "P2: a rate coefficient uses a Fortran exponent (2.0+14) missing the E",
        _BASE_HEAD
        + "H+O2=O+OH   2.0+14 0.0 16800\n"          # 2.0+14 -> 2.0E+14
          "O+H2=H+OH   5.0E04 2.67 6290\n"
          "END\n"),
    "noncon.inp": (
        "P3: a NONCON (non-site-conserving) surface keyword with no Cantera semantics",
        _BASE_HEAD
        + "H+O2=O+OH   2.0E14 0.0 16800\n"
          "O+H2=H+OH   5.0E04 2.67 6290\n"
          "NONCON\n"
          "END\n"),
    "no_end.inp": (
        "P18: the SPECIES block is never closed by END before REACTIONS",
        "ELEMENTS\nH O AR\nEND\n"
        "SPECIES\nH2 O2 H2O H O OH AR\n"            # <-- no END here
        "REACTIONS\n"
        "H+O2=O+OH   2.0E14 0.0 16800\nEND\n"),
    "tab_delim.inp": (
        "P2: a tab character separates the equation from the rate coefficients",
        _BASE_HEAD
        + "H+O2=O+OH\t2.0E14 0.0 16800\n"           # literal tab
          "O+H2=H+OH   5.0E04 2.67 6290\n"
          "END\n"),
    "undeclared_species.inp": (
        "a reaction participant ('HO2X') is never declared in SPECIES",
        _BASE_HEAD
        + "H+O2=O+OH         2.0E14 0.0 16800\n"
          "O+H2=H+OH         5.0E04 2.67 6290\n"
          "H2O+O2=HO2X+OH    1.0E13 0.0 20000\n"     # HO2X never declared
          "END\n"),
    "species_hyphen_locant.inp": (
        "KRUNAL_JULY_23.md / KONNOV_2019 regression: a real-style ring-locant "
        "species name (digits-hyphen-digit, no letters after the hyphen) "
        "declared alone on its own SPECIES line must survive Fortran-exponent "
        "auto-fixing unchanged -- the old char-class regex mistook the "
        "trailing '-2' for a missing exponent and corrupted it to '...E-2', "
        "desyncing the declaration from its correctly-spelled reaction usages",
        "ELEMENTS\nC H O\nEND\n"
        "SPECIES\nCH4 O2 CO2 H2O C2H4O1-2\nEND\n"
        "REACTIONS\n"
        "CH4+O2=CO2+H2O            2.0E14 0.0 16800\n"
        "CH4+O2=C2H4O1-2+H2O       1.0E13 0.0 20000\n"
        "END\n"),
}


def write_all(dest: str | None = None) -> str:
    here = dest or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(here, exist_ok=True)
    for name, (_desc, body) in FIXTURES.items():
        with open(os.path.join(here, name), "w", newline="\n") as fh:
            fh.write(body)
    return here


if __name__ == "__main__":
    where = write_all()
    print("wrote", ", ".join(FIXTURES), "to", where)
