"""
Generator for the W3 OpenSmoke-importer test fixture.

Builds ``opensmoke_pe.inp`` — an OpenSmoke-flavoured CHEMKIN file: a valid
H2/O2 gas-phase subset (thermo cards generated from Cantera's shipped
``h2o2.yaml``) plus the P5-style condensed-phase extensions Cantera cannot
represent: a ``MATERIAL PE`` header, a ``LIQUID … END`` block declaring the
polymer species ``PE``, and a heterogeneous pyrolysis reaction ``PE=>H2+H2``.

Run:  python src/rmip/tests/fixtures/chemkin/_make_opensmoke_fixture.py
"""
from __future__ import annotations

import os

_GAS_SPECIES = ["H2", "O2", "H2O", "OH", "H", "O", "AR"]


def _nasa7_card(name: str, comp: dict, coeffs_low, coeffs_high,
                tlow: float, tmid: float, thigh: float) -> str:
    """Format one 4-line CHEMKIN NASA-7 thermo card (col-80 flags 1–4)."""
    elems = "".join(f"{el:<2}{int(round(c)):>3}" for el, c in comp.items())
    l1 = f"{name:<16}{'':8}{elems:<20}G{tlow:10.3f}{thigh:10.3f}{tmid:8.3f}"
    l1 = f"{l1:<79}1"
    coef = list(coeffs_high) + list(coeffs_low)     # card order: high then low
    def row(vals, flag):
        body = "".join(f"{v: 15.8E}" for v in vals)
        return f"{body:<79}{flag}"
    return "\n".join([l1, row(coef[0:5], "2"), row(coef[5:10], "3"),
                      row(coef[10:14], "4")])


def build_text() -> str:
    import cantera as ct
    import warnings
    warnings.filterwarnings("ignore")
    gas = ct.Solution(os.path.join(ct.__path__[0], "data", "h2o2.yaml"))
    cards = []
    for name in _GAS_SPECIES:
        sp = gas.species(name)
        c = list(sp.thermo.coeffs)                  # [Tmid, high7, low7]
        cards.append(_nasa7_card(name, dict(sp.composition),
                                 c[8:15], c[1:8],
                                 sp.thermo.min_temp, c[0], sp.thermo.max_temp))
    return (
        "! OpenSmoke-flavoured CHEMKIN fixture (W3/P5) — gas H2/O2 subset\n"
        "! plus MATERIAL/LIQUID condensed-phase extensions.\n"
        "ELEMENTS\nH O AR\nEND\n"
        "MATERIAL PE\n"
        "LIQUID\nPE\nEND\n"
        "SPECIES\n" + " ".join(_GAS_SPECIES) + "\nEND\n"
        "THERMO ALL\n   200.000  1000.000  6000.000\n"
        + "\n".join(cards) + "\nEND\n"
        "REACTIONS\n"
        "H+O2=O+OH             2.644E16 -0.6707 17041.0\n"
        "O+H2=H+OH             4.589E04  2.700   6260.0\n"
        "OH+H2=H+H2O           1.734E08  1.510   3430.0\n"
        "PE=>H2+H2             1.000E13  0.000  40000.0   ! liquid pyrolysis\n"
        "END\n")


def write(dest: str | None = None) -> str:
    here = dest or os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "opensmoke_pe.inp")
    with open(path, "w", newline="\n") as fh:
        fh.write(build_text())
    return path


if __name__ == "__main__":
    print("wrote", write())
