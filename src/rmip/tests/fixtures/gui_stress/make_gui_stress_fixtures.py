#!/usr/bin/env python3
"""
make_gui_stress_fixtures.py — build the checked-in dummy fixture set for the
GUI stress-test sweep (GUI_STRESS_TEST_PLAN.md).

Deterministically regenerates, under this same directory
(src/rmip/tests/fixtures/gui_stress/):

  mech_A.inp, mech_B.inp, mech_C.inp
      3 dummy H2/O2 CHEMKIN mechanisms, 8 species / 8 reactions each, with
      ZERO cross-mechanism species-name string overlap (forcing SMILES-only
      identity resolution) and a deliberate reaction-overlap structure:
      R1 (H+O2<=>O+OH) common to all 3; R2 (O+H2<=>H+OH) shared by A,B only;
      R3 (OH+H2<=>H+H2O) shared by B,C only; the rest unique per mechanism.
      This is the exact shape that already found the ChemkinWriter
      species-staleness bug (test_chemkin_writer_species_staleness.py) --
      keep it, don't simplify it away.

  thermo.dat
      One shared NASA-7 THERMO block (real Connaire-2004 H2/O2 coefficients)
      emitted under every spelling used by mech_A/B/C AND the canonical
      names used by mech_std (below) -- same physical species, many name
      variants, one set of real coefficients per identity.

  glossary.csv
      27 rows (9 identities x 3 mechanisms), header `name,smiles`, every
      spelling mapped to its identity's SMILES -> 100% glossary_matched.

  mech_std.inp / mech_std.yaml
      A 4th, CANONICALLY-named mechanism (H, O2, O, OH, H2, H2O, HO2, H2O2,
      AR -- no renaming) containing R1/R2/R3 spelled to match
      h2_o2_crosscheck_unsrt.xml's `rxn=` strings exactly, converted to
      Cantera YAML via ck2yaml. This is the fixture for every Cantera/
      MUQ-SAC-Uncertainty consumer (forward_uq, nominal_sim, brute_force_sa):
      Uncertainty.py's `reaction.__init__` hard-asserts each XML reaction
      string is present in `MechanismParser.Parser(mechPath).rxnList`, which
      loads the mechanism as a Cantera YAML dict and reads
      `mech["reactions"][i]["equation"]` -- a mis-spelled dummy (mech_A/B/C)
      would fail that assertion immediately, so those two fixture tiers are
      deliberately NOT interchangeable.

  project.db
      Built by parsing mech_A/B/C through the real production parser
      (`rmip_cli.py parse --glossary --thermo`), giving the >=2 mechanism
      rows the fusion/compare-graph/reduce/gsa-run apps need.

  target.opt, targets/gui_stress.target, gui_stress.add
      A minimal target/opt/addendum trio (2 Tig targets, adapted from
      DATASET/H2_O2_TEST) pointed at mech_std.yaml + the crosscheck XML,
      Counts.targets_count=2, small enough to run in seconds.

Usage:
    python src/rmip/tests/fixtures/gui_stress/make_gui_stress_fixtures.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[4]           # .../src/rmip/tests/fixtures/gui_stress -> repo root
assert (_REPO / "rmip_cli.py").is_file(), f"repo root mis-detected: {_REPO}"

_DATASET = _REPO / "DATASET" / "H2_O2_TEST"
_CROSSCHECK_XML = _HERE.parent / "h2_o2_crosscheck_unsrt.xml"
assert _CROSSCHECK_XML.is_file(), _CROSSCHECK_XML


# =============================================================================
# 1. Identity table: 9 real H2/O2 species -> SMILES + real NASA-7 (Connaire
#    2004, verbatim from MECHANISM/Connaire/h2_v1a_therm.txt) + per-mechanism
#    spelling (mech_std uses the canonical/identity name itself, unchanged).
# =============================================================================

# (identity, smiles, canonical CHEMKIN name, {A,B,C: spelling})
IDENTITIES = [
    ("H2",   "[H][H]", "H2",  {"A": "diprotium",   "B": "wstoff2",   "C": "sp_hh"}),
    ("O2",   "O=O",    "O2",  {"A": "dioxygenium", "B": "souer2",    "C": "sp_oo"}),
    ("H",    "[H]",    "H",   {"A": "protium",     "B": "wstoff1",   "C": "sp_h"}),
    ("O",    "[O]",    "O",   {"A": "oxo",         "B": "souer1",    "C": "sp_o"}),
    ("OH",   "[OH]",   "OH",  {"A": "hydroxyl",    "B": "hydroxyd",  "C": "sp_oh"}),
    ("HO2",  "[O]O",   "HO2", {"A": "hydroperox",  "B": "hydperox",  "C": "sp_ho2"}),
    ("H2O",  "O",      "H2O", {"A": "aqua",        "B": "wasser",    "C": "sp_h2o"}),
    ("H2O2", "OO",     "H2O2", {"A": "peroxidum",  "B": "perhydrol", "C": "sp_h2o2"}),
    ("AR",   "[Ar]",   "AR",  {"A": "argon_a",     "B": "argon_b",   "C": "sp_ar"}),
]

# Real NASA-7 CHEMKIN thermo cards (GRI-Mech 3.0, MECHANISM/GRI3/grimech30/
# thermo30.dat -- Cantera's own reference mechanism, strictly ck2yaml-clean
# column formatting), keyed by identity. suffix is everything from column 19
# onward (date/elements/phase/T-ranges) -- verbatim from the source file,
# name-independent. Lines 2-4 are the fixed-width E15.8 coefficient lines,
# untouched (no name field).
_THERMO_RAW = {
    "H2": {
        "suffix": "TPIS78H   2               G   200.000  3500.000  1000.000    1",
        "l2": " 3.33727920E+00-4.94024731E-05 4.99456778E-07-1.79566394E-10 2.00255376E-14    2",
        "l3": "-9.50158922E+02-3.20502331E+00 2.34433112E+00 7.98052075E-03-1.94781510E-05    3",
        "l4": " 2.01572094E-08-7.37611761E-12-9.17935173E+02 6.83010238E-01                   4",
    },
    "O2": {
        "suffix": "TPIS89O   2               G   200.000  3500.000  1000.000    1",
        "l2": " 3.28253784E+00 1.48308754E-03-7.57966669E-07 2.09470555E-10-2.16717794E-14    2",
        "l3": "-1.08845772E+03 5.45323129E+00 3.78245636E+00-2.99673416E-03 9.84730201E-06    3",
        "l4": "-9.68129509E-09 3.24372837E-12-1.06394356E+03 3.65767573E+00                   4",
    },
    "H": {
        "suffix": "L 7/88H   1               G   200.000  3500.000  1000.000    1",
        "l2": " 2.50000001E+00-2.30842973E-11 1.61561948E-14-4.73515235E-18 4.98197357E-22    2",
        "l3": " 2.54736599E+04-4.46682914E-01 2.50000000E+00 7.05332819E-13-1.99591964E-15    3",
        "l4": " 2.30081632E-18-9.27732332E-22 2.54736599E+04-4.46682853E-01                   4",
    },
    "O": {
        "suffix": "L 1/90O   1               G   200.000  3500.000  1000.000    1",
        "l2": " 2.56942078E+00-8.59741137E-05 4.19484589E-08-1.00177799E-11 1.22833691E-15    2",
        "l3": " 2.92175791E+04 4.78433864E+00 3.16826710E+00-3.27931884E-03 6.64306396E-06    3",
        "l4": "-6.12806624E-09 2.11265971E-12 2.91222592E+04 2.05193346E+00                   4",
    },
    "OH": {
        "suffix": "RUS 78O   1H   1          G   200.000  3500.000  1000.000    1",
        "l2": " 3.09288767E+00 5.48429716E-04 1.26505228E-07-8.79461556E-11 1.17412376E-14    2",
        "l3": " 3.85865700E+03 4.47669610E+00 3.99201543E+00-2.40131752E-03 4.61793841E-06    3",
        "l4": "-3.88113333E-09 1.36411470E-12 3.61508056E+03-1.03925458E-01                   4",
    },
    "HO2": {
        "suffix": "L 5/89H   1O   2          G   200.000  3500.000  1000.000    1",
        "l2": " 4.01721090E+00 2.23982013E-03-6.33658150E-07 1.14246370E-10-1.07908535E-14    2",
        "l3": " 1.11856713E+02 3.78510215E+00 4.30179801E+00-4.74912051E-03 2.11582891E-05    3",
        "l4": "-2.42763894E-08 9.29225124E-12 2.94808040E+02 3.71666245E+00                   4",
    },
    "H2O": {
        "suffix": "L 8/89H   2O   1          G   200.000  3500.000  1000.000    1",
        "l2": " 3.03399249E+00 2.17691804E-03-1.64072518E-07-9.70419870E-11 1.68200992E-14    2",
        "l3": "-3.00042971E+04 4.96677010E+00 4.19864056E+00-2.03643410E-03 6.52040211E-06    3",
        "l4": "-5.48797062E-09 1.77197817E-12-3.02937267E+04-8.49032208E-01                   4",
    },
    "H2O2": {
        "suffix": "L 7/88H   2O   2          G   200.000  3500.000  1000.000    1",
        "l2": " 4.16500285E+00 4.90831694E-03-1.90139225E-06 3.71185986E-10-2.87908305E-14    2",
        "l3": "-1.78617877E+04 2.91615662E+00 4.27611269E+00-5.42822417E-04 1.67335701E-05    3",
        "l4": "-2.15770813E-08 8.62454363E-12-1.77025821E+04 3.43505074E+00                   4",
    },
    "AR": {
        "suffix": "120186AR  1               G   300.000  5000.000  1000.000    1",
        "l2": " 0.02500000E+02 0.00000000E+00 0.00000000E+00 0.00000000E+00 0.00000000E+00    2",
        "l3": "-0.07453750E+04 0.04366000E+02 0.02500000E+02 0.00000000E+00 0.00000000E+00    3",
        "l4": " 0.00000000E+00 0.00000000E+00-0.07453750E+04 0.04366000E+02                   4",
    },
}


def _thermo_card(name: str, identity: str) -> str:
    """One 4-line NASA-7 CHEMKIN thermo card for `name`, using `identity`'s
    real coefficients. Only the column-1..18 name field changes."""
    raw = _THERMO_RAW[identity]
    l1 = name.ljust(18) + raw["suffix"]
    return "\n".join([l1, raw["l2"], raw["l3"], raw["l4"]])


# =============================================================================
# 2. Reaction pool (real Arrhenius params, MECHANISM/FFCM1/FFCM1.inp,
#    restricted to species in the 9-identity set above) + overlap assignment.
# =============================================================================

# Each reaction: (key, equation_template using identity tokens, A, n, Ea,
#                 extra) where extra is None, or a dict with optional keys:
#   "eff"  -- third-body efficiency line, SPECIES-TOKEN text (gets
#             identity->spelling substitution applied)
#   "low"  -- LOW/... falloff line, PURE NUMBERS (no substitution -- must
#             NOT be run through the identity substitution pass, since the
#             single-letter identity "O" would otherwise corrupt the literal
#             keywords "LOW"/"TROE" themselves, e.g. "LOW" -> "LoxoW")
#   "troe" -- TROE/... line, PURE NUMBERS (same reason)
# equation_template uses identity names directly (e.g. "H+O2<=>O+OH");
# _render_equation() substitutes the per-mechanism spelling for each token.
REACTIONS = {
    "R1_shared_ABC": ("H+O2<=>O+OH",        9.841E+13,  0.000,  15310.00, None),
    "R2_shared_AB":  ("O+H2<=>H+OH",        3.848E+12,  0.000,   7950.00, None),
    "R3_shared_BC":  ("OH+H2<=>H+H2O",      2.256E+08,  1.510,   3437.00, None),

    # -- A-unique (6) --
    "A1_2OH_O_H2O":        ("2OH<=>O+H2O",           3.161E+04,  2.420,  -1928.00, None),
    "A2_H2_M":              ("H2+M<=>2H+M",           4.580E+19, -1.400, 104390.00,
                              {"eff": "AR/0.00/ H2/2.55/ H2O/12.02/"}),
    "A3_HO2falloff":        ("H+O2(+M)<=>HO2(+M)",    4.565E+12,  0.440,      0.00,
                              {"low": "LOW/6.370E+20 -1.720 525.00/",
                               "troe": "TROE/0.5000 30.00 90000.00 90000.00/",
                               "eff": "AR/0.60/ H2/1.87/ O2/0.75/ H2O/15.81/"}),
    "A4_HO2_H_2OH":          ("HO2+H<=>2OH",           5.888E+13,  0.000,    300.00, None),
    "A5_HO2_H_H2_O2":        ("HO2+H<=>H2+O2",         2.945E+06,  2.087,  -1455.00, None),
    "A6_H2O2falloff":        ("H2O2(+M)<=>2OH(+M)",    2.187E+12,  0.900,  48750.00,
                              {"low": "LOW/2.490E+24 -2.300 48750.00/",
                               "troe": "TROE/0.5800 30.00 90000.00 90000.00/",
                               "eff": "AR/0.85/ H2/3.27/ O2/1.20/ H2O/6.63/ H2O2/6.61/"}),

    # -- B-unique (5) --
    "B1_2O_M":               ("2O+M<=>O2+M",           6.160E+15, -0.500,      0.00,
                              {"eff": "AR/0.00/ H2/2.50/ H2O/12.00/"}),
    "B2_O_H_M":               ("O+H+M<=>OH+M",          4.710E+18, -1.000,      0.00,
                              {"eff": "AR/0.75/ H2/2.50/ H2O/15.80/"}),
    "B3_HO2_O":               ("HO2+O<=>OH+O2",         1.609E+13,  0.000,   -445.00, None),
    "B4_HO2_OH":              ("HO2+OH<=>H2O+O2",       7.347E+12,  0.000,  -1093.00, None),
    "B5_H2O2_H_OH_H2O":       ("H2O2+H<=>OH+H2O",       2.045E+13,  0.000,   3970.00, None),

    # -- C-unique (6) --
    "C1_H2O_M":               ("H2O+M<=>H+OH+M",        6.060E+27, -3.322, 120800.00,
                              {"eff": "AR/1.23/ H2/3.77/ H2O/0.00/"}),
    "C2_HO2_H_O_H2O":         ("HO2+H<=>O+H2O",         1.632E+12,  0.000,      0.00, None),
    "C3_2HO2":                ("2HO2<=>H2O2+O2",        1.958E+11,  0.000,  -1409.00, None),
    "C4_H2O2_H_HO2_H2":       ("H2O2+H<=>HO2+H2",       5.856E+13,  0.000,   7950.00, None),
    "C5_H2O2_O":              ("H2O2+O<=>OH+HO2",       8.513E+06,  2.000,   3970.00, None),
    "C6_H2O2_OH":             ("H2O2+OH<=>H2O+HO2",     1.565E+12,  0.000,    318.00, None),
}

MECH_REACTIONS = {
    "A": ["R1_shared_ABC", "R2_shared_AB",
          "A1_2OH_O_H2O", "A2_H2_M", "A3_HO2falloff",
          "A4_HO2_H_2OH", "A5_HO2_H_H2_O2", "A6_H2O2falloff"],
    "B": ["R1_shared_ABC", "R2_shared_AB", "R3_shared_BC",
          "B1_2O_M", "B2_O_H_M", "B3_HO2_O", "B4_HO2_OH", "B5_H2O2_H_OH_H2O"],
    "C": ["R1_shared_ABC", "R3_shared_BC",
          "C1_H2O_M", "C2_HO2_H_O_H2O", "C3_2HO2", "C4_H2O2_H_HO2_H2",
          "C5_H2O2_O", "C6_H2O2_OH"],
}
assert all(len(v) == 8 for v in MECH_REACTIONS.values())


def _render_equation(template: str, spelling: dict) -> str:
    """Substitute identity tokens in an equation template for a mechanism's
    spellings. Longest identity names first so 'H2O2' doesn't get clobbered
    by a partial match on 'H2' or 'O2' or 'H2O'."""
    out = template
    for identity in sorted(spelling, key=len, reverse=True):
        out = out.replace(identity, spelling[identity])
    return out


def _render_extra_lines(extra: dict, spelling: dict) -> list:
    """Render a reaction's extra CHEMKIN lines in the fixed order LOW/TROE/
    efficiency. LOW and TROE are pure-number lines and must NOT go through
    identity substitution (the single-letter identity "O" would otherwise
    match inside the literal keywords "LOW"/"TROE" themselves, corrupting
    them into "LoxoW"/"TRoxoE"). Only the third-body efficiency line
    ("eff") contains species tokens and needs substitution."""
    if extra is None:
        return []
    out = []
    if "low" in extra:
        out.append(extra["low"])          # verbatim, no substitution
    if "troe" in extra:
        out.append(extra["troe"])         # verbatim, no substitution
    if "eff" in extra:
        eff = extra["eff"]
        for identity in sorted(spelling, key=len, reverse=True):
            eff = eff.replace(identity, spelling[identity])
        out.append(eff)
    return out


def build_mechanism_inp(mech_letter: str) -> str:
    spelling = {identity: names[mech_letter] for identity, _, _, names in IDENTITIES}
    species_names = [spelling[identity] for identity, *_ in IDENTITIES]

    lines = [
        f"! Dummy GUI-stress-test mechanism {mech_letter} -- generated by",
        "! make_gui_stress_fixtures.py. Real H2/O2 Arrhenius params (FFCM-1/",
        "! Connaire-2004), species spellings deliberately unique to this",
        "! mechanism to force SMILES/InChI-only identity resolution.",
        "ELEMENTS",
        "H O AR",
        "END",
        "SPECIES",
    ]
    for name in species_names:
        lines.append("  " + name)
    lines.append("END")
    lines.append("REACTIONS")
    for rkey in MECH_REACTIONS[mech_letter]:
        template, A, n, Ea, extra = REACTIONS[rkey]
        eq = _render_equation(template, spelling)
        lines.append(f"!{rkey}")
        lines.append(f"{eq:<48}{A:.3E}  {n:6.3f}  {Ea:10.2f}")
        for eline in _render_extra_lines(extra, spelling):
            lines.append(eline)
        lines.append("")
    lines.append("END")
    return "\n".join(lines) + "\n"


def build_mech_std_inp() -> str:
    """Canonically-named mechanism, spelled to match h2_o2_crosscheck_unsrt.
    xml's `rxn=` strings exactly once run through ck2yaml (equation strings
    render as 'H + O2 <=> O + OH', etc.) for R1/R2/R3.

    Includes ALL 20 reactions (not just R1/R2/R3): a 3-reaction mechanism
    has no chain-branching/HO2/H2O2 network and physically CANNOT ignite an
    H2/O2/AR mixture (confirmed empirically -- Cantera's IdealGasReactor
    integration ran to completion with zero OH buildup, "No ignition!!" from
    cantera_.py's ignitionDelay()). The full real Burke-style H2/O2 network
    (same reactions as mech_A/B/C combined) gives genuine chain-branching
    ignition at the fixture's target conditions (T=1450K, P=2.2atm, H2/O2/AR)
    while keeping R1/R2/R3's equation strings byte-identical to the XML."""
    spelling = {identity: canon for identity, _, canon, _ in IDENTITIES}
    species_names = [spelling[identity] for identity, *_ in IDENTITIES]
    lines = [
        "! Canonically-named GUI-stress-test mechanism -- generated by",
        "! make_gui_stress_fixtures.py. R1/R2/R3 equations match",
        "! h2_o2_crosscheck_unsrt.xml's rxn= strings exactly (post-ck2yaml);",
        "! remaining reactions give it a complete, actually-ignitable H2/O2",
        "! chain-branching network (see build_mech_std_inp() docstring).",
        "ELEMENTS",
        "H O AR",
        "END",
        "SPECIES",
    ]
    for name in species_names:
        lines.append("  " + name)
    lines.append("END")
    lines.append("REACTIONS")
    for rkey in REACTIONS:
        template, A, n, Ea, extra = REACTIONS[rkey]
        eq = _render_equation(template, spelling)
        lines.append(f"!{rkey}")
        lines.append(f"{eq:<48}{A:.3E}  {n:6.3f}  {Ea:10.2f}")
        for eline in _render_extra_lines(extra, spelling):
            lines.append(eline)
        lines.append("")
    lines.append("END")
    return "\n".join(lines) + "\n"


def build_thermo_dat_for(name_by_identity: dict) -> str:
    """A THERMO block scoped to exactly the given {identity: name} names.

    Deliberately NOT a single shared file covering every mechanism's
    spellings: the CHEMKIN parser unions SPECIES-block-declared species
    with every THERMO-block entry it finds (rmip/io/_chemkin_parser.py's
    `_parse_nasa7_entry`, the `else` branch at :430-431 adds any thermo
    entry not already in `self._species`), so a shared 36-entry thermo.dat
    would silently inflate every mechanism's stored species count to 36
    and its glossary-hit fraction down to 27/36=75% instead of the correct
    9 species / 100% -- caught while first building this fixture set.
    """
    lines = [
        "THERMO ALL",
        "   300.000  1000.000  5000.000",
    ]
    for identity, name in name_by_identity.items():
        lines.append(_thermo_card(name, identity))
    lines.append("END")
    return "\n".join(lines) + "\n"


# Real transport params (Connaire 2004, MECHANISM/Connaire/h2_v1a_tran.txt),
# keyed by identity: "geometry eps/k sigma dipole polariz Zrot" (ScanMan/
# CHEMKIN TRANSPORT format).
_TRANSPORT_RAW = {
    "AR":   "0   136.500     3.330     0.000     0.000     0.000",
    "O":    "0    80.000     2.750     0.000     0.000     0.000",
    "O2":   "1   107.400     3.458     0.000     1.600     3.800",
    "OH":   "1    80.000     2.750     0.000     0.000     0.000",
    "H2O":  "2   572.400     2.605     1.844     0.000     4.000",
    "H2O2": "2   107.400     3.458     0.000     0.000     3.800",
    "H":    "0   145.000     2.050     0.000     0.000     0.000",
    "H2":   "1    38.000     2.920     0.000     0.790   280.000",
    "HO2":  "2   107.400     3.458     0.000     0.000     1.000",
}


def build_transport_dat_for(name_by_identity: dict) -> str:
    lines = [f"{name:<18} {_TRANSPORT_RAW[identity]}"
             for identity, name in name_by_identity.items()]
    return "\n".join(lines) + "\n"


def build_glossary_csv() -> str:
    rows = ["name,smiles"]
    for identity, smiles, _canon, spelling in IDENTITIES:
        for mech_letter in ("A", "B", "C"):
            rows.append(f"{spelling[mech_letter]},{smiles}")
    return "\n".join(rows) + "\n"


# =============================================================================
# 3. target/opt/addendum trio (adapted from DATASET/H2_O2_TEST, repointed at
#    mech_std.yaml + the crosscheck XML, 2 Tig targets only).
# =============================================================================

def build_target_file() -> str:
    src = (_DATASET / "targets" / "H2_IDT.target").read_text().splitlines(keepends=True)
    # Keep only the first 2 dataset_001 (Tig) lines -- small, seconds-fast.
    return "".join(src[:2])


def build_addendum_file() -> str:
    """Adapted from DATASET/H2_O2_TEST/H2_O2.add's dataset_001 block, with
    one deliberate change: `ign_delay_def: OH*` -> `OH` / `ign_cond: onset`
    -> `max`. The real addendum tracks the excited-state OH* chemiluminescence
    species for ignition onset, which mech_std (a minimal 9-species dummy)
    doesn't carry -- ct.Solution's species-column lookup in cantera_.py's
    ignitionDelay() KeyErrors on 'OH*' otherwise. Peak ground-state [OH] is
    the standard fallback ignition-delay definition and needs no extra
    species, so it's what every dummy-mechanism target in this fixture set
    uses."""
    return (
        "dataset_001:\n"
        "  solver: cantera\n"
        "  ign_delay_def: OH\n"
        "  ign_cond: max\n"
        "  specific_cond: ''\n"
        "  mol_wt: 2\n"
    )


def build_target_opt(gui_stress_dir: Path) -> str:
    mech_yaml = gui_stress_dir / "mech_std.yaml"
    unsrt_xml = _CROSSCHECK_XML
    targets_f = gui_stress_dir / "targets" / "gui_stress.target"
    addendum_f = gui_stress_dir / "gui_stress.add"
    db_f = gui_stress_dir / "project.db"
    glossary_f = gui_stress_dir / "glossary.csv"
    pre_f = gui_stress_dir / "mech_std.pre"

    return f"""\
##############################################################################
# GUI stress-test dummy project config -- generated by
# make_gui_stress_fixtures.py. See GUI_STRESS_TEST_PLAN.md.
##############################################################################

project:
  name: GUI_Stress_Test
  fuel_name: H2
  fuel_formula: H2
  output_base_dir: outputs/gui_stress

Type:
  optimization_type: PRS
  Algorithm_type: gradient-based
  Algorithm: SLSQP

Counts:
  total_reactions: 3
  parallel_threads: 2
  targets_count: 2

SA:
  carbon_number: 0

Bin:
  bin: {gui_stress_dir}
  solver_bin: /home/krithika/FlameMaster/Bin/bin
  yaml_writer: {_REPO}/src/Sensitivity_Analysis/parallel_yaml_writer.so
  samap_executable: PLACEHOLDER_SAMAP
  jpdap_executable: PLACEHOLDER_JPDAP
  SA_tool: {_REPO}/src/Sensitivity_Analysis/sens.py
  SA_tool_3p: {_REPO}/src/Sensitivity_Analysis/sens_3_params.py

Inputs:
  fileType: cantera_yaml
  fuelClass: Mono
  fuel:
    a: H2
  global_reaction: 2H2+O2==2H2O
  AnalysisType: reaction

Locations:
  mechanism: {mech_yaml}
  Initial_pre_file: {pre_f}
  uncertainty_data: {unsrt_xml}
  thermo_file: ''
  trans_file: ''
  targets: {targets_f}
  addendum: {addendum_f}

# Required by sens_3_params.py's module-level load even though no Fls/Flf
# targets are in this fixture's 2-target trio -- mirrors DATASET/H2_O2_TEST's
# placeholder shape.
StartProfilesData:
  Fls:
    Flame: UnstretchedPremixed
    StartProfilesFile: PLACEHOLDER_FLS_INIT
    CopyTo: PLACEHOLDER_FLS_DEST
    fuel:
      key: Mono
      From: H2
    oxidizer:
      key: Mono
      From: O2
    bathGas:
      key: Mono
      From: AR
    globalReaction:
      FromRxn: 2H2+O2==2H2O
    units:
      pressure: Pa
      temperature: K
      concentration: mol_fraction
  Flf:
    Flame: UnstretchedPremixed
    StartProfilesFile: PLACEHOLDER_FLF_INIT
    CopyTo: PLACEHOLDER_FLF_DEST
    fuel:
      key: Mono
      From: H2
    oxidizer:
      key: Mono
      From: O2
    bathGas:
      key: Mono
      From: AR
    globalReaction:
      FromRxn: 2H2+O2==2H2O
    units:
      pressure: Pa
      temperature: K
      concentration: mol_fraction

Stats:
  Design_of_PRS: A-facto
  Order_of_PRS: 2
  Sampling_of_PRS: 4
  PRS_type: Partial
  cut_off_percentage: 1
  MRE_criteria: 5.0
  Objective: powerMeanWMRE
  pm_p: 0.1
  pm_anneal: true
  pm_p_start: 1.0
  num_generations: 100
  Weights:
    Tig: 1.0

RMIP:
  database: {db_f}
  glossary: {glossary_f}
  mechanisms:
    - name: mech_A
      path: {gui_stress_dir / 'mech_A.inp'}
      thermo: {gui_stress_dir / 'thermo_A.dat'}
      format: chemkin
    - name: mech_B
      path: {gui_stress_dir / 'mech_B.inp'}
      thermo: {gui_stress_dir / 'thermo_B.dat'}
      format: chemkin
    - name: mech_C
      path: {gui_stress_dir / 'mech_C.inp'}
      thermo: {gui_stress_dir / 'thermo_C.dat'}
      format: chemkin
  fusion:
    name: gui_stress_fused
    description: Fused dummy H2/O2 mechanism (GUI stress test)
    uf_warn: 10.0
    out_dir: outputs/gui_stress/fusion
  reduction:
    method: DRG
    target_species: H2,O2,H2O
    eps_max: 0.5
    T: 1200.0
    P: 1.0
    phi: 1.0
    sa_json: null
    out_dir: outputs/gui_stress/reduction
  comparison:
    tol_A: 0.01
    tol_n: 0.001
    tol_Ea: 100.0
    out_dir: outputs/gui_stress/comparison

ForwardUQ:
  n_samples: 2048
  method: sobol
  output_dir: outputs/gui_stress/forward_uq
  plot_histogram: false
  plot_sobol: false
  ci_level: 0.95
"""


# =============================================================================
# 4. Driver
# =============================================================================

def _run(cmd, **kw):
    print("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed (exit {r.returncode}): {cmd}")
    return r


def main() -> int:
    out = _HERE
    out.mkdir(parents=True, exist_ok=True)
    (out / "targets").mkdir(exist_ok=True)

    print("=== Writing dummy CHEMKIN mechanisms (A/B/C) ===")
    for letter in ("A", "B", "C"):
        (out / f"mech_{letter}.inp").write_text(build_mechanism_inp(letter))

    print("=== Writing mech_std.inp (canonical names) ===")
    (out / "mech_std.inp").write_text(build_mech_std_inp())

    print("=== Writing per-mechanism thermo files (A/B/C + std) ===")
    # One THERMO block per mechanism, scoped to only that mechanism's own
    # spellings -- see build_thermo_dat_for()'s docstring for why a single
    # shared file across all 4 mechanisms is wrong here.
    for letter in ("A", "B", "C"):
        name_by_identity = {identity: names[letter] for identity, _, _, names in IDENTITIES}
        (out / f"thermo_{letter}.dat").write_text(build_thermo_dat_for(name_by_identity))
    std_name_by_identity = {identity: canon for identity, _, canon, _ in IDENTITIES}
    (out / "thermo_std.dat").write_text(build_thermo_dat_for(std_name_by_identity))
    (out / "trans_std.txt").write_text(build_transport_dat_for(std_name_by_identity))

    print("=== Writing glossary.csv ===")
    (out / "glossary.csv").write_text(build_glossary_csv())

    print("=== Writing target/addendum trio ===")
    (out / "targets" / "gui_stress.target").write_text(build_target_file())
    (out / "gui_stress.add").write_text(build_addendum_file())
    (out / "target.opt").write_text(build_target_opt(out))

    print("=== ck2yaml: mech_std.inp -> mech_std.yaml ===")
    ck2yaml = shutil.which("ck2yaml") or "ck2yaml"
    _run([ck2yaml, f"--input={out / 'mech_std.inp'}",
          f"--thermo={out / 'thermo_std.dat'}",
          f"--output={out / 'mech_std.yaml'}", "--permissive"])

    print("=== ScanMan: mech_std.inp -> mech_std.pre "
          "(Locations.Initial_pre_file, needed by simulation_manager2_0.py) ===")
    scanman = "/home/krithika/FlameMaster/Bin/bin/ScanMan"
    if Path(scanman).exists():
        _run([scanman, "-i", str(out / "mech_std.inp"),
              "-t", str(out / "thermo_std.dat"),
              "-m", str(out / "trans_std.txt"),
              "-f", "chemkin", "-E", "-c", "-N", "0.05",
              "-o", str(out / "mech_std.pre")])
    else:
        print(f"  SKIPPED: ScanMan not found at {scanman} "
              "(Locations.Initial_pre_file in target.opt will dangle)")

    print("=== rmip_cli.py parse: mech_A/B/C -> project.db ===")
    db_path = out / "project.db"
    if db_path.exists():
        db_path.unlink()
    for letter in ("A", "B", "C"):
        _run([sys.executable, str(_REPO / "rmip_cli.py"), "parse",
              str(out / f"mech_{letter}.inp"),
              "--thermo", str(out / f"thermo_{letter}.dat"),
              "--glossary", str(out / "glossary.csv"),
              "--db", str(db_path),
              "--name", f"mech_{letter}"])

    print("\n=== Fixture build complete ===")
    print(f"Directory: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
