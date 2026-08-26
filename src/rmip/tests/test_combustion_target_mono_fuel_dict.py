"""
rmip/tests/test_combustion_target_mono_fuel_dict.py
=====================================================
Regression coverage for a bug found 2026-07-17 while executing
GUI_STRESS_TEST_PLAN.md's sweep of `brute_force_sa_gui.py` against the real
shipped `DATASET/H2_O2_TEST/targets/H2_IDT.target` file (re-verifying B12).

`combustion_target_class.combustion_target.__init__` (both
`src/MUQ-SAC/combustion_target_class.py` and
`src/Sensitivity_Analysis/combustion_target_class.py` -- byte-identical for
this code path) dispatches the "Fuel" field's parsing on the declared
`Fuel_type` value: "Multi" -> parse `content` as a dict-wrapped JSON blob,
anything else (including "Mono") -> `float(content.split("->")[1].split("=")[1])`.

But every REAL generated `.target` file -- including the checked-in
`DATASET/H2_O2_TEST/targets/H2_IDT.target`, `Fuel_type -- Mono` -- writes the
fuel spec as a dict even for a single-species fuel:
`Fuel -- x->{'a': 'H2'} = {'a': 0.0050}`. The Mono branch's `float()` call
then receives the literal string `" {'a': 0.0050}"` and crashes with
`ValueError: could not convert string to float`. This is not a contrived
edge case: it crashes on the FIRST LINE of the shipped fixture the moment
anything (brute_force_sa_gui, forward_uq.py, ...) tries to actually load a
Mono-fuel target through this parser -- i.e. the Mono branch of this 30-year
code path has likely never successfully parsed a single real generated
target file.

Fixed by dispatching on the actual RHS shape (does it start with "{") rather
than solely on the declared Fuel_type, so a dict-wrapped single-key Mono
fuel spec takes the (already-correct, already-tested-via-Multi) dict-parsing
branch instead of the bare-scalar one. A genuine bare-scalar Mono line (no
dict) still takes the original branch, so no existing behaviour changes.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("subdir", ["MUQ-SAC", "Sensitivity_Analysis"])
def test_mono_fuel_dict_format_parses(subdir):
    """Both copies of combustion_target_class.py must parse a real
    Mono-fuel-type target line whose Fuel field is dict-wrapped (the shape
    every real generated .target file actually uses), not just a Multi one."""
    mod_dir = str(_REPO / "src" / subdir)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    # Force a fresh import even if a same-named module from a different
    # subdir is already cached (both dirs define a top-level
    # `combustion_target_class` module).
    sys.modules.pop("combustion_target_class", None)
    import combustion_target_class

    addendum = yaml.safe_load(
        (_REPO / "DATASET" / "H2_O2_TEST" / "H2_O2.add").read_text()
    )
    line = (_REPO / "DATASET" / "H2_O2_TEST" / "targets" / "H2_IDT.target").read_text().splitlines()[0]

    t = combustion_target_class.combustion_target(line, deepcopy(addendum), 0)

    assert t.fuel_type == "Mono"
    assert t.fuel_dict == {"H2": pytest.approx(0.0050)}
    assert t.temperature == pytest.approx(1450.0)
    assert t.observed == pytest.approx(56.0)


@pytest.mark.parametrize("subdir", ["MUQ-SAC", "Sensitivity_Analysis"])
def test_mono_fuel_bare_scalar_format_still_works(subdir):
    """A genuine bare-scalar Mono fuel line (no dict wrapping) must still
    parse via the original branch -- confirms the fix is additive, not a
    behaviour change for whatever (if anything) already relied on it."""
    mod_dir = str(_REPO / "src" / subdir)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    sys.modules.pop("combustion_target_class", None)
    import combustion_target_class

    addendum = yaml.safe_load(
        (_REPO / "DATASET" / "H2_O2_TEST" / "H2_O2.add").read_text()
    )
    real_line = (_REPO / "DATASET" / "H2_O2_TEST" / "targets" / "H2_IDT.target").read_text().splitlines()[0]
    bare_line = real_line.replace("x->{'a': 'H2'} = {'a': 0.0050}", "x->H2 = 0.0050")
    assert bare_line != real_line

    t = combustion_target_class.combustion_target(bare_line, deepcopy(addendum), 0)
    # Pre-existing (unrelated to this fix) quirk of the bare-scalar branch:
    # fuel_id keeps a trailing space from "H2 =" since it isn't stripped.
    assert {k.strip(): v for k, v in t.fuel_dict.items()} == {"H2": pytest.approx(0.0050)}
