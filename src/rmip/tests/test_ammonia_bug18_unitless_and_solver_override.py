"""
rmip/tests/test_ammonia_bug18_unitless_and_solver_override.py
================================================================
Regression coverage for Ammonia campaign Bug 18 (AMMONIA_CAMPAIGN_2026-07-22.md,
bug.md): the MUQ-SAC engine invoked ANSYS Chemkin-Pro (rc=41) for an ammonia
Tig target declaring `solver: cantera`, while the identical mechanism+targets
ran successfully via pure Cantera through the Sensitivity_Analysis engine.
Two independent, stacked causes, both fixed:

1. `src/MUQ-SAC/data_management.py::run_sim`'s retry path hardcoded
   `solver="CHEMKIN_PRO"` in its `create_input_file` call, bypassing the
   `solver=None -> target.add["solver"]` fallback (`make_input_file.py:
   329-330`). The sibling `src/Sensitivity_Analysis/data_management.py`
   never passed a solver kwarg at all -- the correct reference behavior.
2. `src/MUQ-SAC/make_input_file.py`'s embedded `ignitionDelay()` template
   (emitted into every generated Tig/RCM run script) had ZERO `unitless`
   branches, while `src/Sensitivity_Analysis/make_input_file.py` has 8
   (identical) copies of the branch. Every ammonia target uses
   `specific_cond: 0.95;unitless` (the only simulatable measurement type
   for a standard ground-state NH3 mechanism), so the generated script's
   `ignitionDelay()` fell through every elif with `tau` never assigned,
   raising `UnboundLocalError` -> `eta="N/A"` -> triggered the retry path
   -> hit cause 1's hardcoded Chemkin-Pro -> rc=41. Ported the identical
   29-line block (verified byte-identical across all 8 SA occurrences) into
   all 8 of MUQ-SAC's matching `ignitionDelay(df,pList,...)` occurrences
   (the 2 differently-shaped FlameMaster-profile `ignitionDelay(df,species,
   ...)` occurrences were correctly left untouched -- no `unitless` concept
   applies there).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_MUQSAC_MIF = _ROOT / "src" / "MUQ-SAC" / "make_input_file.py"
_SA_MIF = _ROOT / "src" / "Sensitivity_Analysis" / "make_input_file.py"


def _extract_ignition_delay_and_helper(path: Path) -> str:
    """Extract the first `ignitionDelay(df,pList,...)` function plus its
    sibling `fast_nearest_interp` helper as literal source text (both are
    embedded string-literal content inside make_input_file.py's templates,
    not top-level module code -- see the file's own generation logic)."""
    text = path.read_text()
    lines = text.splitlines(keepends=True)

    fn_start = next(
        i for i, l in enumerate(lines)
        if l.startswith("def ignitionDelay(") and "pList" in l
    )
    fn_end = fn_start
    while lines[fn_end].strip() != "return tau":
        fn_end += 1
    ignition_delay_src = "".join(lines[fn_start:fn_end + 1])

    helper_start = next(
        i for i, l in enumerate(lines) if l.startswith("def fast_nearest_interp(")
    )
    helper_end = helper_start
    while lines[helper_end].strip() != "return y[np.searchsorted(x, xi)]":
        helper_end += 1
    helper_src = "".join(lines[helper_start:helper_end + 1])

    # Both blocks are indented with a leading tab (embedded at that level in
    # the real templates) -- dedent for standalone exec.
    return textwrap_dedent(helper_src) + "\n" + textwrap_dedent(ignition_delay_src)


def textwrap_dedent(src: str) -> str:
    import textwrap
    return textwrap.dedent(src)


def _run_ignition_delay(path: Path):
    """exec the extracted ignitionDelay/fast_nearest_interp pair in an
    isolated namespace and return the callable, for direct invocation."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    scipy_interp = pytest.importorskip("scipy.interpolate")

    src = _extract_ignition_delay_and_helper(path)
    ns = {
        "np": np, "Akima1DInterpolator": scipy_interp.Akima1DInterpolator,
    }
    exec(compile(src, str(path), "exec"), ns)
    return ns["ignitionDelay"]


def _synthetic_consumption_profile(pd, np):
    """A synthetic monotonically-decreasing fuel-concentration trace, like a
    real Tig species mole-fraction history: starts at 1.0, decays smoothly
    to near-zero as the mixture ignites and consumes the fuel."""
    time = np.linspace(0.0, 1.0, 200)
    conc = np.exp(-5.0 * time)  # smooth decay, well past 95% consumption by t=1.0
    df = pd.DataFrame({"NH3": conc}, index=time)
    return df


class TestUnitlessBranchParity:
    """Both MUQ-SAC's (now-ported) and Sensitivity_Analysis's (original)
    ignitionDelay() must handle specified_value="0.95;unitless" identically
    -- no UnboundLocalError, and physically the SAME answer (same formula:
    target_conc = conc[0] * (1 - target), i.e. 95% of the initial reactant
    consumed)."""

    @pytest.mark.parametrize("mif_path", [_MUQSAC_MIF, _SA_MIF], ids=["MUQ-SAC", "Sensitivity_Analysis"])
    def test_unitless_does_not_raise_and_returns_finite_tau(self, mif_path):
        import numpy as np
        import pandas as pd

        ignition_delay = _run_ignition_delay(mif_path)
        df = _synthetic_consumption_profile(pd, np)
        tau = ignition_delay(
            df, pList=None, species="NH3", cond="specific",
            specified_value="0.95;unitless",
        )
        # The point of this assertion is simply that `tau` got assigned at
        # all (the pre-fix bug was UnboundLocalError -- `tau` never set by
        # any elif branch) -- not a claim about fast_nearest_interp's
        # numerical behavior on a monotonically-decreasing profile, which is
        # separate, pre-existing code this fix doesn't touch.
        assert np.isfinite(tau)

    def test_muqsac_and_sa_agree_on_the_same_synthetic_profile(self):
        """Direct parity check: both copies must compute the same tau for
        the identical input, confirming the ported block is byte-identical
        in behavior, not just byte-identical in text."""
        import numpy as np
        import pandas as pd

        muqsac_fn = _run_ignition_delay(_MUQSAC_MIF)
        sa_fn = _run_ignition_delay(_SA_MIF)

        df = _synthetic_consumption_profile(pd, np)
        tau_muqsac = muqsac_fn(df, pList=None, species="NH3", cond="specific",
                                specified_value="0.95;unitless")
        tau_sa = sa_fn(df, pList=None, species="NH3", cond="specific",
                        specified_value="0.95;unitless")
        assert tau_muqsac == pytest.approx(tau_sa)

    def test_all_8_muqsac_occurrences_have_the_unitless_branch(self):
        """Every one of MUQ-SAC's 8 pList-signature ignitionDelay() copies
        (one per solver/target-type combination that reaches this template)
        must have gained the branch -- not just the first one exec'd above."""
        text = _MUQSAC_MIF.read_text()
        assert text.count('elif unit == "unitless"') == 8

    def test_flamemaster_profile_variant_untouched(self):
        """The 2 differently-shaped `ignitionDelay(df,species,...)`
        occurrences (FlameMaster's own profile format -- no `pList`/
        `exp_conc` params, no molecule/cm3 unit-conversion branches at all)
        must NOT have gained a spurious unitless branch -- there's no
        matching concept there."""
        text = _MUQSAC_MIF.read_text()
        other_sig_count = len(re.findall(
            r'def ignitionDelay\(df,species,cond="max"', text,
        ))
        assert other_sig_count == 2


class TestSolverOverrideRemoved:
    """data_management.py::run_sim's retry path must no longer hardcode
    solver="CHEMKIN_PRO" -- it must fall through to create_input_file's own
    solver=None -> target.add["solver"] default, matching the
    Sensitivity_Analysis sibling exactly."""

    def test_muqsac_data_management_no_longer_hardcodes_chemkin_pro(self):
        """No CODE line (as opposed to the fix's own explanatory comment,
        which legitimately names the old value for documentation) still
        passes solver="CHEMKIN_PRO" to create_input_file."""
        text = (_ROOT / "src" / "MUQ-SAC" / "data_management.py").read_text()
        code_lines = [
            line for line in text.splitlines() if not line.strip().startswith("#")
        ]
        assert not any('solver="CHEMKIN_PRO"' in line for line in code_lines)
        assert not any("solver='CHEMKIN_PRO'" in line for line in code_lines)

    def test_create_input_file_call_matches_sa_sibling_shape(self):
        """The MUQ-SAC call site's kwargs must now match the SA sibling's
        (no solver= kwarg at all), confirmed via a direct source match of
        the call site itself, not just an absence-of-string check above."""
        muqsac_text = (_ROOT / "src" / "MUQ-SAC" / "data_management.py").read_text()
        m = re.search(
            r"MakeFile\.create_input_file\(\s*caseID,\s*input_,\s*case,\s*mech_file=mech_path,?\s*\)",
            muqsac_text,
        )
        assert m is not None, "expected create_input_file call with no solver kwarg"

    def test_create_input_file_still_accepts_no_solver_and_falls_back(self):
        """End-to-end confirmation that dropping the hardcoded kwarg
        actually restores the intended fallback behavior: calling
        create_input_file with solver=None (its default) must pick up
        target.add["solver"], not silently default to something else."""
        import sys
        sys.path.insert(0, str(_ROOT / "src" / "MUQ-SAC"))
        import make_input_file as mif

        class _FakeTarget:
            target = "JSR"
            input_file = None
            fuel = "CH4"
            pressure = 1013250.0
            temperature = 1200.0
            species_dict = {"CH4": 0.1, "O2": 0.2, "N2": 0.7}
            add = {
                "solver": "cantera", "species": "CH4", "residenceTime": 0.05,
                "MaxSimulationTime": 0.2, "pfr_length": 0.1, "pfr_diameter": 0.02,
                "pfr_mdot": 1e-3, "pfr_n_points": 50,
                "flux_analysis": False, "flux_element": "C",
            }

        opt_dict = {
            "Locations": {"thermo_file": "x", "trans_file": "y"},
            "StartProfilesData": {}, "Inputs": {"global_reaction": ""},
            "Bin": {"bin": "x", "solver_bin": "y"},
        }
        pytest.importorskip("yaml")
        instring, *_ = mif.create_input_file(
            0, opt_dict, _FakeTarget(), mech_file="mechanism.yaml",
        )  # no solver kwarg at all, matching the fixed call site
        assert "import cantera as ct" in instring
