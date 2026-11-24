"""
Unit tests for the 2026-08-29 SA-vs-p-PRS cutoff decoupling work:

1. correlation_analysis.final_active_set_per_target's rescue test used to be
   `sens_dict[case][rxn] > threshold_corr` (no abs()) -- a reaction with
   strongly NEGATIVE sensitivity could never be rescued by correlation, even
   though the function's own docstring says "rescued only if |S| >
   threshold_corr". Fixed to `abs(sens_dict[case][rxn]) > threshold_corr`.

2. PartialPRS.__init__ now detects when a case's freshly-computed selection
   (current cut_off_percentage + correlation_list) no longer matches a
   cached DM_FOR_PARTIAL_PRS/<case>/selected_parameters.csv, and records it
   in `self.stale_selection` (None when it matches or no cache exists yet;
   else (added, removed) parameter lists) -- WITHOUT deleting, renaming, or
   rebuilding anything itself (see run_script.py's Phase 6 loop for what
   happens with it).

Run: python3.9 tests/test_sa_pprs_cutoffs.py
"""
import os
import sys
import shutil
import tempfile

import numpy as np
import pandas as pd

_MUQSAC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MUQSAC_ROOT not in sys.path:
    sys.path.append(_MUQSAC_ROOT)

import correlation_analysis as CA
import partial_PRS_system as P_PRS


# ─── 1. correlation_analysis abs() rescue fix ──────────────────────────────

def test_negative_sensitivity_reaction_can_be_rescued():
    """RXN2 has strong NEGATIVE sensitivity (-0.3) and is highly correlated
    (0.95) with RXN1, which clears the raw threshold. Before the abs() fix,
    RXN2 could never be rescued (bare `> threshold_corr` on a negative number
    is always False for any positive threshold_corr); after the fix it must
    be, since |-0.3| = 0.3 > threshold_corr = 0.01."""
    tmp = tempfile.mkdtemp()
    try:
        pickle_path = os.path.join(tmp, "sens.pkl")
        # 2 cases so wide_frames has >1 row (not required by this function,
        # but matches realistic shape); RXN1 always clears threshold=0.05,
        # RXN2 never does on its own.
        sens_dict = {
            "case0": {"RXN1": 0.20, "RXN2": -0.30},
            "case1": {"RXN1": 0.18, "RXN2": -0.28},
        }
        import pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(sens_dict, f)

        corr_csv_path = os.path.join(tmp, "correlation_matrix_A.csv")
        rho = pd.DataFrame([[1.0, 0.95], [0.95, 1.0]],
                            index=["RXN1", "RXN2"], columns=["RXN1", "RXN2"])
        rho.to_csv(corr_csv_path)

        final_sets = CA.final_active_set_per_target(
            pickle_path, corr_csv_path, tmp, param_name="A",
            threshold=0.05, corr_cutoff=0.9, threshold_corr=0.01)

        assert "RXN2" in final_sets["case0"], (
            f"RXN2 (negative sensitivity, correlated 0.95 with selected RXN1) "
            f"should be rescued; final set was {final_sets['case0']}")
        print("PASS: negative-sensitivity reaction IS rescued by correlation "
              "(abs() fix)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── 2. PartialPRS stale-selection detection ───────────────────────────────

class _MockUnsrt:
    """Minimal stand-in exposing only what PartialPRS's A-facto branch reads
    directly off an unsrt_data value: activeParameters[0] and linked_rIndex."""
    def __init__(self, a_param, linked_rIndex=None):
        self.activeParameters = [a_param]
        self.linked_rIndex = linked_rIndex


def _make_partial_prs(sensitivity_dict, cut_off_percentage=5):
    unsrt_data = {"rxn1": _MockUnsrt("rxn1_A")}
    optInputs = {"Stats": {"cut_off_percentage": cut_off_percentage}}
    return P_PRS.PartialPRS(
        sensitivity_dict, [], unsrt_data, optInputs, [], "0", [],
        "A-facto", status="Pending")


def test_stale_selection_none_when_no_cache_or_matching_cache():
    tmp = tempfile.mkdtemp()
    cwd0 = os.getcwd()
    os.chdir(tmp)
    try:
        # Well above cutoff (5%) -> selected.
        pp = _make_partial_prs({"rxn1": 0.5})
        assert pp.stale_selection is None, "no cache yet -- must not be stale"
        assert pp.selected_rxn_string.strip() == "rxn1_A"

        os.makedirs("DM_FOR_PARTIAL_PRS/0", exist_ok=True)
        with open("DM_FOR_PARTIAL_PRS/0/selected_parameters.csv", "w") as f:
            f.write("rxn1_A\n")

        pp2 = _make_partial_prs({"rxn1": 0.5})
        assert pp2.stale_selection is None, "cache matches fresh selection -- must not be stale"
        print("PASS: stale_selection is None when there's no cache, and when "
              "the cache matches the current selection")
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp, ignore_errors=True)


def test_stale_selection_detects_threshold_change():
    tmp = tempfile.mkdtemp()
    cwd0 = os.getcwd()
    os.chdir(tmp)
    try:
        os.makedirs("DM_FOR_PARTIAL_PRS/0", exist_ok=True)
        with open("DM_FOR_PARTIAL_PRS/0/selected_parameters.csv", "w") as f:
            f.write("rxn1_A\n")   # cached selection: rxn1_A WAS selected

        # Below the 5% cutoff now (simulates a stricter/looser threshold
        # change since the cache was built) -> no longer selected.
        pp = _make_partial_prs({"rxn1": 0.001})
        assert pp.stale_selection is not None, "selection changed -- must be flagged stale"
        added, removed = pp.stale_selection
        assert added == [] and removed == ["rxn1_A"], (added, removed)
        print("PASS: stale_selection correctly detects a selection that "
              "dropped a parameter the cache had")

        # And PartialPRS must not have touched the cached file itself --
        # detection only, never action.
        with open("DM_FOR_PARTIAL_PRS/0/selected_parameters.csv") as f:
            assert f.read() == "rxn1_A\n", "cached file must be untouched by detection"
        print("PASS: detection never modifies the cached selected_parameters.csv")
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_negative_sensitivity_reaction_can_be_rescued()
    test_stale_selection_none_when_no_cache_or_matching_cache()
    test_stale_selection_detects_threshold_change()
    print("ALL TESTS PASSED")
