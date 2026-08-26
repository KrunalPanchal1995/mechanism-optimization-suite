"""
Regression tests for mso_compare/merge.py's φ/P-less dataset fallback.

Bug found stress-testing the launcher (2026-08): merge_simulation_results()
required EVERY dataset CSV filename to encode 'phi<N>_p<N>' (the convention
built for flame-speed/Fls datasets), but real ignition-delay (Tig/RCM/JSR)
'.target' files in this repo use opaque dataset ids (e.g. 'x10199020', Phi
literally 'N/A') -- so merge phases silently skipped every one of them
("Cannot parse φ/P from '...' — skipped."), producing an empty or
near-empty merged output with no error. Fixed by grouping unparseable
dataset ids under 'by_dataset_id/<id>.csv' instead of dropping them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from mso_compare.merge import merge_simulation_results, _fallback_group_and_filename  # noqa: E402


def _write_csv(folder: Path, name: str, ds_id: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "DS_ID": [ds_id, ds_id],
        "T": [1651.0, 1729.0],
        "Obs(us)": [1340.0, 841.0],
        "Nominal": [1300.0, 800.0],
    })
    df.to_csv(folder / name, index=False)


class TestFallbackGroupAndFilename:
    def test_sanitizes_an_opaque_dataset_id(self):
        folder, fname = _fallback_group_and_filename("x10199020")
        assert folder == "by_dataset_id"
        assert fname == "x10199020.csv"

    def test_sanitizes_unsafe_characters(self):
        _folder, fname = _fallback_group_and_filename("some/weird id?.csv")
        assert "/" not in fname and "?" not in fname and " " not in fname


class TestMergeDoesNotSilentlyDropOpaqueDatasetIds:
    def test_real_ignition_delay_style_dataset_id_survives_nominal_only(self, tmp_path):
        """The exact repro shape from real .target files: an opaque
        dataset id like 'x10199020', no phi/p token anywhere."""
        nominal = tmp_path / "nominal"
        _write_csv(nominal, "x10199020.csv", "x10199020")

        output_root = tmp_path / "out"
        merge_simulation_results(nominal, None, None, output_root)

        out_file = output_root / "nominal_only" / "by_dataset_id" / "x10199020.csv"
        assert out_file.exists(), (
            "Dataset with no φ/P in its id was silently dropped instead of "
            "being grouped under by_dataset_id/"
        )
        merged = pd.read_csv(out_file)
        assert list(merged["Nominal"]) == [1300.0, 800.0]

    def test_mixed_phi_and_opaque_dataset_ids_both_survive(self, tmp_path):
        """A flame-speed-style φ/P dataset and an ignition-delay-style
        opaque-id dataset in the same run must both come through."""
        nominal = tmp_path / "nominal"
        _write_csv(nominal, "ing_FUEL_HTC_phi0_5_p10.csv", "ing_FUEL_HTC_phi0_5_p10")
        _write_csv(nominal, "x10199020.csv", "x10199020")

        output_root = tmp_path / "out"
        merge_simulation_results(nominal, None, None, output_root)

        assert (output_root / "nominal_only" / "phi_0_5" / "Phi_0_5_P_10.csv").exists()
        assert (output_root / "nominal_only" / "by_dataset_id" / "x10199020.csv").exists()

    def test_stage1_merge_also_groups_opaque_dataset_ids(self, tmp_path):
        nominal = tmp_path / "nominal"
        stage1 = tmp_path / "stage1"
        _write_csv(nominal, "x10199020.csv", "x10199020")
        _write_csv(stage1, "x10199020.csv", "x10199020")

        output_root = tmp_path / "out"
        merge_simulation_results(nominal, stage1, None, output_root)

        out_file = output_root / "nominal_s1" / "by_dataset_id" / "x10199020.csv"
        assert out_file.exists()
        merged = pd.read_csv(out_file)
        assert "Stage-1" in merged.columns

    def test_two_opaque_dataset_ids_do_not_collide(self, tmp_path):
        """Two different opaque-id datasets must not overwrite each other --
        unlike the φ/P case, no explicit collision detection is needed here
        since the fallback filename IS the dataset id itself."""
        nominal = tmp_path / "nominal"
        _write_csv(nominal, "x10199020.csv", "x10199020")
        _write_csv(nominal, "x10199006.csv", "x10199006")

        output_root = tmp_path / "out"
        merge_simulation_results(nominal, None, None, output_root)

        assert (output_root / "nominal_only" / "by_dataset_id" / "x10199020.csv").exists()
        assert (output_root / "nominal_only" / "by_dataset_id" / "x10199006.csv").exists()
