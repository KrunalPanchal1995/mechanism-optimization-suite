"""
Tests for mso_studio.core.target_workbench -- the subprocess bridge to
scripts/target_workbench_worker.py (VisualAid.TargetSet/TargetChecker/
DummyPointGenerator/TargetPruner need sklearn/scipy, not installed in the
light GUI venv). Runs the real bridge (.venv-gui -> anaconda python3.9)
against the shipped DATASET/AMMONIA/AMMONIA.target fixture -- same
real-bridge convention as test_forward_uq.py's
test_run_forward_uq_real_bridge_with_synthetic_prs_cache, no mocking.
"""
from pathlib import Path

import pytest

from mso_studio.core.backends import REPO_ROOT
from mso_studio.core import target_workbench as twb_core

AMMONIA_TARGET = REPO_ROOT / "DATASET" / "AMMONIA" / "AMMONIA.target"
AMMONIA_ADD = REPO_ROOT / "DATASET" / "AMMONIA" / "addundem.add"


def _skip_if_missing():
    if not AMMONIA_TARGET.exists():
        pytest.skip("DATASET/AMMONIA/AMMONIA.target fixture not present")


def test_load_target_set_real_bridge():
    _skip_if_missing()
    result = twb_core.load_target_set(str(AMMONIA_TARGET), str(AMMONIA_ADD))
    assert len(result.rows) > 0
    assert len(result.dataset_ids) > 0
    first = result.rows[0]
    assert first.line_no == 1
    assert first.dataset_id


def test_run_checks_real_bridge():
    _skip_if_missing()
    findings = twb_core.run_checks(str(AMMONIA_TARGET), str(AMMONIA_ADD))
    assert len(findings) > 0
    assert all(f.severity in ("error", "warning", "info") for f in findings)


def test_generate_dummy_points_real_bridge():
    """Known-good dataset from tests/test_target_workbench.py's own
    TestDummyDataGenerationRegression -- has a real, detectable NTC region."""
    mb_target = REPO_ROOT / "DATASET" / "METHYL_BUTANOATE_2026" / "MB_LTC_no_dummy.target"
    if not mb_target.exists():
        pytest.skip("MB_LTC_no_dummy.target fixture not present")
    result = twb_core.generate_dummy_points(
        str(mb_target), "", "ing_Ethyl_Butanoate_PHI_1_P_30", points_per_zone=5)
    assert len(result.rows) > 0
    assert result.template_dataset_id == "ing_Ethyl_Butanoate_PHI_1_P_30"
    for row in result.rows:
        assert "Temperature_K" in row and "observed" in row


def test_coverage_prune_real_bridge():
    _skip_if_missing()
    result = twb_core.coverage_prune(str(AMMONIA_TARGET), str(AMMONIA_ADD), n_drop=5, seed=0)
    assert len(result.drop_line_nos) == 5
    assert 0.0 <= result.coverage <= 1.0
    assert set(result.drop_line_nos).isdisjoint(result.keep_line_nos)


def test_write_curated_real_bridge_gates_clean(tmp_path):
    mb_target = REPO_ROOT / "DATASET" / "METHYL_BUTANOATE_2026" / "MB_LTC_no_dummy.target"
    if not mb_target.exists():
        pytest.skip("MB_LTC_no_dummy.target fixture not present")
    out_dir = tmp_path / "out"
    result = twb_core.write_curated(
        str(mb_target), "", "test_curated", keep_line_nos=list(range(1, 20)),
        out_dir=str(out_dir),
    )
    assert Path(result.new_target_path).exists()
    assert Path(result.new_add_path).exists()
    assert Path(result.log_path).exists()
    assert result.gate_errors == []


def test_render_plot_real_bridge(tmp_path):
    _skip_if_missing()
    out_png = tmp_path / "plot.png"
    png_path = twb_core.render_plot(
        str(AMMONIA_TARGET), str(AMMONIA_ADD), "t_p_dataset", str(out_png))
    assert Path(png_path).exists()
    assert Path(png_path).stat().st_size > 0


def test_unknown_worker_mode_raises():
    with pytest.raises(RuntimeError, match="Unknown mode"):
        twb_core._call_worker("nope", {})
