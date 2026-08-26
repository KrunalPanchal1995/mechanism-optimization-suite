"""
Tests for mso_studio.core.automation. These exercise the integrated
src/automation/ code against its bundled example YAML and (where available)
the real external p_PRS factorial reports -- all read-only operations
(--validate, --status, preflight, meta-PRS fit all only read files).
"""
from pathlib import Path

import pytest

from mso_studio.core.automation import (
    AUTOMATION_DIR,
    available_fuels,
    fit_meta_prs,
    run_preflight,
    validate_yaml,
)

EXAMPLE_YAML = AUTOMATION_DIR / "examples" / "compare_only_minimal.yaml"


def test_automation_dir_has_expected_files():
    assert (AUTOMATION_DIR / "run_tasks.py").is_file()
    assert (AUTOMATION_DIR / "TASK_SCHEMA.md").is_file()
    assert EXAMPLE_YAML.is_file()


def test_validate_yaml_against_bundled_example():
    result = validate_yaml(EXAMPLE_YAML)
    assert result.ok, result.message
    assert "validation OK" in result.message


def test_run_preflight_scans_multiple_automation_dirs():
    result = run_preflight(planned_max_threads=1, cap=160)
    threads_detail = result.detail["threads"]
    assert len(threads_detail["scanned_dirs"]) >= 1
    assert any("automation" in d for d in threads_detail["scanned_dirs"])


def test_available_fuels_lists_the_three_known_fuels():
    fuels = available_fuels()
    assert set(fuels) == {"1_3_Butadiene", "n_Heptane_159", "MB"}


@pytest.mark.parametrize("fuel", ["1_3_Butadiene", "n_Heptane_159", "MB"])
def test_fit_meta_prs_against_real_data_if_present(fuel):
    """The factorial report .tex files live on external, multi-GB storage
    (MSO_PPRS_DATA_ROOT) that may not exist on every machine this test
    suite runs on -- skip gracefully rather than failing when it's absent."""
    import prs_extract_doe  # noqa: PLC0415 - only importable once AUTOMATION_DIR is on sys.path

    tex_path = prs_extract_doe.FUEL_REPORTS[fuel]
    if not Path(tex_path).is_file():
        pytest.skip(f"external factorial report not present on this machine: {tex_path}")

    result = fit_meta_prs(fuel)
    assert result.fuel == fuel
    assert result.n_points == 15
    assert isinstance(result.trusted, bool)
    assert result.loocv_max_abs_pct >= 0
