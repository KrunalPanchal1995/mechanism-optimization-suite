from pathlib import Path

import pytest

from mso_studio.core.backends import REPO_ROOT
from mso_studio.core.mso_reports import TargetTablesParams, generate_target_tables

SAMPLE_TARGET = REPO_ROOT / "DATASET" / "AMMONIA" / "AMMONIA.target"


@pytest.mark.skipif(not SAMPLE_TARGET.is_file(), reason="sample .target file not present in this checkout")
def test_generate_target_tables_end_to_end(tmp_path):
    """Exercises the real subprocess (target_tables.py under anaconda) against
    a real .target file already in the repo -- not mocked."""
    params = TargetTablesParams(
        target_csv=str(SAMPLE_TARGET), out_dir=str(tmp_path / "latex_tables"), compile_pdf=False,
    )
    tex_path = generate_target_tables(params)
    assert tex_path.exists()
    assert tex_path.suffix == ".tex"
    assert "\\begin" in tex_path.read_text()
