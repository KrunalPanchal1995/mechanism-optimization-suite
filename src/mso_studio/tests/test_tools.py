from pathlib import Path

import numpy as np
import pytest

from mso_studio.core.backends import REPO_ROOT
from mso_studio.core.tools import gemini_available, pdf_extract_text, poly_fit

SAMPLE_PDF = REPO_ROOT / "MECHANISM" / "Davis_2005" / "Wang_Notes_on_Updated_Diffusion_Coefficients.pdf"


def test_poly_fit_linear_recovers_known_line():
    x = np.linspace(0, 10, 20)
    y = 3.0 * x + 2.0
    result = poly_fit(x, y, degree=1)
    assert result.degree == 1
    assert result.rmse < 1e-6
    # coefficients are high->low degree: [slope, intercept]
    assert result.coefficients[0] == pytest.approx(3.0, abs=1e-6)
    assert result.coefficients[1] == pytest.approx(2.0, abs=1e-6)
    assert "x" in result.latex_equation


def test_poly_fit_produces_fine_grid_spanning_input_range():
    x = np.array([0.0, 5.0, 10.0])
    y = np.array([1.0, 2.0, 1.0])
    result = poly_fit(x, y, degree=2)
    assert len(result.x_fine) == 300
    assert min(result.x_fine) == pytest.approx(0.0)
    assert max(result.x_fine) == pytest.approx(10.0)


def test_gemini_available_reflects_env_var(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert gemini_available() is False
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")
    assert gemini_available() is True


@pytest.mark.skipif(not SAMPLE_PDF.is_file(), reason="sample reference PDF not present in this checkout")
def test_pdf_extract_text_end_to_end_via_anaconda_subprocess():
    """Exercises the real subprocess bridge (tools_worker.py under the
    anaconda interpreter, pdfplumber) against a real PDF already in the repo
    -- not mocked, matching how the automation meta-PRS tests are verified
    against real data rather than fixtures.
    """
    result = pdf_extract_text(str(SAMPLE_PDF))
    assert result.n_pages >= 1
    assert len(result.text) > 0
