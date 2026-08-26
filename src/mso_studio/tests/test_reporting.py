from mso_studio.core.reporting import build_report, tex_escape
from mso_studio.core.runrecord import RunConfig, new_record


def _sample_record():
    cfg = RunConfig(
        module="nominal_sim", label="nominal_htc_100%_run",
        backend_command=["python3.9", "run_nominal_sim.py", "--config", "x.yaml"],
        working_dir="/tmp/work",
        inputs={"mechanism": "/a/mech_v2.yaml"},
        parameters={"targets_count": 52},
    )
    record = new_record(cfg)
    record.results["n_datasets"] = 4
    record.warnings.append("Case 3 produced no result")
    record.log_tail = ["line one", "line two"]
    record.mark_finished("done", return_code=0)
    return record


def test_tex_escape_handles_specials():
    assert tex_escape("50%_done") == r"50\%\_done"
    assert tex_escape("a_b#c") == r"a\_b\#c"


def test_build_report_never_dead_ends(tmp_path):
    """No LaTeX toolchain is installed in this environment -- this exercises
    the guidelines doc's rung-3 fallback: a .tex + report_bundle.zip that
    lets the user compile locally or on Overleaf, never a bare error.
    """
    record = _sample_record()
    result = build_report(record, tmp_path)

    assert result.tex_path.exists()
    tex_source = result.tex_path.read_text()
    # The label contains '%' and '_' -- must be escaped, not raw, in the body.
    assert r"nominal\_htc\_100\%\_run" in tex_source

    if result.pdf_path is None:
        # No pdflatex/tectonic/latexmk on this machine -> bundle fallback.
        assert result.bundle_path is not None
        assert result.bundle_path.exists()
        assert result.engine == "none"
    else:
        assert result.pdf_path.exists()
