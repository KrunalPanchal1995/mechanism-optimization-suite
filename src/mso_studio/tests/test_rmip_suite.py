"""
Tests for mso_studio.core.rmip_suite -- the PySide6 port of RMIP Suite's 6
tabs (KRUNAL_JULY_23.md: PySide6 conversion). None of these 6 needed a new
headless CLI mode (5 already had `rmip_cli.py` subcommands, 1 already had
its own argparse CLI) -- unlike the converter/fusion/uncertainty-generator
ports.
"""
import sys
from pathlib import Path

import pytest

from mso_studio.core.backends import REPO_ROOT, SCIENCE_PYTHON
from mso_studio.core.rmip_suite import (
    GraphCompareParams, run_graph_compare, serve_graph_compare,
    ReduceParams, run_reduce,
    GsaParams, run_gsa,
    ActiveParamSelectionParams, run_active_param_selection,
    TargetBuilderParams, run_build_target,
    RespecthImportParams, run_import_respecth,
)


def _fake_backend(monkeypatch, module, tail=None, return_code=0, status="done"):
    captured = {}

    def fake_run_backend(command, **kwargs):
        captured["command"] = command
        from mso_studio.core.backends import BackendResult
        return BackendResult(command=list(command), return_code=return_code,
                             status=status, tail=tail or [], log_path=None)

    monkeypatch.setattr(f"mso_studio.core.rmip_suite.run_backend", fake_run_backend)
    return captured


class TestGraphCompare:
    def test_builds_expected_argv(self, monkeypatch, tmp_path):
        captured = _fake_backend(monkeypatch, "graph_compare")
        params = GraphCompareParams(db_path="/a/p.db", mechanism_ids=[2, 3, 4],
                                    output_dir=str(tmp_path))
        record = run_graph_compare(params)
        cmd = captured["command"]
        assert cmd[0] == SCIENCE_PYTHON
        assert str(REPO_ROOT / "rmip_cli.py") in cmd
        assert "compare-graph" in cmd
        assert "--mechs" in cmd and "2" in cmd and "4" in cmd
        assert record.status == "done"

    def test_collects_output_jsons(self, monkeypatch, tmp_path):
        (tmp_path / "graph_classification.json").write_text("{}")
        (tmp_path / "union_graph_spec.json").write_text("{}")
        _fake_backend(monkeypatch, "graph_compare")
        params = GraphCompareParams(db_path="/a/p.db", mechanism_ids=[2, 3],
                                    output_dir=str(tmp_path))
        record = run_graph_compare(params)
        assert record.results["classification_json"] == str(tmp_path / "graph_classification.json")
        assert record.results["union_graph_spec_json"] == str(tmp_path / "union_graph_spec.json")


class TestServeGraphCompare:
    """KRUNAL_JULY_23.md point 7 PySide6 link-surfacing (Phase 5c, bug.md):
    serve_graph_compare launches a DETACHED subprocess (never routed through
    run_backend, which waits for EOF on the child's stdout -- a Dash dev
    server never exits, so that would block until DEFAULT_TIMEOUT_S)."""

    def test_builds_expected_argv_with_serve_flags(self, monkeypatch, tmp_path):
        captured = {}

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            class _FakeProc:
                pid = 12345
                def poll(self): return None
                def terminate(self): pass
                def wait(self, timeout=None): pass
            return _FakeProc()

        monkeypatch.setattr("mso_studio.core.rmip_suite.subprocess.Popen", fake_popen)
        params = GraphCompareParams(db_path="/a/p.db", mechanism_ids=[2, 3, 4],
                                    output_dir=str(tmp_path))
        proc = serve_graph_compare(params, port=8099, label_mode="id")

        cmd = captured["command"]
        assert "--serve" in cmd
        assert "--port" in cmd and "8099" in cmd
        assert "--label" in cmd and "id" in cmd
        assert captured["kwargs"]["cwd"] == str(tmp_path)
        assert proc.pid == 12345

    @pytest.mark.skipif(not Path(SCIENCE_PYTHON).exists() or SCIENCE_PYTHON == "python3.9",
                        reason="anaconda science interpreter not found on this machine")
    def test_real_server_starts_and_is_reachable_then_stops(self, tmp_path):
        """Real end-to-end: launches an actual Dash server via the real
        bridge (.venv-gui -> anaconda -> rmip_cli.py compare-graph --serve),
        polls it for a real HTTP response, then terminates it and confirms
        the process is actually gone -- not mocked."""
        import time
        import urllib.request

        sys.path.insert(0, str(REPO_ROOT / "src" / "rmip" / "tests"))
        from test_stubs_fusion import _build_h2_fixture

        db, mid1, mid2 = _build_h2_fixture(tmp_path)
        params = GraphCompareParams(
            db_path=db, mechanism_ids=[mid1, mid2], output_dir=str(tmp_path / "out"))

        port = 18077
        proc = serve_graph_compare(params, port=port, label_mode="formula")
        try:
            url = f"http://localhost:{port}"
            reached = False
            for _ in range(30):  # up to ~15s for the Dash dev server to bind
                if proc.poll() is not None:
                    break  # died early -- fail fast below with useful state
                try:
                    resp = urllib.request.urlopen(url, timeout=1)
                    reached = resp.status == 200
                    break
                except Exception:
                    time.sleep(0.5)
            assert reached, f"server never answered on {url} (proc alive: {proc.poll() is None})"
        finally:
            proc.terminate()
            proc.wait(timeout=5)

        assert proc.poll() is not None  # confirmed actually terminated, not leaked


class TestReduce:
    def test_requires_target_species(self, tmp_path):
        params = ReduceParams(db_path="/a/p.db", mechanism_id="2", target_species="",
                              output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="target_species"):
            run_reduce(params)

    def test_builds_expected_argv_with_conditions(self, monkeypatch, tmp_path):
        captured = _fake_backend(monkeypatch, "reduce")
        params = ReduceParams(
            db_path="/a/p.db", mechanism_id="2", target_species="H2,O2",
            conditions=[(1000.0, 1.0, 1.0, "H2:1.0")], output_dir=str(tmp_path))
        run_reduce(params)
        cmd = captured["command"]
        assert "reduce" in cmd and "2" in cmd
        assert "--condition" in cmd and "1000.0,1.0,1.0,H2:1.0" in cmd


class TestGsa:
    def test_requires_fuel(self, tmp_path):
        params = GsaParams(db_path="/a/p.db", mechanism_id="2", fuel="", output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="fuel"):
            run_gsa(params)

    def test_builds_expected_argv(self, monkeypatch, tmp_path):
        captured = _fake_backend(monkeypatch, "gsa")
        params = GsaParams(db_path="/a/p.db", mechanism_id="2", fuel="H2:1.0",
                           output_dir=str(tmp_path))
        run_gsa(params)
        cmd = captured["command"]
        assert "gsa-run" in cmd and "H2:1.0" in cmd


class TestActiveParamSelection:
    def test_requires_classification_for_xml(self, tmp_path):
        params = ActiveParamSelectionParams(
            sens_dir=".", mechanism="m.yaml", generate_xml=True,
            results_dir=str(tmp_path))
        with pytest.raises(ValueError, match="classification"):
            run_active_param_selection(params)

    def test_builds_expected_argv(self, monkeypatch, tmp_path):
        captured = _fake_backend(monkeypatch, "active_param_selection")
        params = ActiveParamSelectionParams(
            sens_dir="/a/sa", mechanism="/a/m.yaml", results_dir=str(tmp_path))
        run_active_param_selection(params)
        cmd = captured["command"]
        assert cmd[0] == SCIENCE_PYTHON
        assert str(REPO_ROOT / "src" / "ActiveParameterScreen.py") in cmd
        assert "--sens-dir" in cmd and "/a/sa" in cmd

    def test_collects_result_csvs(self, monkeypatch, tmp_path):
        (tmp_path / "active_species.csv").write_text("a,b\n")
        _fake_backend(monkeypatch, "active_param_selection")
        params = ActiveParamSelectionParams(
            sens_dir="/a/sa", mechanism="/a/m.yaml", results_dir=str(tmp_path))
        record = run_active_param_selection(params)
        assert record.tables["active_species.csv"] == str(tmp_path / "active_species.csv")


class TestTargetBuilder:
    def test_requires_fuel_blend(self, tmp_path):
        params = TargetBuilderParams(
            db_path="/a/p.db", mechanism_id="2", base_target_opt="/a/t.opt",
            fuel_blend=[], work_dir=str(tmp_path))
        with pytest.raises(ValueError, match="fuel-blend"):
            run_build_target(params)

    def test_builds_expected_argv(self, monkeypatch, tmp_path):
        captured = _fake_backend(monkeypatch, "target_builder")
        params = TargetBuilderParams(
            db_path="/a/p.db", mechanism_id="2", base_target_opt="/a/t.opt",
            fuel_blend=[("H2", 1.0)], work_dir=str(tmp_path),
            extra_fields={"--residence-time": "1.0", "--flux": True})
        run_build_target(params)
        cmd = captured["command"]
        assert "build-target" in cmd and "H2:1.0" in cmd
        assert "--residence-time" in cmd and "1.0" in cmd
        assert "--flux" in cmd


class TestRespecthImport:
    def test_requires_xml_path(self, tmp_path):
        params = RespecthImportParams(db_path="/a/p.db", xml_path="")
        with pytest.raises(ValueError, match="XML"):
            run_import_respecth(params)

    def test_builds_expected_argv(self, monkeypatch, tmp_path):
        captured = _fake_backend(monkeypatch, "respecth_import")
        params = RespecthImportParams(db_path="/a/p.db", xml_path="/a/data.xml",
                                      out_dir=str(tmp_path), no_db=True)
        run_import_respecth(params)
        cmd = captured["command"]
        assert "import-respecth" in cmd and "/a/data.xml" in cmd
        assert "--no-db" in cmd


@pytest.mark.skipif(not Path(SCIENCE_PYTHON).exists() or SCIENCE_PYTHON == "python3.9",
                    reason="anaconda science interpreter not found on this machine")
def test_run_graph_compare_real_bridge_h2_fixture(tmp_path):
    """The actual bridge: .venv-gui -> anaconda python3.9 -> rmip_cli.py
    compare-graph, against the same H2-combustion DB fixture the fusion
    port's real-bridge test uses."""
    sys.path.insert(0, str(REPO_ROOT / "src" / "rmip" / "tests"))
    from test_stubs_fusion import _build_h2_fixture

    db, mid1, mid2 = _build_h2_fixture(tmp_path)
    params = GraphCompareParams(
        db_path=db, mechanism_ids=[mid1, mid2], output_dir=str(tmp_path / "out"))
    record = run_graph_compare(params)

    assert record.status == "done"
    assert Path(record.results["classification_json"]).exists()
    assert Path(record.results["union_graph_spec_json"]).exists()
