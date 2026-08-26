"""Widget-level tests for RMIPSuitePage (KRUNAL_JULY_23.md PySide6 conversion)."""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_page_registered_in_page_factories():
    from mso_studio.gui.modules import PAGE_FACTORIES
    ids = [mid for mid, _label, _factory in PAGE_FACTORIES]
    assert "rmip_suite" in ids


def test_page_has_all_six_tabs(qapp):
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage
    page = RMIPSuitePage()
    for attr in ("graph_compare_tab", "reduce_tab", "gsa_tab", "active_param_tab",
                "target_builder_tab", "respecth_import_tab"):
        assert hasattr(page, attr)


def test_graph_compare_tab_requires_db_and_two_ids(qapp, tmp_path):
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage
    page = RMIPSuitePage()
    with pytest.raises(ValueError, match="database"):
        page.graph_compare_tab.build_core_call()

    db = tmp_path / "p.db"
    db.write_text("")
    page.db_picker.set_path(str(db))
    page.graph_compare_tab.mech_ids_edit.setText("2")
    with pytest.raises(ValueError, match="at least 2"):
        page.graph_compare_tab.build_core_call()


def test_graph_compare_tab_returns_run_graph_compare(qapp, tmp_path):
    from mso_studio.core.rmip_suite import GraphCompareParams, run_graph_compare
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage

    db = tmp_path / "p.db"
    db.write_text("")
    page = RMIPSuitePage()
    page.db_picker.set_path(str(db))
    page.graph_compare_tab.mech_ids_edit.setText("2,3,4")

    fn, args, kwargs = page.graph_compare_tab.build_core_call()
    assert fn is run_graph_compare
    assert kwargs == {}
    (params,) = args
    assert isinstance(params, GraphCompareParams)
    assert params.mechanism_ids == [2, 3, 4]


def test_reduce_tab_requires_mechanism_id(qapp):
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage
    page = RMIPSuitePage()
    with pytest.raises(ValueError):
        page.reduce_tab.build_core_call()


def test_active_param_tab_requires_mechanism(qapp):
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage
    page = RMIPSuitePage()
    with pytest.raises(ValueError, match="mechanism"):
        page.active_param_tab.build_core_call()


def test_target_builder_tab_fuel_row_and_build_call(qapp, tmp_path):
    from mso_studio.core.rmip_suite import TargetBuilderParams, run_build_target
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage

    db = tmp_path / "p.db"
    db.write_text("")
    page = RMIPSuitePage()
    page.db_picker.set_path(str(db))
    tab = page.target_builder_tab
    tab.mech_id_edit.setText("2")
    tab.base_opt_picker.set_path(str(tmp_path / "t.opt"))
    tab.work_dir_picker.set_path(str(tmp_path / "work"))
    tab._fuel_sp_edit.setText("H2")
    tab._fuel_frac_edit.setText("1.0")
    tab._on_add_fuel_row()
    assert tab.fuel_table.rowCount() == 1

    fn, args, kwargs = tab.build_core_call()
    assert fn is run_build_target
    (params,) = args
    assert isinstance(params, TargetBuilderParams)
    assert params.fuel_blend == [("H2", 1.0)]


def test_respecth_import_tab_requires_xml(qapp, tmp_path):
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage
    db = tmp_path / "p.db"
    db.write_text("")
    page = RMIPSuitePage()
    page.db_picker.set_path(str(db))
    with pytest.raises(ValueError, match="XML"):
        page.respecth_import_tab.build_core_call()


# ---------------------------------------------------------------------------
# KRUNAL_JULY_23.md point 7 -- graph-viewer link surfacing (Phase 5c, bug.md)
# ---------------------------------------------------------------------------

class _FakeServeProc:
    """Stand-in for the real subprocess.Popen serve_graph_compare returns."""
    _next_pid = 1000

    def __init__(self):
        self.pid = _FakeServeProc._next_pid
        _FakeServeProc._next_pid += 1
        self.terminated = False
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def wait(self, timeout=None):
        pass

    def kill(self):
        self._alive = False


def test_graph_compare_tab_has_viewer_controls(qapp, tmp_path):
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage
    page = RMIPSuitePage()
    tab = page.graph_compare_tab
    for attr in ("viewer_port_spin", "viewer_label_combo", "launch_viewer_button",
                "viewer_link_label"):
        assert hasattr(tab, attr)
    assert tab.viewer_port_spin.value() == 8050


def test_launch_viewer_requires_db_and_two_ids(qapp, monkeypatch):
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage
    called = {"serve": False}
    monkeypatch.setattr(
        "mso_studio.gui.modules.rmip_suite.core.serve_graph_compare",
        lambda *a, **kw: called.__setitem__("serve", True) or _FakeServeProc(),
    )
    page = RMIPSuitePage()
    page.graph_compare_tab._on_launch_viewer_clicked()  # no db, no ids
    assert called["serve"] is False
    assert page.graph_compare_tab._serve_proc is None


def test_launch_viewer_starts_process_and_sets_link(qapp, tmp_path, monkeypatch):
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage

    captured = {}

    def fake_serve(params, port, label_mode):
        captured["params"] = params
        captured["port"] = port
        captured["label_mode"] = label_mode
        return _FakeServeProc()

    monkeypatch.setattr("mso_studio.gui.modules.rmip_suite.core.serve_graph_compare", fake_serve)
    # Avoid the real webbrowser.open / 2.5s QTimer firing during the test.
    monkeypatch.setattr("mso_studio.gui.modules.rmip_suite.QTimer.singleShot",
                        lambda *a, **kw: None)

    db = tmp_path / "p.db"
    db.write_text("")
    page = RMIPSuitePage()
    page.db_picker.set_path(str(db))
    tab = page.graph_compare_tab
    tab.mech_ids_edit.setText("2,3,4")
    tab.viewer_port_spin.setValue(9090)
    tab.viewer_label_combo.setCurrentText("id")

    tab._on_launch_viewer_clicked()

    assert captured["port"] == 9090
    assert captured["label_mode"] == "id"
    assert tab._serve_proc is not None
    assert "http://localhost:9090" in tab.viewer_link_label.text()


def test_launch_viewer_twice_stops_the_first_process(qapp, tmp_path, monkeypatch):
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage

    procs = []

    def fake_serve(params, port, label_mode):
        p = _FakeServeProc()
        procs.append(p)
        return p

    monkeypatch.setattr("mso_studio.gui.modules.rmip_suite.core.serve_graph_compare", fake_serve)
    monkeypatch.setattr("mso_studio.gui.modules.rmip_suite.QTimer.singleShot",
                        lambda *a, **kw: None)

    db = tmp_path / "p.db"
    db.write_text("")
    page = RMIPSuitePage()
    page.db_picker.set_path(str(db))
    tab = page.graph_compare_tab
    tab.mech_ids_edit.setText("2,3,4")

    tab._on_launch_viewer_clicked()
    tab._on_launch_viewer_clicked()

    assert len(procs) == 2
    assert procs[0].terminated is True    # replaced -- must not leak
    assert procs[1].terminated is False   # the current one stays running
    assert tab._serve_proc is procs[1]


def test_stop_viewer_terminates_and_clears_handle(qapp, tmp_path, monkeypatch):
    from mso_studio.gui.modules.rmip_suite import RMIPSuitePage

    monkeypatch.setattr("mso_studio.gui.modules.rmip_suite.core.serve_graph_compare",
                        lambda *a, **kw: _FakeServeProc())
    monkeypatch.setattr("mso_studio.gui.modules.rmip_suite.QTimer.singleShot",
                        lambda *a, **kw: None)

    db = tmp_path / "p.db"
    db.write_text("")
    page = RMIPSuitePage()
    page.db_picker.set_path(str(db))
    tab = page.graph_compare_tab
    tab.mech_ids_edit.setText("2,3,4")
    tab._on_launch_viewer_clicked()
    proc = tab._serve_proc

    tab._stop_viewer()

    assert proc.terminated is True
    assert tab._serve_proc is None
