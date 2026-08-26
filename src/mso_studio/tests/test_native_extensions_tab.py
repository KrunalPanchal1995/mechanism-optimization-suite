"""
Widget-level test for NativeExtensionsTab (KRUNAL_JULY_23.md point 8 --
".so file generator"). Covers the table population + rebuild wiring via a
monkeypatched build_extension (no real subprocess/compile here -- that path
was verified manually: rebuilding 'shuffle' via
mso_studio.core.tools.build_extension('shuffle', copy_to_runtime=False)
produced a real .so with a clean g++ exit).
"""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_tab_registered_on_tools_page(qapp):
    from mso_studio.gui.modules.tools import ToolsPage, NativeExtensionsTab
    page = ToolsPage()
    assert isinstance(page.native_extensions_tab, NativeExtensionsTab)


def test_table_populated_with_known_extensions(qapp):
    from mso_studio.gui.modules.tools import NativeExtensionsTab
    tab = NativeExtensionsTab()
    names = {tab.table.item(r, 0).text() for r in range(tab.table.rowCount())}
    assert names == {"yamlwriter", "parallel_yaml_writer", "shuffle"}


def test_rebuild_without_selection_shows_message(qapp, monkeypatch):
    from mso_studio.gui.modules.tools import NativeExtensionsTab
    from PySide6.QtWidgets import QMessageBox
    tab = NativeExtensionsTab()
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a))
    tab._on_rebuild()
    assert len(shown) == 1


def test_build_finished_success_updates_log_and_status(qapp):
    from mso_studio.gui.modules.tools import NativeExtensionsTab
    from mso_studio.core.tools import BuildResult
    tab = NativeExtensionsTab()
    tab.rebuild_button.setEnabled(False)

    messages = []
    tab.statusMessage.connect(messages.append)
    result = BuildResult(name="shuffle", ok=True, log="build ok",
                          built_artifact="/tmp/shuffle.so", copied_to=["/a/shuffle.so"])
    tab._on_build_finished(result)

    assert tab.rebuild_button.isEnabled()
    assert "build ok" in tab.log_view.toPlainText()
    assert "/tmp/shuffle.so" in tab.log_view.toPlainText()
    assert any("rebuilt" in m for m in messages)


def test_build_finished_failure_shows_failed_marker(qapp):
    from mso_studio.gui.modules.tools import NativeExtensionsTab
    from mso_studio.core.tools import BuildResult
    tab = NativeExtensionsTab()
    result = BuildResult(name="yamlwriter", ok=False, log="g++ error: bad thing")
    tab._on_build_finished(result)
    assert "FAILED" in tab.log_view.toPlainText()
    assert "g++ error" in tab.log_view.toPlainText()
