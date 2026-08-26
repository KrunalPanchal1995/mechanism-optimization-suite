"""
Tests for the structured module-help content (content.MODULE_HELP) and its
Qt rendering (help_panel.ModuleHelpView / ModuleHelpDialog).
"""
import pytest

from mso_studio.launcher.catalog import CATALOG, CATALOG_BY_ID
from mso_studio.launcher.content import MODULE_HELP


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_every_catalog_entry_has_module_help():
    for entry in CATALOG:
        assert entry.id in MODULE_HELP, entry.id


def test_every_module_help_field_is_non_empty():
    for module_id, help_ in MODULE_HELP.items():
        assert help_.purpose.strip(), module_id
        assert help_.typical_use.strip(), module_id
        # inputs/outputs/prerequisites/tips/see_also may legitimately be
        # empty (e.g. Doctor has no prerequisites, Tools has none either) --
        # but purpose and typical_use are required for every module.


def test_see_also_ids_resolve_to_a_known_module():
    from mso_studio.launcher.content import MODULE_META

    for module_id, help_ in MODULE_HELP.items():
        for other_id in help_.see_also:
            assert other_id in CATALOG_BY_ID or other_id in MODULE_META, (
                f"{module_id} references unknown see_also id {other_id!r}"
            )


def test_see_also_does_not_self_reference():
    for module_id, help_ in MODULE_HELP.items():
        assert module_id not in help_.see_also


def test_module_help_view_renders_all_sections(qapp):
    from mso_studio.launcher.help_panel import ModuleHelpView

    view = ModuleHelpView()
    view.set_module("mechanism_check")
    # purpose + typical_use headings always render, plus one per non-empty
    # tuple field for this particular module (inputs/outputs/prerequisites/
    # tips/see_also are all populated for mechanism_check).
    assert view._section_widgets
    rendered_text = " ".join(
        w.text() for w in view._section_widgets if hasattr(w, "text")
    )
    assert "PURPOSE" in rendered_text
    assert "OUTPUTS" in rendered_text
    # Spot-check the purpose body text made it into the rendered widgets.
    from mso_studio.launcher.content import MODULE_HELP
    assert MODULE_HELP["mechanism_check"].purpose in rendered_text


def test_module_help_view_navigate_signal_on_see_also_click(qapp):
    from mso_studio.launcher.help_panel import ModuleHelpView

    view = ModuleHelpView()
    view.set_module("mechanism_converter")
    received = []
    view.navigateRequested.connect(received.append)

    # Find a see-also link button and click it programmatically.
    from PySide6.QtWidgets import QPushButton
    buttons = [w for w in view.findChildren(QPushButton)]
    assert buttons, "expected at least one see-also link button"
    buttons[0].click()
    assert received


def test_module_help_dialog_lists_every_help_module(qapp):
    from mso_studio.launcher.help_panel import HELP_MODULE_IDS, ModuleHelpDialog

    dialog = ModuleHelpDialog()
    assert dialog.module_list.count() == len(HELP_MODULE_IDS)


def test_module_help_dialog_show_module_selects_matching_item(qapp):
    from mso_studio.launcher.help_panel import ModuleHelpDialog

    dialog = ModuleHelpDialog()
    dialog.show_module("forward_uq")
    current = dialog.module_list.currentItem()
    assert current is not None
    from PySide6.QtCore import Qt
    assert current.data(Qt.ItemDataRole.UserRole) == "forward_uq"
