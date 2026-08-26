"""
Widget-level tests for SensitivityPage's KRUNAL_JULY_23.md point 4 additions:
the target-preview table (with live-engine gating) and the plot-style
panel, on the "Brute-force (target.opt)" tab. Same minimal convention as
test_converter_page.py: construct + drive methods directly, no click
simulation, no real sens_3_params.py run.
"""
import yaml
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _write_target_opt(tmp_path, targets_path, addendum_path):
    cfg = {
        "Locations": {"mechanism": "/fake/mech.yaml",
                       "targets": str(targets_path),
                       "addendum": str(addendum_path)},
        "Stats": {"SA_scheme": "central_difference"},
    }
    cfg_path = tmp_path / "target.opt"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_plot_style_defaults_match_dataclass(qapp):
    from mso_studio.gui.modules.sensitivity import SensitivityPage
    from mso_io.plot_style import PlotStyle
    page = SensitivityPage()
    style = page._collect_plot_style()
    assert style.export_format == "png"
    assert style.dpi == 150
    assert style.grid == "none"


def test_plot_style_widgets_reflect_edits(qapp):
    from mso_studio.gui.modules.sensitivity import SensitivityPage
    page = SensitivityPage()
    page._style_widgets["dpi"].setValue(300)
    page._style_widgets["export_format"].setCurrentText("pdf")
    page._style_widgets["title"].setText("My SA plot")
    style = page._collect_plot_style()
    assert style.dpi == 300
    assert style.export_format == "pdf"
    assert style.title == "My SA plot"


def test_preview_targets_populates_table_and_gate_label(qapp, tmp_path):
    from mso_studio.gui.modules.sensitivity import SensitivityPage
    target_path = tmp_path / "t.target"
    target_path.write_text(
        "1 |a| target -- Tig| T -- 900| P -- 1| species -- OH\n"
        "2 |b| target -- Tig| T -- 950| P -- 1| species -- OH\n"
    )
    add_path = tmp_path / "t.add"
    add_path.write_text(yaml.safe_dump({"b": {"solver": "zerork"}}))
    cfg_path = _write_target_opt(tmp_path, target_path, add_path)

    page = SensitivityPage()
    page.bf_target_opt.set_path(str(cfg_path))
    page._preview_bf_targets()

    assert page.bf_targets_table.rowCount() == 2
    assert page.bf_targets_table.item(0, 0).text() == "a"
    assert page.bf_targets_table.item(0, 5).text() == "✓"
    assert page.bf_targets_table.item(1, 5).text() == "⚠"   # zerork, gated
    assert "1/2" in page.bf_gate_label.text()


def test_preview_targets_missing_locations_shows_message(qapp, tmp_path):
    from mso_studio.gui.modules.sensitivity import SensitivityPage
    cfg_path = tmp_path / "target.opt"
    cfg_path.write_text(yaml.safe_dump({"Locations": {"mechanism": "/fake.yaml"}}))
    page = SensitivityPage()
    page.bf_target_opt.set_path(str(cfg_path))
    page._preview_bf_targets()
    assert page.bf_targets_table.rowCount() == 0
    assert "not found" in page.bf_gate_label.text()


def test_build_core_call_writes_plot_style_json_and_sets_param(qapp, tmp_path):
    from mso_studio.gui.modules.sensitivity import SensitivityPage
    target_path = tmp_path / "t.target"
    target_path.write_text("1 |a| target -- Tig| T -- 900| P -- 1\n")
    add_path = tmp_path / "t.add"
    add_path.write_text(yaml.safe_dump({}))
    cfg_path = _write_target_opt(tmp_path, target_path, add_path)

    page = SensitivityPage()
    page.bf_target_opt.set_path(str(cfg_path))
    page.bf_output_dir.set_path(str(tmp_path / "out"))
    page._style_widgets["dpi"].setValue(220)

    fn, args, kwargs = page.build_core_call()
    params = args[0]
    assert params.plot_style_json.endswith("plot_style.json")
    import json
    saved = json.loads((tmp_path / "out" / "plot_style.json").read_text())
    assert saved["dpi"] == 220
