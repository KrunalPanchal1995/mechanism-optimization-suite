"""
FilePicker: baseline behavior (had zero test coverage before Session B1)
plus the new opt-in recent-paths dropdown, redirected to a tmp QSettings
INI file so tests never touch the real user's ~/.config (same pattern as
test_launcher_window.py).
"""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def redirected_settings(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    ini_path = str(tmp_path / "studio.ini")

    def _factory(*_args, **_kwargs):
        return QSettings(ini_path, QSettings.Format.IniFormat)

    monkeypatch.setattr("mso_studio.gui.widgets.file_picker.QSettings", _factory)
    return ini_path


def test_empty_path_is_invalid_when_must_exist(qapp):
    from mso_studio.gui.widgets.file_picker import FilePicker

    picker = FilePicker(mode="file", must_exist=True)
    assert not picker.is_valid


def test_nonexistent_path_is_invalid_when_must_exist(qapp):
    from mso_studio.gui.widgets.file_picker import FilePicker

    picker = FilePicker(mode="file", must_exist=True)
    picker.set_path("/no/such/path/at/all")
    assert not picker.is_valid


def test_existing_path_is_valid(qapp, tmp_path):
    from mso_studio.gui.widgets.file_picker import FilePicker

    real_file = tmp_path / "x.txt"
    real_file.write_text("hi")
    picker = FilePicker(mode="file", must_exist=True)
    picker.set_path(str(real_file))
    assert picker.is_valid


def test_must_exist_false_accepts_any_nonempty_path(qapp):
    from mso_studio.gui.widgets.file_picker import FilePicker

    picker = FilePicker(mode="save", must_exist=False)
    picker.set_path("/wherever/output.csv")
    assert picker.is_valid


def test_save_mode_forces_must_exist_false(qapp):
    from mso_studio.gui.widgets.file_picker import FilePicker

    picker = FilePicker(mode="save", must_exist=True)  # should still not require existence
    picker.set_path("/wherever/output.csv")
    assert picker.is_valid


def test_path_changed_signal_fires(qapp):
    from mso_studio.gui.widgets.file_picker import FilePicker

    picker = FilePicker(mode="file", must_exist=False)
    received = []
    picker.pathChanged.connect(received.append)
    picker.set_path("/some/path")
    assert received[-1] == "/some/path"


def test_no_recent_button_without_recent_key(qapp):
    from mso_studio.gui.widgets.file_picker import FilePicker

    picker = FilePicker(mode="file", must_exist=False)
    assert picker.recent_button is None


def test_recent_button_present_with_recent_key(qapp, redirected_settings):
    from mso_studio.gui.widgets.file_picker import FilePicker

    picker = FilePicker(mode="file", must_exist=False, recent_key="test_thing")
    assert picker.recent_button is not None


def test_browsing_remembers_path_across_new_pickers(qapp, redirected_settings, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    from mso_studio.gui.widgets.file_picker import FilePicker

    chosen = str(tmp_path / "mech.yaml")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (chosen, ""))

    first = FilePicker(mode="file", must_exist=False, recent_key="mechanism")
    first._browse()
    assert first.path() == chosen

    # A fresh picker instance with the same recent_key should see it.
    second = FilePicker(mode="file", must_exist=False, recent_key="mechanism")
    assert chosen in second._recent_paths()


def test_recent_list_caps_at_eight_and_dedupes(qapp, redirected_settings, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    from mso_studio.gui.widgets.file_picker import FilePicker

    picker = FilePicker(mode="file", must_exist=False, recent_key="many")
    paths = [str(tmp_path / f"f{i}.txt") for i in range(10)]
    for p in paths:
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, _p=p, **k: (_p, ""))
        picker._browse()
    # Re-choosing the very first one again should move it to the front, not duplicate it.
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (paths[0], ""))
    picker._browse()

    recents = picker._recent_paths()
    assert len(recents) == 8
    assert recents[0] == paths[0]
    assert recents.count(paths[0]) == 1


def test_recent_paths_isolated_by_key(qapp, redirected_settings, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    from mso_studio.gui.widgets.file_picker import FilePicker

    mech_path = str(tmp_path / "mech.yaml")
    thermo_path = str(tmp_path / "thermo.dat")

    mech_picker = FilePicker(mode="file", must_exist=False, recent_key="mechanism")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (mech_path, ""))
    mech_picker._browse()

    thermo_picker = FilePicker(mode="file", must_exist=False, recent_key="thermo")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (thermo_path, ""))
    thermo_picker._browse()

    assert mech_path in mech_picker._recent_paths()
    assert mech_path not in thermo_picker._recent_paths()
