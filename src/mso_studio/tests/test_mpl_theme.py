"""
Verifies embedded matplotlib canvases actually pick up the app's theme
colors instead of defaulting to a plain white figure inside the dark shell
(mpl_canvas.py previously created a bare Figure with no facecolor override).
"""
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_rc_params_differ_between_modes():
    from mso_io.mpl_theme import rc_params_for
    from mso_studio.gui.tokens import DARK, LIGHT

    dark_rc = rc_params_for(DARK)
    light_rc = rc_params_for(LIGHT)
    assert dark_rc["figure.facecolor"] == DARK.surface
    assert light_rc["figure.facecolor"] == LIGHT.surface
    assert dark_rc["figure.facecolor"] != light_rc["figure.facecolor"]


def test_mpl_canvas_figure_facecolor_matches_mode(qapp):
    from mso_studio.gui.tokens import DARK, LIGHT
    from mso_studio.gui.widgets.mpl_canvas import MplCanvas

    canvas = MplCanvas()
    canvas.apply_theme("dark")
    assert canvas.figure.get_facecolor() == _rgba(DARK.surface)

    canvas.apply_theme("light")
    assert canvas.figure.get_facecolor() == _rgba(LIGHT.surface)


def test_mpl_canvas_axes_are_rethemed_after_creation(qapp):
    from mso_studio.gui.tokens import DARK
    from mso_studio.gui.widgets.mpl_canvas import MplCanvas

    canvas = MplCanvas()
    canvas.apply_theme("dark")
    ax = canvas.figure.add_subplot(111)
    canvas.apply_theme("dark")  # re-apply after an axes exists
    assert ax.get_facecolor() == _rgba(DARK.surface)


def _rgba(hex_color: str):
    from matplotlib.colors import to_rgba
    return to_rgba(hex_color)
