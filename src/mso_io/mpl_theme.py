"""mpl_theme.py -- dark/light rcParams for embedded matplotlib canvases.

``mso_studio.gui.widgets.mpl_canvas.MplCanvas`` used to create a bare
``Figure`` with no facecolor override, so every embedded plot rendered as a
white rectangle inside the app's dark shell regardless of theme. This module
derives matplotlib's figure/axes colors from the same design tokens
(``mso_studio.gui.tokens``) that drive the Qt QSS, so in-app plots and (later)
notebook-generated figures via ``mso_viz`` match the surrounding chrome.

Pure Python, no PySide6 import (``mso_studio.gui.tokens`` has none either), so
this is importable from the anaconda interpreter too -- the same
cross-interpreter convention documented in ``mso_io/notebook_gen.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from mso_studio.gui.tokens import DARK, LIGHT, Mode, Tokens


def rc_params_for(tokens: Tokens) -> Dict[str, Any]:
    """matplotlib rcParams dict that makes a figure match *tokens*."""
    cycle = [tokens.accent, tokens.accent2, tokens.accent3,
              tokens.success, tokens.warning, tokens.error, tokens.info]
    return {
        "figure.facecolor": tokens.surface,
        "figure.edgecolor": tokens.surface,
        "axes.facecolor": tokens.surface,
        "axes.edgecolor": tokens.border,
        "axes.labelcolor": tokens.text,
        "axes.titlecolor": tokens.text,
        "axes.prop_cycle": _cycler(cycle),
        "xtick.color": tokens.text_dim,
        "ytick.color": tokens.text_dim,
        "text.color": tokens.text,
        "grid.color": tokens.border,
        "grid.alpha": 0.5,
        "legend.facecolor": tokens.surface_raised,
        "legend.edgecolor": tokens.border,
        "legend.labelcolor": tokens.text,
        "savefig.facecolor": tokens.surface,
        "savefig.edgecolor": tokens.surface,
    }


def _cycler(colors: List[str]):
    from cycler import cycler
    return cycler(color=colors)


def apply_global_rc_params(tokens: Tokens) -> None:
    """Update matplotlib's process-wide rcParams so any axes/figure created
    from this point on (e.g. after MplCanvas.clear() + a fresh add_subplot)
    picks up the theme automatically, not just the figure re-themed in
    apply_rc_params()."""
    import matplotlib
    matplotlib.rcParams.update(rc_params_for(tokens))


def apply_rc_params(figure, tokens: Tokens) -> None:
    """Apply theme colors directly to an existing Figure + its Axes, for
    canvases that are re-themed after creation rather than at Figure()
    construction time (rcParams only affects new figures)."""
    figure.set_facecolor(tokens.surface)
    for ax in figure.get_axes():
        ax.set_facecolor(tokens.surface)
        ax.tick_params(colors=tokens.text_dim)
        ax.xaxis.label.set_color(tokens.text)
        ax.yaxis.label.set_color(tokens.text)
        ax.title.set_color(tokens.text)
        for spine in ax.spines.values():
            spine.set_color(tokens.border)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(tokens.surface_raised)
            legend.get_frame().set_edgecolor(tokens.border)
            for text in legend.get_texts():
                text.set_color(tokens.text)


def tokens_for_mode(mode: Mode) -> Tokens:
    if mode == "light":
        return LIGHT
    return DARK
