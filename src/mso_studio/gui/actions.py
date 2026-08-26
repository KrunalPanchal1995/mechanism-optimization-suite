"""
mso_studio.gui.actions
========================
The action registry (guidelines doc, Part 3.2): every user-visible action is
registered once here, menus/toolbars are built *from* this list, and a docs
script (``docs/generate_action_table.py``) walks it to emit the
action-to-function traceability table reviewers ask for. This keeps that
document honest by construction instead of hand-maintained.

Module pages register their own actions via :func:`register` at import time
(see ``gui/modules/*.py``); ``main_window.py`` then builds the Simulation /
Results menus from whatever is registered instead of hardcoding them, so a
new module page's actions show up in the menu bar automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtGui import QAction, QKeySequence


@dataclass
class Action:
    """One user-visible action.

    Parameters
    ----------
    id : str
        Unique, stable identifier (used as the dict key in the registry).
    label : str
        Menu/toolbar text, active voice, sentence case
        (guidelines doc, Part 1.11 / Part 6.7), e.g. ``"Generate LaTeX report"``.
    calls : str
        Dotted path to the ``core`` function this action executes, e.g.
        ``"core.nominal.run_nominal"``. Purely documentation -- not resolved
        at runtime -- so it stays valid even for actions wired to a GUI-only
        handler that calls several core functions.
    shortcut : str, optional
        Qt key-sequence string, e.g. ``"F5"``, ``"Ctrl+R"``.
    menu : str, optional
        Which top-level menu this belongs under (``"Simulation"``,
        ``"Results"``, ...), for grouping when the traceability table or a
        menu is generated from the registry.
    tooltip : str, optional
        Shown on hover; also the text used when the action is disabled and
        needs to explain why (guidelines doc, Part 6.8.4) -- callers should
        set a more specific "why disabled" tooltip on the QAction/button
        itself at that point, this is just the baseline description.
    """

    id: str
    label: str
    calls: str
    shortcut: Optional[str] = None
    menu: Optional[str] = None
    tooltip: str = ""


ACTIONS: dict[str, Action] = {}


def register(action: Action) -> Action:
    """Register *action* in the global :data:`ACTIONS` table. Idempotent by id."""
    ACTIONS[action.id] = action
    return action


def make_qaction(action: Action, parent, handler: Callable[[], None]) -> QAction:
    """Build a ``QAction`` from a registered :class:`Action`, wired to *handler*."""
    qaction = QAction(action.label, parent)
    if action.shortcut:
        qaction.setShortcut(QKeySequence(action.shortcut))
    if action.tooltip:
        qaction.setToolTip(action.tooltip)
    qaction.triggered.connect(handler)
    return qaction


def traceability_rows() -> list[tuple[str, str, str, str]]:
    """Return ``(menu, label, shortcut, calls)`` rows, sorted for the docs table."""
    rows = [
        (a.menu or "", a.label, a.shortcut or "", a.calls)
        for a in ACTIONS.values()
    ]
    return sorted(rows, key=lambda r: (r[0], r[1]))
