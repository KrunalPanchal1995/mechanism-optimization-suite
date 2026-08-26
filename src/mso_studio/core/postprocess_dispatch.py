"""
mso_studio.core.postprocess_dispatch
=======================================
Maps a :class:`~mso_studio.gui.widgets.module_page.ModulePage` subclass's
``module_id`` to the matching repo's ``postprocess()`` CLI entry point (see
``src/MUQ-SAC/VisualAid.py``'s module docstring for the shared contract
every one of these implements) -- so :meth:`ModulePage._on_postprocess_requested`
has one place to look up "which script do I run for this page" instead of
each page hand-wiring its own subprocess call.

Only module_ids with a real postprocess() implementation are listed here;
:func:`postprocess_command_for` returns ``None`` for anything else (e.g.
``mechanism_check``, ``mechanism_converter`` -- these don't produce the kind
of multi-run scientific output a postprocess step applies to) and the GUI
disables the button with an explanatory tooltip rather than guessing.

``rmip_suite`` (RMIPSuitePage) and ``automation`` (AutomationPage) are
deliberately NOT :class:`ModulePage` subclasses (see their own module
docstrings) so they aren't wired through this generic mechanism even though
``rmip.postprocess`` and ``automation.prs_report`` both exist -- a
follow-up item, not done this session (see the workbench audit doc).
"""
from __future__ import annotations

from typing import List, Optional

from mso_studio.core.backends import SCIENCE_PYTHON, REPO_ROOT

# module_id -> script path (relative to REPO_ROOT), matching each repo's
# postprocess() CLI (add_postprocess_args: --postprocess, --out, --journal,
# --csv-only, --formats, --intermediate).
_DISPATCH = {
    "optimization": "src/MUQ-SAC/VisualAid.py",
    "sensitivity": "src/Sensitivity_Analysis/VisualAid.py",
    "forward_uq": "src/forward_uq/uq_plots.py",
    "mso_compare": "src/mso_compare/run_plots.py",
}


def has_postprocess(module_id: str) -> bool:
    return module_id in _DISPATCH


def postprocess_command(
    module_id: str, run_dir: str, *, out_dir: Optional[str] = None,
    journal: bool = True, intermediate: bool = False,
) -> Optional[List[str]]:
    """Build the ``[SCIENCE_PYTHON, script, --postprocess, run_dir, ...]``
    argv for *module_id*, or ``None`` if that module has no postprocess()
    entry point registered (see module docstring)."""
    script = _DISPATCH.get(module_id)
    if script is None:
        return None
    command = [SCIENCE_PYTHON, str(REPO_ROOT / script), "--postprocess", run_dir]
    if out_dir:
        command += ["--out", out_dir]
    if journal:
        command.append("--journal")
    if intermediate:
        command.append("--intermediate")
    return command
