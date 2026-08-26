"""
mso_studio.launcher.doctor
=============================
Doctor / Recipes as a proper Qt panel, replacing the ``tk.Toplevel`` version
in ``launcher_gui.py``.

Recipes (``rmip.config.recipes``) is static text lookup with no environment
probing, so it's imported and run in-process (verified importable under
``.venv-gui`` with no Cantera dependency).

Doctor is different: ``rmip.config.preflight.check_environment`` probes
prerequisites by doing a plain ``import cantera`` *in the current process*
(see ``_probe_import`` in that module). If this ran in-process here, under
``.venv-gui``, it would always report Cantera missing -- even though the
science interpreter that runs every other app in this launcher has it fine
-- a false negative that would actively contradict the boot splash's own
(correctly cross-interpreter) Cantera check. So Doctor shells out to
``rmip_cli.py doctor`` under ``SCIENCE_PYTHON``, the same interpreter the
catalog's science-runner apps launch under (project convention: always
``rmip_cli.py``, never ``rmip/cli.py`` directly -- see the ``io``
package-shadowing rule).
"""
from __future__ import annotations

import subprocess

from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QTabWidget, QVBoxLayout, QWidget,
)

from mso_studio.core.backends import SCIENCE_PYTHON, sanitized_subprocess_env
from mso_studio.gui.workers import call_in_thread
from mso_studio.launcher.catalog import REPO_ROOT


class DoctorPanel(QTabWidget):
    """Doctor (toolchain preflight) + Recipes (task -> command lookup)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None
        self._build_doctor_tab()
        self._build_recipes_tab()

    # -- Doctor tab -----------------------------------------------------
    def _build_doctor_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Task:"))
        self.task_combo = QComboBox(page)
        try:
            from rmip.config.preflight import TASKS
            self.task_combo.addItems(sorted(TASKS))
        except Exception as exc:  # noqa: BLE001 - shown inline, not fatal
            self.task_combo.addItem("all")
            self._import_error = str(exc)
        else:
            self._import_error = None
        bar.addWidget(self.task_combo)

        self.run_doctor_button = QPushButton("Run doctor", page)
        self.run_doctor_button.clicked.connect(self._on_run_doctor)
        bar.addWidget(self.run_doctor_button)
        bar.addStretch(1)
        layout.addLayout(bar)

        self.doctor_output = QPlainTextEdit(page)
        self.doctor_output.setObjectName("LogView")
        self.doctor_output.setReadOnly(True)
        layout.addWidget(self.doctor_output, 1)

        self.addTab(page, "Doctor — toolchain preflight")
        if self._import_error:
            self.doctor_output.setPlainText(f"Could not import rmip.config: {self._import_error}")
            self.run_doctor_button.setEnabled(False)

    def _on_run_doctor(self) -> None:
        task = self.task_combo.currentText()
        self.doctor_output.setPlainText(f"Checking prerequisites for task '{task}' …")
        self.run_doctor_button.setEnabled(False)

        self._thread, self._worker = call_in_thread(_run_doctor_check, task)
        self._worker.finished.connect(self._on_doctor_finished)
        self._worker.error.connect(self._on_doctor_error)

    def _on_doctor_finished(self, text: str) -> None:
        self.doctor_output.setPlainText(text)
        self.run_doctor_button.setEnabled(True)

    def _on_doctor_error(self, message: str) -> None:
        self.doctor_output.setPlainText(f"doctor failed: {message}")
        self.run_doctor_button.setEnabled(True)

    # -- Recipes tab -----------------------------------------------------
    def _build_recipes_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        bar = QHBoxLayout()
        self.query_edit = QLineEdit(page)
        self.query_edit.setPlaceholderText("How do I… (e.g. \"convert a mechanism\")")
        self.query_edit.returnPressed.connect(self._search)
        bar.addWidget(self.query_edit, 1)
        find_button = QPushButton("Find command", page)
        find_button.clicked.connect(self._search)
        bar.addWidget(find_button)
        show_all_button = QPushButton("Show all", page)
        show_all_button.clicked.connect(self._show_all)
        bar.addWidget(show_all_button)
        layout.addLayout(bar)

        self.recipes_output = QPlainTextEdit(page)
        self.recipes_output.setObjectName("LogView")
        self.recipes_output.setReadOnly(True)
        layout.addWidget(self.recipes_output, 1)

        self.addTab(page, "Recipes — how do I…?")
        self._show_all()

    def _search(self) -> None:
        query = self.query_edit.text().strip()
        try:
            from rmip.config.recipes import RECIPES, find_recipes, render_recipes
        except Exception as exc:  # noqa: BLE001
            self.recipes_output.setPlainText(f"Could not import rmip.config.recipes: {exc}")
            return
        if query:
            self.recipes_output.setPlainText(render_recipes(find_recipes(query), query=query))
        else:
            self.recipes_output.setPlainText(render_recipes(RECIPES))

    def _show_all(self) -> None:
        self.query_edit.clear()
        self._search()


def _run_doctor_check(task: str) -> str:
    """core_fn for call_in_thread -- shells out to ``rmip_cli.py doctor``
    under SCIENCE_PYTHON (see module docstring for why this must not run
    in-process here) and returns its printed report."""
    argv = [SCIENCE_PYTHON, str(REPO_ROOT / "rmip_cli.py"), "doctor", "--task", task]
    try:
        result = subprocess.run(
            argv, cwd=str(REPO_ROOT), env=sanitized_subprocess_env(),
            capture_output=True, text=True, timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Could not run doctor: {exc}\n\nCommand: {' '.join(argv)}"
    output = result.stdout
    if result.stderr:
        output += f"\n{result.stderr}"
    return output or f"(no output, exit code {result.returncode})"
