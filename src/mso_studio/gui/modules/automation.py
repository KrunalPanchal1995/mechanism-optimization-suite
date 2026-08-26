"""
mso_studio.gui.modules.automation
====================================
Mechanism-comparison automation module page -- drives the integrated
``src/automation/run_tasks.py`` task runner and its meta-PRS toolchain
(:mod:`mso_studio.core.automation`).

Deliberately NOT a :class:`~mso_studio.gui.widgets.module_page.ModulePage`
subclass: that base class models one blocking "press Run, get a RunRecord"
action, but a study here runs unsupervised for days after ``--daemon``
detaches it. This page instead exposes four separate, mostly read-only
operations (validate / preflight / launch / status) plus a read-only
meta-PRS fit tab, each on its own short-lived background thread via
:func:`~mso_studio.gui.workers.call_in_thread`.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget,
)

from mso_studio.core.automation import (
    AUTOMATION_DIR, CheckResult, MetaPrsFitResult, available_fuels, fit_meta_prs,
    get_status, launch_daemon, run_preflight, validate_yaml,
)
from mso_studio.gui.widgets.file_picker import FilePicker
from mso_studio.gui.widgets.log_view import LogView
from mso_studio.gui.widgets.results_table import ResultsTable
from mso_studio.gui.widgets.module_hero import make_module_hero
from mso_studio.gui.workers import call_in_thread, run_in_thread


class AutomationPage(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._threads = []  # keep (thread, worker) pairs alive until they finish

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.hero = make_module_hero(
            "automation", "Mechanism-Comparison Automation",
            "Drives the integrated task runner (src/automation/run_tasks.py) for "
            "long-running, unsupervised mechanism-comparison studies, plus a "
            "read-only view of the meta-PRS hyperparameter fit.",
            self,
        )
        root.addWidget(self.hero)

        tabs = QTabWidget(self)
        root.addWidget(tabs, 1)
        tabs.addTab(self._build_task_runner_tab(), "Task Runner")
        tabs.addTab(self._build_meta_prs_tab(), "Meta-PRS Fit")

    # -- Task Runner tab ---------------------------------------------------

    def _build_task_runner_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QHBoxLayout(tab)

        left = QWidget(tab)
        form = QFormLayout(left)
        form.setContentsMargins(16, 16, 16, 16)

        self.yaml_picker = FilePicker(mode="file", filter_="YAML files (*.yaml *.yml)", parent=left)
        default_example = AUTOMATION_DIR / "examples" / "compare_only_minimal.yaml"
        if default_example.exists():
            self.yaml_picker.set_path(str(default_example))
        form.addRow("Task YAML", self.yaml_picker)

        self.thread_cap_spin = QSpinBox(left)
        self.thread_cap_spin.setRange(1, 1000)
        self.thread_cap_spin.setValue(160)
        form.addRow("Thread cap", self.thread_cap_spin)

        self.planned_threads_spin = QSpinBox(left)
        self.planned_threads_spin.setRange(1, 1000)
        self.planned_threads_spin.setValue(40)
        form.addRow("Planned max threads", self.planned_threads_spin)

        self.validate_button = QPushButton("Validate", left)
        self.validate_button.clicked.connect(self._on_validate)
        form.addRow(self.validate_button)

        self.preflight_button = QPushButton("Preflight check", left)
        self.preflight_button.setToolTip(
            "Checks disk space and thread headroom across this repo's "
            "automation directory AND any live external one -- never "
            "launches or deletes anything."
        )
        self.preflight_button.clicked.connect(self._on_preflight)
        form.addRow(self.preflight_button)

        self.launch_button = QPushButton("Launch (--daemon)", left)
        self.launch_button.setObjectName("PrimaryAction")
        self.launch_button.setToolTip(
            "Detaches and runs unsupervised, potentially for days. "
            "Run Validate and Preflight first."
        )
        self.launch_button.clicked.connect(self._on_launch)
        form.addRow(self.launch_button)

        self.status_button = QPushButton("Refresh status", left)
        self.status_button.clicked.connect(self._on_status)
        form.addRow(self.status_button)

        layout.addWidget(left, 0)

        right = QWidget(tab)
        right_layout = QVBoxLayout(right)
        self.task_table = ResultsTable(["Task", "Type", "Status"], right)
        right_layout.addWidget(self.task_table, 1)

        self.output_view = LogView(right)
        right_layout.addWidget(self.output_view, 1)
        layout.addWidget(right, 1)

        return tab

    def _current_yaml(self) -> str | None:
        if not self.yaml_picker.is_valid:
            self.statusMessage.emit("Pick an existing task YAML first.")
            self.output_view.append_status("No task YAML selected.")
            return None
        return self.yaml_picker.path()

    def _on_validate(self) -> None:
        yaml_path = self._current_yaml()
        if yaml_path is None:
            return
        self.output_view.append_status("Validating…")
        thread, worker = call_in_thread(validate_yaml, yaml_path)
        worker.finished.connect(self._on_check_finished)
        worker.error.connect(self._on_check_error)
        self._threads.append((thread, worker))

    def _on_preflight(self) -> None:
        thread, worker = call_in_thread(
            run_preflight, self.planned_threads_spin.value(), cap=self.thread_cap_spin.value(),
        )
        self.output_view.append_status("Running preflight check…")
        worker.finished.connect(self._on_check_finished)
        worker.error.connect(self._on_check_error)
        self._threads.append((thread, worker))

    def _on_status(self) -> None:
        yaml_path = self._current_yaml()
        if yaml_path is None:
            return
        thread, worker = call_in_thread(get_status, yaml_path)
        self.output_view.append_status("Fetching status…")
        worker.finished.connect(self._on_check_finished)
        worker.error.connect(self._on_check_error)
        self._threads.append((thread, worker))

    def _on_launch(self) -> None:
        yaml_path = self._current_yaml()
        if yaml_path is None:
            return
        self.output_view.append_status("Launching daemon (validate/preflight first if you haven't)…")
        self.launch_button.setEnabled(False)
        thread, worker = run_in_thread(launch_daemon, yaml_path)
        worker.progress.connect(self.output_view.append_line)
        worker.finished.connect(self._on_launch_finished)
        worker.error.connect(self._on_check_error)
        self._threads.append((thread, worker))

    def _on_launch_finished(self, record) -> None:
        self.launch_button.setEnabled(True)
        self.output_view.append_status(f"Launch: {record.status} — {record.results.get('note', '')}")
        self.statusMessage.emit("Study launched — check back with Refresh status")

    def _on_check_finished(self, result: CheckResult) -> None:
        self.output_view.append_line(result.message)
        self._populate_task_table(result.detail.get("tasks", {}))
        self.statusMessage.emit("OK" if result.ok else "Check failed — see output")

    def _on_check_error(self, message: str) -> None:
        self.output_view.append_status(f"Failed: {message}")
        self.statusMessage.emit("Failed — see output")

    def _populate_task_table(self, tasks: dict) -> None:
        if not tasks:
            return
        rows = []
        for task_id, info in sorted(tasks.items()):
            status_text = info.get("status")
            if status_text is None:
                counts = info.get("counts", {})
                status_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no runs yet"
                failed = info.get("failed", [])
                if failed:
                    status_text += f"  ({len(failed)} FAILED)"
            rows.append([task_id, info.get("type", ""), status_text])
        self.task_table.set_rows(rows)
        self.task_table.resizeColumnsToContents()

    # -- Meta-PRS Fit tab ---------------------------------------------------

    def _build_meta_prs_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QHBoxLayout(tab)

        left = QWidget(tab)
        form = QFormLayout(left)
        form.setContentsMargins(16, 16, 16, 16)

        self.fuel_combo = QComboBox(left)
        try:
            self.fuel_combo.addItems(available_fuels())
        except Exception:  # noqa: BLE001 - automation dir might not be importable yet at page build time
            pass
        form.addRow("Fuel", self.fuel_combo)

        self.precision_spin = QDoubleSpinBox(left)
        self.precision_spin.setRange(0.0001, 1.0)
        self.precision_spin.setDecimals(4)
        self.precision_spin.setSingleStep(0.001)
        self.precision_spin.setValue(0.001)
        form.addRow("Precision", self.precision_spin)

        self.fit_button = QPushButton("Fit surface", left)
        self.fit_button.setObjectName("PrimaryAction")
        self.fit_button.clicked.connect(self._on_fit)
        form.addRow(self.fit_button)

        note = QLabel(
            "Read-only: parses the already-validated factorial .tex reports, "
            "fits a 2-variable quadratic response surface with a leave-one-out "
            "cross-validation trust gate, and cheaply optimizes it. Launches "
            "no simulations.",
            left,
        )
        note.setObjectName("Caption")
        note.setWordWrap(True)
        form.addRow(note)

        layout.addWidget(left, 0)

        right = QWidget(tab)
        right_form = QFormLayout(right)
        self.fit_points_label = QLabel("—", right)
        self.fit_trusted_label = QLabel("—", right)
        self.fit_loocv_label = QLabel("—", right)
        self.fit_optimum_label = QLabel("—", right)
        for label, widget in (
            ("DoE points", self.fit_points_label),
            ("Trusted (LOOCV ≤ 6%)", self.fit_trusted_label),
            ("LOOCV max / mean |err|", self.fit_loocv_label),
            ("Predicted optimum", self.fit_optimum_label),
        ):
            right_form.addRow(label, widget)
        layout.addWidget(right, 1)

        return tab

    def _on_fit(self) -> None:
        fuel = self.fuel_combo.currentText()
        if not fuel:
            self.statusMessage.emit("No fuel selected.")
            return
        self.fit_button.setEnabled(False)
        thread, worker = call_in_thread(fit_meta_prs, fuel, precision=self.precision_spin.value())
        worker.finished.connect(self._on_fit_finished)
        worker.error.connect(self._on_fit_error)
        self._threads.append((thread, worker))

    def _on_fit_finished(self, result: MetaPrsFitResult) -> None:
        self.fit_button.setEnabled(True)
        self.fit_points_label.setText(str(result.n_points))
        self.fit_trusted_label.setObjectName("StatusSuccess" if result.trusted else "StatusWarning")
        self.fit_trusted_label.setText("Yes" if result.trusted else "No (informational only)")
        self.fit_trusted_label.style().unpolish(self.fit_trusted_label)
        self.fit_trusted_label.style().polish(self.fit_trusted_label)
        self.fit_loocv_label.setText(
            f"{result.loocv_max_abs_pct:.2f}% / {result.loocv_mean_abs_pct:.2f}%"
        )
        self.fit_optimum_label.setText(
            f"p_start={result.optimum_p_start:.4g}, p_final={result.optimum_p_final:.4g}, "
            f"predicted WMRE={result.optimum_predicted_wmre:.4f}%"
            f"{' (grid agrees)' if result.grid_agrees else ' (grid disagrees — check bounds)'}"
        )
        self.statusMessage.emit(f"Fit complete for {result.fuel}")

    def _on_fit_error(self, message: str) -> None:
        self.fit_button.setEnabled(True)
        self.statusMessage.emit(f"Fit failed: {message}")
