"""
gui_common/run_status_export.py
=================================
Shared "Export run status…" button for the standalone Tkinter apps
(KRUNAL_JULY_23.md point 1: "Add option of exporting the status of the run
from any GUI into a txt file").

Each app owns the knowledge of its own current field values and captured
log text; this module only owns the identical boilerplate every app would
otherwise duplicate -- the button, the save dialog, and the
build-report/write/report-back sequence -- so all 11 apps get the exact
same button and the exact same failure handling.

Usage (inside an app's ``_build_ui``)::

    from gui_common.run_status_export import add_export_button

    def _build_run_status(self) -> RunStatusReport:
        return RunStatusReport(
            tool="Mechanism Converter",
            inputs={"mechanism (.inp)": self._vars["inp"].get(), ...},
            output_dir=self._vars["out_dir"].get(),
            options={k: v.get() for k, v in self._flags.items()},
            status="failed" if self._last_errors else "success",
            summary=self._last_errors[0] if self._last_errors else None,
            log=self._out.get("1.0", "end"),
        )

    add_export_button(btn_row, self._build_run_status).pack(side="left", padx=6)
"""
from __future__ import annotations

from tkinter import filedialog, messagebox, ttk
from typing import Callable

from mso_io.run_status import RunStatusReport


def add_export_button(
    parent,
    report_factory: Callable[[], RunStatusReport],
    text: str = "Export run status…",
) -> ttk.Button:
    """Create an "Export run status…" ``ttk.Button`` on *parent*.

    Clicking it calls ``report_factory()`` to build the CURRENT
    :class:`~mso_io.run_status.RunStatusReport` (so it always reflects
    whatever the user has done up to that click, not a stale snapshot),
    prompts for a save location, and writes the plain-text report. Does
    **not** grid/pack the button -- callers place it themselves, same as
    every other button in these apps.
    """
    def _on_click() -> None:
        try:
            report = report_factory()
        except Exception as e:                      # noqa: BLE001
            messagebox.showerror("Export failed",
                                 f"Could not build the run-status report: {e}")
            return
        path = filedialog.asksaveasfilename(
            title="Export run status", defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            report.write_txt(path)
        except OSError as e:
            messagebox.showerror("Export failed", f"Could not write {path}: {e}")
            return
        messagebox.showinfo("Exported", f"Run status written to:\n{path}")

    return ttk.Button(parent, text=text, command=_on_click)
