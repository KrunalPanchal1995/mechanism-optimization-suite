"""
pdf_table_panel.py
------------------
Tkinter panel: load a PDF → extract tables (via pdfplumber) → preview + CSV export.

Usage (embedded)::

    from notebook_gemini.pdf_table_panel import PDFTablePanel
    panel = PDFTablePanel(parent, client=GeminiClient())
    panel.pack(fill=tk.BOTH, expand=True)
"""

from __future__ import annotations

import csv
import io
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional


class PDFTablePanel(ttk.Frame):
    """
    Extract tables from a PDF using pdfplumber and display them in a
    scrollable grid. Each detected table can be previewed and exported
    as CSV. Gemini can optionally clean up OCR artefacts.

    Parameters
    ----------
    parent : tk widget
    client : GeminiClient (may have available=False)
    palette : dict, optional
    """

    def __init__(self, parent, client, palette: Optional[dict] = None):
        super().__init__(parent)
        self._client = client
        self._pal = palette or {"bg": "#1e1e2e", "surface": "#2a2a3e",
                                "accent": "#7aa2f7", "text": "#cdd6f4"}
        self._pdf_path: str = ""
        self._tables: List[List[List[str]]] = []   # list of tables, each a 2D list of str
        self._page_nums: List[int] = []
        self._selected: int = -1
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        self.configure(padding=6)

        # ── top bar ──────────────────────────────────────────────────
        top = ttk.Frame(self)
        top.pack(fill=tk.X, pady=(0, 6))

        self._path_var = tk.StringVar(value="No PDF loaded")
        ttk.Label(top, textvariable=self._path_var, width=55, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Button(top, text="Open PDF…",      command=self._open_pdf).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Extract tables", command=self._extract).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Export CSV…",    command=self._export_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Gemini: clean table", command=self._gemini_clean).pack(side=tk.LEFT, padx=4)

        self._spinner = tk.StringVar(value="")
        ttk.Label(top, textvariable=self._spinner).pack(side=tk.LEFT)

        # ── table selector ───────────────────────────────────────────
        sel_frame = ttk.Frame(self)
        sel_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(sel_frame, text="Detected tables:").pack(side=tk.LEFT)
        self._table_list = tk.Listbox(sel_frame, height=4, exportselection=False,
                                      bg=self._pal["surface"], fg=self._pal["text"],
                                      selectbackground=self._pal["accent"])
        self._table_list.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._table_list.bind("<<ListboxSelect>>", self._on_select)

        # ── table preview (scrollable grid) ──────────────────────────
        prev_frame = ttk.LabelFrame(self, text="Table preview", padding=4)
        prev_frame.pack(fill=tk.BOTH, expand=True)

        hbar = tk.Scrollbar(prev_frame, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = tk.Scrollbar(prev_frame, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._preview = tk.Text(
            prev_frame,
            xscrollcommand=hbar.set, yscrollcommand=vbar.set,
            wrap=tk.NONE,
            font=("DejaVu Sans Mono", 8),
            bg=self._pal["surface"], fg=self._pal["text"],
            relief=tk.FLAT, state=tk.DISABLED,
        )
        self._preview.pack(fill=tk.BOTH, expand=True)
        hbar.config(command=self._preview.xview)
        vbar.config(command=self._preview.yview)

        # ── status ───────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Open a PDF to begin.")
        ttk.Label(self, textvariable=self._status_var, anchor=tk.W).pack(fill=tk.X)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _open_pdf(self):
        path = filedialog.askopenfilename(
            filetypes=[("PDF", "*.pdf"), ("All", "*")],
            title="Open PDF for table extraction",
        )
        if not path:
            return
        self._pdf_path = path
        self._path_var.set(os.path.basename(path))
        self._tables.clear()
        self._page_nums.clear()
        self._table_list.delete(0, tk.END)
        self._clear_preview()
        self._status_var.set("PDF loaded — click 'Extract tables'.")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract(self):
        if not self._pdf_path:
            messagebox.showinfo("No PDF", "Load a PDF first.")
            return
        self._spinner.set("⏳ Extracting…")
        self._status_var.set("Running pdfplumber table extraction…")

        def _worker():
            try:
                import pdfplumber
                tables, pages = [], []
                with pdfplumber.open(self._pdf_path) as pdf:
                    for pno, page in enumerate(pdf.pages, 1):
                        for tbl in page.extract_tables():
                            # Normalise: None → ""
                            clean = [
                                [cell if cell is not None else "" for cell in row]
                                for row in tbl
                            ]
                            tables.append(clean)
                            pages.append(pno)
                self.after(0, lambda t=tables, p=pages: self._on_extracted(t, p))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_extracted(self, tables, pages):
        self._spinner.set("")
        self._tables = tables
        self._page_nums = pages
        self._table_list.delete(0, tk.END)
        if not tables:
            self._status_var.set("No tables found in this PDF.")
            return
        for i, (tbl, pno) in enumerate(zip(tables, pages)):
            nrow = len(tbl)
            ncol = max((len(r) for r in tbl), default=0)
            self._table_list.insert(tk.END, f"Table {i+1}  (page {pno})  {nrow}×{ncol}")
        self._table_list.selection_set(0)
        self._on_select(None)
        self._status_var.set(f"{len(tables)} table(s) extracted. Select one to preview.")

    def _on_error(self, msg):
        self._spinner.set("")
        messagebox.showerror("Extraction error", msg)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_select(self, _event):
        sel = self._table_list.curselection()
        if not sel:
            return
        idx = sel[0]
        self._selected = idx
        self._show_table(self._tables[idx])

    def _show_table(self, tbl: List[List[str]]):
        self._clear_preview()
        self._preview.config(state=tk.NORMAL)
        if not tbl:
            self._preview.insert(tk.END, "(empty)")
            self._preview.config(state=tk.DISABLED)
            return
        # Compute column widths
        ncol = max(len(r) for r in tbl)
        widths = [0] * ncol
        for row in tbl:
            for j, cell in enumerate(row):
                widths[j] = max(widths[j], len(str(cell)))
        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        for i, row in enumerate(tbl):
            self._preview.insert(tk.END, sep + "\n")
            cells = [f" {str(row[j]) if j < len(row) else '':<{widths[j]}} " for j in range(ncol)]
            self._preview.insert(tk.END, "|" + "|".join(cells) + "|\n")
        self._preview.insert(tk.END, sep + "\n")
        self._preview.config(state=tk.DISABLED)

    def _clear_preview(self):
        self._preview.config(state=tk.NORMAL)
        self._preview.delete("1.0", tk.END)
        self._preview.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_csv(self):
        if self._selected < 0 or not self._tables:
            messagebox.showinfo("No selection", "Extract tables and select one first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*")],
            title="Export table as CSV",
        )
        if not path:
            return
        tbl = self._tables[self._selected]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(tbl)
        messagebox.showinfo("Exported", f"Table saved to:\n{path}")

    # ------------------------------------------------------------------
    # Gemini cleanup
    # ------------------------------------------------------------------

    def _gemini_clean(self):
        if self._selected < 0 or not self._tables:
            messagebox.showinfo("No table", "Select a table first.")
            return
        if not self._client.available:
            messagebox.showinfo("Gemini not available", "Set GOOGLE_API_KEY to use Gemini cleanup.")
            return
        tbl = self._tables[self._selected]
        # Render table as CSV text for Gemini
        buf = io.StringIO()
        csv.writer(buf).writerows(tbl)
        csv_text = buf.getvalue()

        self._spinner.set("⏳ Asking Gemini…")

        def _run():
            reply = self._client.chat(
                "Below is a table extracted from a PDF (possibly OCR-noisy). "
                "Fix any obvious OCR errors, merge split cells, and return the "
                "corrected table in CSV format (no extra commentary):\n\n" + csv_text
            )
            self.after(0, lambda: self._on_gemini_reply(reply))

        threading.Thread(target=_run, daemon=True).start()

    def _on_gemini_reply(self, reply: str):
        self._spinner.set("")
        # Parse CSV from Gemini reply and replace selected table
        try:
            reader = csv.reader(reply.strip().splitlines())
            new_tbl = [row for row in reader if row]
            self._tables[self._selected] = new_tbl
            self._show_table(new_tbl)
            self._status_var.set("Table cleaned by Gemini — review and export.")
        except Exception as exc:
            messagebox.showwarning("Parse error",
                                   f"Could not parse Gemini output as CSV:\n{exc}\n\n{reply[:400]}")
