"""
pysr_panel.py
-------------
Tkinter panel for symbolic regression + polynomial fitting over digitized data.

Features
--------
* Load a CSV (from the digitizer or any source).
* Choose X and Y columns.
* Polynomial fit (numpy) — fast, no external dep, shows equation.
* PySR symbolic regression — deeper search, requires pysr + Julia.
* Plot data + fit in a matplotlib canvas embedded in the panel.
* Export the best equation as a LaTeX string.

Usage (embedded)::

    from notebook_gemini.pysr_panel import PySRPanel
    panel = PySRPanel(parent, client=GeminiClient())
    panel.pack(fill=tk.BOTH, expand=True)
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import numpy as np
import pandas as pd


# Matplotlib embedded in Tk
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    import pysr
    _PYSR_OK = True
except ImportError:
    _PYSR_OK = False


class PySRPanel(ttk.Frame):
    """
    Parameters
    ----------
    parent : tk widget
    client : GeminiClient (may be unavailable)
    palette : dict, optional
    """

    def __init__(self, parent, client, palette: Optional[dict] = None):
        super().__init__(parent)
        self._client = client
        self._pal = palette or {"bg": "#1e1e2e", "surface": "#2a2a3e",
                                "accent": "#7aa2f7", "text": "#cdd6f4"}
        self._data: Optional[np.ndarray] = None
        self._headers: List[str] = []
        self._pysr_model = None
        self._pysr_model_ref = None   # set as soon as PySRRegressor() is constructed, for progress polling
        self._pysr_running = False
        self._pysr_cancelled = False
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        self.configure(padding=6)

        # ── toolbar ──────────────────────────────────────────────────
        top = ttk.Frame(self)
        top.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(top, text="Load CSV…",  command=self._load_csv).pack(side=tk.LEFT, padx=2)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(top, text="X col:").pack(side=tk.LEFT)
        self._x_var = tk.StringVar()
        self._x_menu = ttk.Combobox(top, textvariable=self._x_var, width=12, state="readonly")
        self._x_menu.pack(side=tk.LEFT, padx=2)

        ttk.Label(top, text="Y col:").pack(side=tk.LEFT, padx=(6, 0))
        self._y_var = tk.StringVar()
        self._y_menu = ttk.Combobox(top, textvariable=self._y_var, width=12, state="readonly")
        self._y_menu.pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(top, text="Poly degree:").pack(side=tk.LEFT)
        self._deg_var = tk.StringVar(value="3")
        ttk.Spinbox(top, textvariable=self._deg_var, from_=1, to=10, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Poly fit", command=self._poly_fit).pack(side=tk.LEFT, padx=2)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(top, text="PySR iter:").pack(side=tk.LEFT)
        self._iter_var = tk.StringVar(value="40")
        ttk.Spinbox(top, textvariable=self._iter_var, from_=5, to=500, width=5).pack(side=tk.LEFT, padx=2)
        self._pysr_btn = ttk.Button(top, text="PySR fit", command=self._pysr_fit,
                                    state="normal" if _PYSR_OK else "disabled")
        self._pysr_btn.pack(side=tk.LEFT, padx=2)
        self._pysr_cancel_btn = ttk.Button(top, text="Cancel", command=self._cancel_pysr,
                                           state="disabled")
        self._pysr_cancel_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Export LaTeX eq.", command=self._export_latex).pack(side=tk.LEFT, padx=4)

        self._spinner = tk.StringVar(value="" if _PYSR_OK else "[pysr not installed]")
        ttk.Label(top, textvariable=self._spinner).pack(side=tk.LEFT)

        # ── split: left=plot, right=equations ────────────────────────
        pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        if _MPL_OK:
            fig_frame = ttk.Frame(pane)
            pane.add(fig_frame, weight=3)
            self._fig = Figure(figsize=(5, 4), dpi=90, facecolor="#1e1e2e")
            self._ax  = self._fig.add_subplot(111, facecolor="#2a2a3e")
            self._ax.tick_params(colors="#cdd6f4")
            for spine in self._ax.spines.values():
                spine.set_edgecolor("#cdd6f4")
            self._canvas_mpl = FigureCanvasTkAgg(self._fig, master=fig_frame)
            self._canvas_mpl.draw()
            self._canvas_mpl.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            nav = NavigationToolbar2Tk(self._canvas_mpl, fig_frame)
            nav.update()
        else:
            ttk.Label(pane, text="matplotlib not found — install it for plots.").pack()
            pane.add(ttk.Frame(pane), weight=3)

        eq_frame = ttk.LabelFrame(pane, text="Equations / fit results", padding=4)
        pane.add(eq_frame, weight=2)
        eq_scroll = tk.Scrollbar(eq_frame)
        eq_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._eq_text = tk.Text(eq_frame, yscrollcommand=eq_scroll.set, wrap=tk.WORD,
                                font=("DejaVu Sans Mono", 8),
                                bg=self._pal["surface"], fg=self._pal["text"],
                                relief=tk.FLAT, state=tk.DISABLED)
        self._eq_text.pack(fill=tk.BOTH, expand=True)
        eq_scroll.config(command=self._eq_text.yview)

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    # Row cap for PySR fitting speed — declinable, not silent. Only affects
    # the in-memory fitting data; the file on disk is untouched.
    _ROW_CAP = 5000

    def _load_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("All", "*")],
            title="Load data CSV",
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
            n_total = len(df)
            if n_total > self._ROW_CAP:
                if messagebox.askyesno(
                    "Large file",
                    f"{n_total} rows found; PySR fitting is slow beyond ~{self._ROW_CAP} rows.\n\n"
                    f"Randomly subsample to {self._ROW_CAP} rows for fitting?\n"
                    "Yes = subsample (faster). No = keep all rows (slower)."
                ):
                    df = df.sample(n=self._ROW_CAP, random_state=0).reset_index(drop=True)
                    self._log_eq(f"Subsampled {n_total} → {self._ROW_CAP} rows "
                                 f"(random_state=0) for fit speed.")
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] == 0:
                messagebox.showwarning("No numeric columns", "CSV has no numeric columns to fit.")
                return
            self._headers = list(numeric_df.columns)
            self._data = numeric_df.to_numpy()
            self._x_menu["values"] = self._headers
            self._y_menu["values"] = self._headers
            self._x_var.set(self._headers[0])
            self._y_var.set(self._headers[1] if len(self._headers) > 1 else self._headers[0])
            self._log_eq(f"Loaded: {os.path.basename(path)}  —  {len(numeric_df)} rows, "
                         f"{len(self._headers)} numeric columns\nColumns: {', '.join(self._headers)}")
            self._replot_data()
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    # ------------------------------------------------------------------
    # Polynomial fit
    # ------------------------------------------------------------------

    def _poly_fit(self):
        X, Y = self._get_xy()
        if X is None:
            return
        deg = int(self._deg_var.get())
        coeffs = np.polyfit(X, Y, deg)
        p = np.poly1d(coeffs)
        Y_fit = p(X)
        rmse = float(np.sqrt(np.mean((Y - Y_fit) ** 2)))

        # Build LaTeX equation
        terms = []
        for i, c in enumerate(coeffs):
            exp = deg - i
            if abs(c) < 1e-15:
                continue
            if exp == 0:
                terms.append(f"{c:.4g}")
            elif exp == 1:
                terms.append(f"{c:.4g}x")
            else:
                terms.append(f"{c:.4g}x^{{{exp}}}")
        latex_eq = " + ".join(terms).replace("+ -", "- ")

        self._log_eq(
            f"\n─── Polynomial fit (degree {deg}) ───\n"
            f"RMSE = {rmse:.4g}\n"
            f"Coefficients (high→low): {[f'{c:.4g}' for c in coeffs]}\n"
            f"LaTeX: $y = {latex_eq}$"
        )
        self._replot_data(x_fit=np.linspace(X.min(), X.max(), 300),
                          y_fit=p(np.linspace(X.min(), X.max(), 300)),
                          fit_label=f"Poly deg={deg}")

    # ------------------------------------------------------------------
    # PySR
    # ------------------------------------------------------------------

    def _pysr_fit(self):
        if not _PYSR_OK:
            messagebox.showinfo("PySR not available",
                                "Install PySR: pip install pysr  (also needs Julia)")
            return
        X, Y = self._get_xy()
        if X is None:
            return
        n_iter = int(self._iter_var.get())
        self._pysr_running = True
        self._pysr_cancelled = False
        self._pysr_model = None
        self._pysr_model_ref = None
        # "Checking Julia/PySR install…" first: PySRRegressor() construction is
        # where first-run Julia package installation/precompilation actually
        # happens (can take minutes) — this is what used to look like a hang
        # with only a static spinner label and no distinguishing step text.
        self._spinner.set("Checking Julia/PySR install…")
        self._pysr_btn.config(state="disabled")
        self._pysr_cancel_btn.config(state="normal")

        def _run():
            try:
                from pysr import PySRRegressor
                model = PySRRegressor(
                    niterations=n_iter,
                    binary_operators=["+", "-", "*", "/", "^"],
                    unary_operators=["exp", "log", "sqrt", "abs"],
                    populations=10,
                    population_size=15,
                    maxsize=20,
                    verbosity=0,
                    # progress=False: PySR's own tqdm-style console progress
                    # would corrupt the Tk log widget — poll equation_file_
                    # instead (see _poll_pysr_progress) for incremental feedback.
                    progress=False,
                )
                self._pysr_model_ref = model
                if self._pysr_cancelled:
                    return
                self.after(0, lambda: self._spinner.set(f"Fitting (iteration budget: {n_iter})…"))
                self.after(500, self._poll_pysr_progress)
                model.fit(X.reshape(-1, 1), Y)
                if self._pysr_cancelled:
                    return
                self._pysr_model = model
                self.after(0, self._on_pysr_done)
            except Exception as exc:
                if not self._pysr_cancelled:
                    self.after(0, lambda e=exc: self._on_pysr_error(str(e)))
            finally:
                self._pysr_running = False

        threading.Thread(target=_run, daemon=True).start()

    def _poll_pysr_progress(self):
        """Cheap incremental-progress proxy: PySR writes its Pareto-front
        equations to `model.equation_file_` as the search runs — poll that
        file's row count instead of enabling PySR's own progress bar
        (which would corrupt the Tk log widget).
        """
        if not self._pysr_running:
            return
        eq_file = getattr(self._pysr_model_ref, "equation_file_", None)
        if eq_file and os.path.exists(eq_file):
            try:
                n_found = max(sum(1 for _ in open(eq_file)) - 1, 0)  # minus header
                self._spinner.set(f"Fitting… {n_found} candidate equation(s) found so far")
            except OSError:
                pass
        self.after(500, self._poll_pysr_progress)

    def _cancel_pysr(self):
        """Stops watching this run — PySR's Julia .fit() call can't be
        cooperatively interrupted mid-run, so it may keep computing in the
        background until it finishes; this just re-enables the UI so the
        user isn't blocked from starting a corrected-parameters run.
        """
        self._pysr_cancelled = True
        self._pysr_running = False
        self._pysr_btn.config(state="normal")
        self._pysr_cancel_btn.config(state="disabled")
        self._spinner.set("Cancelled — a background PySR run may still be finishing")

    def _on_pysr_done(self):
        self._pysr_running = False
        self._spinner.set("")
        self._pysr_btn.config(state="normal")
        self._pysr_cancel_btn.config(state="disabled")
        if self._pysr_cancelled or self._pysr_model is None:
            return
        eqs = self._pysr_model.equations_
        lines = ["\n─── PySR Symbolic Regression Results ───"]
        for i, row in eqs.iterrows():
            lines.append(f"[{i}] complexity={row.get('complexity','?'):>3}  "
                         f"loss={row.get('loss',0):.4g}  "
                         f"  {row.get('equation','')}")
        best = self._pysr_model.get_best()
        lines.append(f"\nBest: {best.get('equation','?')}")
        self._log_eq("\n".join(lines))

        # Plot best fit
        X, Y = self._get_xy()
        if X is not None:
            X_fine = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
            Y_fine = self._pysr_model.predict(X_fine)
            self._replot_data(x_fit=X_fine.ravel(), y_fit=Y_fine, fit_label="PySR best")

    def _on_pysr_error(self, msg):
        self._pysr_running = False
        self._spinner.set("")
        self._pysr_btn.config(state="normal")
        self._pysr_cancel_btn.config(state="disabled")
        messagebox.showerror("PySR error", msg)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_latex(self):
        content = self._eq_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showinfo("Nothing to export", "Run a fit first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".tex",
            filetypes=[("LaTeX", "*.tex"), ("Text", "*.txt"), ("All", "*")],
            title="Export equation",
        )
        if not path:
            return
        with open(path, "w") as fh:
            fh.write(content)
        messagebox.showinfo("Saved", f"Equations written to:\n{path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_xy(self):
        if self._data is None:
            messagebox.showinfo("No data", "Load a CSV first.")
            return None, None
        try:
            xi = self._headers.index(self._x_var.get())
            yi = self._headers.index(self._y_var.get())
        except ValueError:
            messagebox.showwarning("Column not found", "Select valid X and Y columns.")
            return None, None
        X = self._data[:, xi]
        Y = self._data[:, yi]
        mask = np.isfinite(X) & np.isfinite(Y)
        return X[mask], Y[mask]

    def _log_eq(self, text: str):
        self._eq_text.config(state=tk.NORMAL)
        self._eq_text.insert(tk.END, text + "\n")
        self._eq_text.see(tk.END)
        self._eq_text.config(state=tk.DISABLED)

    def _replot_data(self, x_fit=None, y_fit=None, fit_label="Fit"):
        if not _MPL_OK:
            return
        X, Y = self._get_xy()
        if X is None:
            return
        self._ax.clear()
        self._ax.set_facecolor("#2a2a3e")
        self._ax.scatter(X, Y, color="#7aa2f7", s=20, label="Data", zorder=3)
        if x_fit is not None and y_fit is not None:
            self._ax.plot(x_fit, y_fit, color="#f38ba8", lw=1.5, label=fit_label)
        self._ax.set_xlabel(self._x_var.get(), color="#cdd6f4")
        self._ax.set_ylabel(self._y_var.get(), color="#cdd6f4")
        self._ax.tick_params(colors="#cdd6f4")
        self._ax.legend(facecolor="#2a2a3e", labelcolor="#cdd6f4")
        self._fig.tight_layout()
        self._canvas_mpl.draw()
