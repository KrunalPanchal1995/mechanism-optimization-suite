"""
digitizer.py
------------
Tkinter-based plot digitizer.

Workflow:
1. Load an image (PNG / JPG / PDF page).
2. Calibrate X axis: click two known X-value reference points.
3. Calibrate Y axis: click two known Y-value reference points.
4. Click data points on the graph — each click appends a (x, y) row.
5. Export to CSV; optionally ask Gemini to read axis labels.

Usage (standalone)::

    python digitizer.py [image_path]

Usage (embedded)::

    from notebook_gemini.digitizer import DigitizerPanel
    panel = DigitizerPanel(parent_frame, client=GeminiClient())
    panel.pack(fill=tk.BOTH, expand=True)
"""

from __future__ import annotations

import csv
import io
import math
import os
import threading
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Optional, List, Tuple

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class CalibPoint:
    """One calibration click: pixel (px, py) + known data value."""
    def __init__(self, px: int, py: int, value: float):
        self.px, self.py, self.value = px, py, value


class DataPoint:
    def __init__(self, px: int, py: int, x: float, y: float,
                 dy_plus: float = 0.0, dy_minus: float = 0.0):
        self.px, self.py, self.x, self.y = px, py, x, y
        self.dy_plus  = dy_plus    # upper uncertainty (+)
        self.dy_minus = dy_minus   # lower uncertainty (-)


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

def _pixel_to_data(
    px: int, py: int,
    x_cal: Tuple[CalibPoint, CalibPoint],
    y_cal: Tuple[CalibPoint, CalibPoint],
) -> Tuple[float, float]:
    x0, x1 = x_cal
    y0, y1 = y_cal
    dx_pix = x1.px - x0.px
    dy_pix = y1.py - y0.py
    if dx_pix == 0 or dy_pix == 0:
        return float("nan"), float("nan")
    x = x0.value + (px - x0.px) / dx_pix * (x1.value - x0.value)
    y = y0.value + (py - y0.py) / dy_pix * (y1.value - y0.value)
    return x, y


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class DigitizerPanel(ttk.Frame):
    """
    Parameters
    ----------
    parent : tk widget
    client : GeminiClient
        May be unavailable; Gemini features degrade gracefully.
    palette : dict, optional
    """

    _MODES = ("x_cal", "y_cal", "data", "err_bar")
    _MODE_LABELS = {
        "x_cal":   "Mode: calibrate X axis (click 2 reference points)",
        "y_cal":   "Mode: calibrate Y axis (click 2 reference points)",
        "data":    "Mode: pick data points (click on the curve)",
        "err_bar": "Mode: pick error bars — click 1=upper, 2=lower of bar; pairs define ±uncertainty",
    }
    _CAL_COLORS = {"x_cal": "red", "y_cal": "blue", "data": "lime", "err_bar": "orange"}

    def __init__(self, parent, client, palette: Optional[dict] = None):
        super().__init__(parent)
        self._client = client
        self._pal = palette or {"bg": "#1e1e2e", "surface": "#2a2a3e",
                                "accent": "#7aa2f7", "text": "#cdd6f4"}

        self._image_orig: Optional[Image.Image] = None
        self._image_tk: Optional[ImageTk.PhotoImage] = None
        self._scale: float = 1.0

        self._mode: str = "x_cal"
        self._x_cal: List[CalibPoint] = []
        self._y_cal: List[CalibPoint] = []
        self._data_pts: List[DataPoint] = []
        self._canvas_items: List[int] = []   # canvas IDs for markers
        # Error-bar picking state: each pair (upper, lower) defines one bar on the last data pt
        self._err_upper: Optional[float] = None   # y value of upper bound

        self._log_x = tk.BooleanVar(value=False)
        self._log_y = tk.BooleanVar(value=False)

        if not _PIL_OK:
            ttk.Label(self, text="Pillow (PIL) is required for the digitizer.\n"
                      "Install with: pip install Pillow").pack(expand=True)
            return

        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── toolbar ──────────────────────────────────────────────────
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, pady=(0, 4))

        ttk.Button(bar, text="Open image…",  command=self._open_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Open PDF page…", command=self._open_pdf_page).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self._mode_var = tk.StringVar(value="x_cal")
        for m in self._MODES:
            ttk.Radiobutton(bar, text=m.replace("_", " ").title(),
                            variable=self._mode_var, value=m,
                            command=self._on_mode_change).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Checkbutton(bar, text="log X", variable=self._log_x).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(bar, text="log Y", variable=self._log_y).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(bar, text="Undo last", command=self._undo_last).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Clear all", command=self._clear_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Export CSV…",     command=self._export_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Export XML (RKD)…", command=self._export_rkd_xml).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Export YAML…",    command=self._export_yaml).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Gemini: read axes", command=self._gemini_read_axes).pack(side=tk.LEFT, padx=4)

        # ── status line ──────────────────────────────────────────────
        self._status_var = tk.StringVar(value=self._MODE_LABELS["x_cal"])
        ttk.Label(self, textvariable=self._status_var, anchor=tk.W).pack(fill=tk.X, pady=2)

        # ── main canvas + scrollbars ──────────────────────────────────
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        hbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas = tk.Canvas(canvas_frame, bg="#111122", cursor="crosshair",
                                 xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        hbar.config(command=self._canvas.xview)
        vbar.config(command=self._canvas.yview)
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<MouseWheel>",   self._on_scroll)
        self._canvas.bind("<Button-4>",     self._on_scroll)  # Linux scroll up
        self._canvas.bind("<Button-5>",     self._on_scroll)  # Linux scroll down

        # ── data table (bottom) ───────────────────────────────────────
        tbl_frame = ttk.Frame(self)
        tbl_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(tbl_frame, text="Digitized points:").pack(anchor=tk.W)
        self._table_var = tk.StringVar(value="(none yet)")
        ttk.Label(tbl_frame, textvariable=self._table_var, anchor=tk.W,
                  font=("DejaVu Sans Mono", 8)).pack(anchor=tk.W)

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All", "*")]
        )
        if path:
            self._load_image(Image.open(path).convert("RGB"))

    def _open_pdf_page(self):
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf"), ("All", "*")])
        if not path:
            return
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                n = len(pdf.pages)
                page_no = simpledialog.askinteger(
                    "Page", f"PDF has {n} page(s). Which page? (1–{n})",
                    minvalue=1, maxvalue=n, initialvalue=1
                )
                if page_no is None:
                    return
                img = pdf.pages[page_no - 1].to_image(resolution=150).original
            self._load_image(img.convert("RGB"))
        except Exception as exc:
            messagebox.showerror("PDF error", str(exc))

    def _load_image(self, img: "Image.Image"):
        self._image_orig = img
        self._clear_all(keep_image=False)
        self._redraw_image()
        self._status_var.set("Image loaded. " + self._MODE_LABELS[self._mode])

    def _redraw_image(self):
        if self._image_orig is None:
            return
        w = int(self._image_orig.width  * self._scale)
        h = int(self._image_orig.height * self._scale)
        img_rs = self._image_orig.resize((w, h), Image.LANCZOS)
        self._image_tk = ImageTk.PhotoImage(img_rs)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._image_tk)
        self._canvas.config(scrollregion=(0, 0, w, h))
        self._canvas_items.clear()
        self._redraw_markers()

    # ------------------------------------------------------------------
    # Scroll / zoom
    # ------------------------------------------------------------------

    def _on_scroll(self, event):
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            self._scale = min(self._scale * 1.15, 8.0)
        else:
            self._scale = max(self._scale / 1.15, 0.1)
        self._redraw_image()

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def _on_click(self, event):
        if self._image_orig is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        # Convert canvas coords (account for scroll offset)
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        # Map back to original image pixels
        px = int(cx / self._scale)
        py = int(cy / self._scale)
        mode = self._mode_var.get()

        if mode == "x_cal":
            val = simpledialog.askfloat("X calibration",
                                        f"Enter the data X value at this point (px={px}):")
            if val is None:
                return
            self._x_cal.append(CalibPoint(px, py, val))
            self._draw_marker(cx, cy, color="red", label=f"X={val}")
            if len(self._x_cal) == 2:
                self._status_var.set("X axis calibrated ✓  →  switch to Y calibration")

        elif mode == "y_cal":
            val = simpledialog.askfloat("Y calibration",
                                        f"Enter the data Y value at this point (py={py}):")
            if val is None:
                return
            self._y_cal.append(CalibPoint(px, py, val))
            self._draw_marker(cx, cy, color="blue", label=f"Y={val}")
            if len(self._y_cal) == 2:
                self._status_var.set("Y axis calibrated ✓  →  switch to Data mode to pick points")

        elif mode == "data":
            if len(self._x_cal) < 2 or len(self._y_cal) < 2:
                messagebox.showwarning("Not calibrated",
                                       "Calibrate both X and Y axes first (2 points each).")
                return
            x, y = _pixel_to_data(px, py,
                                   tuple(self._x_cal[:2]),
                                   tuple(self._y_cal[:2]))
            if self._log_x.get() and x > 0:
                x = 10 ** x
            if self._log_y.get() and y > 0:
                y = 10 ** y
            self._data_pts.append(DataPoint(px, py, x, y))
            self._draw_marker(cx, cy, color="lime", label=f"({x:.4g},{y:.4g})")
            self._refresh_table()

        elif mode == "err_bar":
            if len(self._x_cal) < 2 or len(self._y_cal) < 2:
                messagebox.showwarning("Not calibrated",
                                       "Calibrate both X and Y axes first.")
                return
            _, y_raw = _pixel_to_data(px, py,
                                      tuple(self._x_cal[:2]),
                                      tuple(self._y_cal[:2]))
            y_val = 10 ** y_raw if (self._log_y.get() and y_raw > 0) else y_raw
            if self._err_upper is None:
                # First click: upper bound (top of error bar)
                self._err_upper = y_val
                self._draw_marker(cx, cy, color="orange", label=f"↑{y_val:.4g}")
                self._status_var.set("Error bar: upper set — now click the LOWER bound")
            else:
                # Second click: lower bound → assign to most recent data point
                y_upper = self._err_upper
                y_lower = y_val
                self._err_upper = None
                if not self._data_pts:
                    messagebox.showwarning("No data point",
                                           "Add a data point in 'Data' mode first, "
                                           "then assign its error bar here.")
                    return
                dp = self._data_pts[-1]
                y_center = (y_upper + y_lower) / 2.0
                dp.dy_plus  = abs(y_upper - dp.y)
                dp.dy_minus = abs(dp.y  - y_lower)
                self._draw_marker(cx, cy, color="orange", label=f"↓{y_lower:.4g}")
                self._status_var.set(
                    f"Error bar set on last point: +{dp.dy_plus:.4g} / -{dp.dy_minus:.4g}")
                self._refresh_table()

    def _on_mode_change(self):
        self._mode = self._mode_var.get()
        self._status_var.set(self._MODE_LABELS[self._mode])

    # ------------------------------------------------------------------
    # Markers / table
    # ------------------------------------------------------------------

    def _draw_marker(self, cx, cy, color: str, label: str):
        r = 5
        iid = self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                       outline=color, width=2)
        tid = self._canvas.create_text(cx + 8, cy - 8, text=label,
                                       fill=color, font=("DejaVu Sans", 7))
        self._canvas_items.extend([iid, tid])

    def _redraw_markers(self):
        for cp in self._x_cal:
            self._draw_marker(cp.px * self._scale, cp.py * self._scale,
                              "red", f"X={cp.value}")
        for cp in self._y_cal:
            self._draw_marker(cp.px * self._scale, cp.py * self._scale,
                              "blue", f"Y={cp.value}")
        for dp in self._data_pts:
            self._draw_marker(dp.px * self._scale, dp.py * self._scale,
                              "lime", f"({dp.x:.4g},{dp.y:.4g})")

    def _refresh_table(self):
        if not self._data_pts:
            self._table_var.set("(none yet)")
            return
        rows = [f"  {'x':>12}  {'y':>12}"]
        for dp in self._data_pts[-8:]:
            rows.append(f"  {dp.x:>12.6g}  {dp.y:>12.6g}")
        if len(self._data_pts) > 8:
            rows.append(f"  … ({len(self._data_pts)} total)")
        self._table_var.set("\n".join(rows))

    # ------------------------------------------------------------------
    # Edit
    # ------------------------------------------------------------------

    def _undo_last(self):
        if self._data_pts:
            self._data_pts.pop()
            self._refresh_table()
            self._redraw_image()
        elif self._y_cal:
            self._y_cal.pop()
            self._redraw_image()
        elif self._x_cal:
            self._x_cal.pop()
            self._redraw_image()

    def _clear_all(self, keep_image: bool = True):
        self._x_cal.clear()
        self._y_cal.clear()
        self._data_pts.clear()
        self._canvas_items.clear()
        if not keep_image:
            self._canvas.delete("all")
        else:
            self._redraw_image()
        self._refresh_table()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_csv(self):
        if not self._data_pts:
            messagebox.showinfo("No data", "No data points to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*")],
            title="Save digitized data",
        )
        if not path:
            return
        x_label = simpledialog.askstring("X label", "X axis label:", initialvalue="T_K")
        y_label = simpledialog.askstring("Y label", "Y axis label:", initialvalue="tau_ms")
        x_label = x_label or "x"
        y_label = y_label or "y"
        has_err = any(dp.dy_plus or dp.dy_minus for dp in self._data_pts)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            if has_err:
                writer.writerow([x_label, y_label, f"{y_label}_plus", f"{y_label}_minus"])
                for dp in self._data_pts:
                    writer.writerow([dp.x, dp.y, dp.dy_plus, dp.dy_minus])
            else:
                writer.writerow([x_label, y_label])
                for dp in self._data_pts:
                    writer.writerow([dp.x, dp.y])
        messagebox.showinfo("Exported", f"Saved {len(self._data_pts)} points to:\n{path}")

    def _export_rkd_xml(self):
        """Export digitized points as a minimal ReSpecTh/RKD-format XML."""
        if not self._data_pts:
            messagebox.showinfo("No data", "No data points to export.")
            return
        x_label = simpledialog.askstring("X property", "X axis property name:", initialvalue="temperature")
        y_label = simpledialog.askstring("Y property", "Y axis property name:", initialvalue="ignition delay")
        x_units  = simpledialog.askstring("X units", "X units:", initialvalue="K")
        y_units  = simpledialog.askstring("Y units", "Y units:", initialvalue="ms")
        if x_label is None or y_label is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xml",
            filetypes=[("XML", "*.xml"), ("All", "*")],
            title="Export RKD XML",
        )
        if not path:
            return

        root = ET.Element("experiment")
        ET.SubElement(root, "fileAuthor").text = "RMIP digitizer"
        dg = ET.SubElement(root, "dataGroup")
        has_err = any(dp.dy_plus or dp.dy_minus for dp in self._data_pts)

        p_x = ET.SubElement(dg, "property", name=x_label or "x",
                             label="x", sourceType="digitizer", units=x_units or "")
        p_y = ET.SubElement(dg, "property", name=y_label or "y",
                             label="y", sourceType="digitizer", units=y_units or "")
        if has_err:
            ET.SubElement(dg, "property", name=f"{y_label}_uncertainty",
                          label="dy", sourceType="digitizer", units=y_units or "")

        for dp in self._data_pts:
            dp_elem = ET.SubElement(dg, "dataPoint")
            ET.SubElement(dp_elem, "x").text = f"{dp.x:.6g}"
            ET.SubElement(dp_elem, "y").text = f"{dp.y:.6g}"
            if has_err:
                ET.SubElement(dp_elem, "dy").text = (
                    f"+{dp.dy_plus:.6g}/-{dp.dy_minus:.6g}"
                )

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        messagebox.showinfo("Exported", f"RKD XML saved to:\n{path}")

    def _export_yaml(self):
        """Export digitized points as a YAML data file."""
        if not self._data_pts:
            messagebox.showinfo("No data", "No data points to export.")
            return
        x_label = simpledialog.askstring("X label", "X axis label:", initialvalue="T_K") or "x"
        y_label = simpledialog.askstring("Y label", "Y axis label:", initialvalue="tau_ms") or "y"
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*")],
            title="Export digitized data as YAML",
        )
        if not path:
            return
        has_err = any(dp.dy_plus or dp.dy_minus for dp in self._data_pts)
        lines = [
            "# Digitized data — RMIP digitizer",
            f"x_label: {x_label}",
            f"y_label: {y_label}",
            "data:",
        ]
        for dp in self._data_pts:
            if has_err:
                lines.append(f"  - {{{x_label}: {dp.x:.6g}, {y_label}: {dp.y:.6g}, "
                             f"dy_plus: {dp.dy_plus:.6g}, dy_minus: {dp.dy_minus:.6g}}}")
            else:
                lines.append(f"  - {{{x_label}: {dp.x:.6g}, {y_label}: {dp.y:.6g}}}")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        messagebox.showinfo("Exported", f"YAML saved to:\n{path}")

    # ------------------------------------------------------------------
    # Gemini axis reading
    # ------------------------------------------------------------------

    def _gemini_read_axes(self):
        if self._image_orig is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if not self._client.available:
            messagebox.showinfo("Gemini not available",
                                "Set GOOGLE_API_KEY to use Gemini axis reading.")
            return

        def _run():
            buf = io.BytesIO()
            self._image_orig.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            reply = self._client.chat_with_image_bytes(
                png_bytes, "image/png",
                "This is a combustion data plot. "
                "Please identify:\n"
                "1. X axis: label, units, min value, max value, scale (linear/log)\n"
                "2. Y axis: label, units, min value, max value, scale (linear/log)\n"
                "3. Number of data series and their legend labels (if any)\n"
                "Be concise."
            )
            self.after(0, lambda: messagebox.showinfo("Gemini axis reading", reply))

        threading.Thread(target=_run, daemon=True).start()
        self._status_var.set("⏳ Asking Gemini to read axes…")


# ---------------------------------------------------------------------------
# Standalone launcher
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from notebook_gemini.gemini_client import GeminiClient

    root = tk.Tk()
    root.title("Plot Digitizer")
    root.geometry("1000x720")
    client = GeminiClient()
    panel = DigitizerPanel(root, client=client)
    panel.pack(fill=tk.BOTH, expand=True)
    if len(sys.argv) > 1:
        from PIL import Image as _Im
        panel._load_image(_Im.open(sys.argv[1]).convert("RGB"))
    root.mainloop()
