"""
mso_studio.core.digitizer
============================
Pure data model and export logic for the plot/image digitizer, ported from
``src/notebook_gemini/digitizer.py`` (Tkinter). The coordinate math and
CSV/RKD-XML/YAML export functions were already framework-agnostic in the
original -- only the interactive click-to-calibrate canvas is genuinely
GUI-specific (see ``gui/modules/tools.py``'s ``DigitizerWidget``, built on
``QGraphicsView``/``QGraphicsScene``).
"""
from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class CalibPoint:
    """One calibration click: pixel (px, py) + the known data value there."""
    px: float
    py: float
    value: float


@dataclass
class DataPoint:
    px: float
    py: float
    x: float
    y: float
    dy_plus: float = 0.0
    dy_minus: float = 0.0


def pixel_to_data(
    px: float, py: float,
    x_cal: Tuple[CalibPoint, CalibPoint],
    y_cal: Tuple[CalibPoint, CalibPoint],
) -> Tuple[float, float]:
    """Map a pixel coordinate to data-space using two calibration points per
    axis. Returns ``(nan, nan)`` if either axis's two calibration points
    coincide in pixel space (degenerate calibration)."""
    x0, x1 = x_cal
    y0, y1 = y_cal
    dx_pix = x1.px - x0.px
    dy_pix = y1.py - y0.py
    if dx_pix == 0 or dy_pix == 0:
        return float("nan"), float("nan")
    x = x0.value + (px - x0.px) / dx_pix * (x1.value - x0.value)
    y = y0.value + (py - y0.py) / dy_pix * (y1.value - y0.value)
    return x, y


def export_csv(points: List[DataPoint], path: str, x_label: str = "x", y_label: str = "y") -> None:
    has_err = any(dp.dy_plus or dp.dy_minus for dp in points)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        if has_err:
            writer.writerow([x_label, y_label, f"{y_label}_plus", f"{y_label}_minus"])
            for dp in points:
                writer.writerow([dp.x, dp.y, dp.dy_plus, dp.dy_minus])
        else:
            writer.writerow([x_label, y_label])
            for dp in points:
                writer.writerow([dp.x, dp.y])


def export_rkd_xml(
    points: List[DataPoint], path: str,
    x_label: str = "temperature", y_label: str = "ignition delay",
    x_units: str = "K", y_units: str = "ms",
) -> None:
    """Minimal ReSpecTh/RKD-format XML export."""
    root = ET.Element("experiment")
    ET.SubElement(root, "fileAuthor").text = "MSO Studio digitizer"
    dg = ET.SubElement(root, "dataGroup")
    has_err = any(dp.dy_plus or dp.dy_minus for dp in points)

    ET.SubElement(dg, "property", name=x_label, label="x", sourceType="digitizer", units=x_units)
    ET.SubElement(dg, "property", name=y_label, label="y", sourceType="digitizer", units=y_units)
    if has_err:
        ET.SubElement(dg, "property", name=f"{y_label}_uncertainty", label="dy",
                       sourceType="digitizer", units=y_units)

    for dp in points:
        dp_elem = ET.SubElement(dg, "dataPoint")
        ET.SubElement(dp_elem, "x").text = f"{dp.x:.6g}"
        ET.SubElement(dp_elem, "y").text = f"{dp.y:.6g}"
        if has_err:
            ET.SubElement(dp_elem, "dy").text = f"+{dp.dy_plus:.6g}/-{dp.dy_minus:.6g}"

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def export_yaml(points: List[DataPoint], path: str, x_label: str = "x", y_label: str = "y") -> None:
    has_err = any(dp.dy_plus or dp.dy_minus for dp in points)
    lines = [
        "# Digitized data -- MSO Studio digitizer",
        f"x_label: {x_label}",
        f"y_label: {y_label}",
        "data:",
    ]
    for dp in points:
        if has_err:
            lines.append(f"  - {{{x_label}: {dp.x:.6g}, {y_label}: {dp.y:.6g}, "
                          f"dy_plus: {dp.dy_plus:.6g}, dy_minus: {dp.dy_minus:.6g}}}")
        else:
            lines.append(f"  - {{{x_label}: {dp.x:.6g}, {y_label}: {dp.y:.6g}}}")
    Path(path).write_text("\n".join(lines) + "\n")
