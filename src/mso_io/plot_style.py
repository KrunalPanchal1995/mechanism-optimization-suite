"""plot_style.py -- shared matplotlib customization contract for GUI-driven
plots (KRUNAL_JULY_23.md point 4: "every option that matplotlib provides
must be available in this GUI").

Framework-free dataclass + apply/save helpers, consumed by:
  - idt_sensitivity_gui.py / brute_force_sa_gui.py's chart panel
  - src/Sensitivity_Analysis/sens_3_params.py's two savefig call sites (via
    an optional --plot-style-json flag)
  - src/modules/plotting.py (Point 5's ActiveParameterScreen plots)
  - src/mso_studio/gui/widgets/mpl_canvas.py (PySide6 side)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

EXPORT_FORMATS = ["png", "pdf", "svg", "eps"]
LEGEND_LOCATIONS = [
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center",
    "center",
]
GRID_AXES = ["both", "x", "y", "none"]


@dataclass
class PlotStyle:
    export_format: str = "png"
    dpi: int = 150
    figsize: Optional[List[float]] = None      # [w, h] inches; None = leave default

    color_scheme: str = "default"               # a matplotlib colormap/style name...
    colors: Optional[List[str]] = None           # ...or an explicit per-series color list

    font_family: str = ""                        # "" = matplotlib default
    font_size_title: int = 12
    font_size_axis_label: int = 10
    font_size_tick_label: int = 8
    font_size_legend: int = 9

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    show_legend: bool = True
    legend_loc: str = "best"

    grid: str = "none"                            # "both"/"x"/"y"/"none"
    grid_alpha: float = 0.3

    xlim: Optional[List[float]] = None
    ylim: Optional[List[float]] = None

    line_width: float = 1.5
    marker: str = ""
    alpha: float = 1.0

    annotations: List[Dict[str, object]] = field(default_factory=list)
    # each: {"text": str, "xy": [x, y], "xytext": [x, y] (optional),
    #        "fontsize": int (optional), "color": str (optional)}

    tight_layout: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PlotStyle":
        return cls(**json.loads(text))

    @classmethod
    def load(cls, path: str) -> "PlotStyle":
        with open(path) as fh:
            return cls.from_json(fh.read())

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write(self.to_json())


def color_cycle(style: PlotStyle, n: int) -> List[str]:
    """Resolve style.colors / style.color_scheme into a concrete list of
    n color specs, falling back to matplotlib's own cycle/colormap."""
    if style.colors:
        cyc = list(style.colors)
        while len(cyc) < n:
            cyc += style.colors
        return cyc[:n]
    if style.color_scheme and style.color_scheme != "default":
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(style.color_scheme)
            return [cmap(i / max(n - 1, 1)) for i in range(n)]
        except Exception:
            pass
    import matplotlib.pyplot as plt
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    base = prop_cycle.by_key().get("color", ["C0"])
    out = list(base)
    while len(out) < n:
        out += base
    return out[:n]


def apply_style(fig, ax, style: PlotStyle) -> None:
    """Apply a PlotStyle to an existing matplotlib Figure/Axes in place."""
    if style.font_family:
        for text in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                     + ax.get_xticklabels() + ax.get_yticklabels()):
            text.set_family(style.font_family)

    if style.title:
        ax.set_title(style.title, fontsize=style.font_size_title)
    elif ax.get_title():
        ax.title.set_fontsize(style.font_size_title)

    if style.xlabel:
        ax.set_xlabel(style.xlabel, fontsize=style.font_size_axis_label)
    elif ax.get_xlabel():
        ax.xaxis.label.set_fontsize(style.font_size_axis_label)

    if style.ylabel:
        ax.set_ylabel(style.ylabel, fontsize=style.font_size_axis_label)
    elif ax.get_ylabel():
        ax.yaxis.label.set_fontsize(style.font_size_axis_label)

    ax.tick_params(axis="both", labelsize=style.font_size_tick_label)

    if style.show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=style.legend_loc, fontsize=style.font_size_legend)
    elif not style.show_legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    if style.grid != "none":
        axis = "both" if style.grid == "both" else style.grid
        ax.grid(True, axis=axis, alpha=style.grid_alpha)
    else:
        ax.grid(False)

    if style.xlim:
        ax.set_xlim(style.xlim[0], style.xlim[1])
    if style.ylim:
        ax.set_ylim(style.ylim[0], style.ylim[1])

    for ann in style.annotations:
        ax.annotate(
            ann.get("text", ""),
            xy=tuple(ann.get("xy", (0, 0))),
            xytext=tuple(ann["xytext"]) if "xytext" in ann else None,
            fontsize=ann.get("fontsize", style.font_size_legend),
            color=ann.get("color", "black"),
        )

    if style.figsize:
        fig.set_size_inches(*style.figsize)

    if style.tight_layout:
        fig.tight_layout()


def save_figure(fig, path: str, style: PlotStyle) -> str:
    """Save fig to path using style's export_format/dpi. If path's
    extension doesn't match export_format, the extension is replaced."""
    import os
    root, _ext = os.path.splitext(path)
    out_path = f"{root}.{style.export_format}"
    fig.savefig(out_path, format=style.export_format, dpi=style.dpi,
                bbox_inches="tight" if style.tight_layout else None)
    return out_path
