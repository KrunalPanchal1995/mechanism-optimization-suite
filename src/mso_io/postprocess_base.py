"""
mso_io.postprocess_base
=========================
Shared contract every repo's post-processing entry point builds on --
``VisualAid.py`` (MUQ-SAC, Sensitivity_Analysis), ``uq_plots.py`` (forward_uq),
``plots.py`` (graph_sa), ``run_plots.py`` (mso_compare), ``prs_report.py``
(automation) and ``rmip/postprocess.py`` all add ``add_postprocess_args()``
to their existing ``argparse`` parser (or build one in ``__main__``) and call
``save_figure``/``write_manifest`` so every repo behaves identically under
the one flag, per the "generate reports for journal papers from the runs"
requirement.

Design rules
------------
* **CSV emission is never optional.** ``save_figure()`` always writes the
  data backing a figure as a CSV next to it, whether or not ``--journal``
  was passed -- ``--journal`` only adds the *extra* paper-table CSVs
  (Arrhenius parameter tables, summary/residual tables) that don't
  correspond to any single figure. ``--csv-only`` suppresses the image
  file, never the CSV.
* **Tolerant of a partial run.** ``discover()`` globs for known output
  filenames and returns whatever subset exists -- it never raises just
  because a run is still in progress (the ``--intermediate`` use case:
  processing results *while* the pipeline keeps running).
* **No Qt.** Matches the ``mso_studio.core`` rule (see
  ``src/mso_studio/tests/test_no_gui_imports.py``) so every one of these
  modules stays importable from a bare, GUI-less interpreter -- including
  the anaconda subprocess environment the GUI itself shells out to.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------

def add_postprocess_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the shared post-processing flags on *parser*.

    Every repo's ``__main__`` block does::

        parser = argparse.ArgumentParser(...)
        add_postprocess_args(parser)
        args = parser.parse_args()
        if args.postprocess:
            result = postprocess(args.postprocess, args.out, journal=args.journal, ...)
    """
    group = parser.add_argument_group("post-processing")
    group.add_argument(
        "--postprocess", metavar="RUN_DIR", default=None,
        help="Process an existing (or still-running) run directory instead of "
             "starting a new run: regenerate plots/tables from whatever output "
             "files are present.",
    )
    group.add_argument(
        "--out", metavar="DIR", default=None,
        help="Output directory for post-processed figures/tables "
             "(default: <RUN_DIR>/postprocess).",
    )
    group.add_argument(
        "--journal", action="store_true",
        help="Also emit the extra tidy CSV tables a paper's results section "
             "needs (per-target predictions/residuals, per-reaction Arrhenius "
             "parameters, cost/convergence summary) under <out>/journal_data/.",
    )
    group.add_argument(
        "--csv-only", action="store_true",
        help="Write only the CSVs backing each figure, skip the image files.",
    )
    group.add_argument(
        "--formats", metavar="FMT,FMT", default="png",
        help="Comma-separated image formats to save each figure as, e.g. "
             "'png,pdf' (default: png). Ignored with --csv-only.",
    )
    group.add_argument(
        "--intermediate", action="store_true",
        help="Tolerate a partially-written run directory (the run is still "
             "in progress): process whatever output files exist so far "
             "instead of requiring the run to have finished.",
    )
    return parser


def formats_from_args(args: argparse.Namespace) -> List[str]:
    return [f.strip() for f in args.formats.split(",") if f.strip()]


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------

@dataclass
class PostProcessResult:
    """What one ``postprocess(...)`` call produced. Every repo's
    ``postprocess()`` returns one of these so the GUI's "Post-process
    results" action (``mso_studio.core.*``) has a uniform shape to render,
    and so :func:`write_manifest` / :meth:`mso_project.Project.scan` can
    classify the output directory the same way everywhere."""

    run_dir: str
    out_dir: str
    figures: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    journal_tables: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)  # human-readable "why" notes
    manifest_path: Optional[str] = None
    partial: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        return d


# ---------------------------------------------------------------------------
# Figure + CSV emission
# ---------------------------------------------------------------------------

def save_figure(
    fig, name: str, out_dir: str | Path, formats: Iterable[str] = ("png",),
    *, df: Optional[pd.DataFrame] = None, csv_only: bool = False,
) -> Dict[str, str]:
    """Save *fig* (a Matplotlib Figure) as ``<out_dir>/<name>.<fmt>`` for
    every format in *formats*, and **always** save the DataFrame backing it
    as ``<out_dir>/<name>.csv`` -- CSV emission is unconditional, per the
    "it must still generate the csv files that stores the data" requirement.

    If *df* is not supplied, the figure's own line/scatter data is pulled
    off its axes (x/y arrays) as a best-effort fallback so a CSV is written
    even when the caller didn't build a DataFrame up front.

    Returns ``{"csv": path, "png": path, ...}`` for whatever was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    csv_path = out_dir / f"{name}.csv"
    frame = df if df is not None else _dataframe_from_figure(fig)
    frame.to_csv(csv_path, index=False)
    written["csv"] = str(csv_path)

    if not csv_only:
        for fmt in formats:
            fmt = fmt.lstrip(".")
            img_path = out_dir / f"{name}.{fmt}"
            fig.savefig(img_path, bbox_inches="tight")
            written[fmt] = str(img_path)

    return written


def _dataframe_from_figure(fig) -> pd.DataFrame:
    """Best-effort extraction of the plotted x/y series when the caller
    didn't pass an explicit DataFrame -- never raises; returns an empty
    frame if the figure has no line/scatter data to pull."""
    rows: Dict[str, Any] = {}
    try:
        for ax_i, ax in enumerate(fig.axes):
            for line_i, line in enumerate(ax.get_lines()):
                xdata, ydata = line.get_xdata(), line.get_ydata()
                label = line.get_label() or f"ax{ax_i}_line{line_i}"
                if str(label).startswith("_"):
                    label = f"ax{ax_i}_line{line_i}"
                rows[f"{label}_x"] = list(xdata)
                rows[f"{label}_y"] = list(ydata)
    except Exception:
        pass
    if not rows:
        return pd.DataFrame()
    # Pad ragged columns to the same length so pd.DataFrame doesn't choke.
    max_len = max((len(v) for v in rows.values()), default=0)
    for k, v in rows.items():
        rows[k] = list(v) + [None] * (max_len - len(v))
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, name: str, out_dir: str | Path) -> str:
    """Write a standalone (not figure-backed) CSV table, e.g. a journal
    summary table. Returns the path written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def write_manifest(result: PostProcessResult, out_dir: Optional[str | Path] = None) -> str:
    """Write ``postprocess_manifest.json`` listing every figure/table this
    call produced, so ``mso_project.Project.scan()`` can classify the
    output directory the same way ``capture()`` classifies a fresh run."""
    out_dir = Path(out_dir or result.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "postprocess_manifest.json"
    payload = result.to_dict()
    payload["generated_at"] = _now_iso()
    path.write_text(json.dumps(payload, indent=2, default=str))
    result.manifest_path = str(path)
    return str(path)


# ---------------------------------------------------------------------------
# Discovery -- tolerant of a partial (still-running) run directory
# ---------------------------------------------------------------------------

def discover(run_dir: str | Path, patterns: Dict[str, str]) -> Dict[str, List[Path]]:
    """Glob *run_dir* for each ``{key: glob_pattern}`` in *patterns* (glob
    is relative to run_dir, ``**`` allowed). Returns ``{key: [matches]}``;
    a key with no matches maps to ``[]`` rather than being omitted or
    raising -- this is what makes ``--intermediate`` safe to call against a
    run directory the pipeline hasn't finished writing yet.
    """
    run_dir = Path(run_dir)
    out: Dict[str, List[Path]] = {}
    for key, pattern in patterns.items():
        if not run_dir.is_dir():
            out[key] = []
            continue
        out[key] = sorted(run_dir.glob(pattern))
    return out


def default_out_dir(run_dir: str | Path, out: Optional[str]) -> Path:
    return Path(out) if out else Path(run_dir) / "postprocess"
