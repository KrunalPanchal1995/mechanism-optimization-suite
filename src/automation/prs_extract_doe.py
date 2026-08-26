#!/usr/bin/env python3
"""
prs_extract_doe.py — extract the already-completed 5x3 full-factorial DoE
(p_start, p_final, WMRE) per fuel out of the three validated
report_<fuel>_factorial.tex tables, into a canonical CSV that every
downstream prs_*.py script reads.

Integrated into the mechanism-optimization-suite repo (2026-07-21). The
fuel-comparison data (MECHANISM_COMPARISON/) is multi-GB external output
that lives outside this repo, not a sibling of this script anymore -- set
MSO_PPRS_DATA_ROOT to the external p_PRS project root (the directory that
contains MECHANISM_COMPARISON/, 1_3_Butadiene/, n_Heptane_159/, MB/).
Defaults to this script's own parent directory, preserving the original
in-place behavior if ever run from inside that external project again.

Why parse the .tex tables instead of recomputing from ERROR_METRICS CSVs:
run_tasks.py's own generic report task sums Error_Function per case (see
render_report_tex's "Objective (sum)" column) and that number does NOT match
the "Mean WMRE (%)" figures in report_<fuel>_factorial.tex (checked by hand
for n-Heptane case powerMeanWMRE_1_0_05_1200_GEN_PM_POLISH_0_05: generic sum
gives ~31.5%, the factorial report says 12.1688%). The exact aggregation
used to produce the factorial reports isn't present anywhere in checked-in
code, so recomputing it here would silently diverge. The three .tex tables
are the validated ground truth (corruption-checked, footnoted, already
used to drive real decisions) and are treated as such.

Usage:
  python3.9 prs_extract_doe.py                 # extract all 3 fuels
  python3.9 prs_extract_doe.py --fuel n_Heptane_159
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

AUTOMATION_DIR = Path(__file__).resolve().parent
PPRS_DATA_ROOT = Path(os.environ.get("MSO_PPRS_DATA_ROOT", str(AUTOMATION_DIR.parent)))
MC_DIR = PPRS_DATA_ROOT / "MECHANISM_COMPARISON"
OUT_DIR = AUTOMATION_DIR / "prs_doe_data"


def _assert_not_under_pprs_root(path: Path) -> None:
    """Refuse to write anywhere under PPRS_DATA_ROOT.

    MSO_PPRS_DATA_ROOT (default: this repo, but in real use set to
    /data2/KRUNAL/FINALIZED_STAGE_OPTIMIZATION_2026/MSO_A_FACTOR/p_PRS --
    see src/mso_studio/core/automation.py) runs its OWN independent,
    long-lived study on this machine. This script only ever needs to READ
    the validated .tex reports under MC_DIR; nothing here currently writes
    there (write_csv's out_dir defaults to the safe, in-repo OUT_DIR), but
    this guard makes that invariant explicit and enforced rather than
    implicit, so a future caller can't accidentally point a write at the
    live campaign's directory.
    """
    resolved = Path(path).resolve()
    root = PPRS_DATA_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        raise PermissionError(
            f"Refusing to write to {resolved} -- it is under PPRS_DATA_ROOT "
            f"({root}), which may be the live external p_PRS campaign's own "
            "directory (read-only, never touched by this script by design)."
        )

FUEL_REPORTS = {
    "1_3_Butadiene": MC_DIR / "1_3_Butadiene" / "report_1_3_Butadiene_factorial.tex",
    "n_Heptane_159": MC_DIR / "n-Heptane" / "report_n_Heptane_159_factorial.tex",
    "MB": MC_DIR / "MB" / "report_MB_factorial.tex",
}

# One data row of the longtable looks like (a numeric-only p_start/p_final
# cell, a WMRE cell that may carry a $^{*}$/$^{\dagger}$ footnote marker or
# be wrapped in \textbf{...} for a highlighted outlier, then a source cell):
#   1 & 0.05 & 12.1688 & dup of original \\
#   5 & 0.10 & 13.3027$^{*}$ & new \\
#   4 & 0.10 & \textbf{13.9960} & new \\
NUMERIC_CELL_RE = re.compile(r"(\d+(?:\.\d+)?)")

GENS_FIXED = 1200  # not varied across the formal 5x3 DoE


def _first_number(cell):
    m = NUMERIC_CELL_RE.search(cell)
    if not m:
        raise ValueError(f"no number found in cell {cell!r}")
    return float(m.group(1))


def parse_factorial_report(tex_path):
    """Return a list of dicts: p_start, p_final, gens, wmre, source."""
    text = Path(tex_path).read_text()
    m = re.search(r"\\begin\{longtable\}.*?\\end\{longtable\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no longtable found in {tex_path}")
    body = m.group(0)
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line.endswith(r"\\"):
            continue
        line = line[:-2].strip()  # drop trailing \\
        if line.startswith((r"\toprule", r"\midrule", r"\bottomrule",
                             r"\begin", r"\end", r"\multicolumn")):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 4:
            continue
        p_start_cell, p_final_cell, wmre_cell, source = cells
        if "p_{" in p_start_cell or "p_{" in p_final_cell:  # header row
            continue
        try:
            p_start = _first_number(p_start_cell)
            p_final = _first_number(p_final_cell)
            wmre = _first_number(wmre_cell)
        except ValueError:
            continue
        rows.append({
            "p_start": p_start,
            "p_final": p_final,
            "gens": GENS_FIXED,
            "wmre": wmre,
            "source": source.strip(),
        })
    return rows


def write_csv(fuel, rows, out_dir=OUT_DIR):
    _assert_not_under_pprs_root(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fuel}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["p_start", "p_final", "gens", "wmre", "source"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuel", choices=list(FUEL_REPORTS), default=None,
                     help="extract just this fuel (default: all 3)")
    args = ap.parse_args()

    fuels = [args.fuel] if args.fuel else list(FUEL_REPORTS)
    for fuel in fuels:
        tex_path = FUEL_REPORTS[fuel]
        if not tex_path.is_file():
            print(f"ERROR: {fuel}: report not found: {tex_path}", file=sys.stderr)
            continue
        rows = parse_factorial_report(tex_path)
        if len(rows) != 15:
            print(f"WARNING: {fuel}: expected 15 DoE rows, parsed {len(rows)} "
                  f"from {tex_path} — check ROW_RE against the table format",
                  file=sys.stderr)
        out_path = write_csv(fuel, rows)
        print(f"{fuel}: {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
