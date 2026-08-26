#!/usr/bin/env python3
"""
prs_report.py — final per-fuel LaTeX report for the PRS-based hyperparameter
refinement, in the same style as the existing report_<fuel>_factorial.tex.

Integrated into the mechanism-optimization-suite repo (2026-07-21) -- see
prs_extract_doe.py's docstring for the MSO_PPRS_DATA_ROOT override this
file also uses (MECHANISM_COMPARISON/ is external, multi-GB data).

Can be run at two points in the workflow:
  - Before the follow-up YAML has been executed: renders the PRS fit
    diagnostics, predicted optimum, and the run PLAN (not yet real results).
  - After run_tasks.py has executed <fuel>_prs_followup.yaml and its own
    'report' task has written report_<fuel>_prs_verify.csv: joins those real
    objective values in against the plan by reconstructing each point's
    expected case-folder name, and reports which points are still pending.

Note on units: the joined-in verification numbers are run_tasks.py's
generic "Objective (sum of Error_Function)" metric, NOT the bespoke
"Mean WMRE (%)" aggregation used in the original report_<fuel>_factorial.tex
tables -- that original formula is not present anywhere in checked-in code
(see prs_extract_doe.py's docstring), so this script does not claim to
reproduce it. Both are stated explicitly, side by side, wherever compared,
so nobody downstream mistakes one unit for the other.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tasks import fmt_token  # noqa: E402
from prs_fit import load_doe_csv, DOE_DIR  # noqa: E402
from prs_plan_runs import build_fuel_plan, GENS_FIXED, FUEL_LOCATIONS  # noqa: E402

# Round 4 fix (WORKBENCH_AUDIT_2026-08-10.md): the insert above only adds
# THIS file's own directory (src/automation/, needed for the sibling
# run_tasks/prs_fit/prs_plan_runs imports above) -- it never added src/
# itself, unlike graph_sa/plots.py and forward_uq/uq_plots.py's identical
# "postprocess() needs mso_io" bootstrap (`normpath(join(this_dir, ".."))`,
# i.e. src/), which both DO add. Running this script directly (`python3
# src/automation/prs_report.py ...`, exactly as its own module docstring
# and every sibling postprocessor's docstring documents) used to fail the
# `from mso_io.postprocess_base import ...` below with ModuleNotFoundError,
# silently setting PostProcessResult = None -- which then made main()'s
# `if PostProcessResult is not None: add_postprocess_args(ap)` guard skip
# registering --postprocess/--out/--journal/etc entirely, so passing them
# produced a confusing argparse "unrecognized arguments" error with zero
# indication the real cause was several layers upstream.
_src_dir_pp = str(Path(__file__).resolve().parent.parent)
if _src_dir_pp not in sys.path:
    sys.path.append(_src_dir_pp)

AUTOMATION_DIR = Path(__file__).resolve().parent
PPRS_DATA_ROOT = Path(os.environ.get("MSO_PPRS_DATA_ROOT", str(AUTOMATION_DIR.parent)))
MC_DIR = PPRS_DATA_ROOT / "MECHANISM_COMPARISON"

FUEL_REPORT_OUT = {
    "1_3_Butadiene": MC_DIR / "1_3_Butadiene" / "report_1_3_Butadiene_prs_optimum.tex",
    "n_Heptane_159": MC_DIR / "n-Heptane" / "report_n_Heptane_159_prs_optimum.tex",
    "MB": MC_DIR / "MB" / "report_MB_prs_optimum.tex",
}


def _folder_name(p_start, p_final, gens, suffix=""):
    return (f"powerMeanWMRE_{fmt_token(p_start)}_{fmt_token(p_final)}_{gens}"
            f"_GEN_PM_POLISH_{fmt_token(p_final)}{suffix}")


def _load_verify_csv(fuel):
    """report_<fuel>_prs_verify.csv, written by the follow-up YAML's own
    'report' task (run_tasks.py render logic) once cases finish. Returns
    {folder_name: {"nominal_obj":..,"stage1_obj":..,"n_conditions":..}}."""
    name = f"{fuel}_prs_verify"
    csv_path = MC_DIR / f"report_{name}.csv"
    if not csv_path.is_file():
        return {}
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["folder"]] = {
                "nominal_obj": float(row["nominal_obj"]),
                "stage1_obj": float(row["stage1_obj"]),
                "n_conditions": int(row["n_conditions"]),
            }
    return out


def _join_points(points, verify_data, suffix_map=None):
    suffix_map = suffix_map or {}
    rows = []
    for pt in points:
        folder = _folder_name(pt["p_start"], pt["p_final"], pt["gens"],
                               suffix=suffix_map.get(id(pt), ""))
        result = verify_data.get(folder)
        rows.append({**pt, "folder": folder, "result": result})
    return rows


def tex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))


def render_final_report(fuel):
    doe_points = load_doe_csv(DOE_DIR / f"{fuel}.csv")
    fit_data = json.loads((DOE_DIR / f"{fuel}_fit.json").read_text())
    optimum_path = DOE_DIR / f"{fuel}_optimum.json"
    optimum = json.loads(optimum_path.read_text()) if optimum_path.is_file() else None
    plan = build_fuel_plan(fuel)
    verify_data = _load_verify_csv(fuel)

    suffix_map = {}
    for pt in plan.get("mb_repeat", []):
        suffix_map[id(pt)] = pt.get("folder_suffix", "")
    plan_rows = _join_points(plan["points"], verify_data)
    mb_rows = _join_points(plan.get("mb_repeat", []), verify_data, suffix_map)
    gens_rows = _join_points(plan["gens_points"], verify_data)

    best_doe = min(doe_points, key=lambda p: p["wmre"])
    all_result_rows = [r for r in plan_rows + mb_rows + gens_rows if r["result"]]
    best_verified = (min(all_result_rows, key=lambda r: r["result"]["stage1_obj"])
                      if all_result_rows else None)

    esc = tex_escape
    lines = []
    a = lines.append
    a(r"\documentclass[11pt]{article}")
    a(r"\usepackage[margin=1in]{geometry}")
    a(r"\usepackage{booktabs}")
    a(r"\usepackage{longtable}")
    a(r"\usepackage{amsmath}")
    a(r"\usepackage{hyperref}")
    a(r"\title{PRS-Based Hyperparameter Optimization --- %s}" % esc(fuel))
    a(r"\author{Automated MSO\_A\_FACTOR pipeline}")
    a(r"\date{\today}")
    a(r"\begin{document}")
    a(r"\maketitle")

    a(r"\section{Methodology}")
    a(r"Starting from the completed $5\times3$ full-factorial DoE (%d points, "
      r"\texttt{report\_%s\_factorial.tex}), a 2nd-order Polynomial Response "
      r"Surface (PRS) was fit over $(p_{\text{start}}, p_{\text{final}})$, "
      r"mirroring this pipeline's own house convention for response-surface "
      r"fitting (\texttt{ResponseSurface.py}: quadratic basis, QR least "
      r"squares) and its 6\%% max-relative-error accept/reject threshold, "
      r"applied here via leave-one-out cross-validation (LOOCV) rather than "
      r"a single train/test split, since 15 points is too few to hold out a "
      r"block and still fit reliably. Where the PRS passes this gate, its "
      r"predicted optimum is verified with a small number of real "
      r"opt\_sweep+compare runs. Where it does not, a single bounded round "
      r"of local refinement around the current empirical best point is run "
      r"instead (never an open-ended iterative search). A separate, "
      r"independent 1D exploration of \texttt{pm\_anneal\_gens} is always "
      r"run at the current best $(p_{\text{start}}, p_{\text{final}})$, "
      r"since this dimension was never varied in the formal DoE (fixed at "
      r"1200 throughout) and so has no data to fit any surface on."
      % (len(doe_points), esc(fuel)))

    a(r"\section{PRS Fit}")
    a(r"\begin{itemize}")
    a(r"\item LOOCV max $|$error$|$: %.2f\%%; mean $|$error$|$: %.2f\%%; "
      r"threshold: %.1f\%%."
      % (fit_data["loocv_max_abs_pct"], fit_data["loocv_mean_abs_pct"],
         fit_data["threshold_pct"]))
    verdict = "TRUSTED" if fit_data["trusted"] else "NOT TRUSTED"
    a(r"\item Verdict: \textbf{%s}." % verdict)
    if optimum:
        a(r"\item PRS-predicted optimum (cheap, no simulation): "
          r"$p_{\text{start}}=%.4g$, $p_{\text{final}}=%.4g$, predicted "
          r"objective $=%.4f$ (grid cross-check agrees: %s)."
          % (optimum["p_start"], optimum["p_final"], optimum["predicted_wmre"],
             "yes" if optimum["grid_agrees"] else "NO"))
    a(r"\item Empirical best point in the DoE itself: "
      r"$p_{\text{start}}=%.4g$, $p_{\text{final}}=%.4g$, WMRE$=%.4f$\%%."
      % (best_doe["p_start"], best_doe["p_final"], best_doe["wmre"]))
    a(r"\end{itemize}")
    a(r"\noindent \textit{Units note:} the PRS was fit to, and its predicted "
      r"objective is in, the same ``Mean WMRE (\%)'' units as the original "
      r"factorial report. The verification/refinement/gens results below "
      r"are read from \texttt{run\_tasks.py}'s generic report task, which "
      r"reports a DIFFERENT quantity (sum of \texttt{Error\_Function} over "
      r"condition CSVs, labeled ``Objective (sum)'' below) -- the exact "
      r"aggregation used for the original ``Mean WMRE (\%)'' figures is not "
      r"recoverable from checked-in code, so this report does not claim the "
      r"two are on the same scale.")

    def render_points_table(title, rows, note=""):
        a(r"\subsection*{%s}" % esc(title))
        if note:
            a(esc(note))
        a(r"\begin{longtable}{llrrrl}")
        a(r"\toprule")
        a(r"$p_{\text{start}}$ & $p_{\text{final}}$ & gens & Objective (sum) & "
          r"Conditions & Status \\")
        a(r"\midrule")
        for r in rows:
            if r["result"]:
                obj = r"%.4f" % r["result"]["stage1_obj"]
                n = str(r["result"]["n_conditions"])
                status = "done"
            else:
                obj, n, status = "---", "---", "PENDING"
            a(r"%.4g & %.4g & %d & %s & %s & %s \\"
              % (r["p_start"], r["p_final"], r["gens"], obj, n, status))
        a(r"\bottomrule")
        a(r"\end{longtable}")

    branch_title = ("Verification Runs (PRS Trusted)" if plan["trusted"]
                     else "Local Refinement Runs (PRS Not Trusted)")
    render_points_table(
        branch_title, plan_rows,
        note=f"Follow-up YAML: _automation/generated/{fuel}_prs_followup.yaml "
             f"(branch={plan['branch']}, {len(plan_rows)} points, run via "
             f"run_tasks.py, not yet auto-launched by this report).")

    if mb_rows:
        render_points_table(
            "Outlier Confirmation Repeat",
            mb_rows,
            note="Independent repeat of the (p_start=4, p_final=0.10) point "
                 "flagged by report_MB_factorial.tex as possibly a "
                 "stochastic fluke rather than a systematic effect.")

    render_points_table(
        "Gens (pm_anneal_gens) 1D Exploration", gens_rows,
        note="No existing DoE data varies this dimension (fixed at 1200 "
             "throughout); these are the first real data points on it.")

    a(r"\section{Final Recommendation}")
    if best_verified:
        pt = best_verified
        a(r"Best verified real point so far: $p_{\text{start}}=%.4g$, "
          r"$p_{\text{final}}=%.4g$, gens$=%d$, objective (sum) $=%.4f$, "
          r"folder \texttt{%s}."
          % (pt["p_start"], pt["p_final"], pt["gens"],
             pt["result"]["stage1_obj"], esc(pt["folder"])))
        case_dir = Path(FUEL_LOCATIONS[fuel]["opt_dir"]) / pt["folder"]
        a(r"New mechanism for downstream use: "
          r"\texttt{%s/new\_mech.yaml}." % esc(str(case_dir)))
    else:
        a(r"No follow-up runs have completed yet -- this section will be "
          r"filled in once \texttt{run\_tasks.py "
          r"generated/%s\_prs\_followup.yaml --daemon} has produced results "
          r"and \texttt{prs\_report.py} is re-run. Until then, the "
          r"provisional best is the empirical DoE point: "
          r"$p_{\text{start}}=%.4g$, $p_{\text{final}}=%.4g$, "
          r"gens$=%d$, WMRE$=%.4f$\%%."
          % (esc(fuel), best_doe["p_start"], best_doe["p_final"],
             GENS_FIXED, best_doe["wmre"]))
    a(r"\end{document}")

    return "\n".join(lines) + "\n"


# =============================================================================
# postprocess() -- "process the results" flag entry point
# =============================================================================
# Same shared CLI contract as src/MUQ-SAC/VisualAid.py's postprocess() (see
# that file's docstring for the full rationale) via mso_io.postprocess_base,
# with one deliberate deviation: this pipeline is addressed by --fuel (one
# of FUEL_REPORT_OUT's three hardcoded fuels, each with its own DOE_DIR /
# MECHANISM_COMPARISON paths resolved internally), not by an arbitrary run
# directory the way every other repo's postprocess() is -- there is no
# single "run_dir" to discover under; --postprocess here takes a FUEL name,
# not a path. Everything render_final_report() already reads (the DoE CSV,
# the PRS fit JSON, the verify CSV once run_tasks.py has produced one) is
# additionally relocated into --out under the same tables/ + journal_data/
# layout as every other repo's postprocess(), so mso_project.Project.scan()
# still finds a uniform shape here.
try:
    from mso_io.postprocess_base import (
        PostProcessResult, add_postprocess_args, default_out_dir,
        formats_from_args, write_manifest, write_table,
    )
    import pandas as _pd
except ImportError:  # pragma: no cover
    PostProcessResult = None


def postprocess(fuel, out_dir=None, journal=False, csv_only=False,
                 formats=("png",), intermediate=False):
    """Render the PRS report for *fuel* and relocate its source CSVs into
    *out_dir* under the shared tables/ + journal_data/ layout. *fuel* is a
    FUEL_REPORT_OUT key (a name, not a directory -- see module note above).
    Never raises on a missing verify CSV (that's the normal "still running"
    state this pipeline itself documents in its own docstring).
    Returns a :class:`mso_io.postprocess_base.PostProcessResult`."""
    if PostProcessResult is None:
        raise ImportError("mso_io.postprocess_base not importable -- is src/ on sys.path?")
    if fuel not in FUEL_REPORT_OUT:
        raise ValueError(f"unknown fuel {fuel!r} -- choices: {list(FUEL_REPORT_OUT)}")

    out_dir = str(default_out_dir(str(AUTOMATION_DIR / fuel), out_dir))
    result = PostProcessResult(run_dir=str(AUTOMATION_DIR / fuel), out_dir=out_dir,
                                partial=bool(intermediate))
    tables_dir = Path(out_dir) / "tables"

    try:
        tex = render_final_report(fuel)
        tex_path = Path(out_dir) / f"{fuel}_prs_report.tex"
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(tex)
        result.figures.append(str(tex_path))  # .tex, not an image -- see save_figure's contract note
    except Exception as exc:
        result.skipped.append(f"render_final_report({fuel}): {type(exc).__name__}: {exc}")

    doe_csv = DOE_DIR / f"{fuel}.csv"
    if doe_csv.is_file():
        result.tables.append(write_table(_pd.read_csv(doe_csv), f"{fuel}_doe", tables_dir))
    else:
        result.skipped.append(f"DoE table: no {doe_csv} found")

    verify_csv = MC_DIR / f"report_{fuel}_prs_verify.csv"
    if verify_csv.is_file():
        result.tables.append(write_table(_pd.read_csv(verify_csv), f"{fuel}_verify", tables_dir))
    else:
        result.skipped.append(
            f"verify table: no {verify_csv} found -- follow-up runs not yet complete"
            if intermediate else f"verify table: no {verify_csv} found")

    if journal and result.tables:
        journal_dir = Path(out_dir) / "journal_data"
        for t in result.tables:
            result.journal_tables.append(
                write_table(_pd.read_csv(t), "journal_" + Path(t).stem, journal_dir))

    write_manifest(result, out_dir)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fuel", choices=list(FUEL_REPORT_OUT))
    if PostProcessResult is not None:
        add_postprocess_args(ap)
    args = ap.parse_args()

    if getattr(args, "postprocess", None):
        # Unlike --fuel, the shared --postprocess flag (add_postprocess_args)
        # has no `choices=` restriction, so an ordinary typo'd fuel key
        # reaches postprocess()'s own `if fuel not in FUEL_REPORT_OUT: raise
        # ValueError` uncaught -- a raw traceback for a predictable, already-
        # validated-by-hand user mistake. Print the same clean message
        # postprocess()'s own docstring/error text already gives instead.
        try:
            result = postprocess(
                args.postprocess, out_dir=args.out, journal=args.journal,
                csv_only=args.csv_only, formats=formats_from_args(args),
                intermediate=args.intermediate,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"[prs_report.postprocess] {len(result.figures)} report(s), "
              f"{len(result.tables)} table(s), {len(result.journal_tables)} journal table(s) "
              f"-> {result.out_dir}")
        for note in result.skipped:
            print(f"  skipped: {note}")
        return

    if not args.fuel:
        ap.error("--fuel is required (or use --postprocess FUEL)")
    tex = render_final_report(args.fuel)
    out_path = FUEL_REPORT_OUT[args.fuel]
    out_path.write_text(tex)
    print(f"{args.fuel}: wrote {out_path}")


if __name__ == "__main__":
    # sys.exit(...), not a bare call -- main()'s ValueError handler above
    # returns 1 on a bad --postprocess fuel key specifically so the process
    # exit code reflects the failure (a caller scripting around this CLI
    # needs $? to be nonzero, not just the printed "ERROR:" text); a bare
    # `main()` call here would have silently discarded that return value
    # and always exited 0 regardless.
    sys.exit(main())
