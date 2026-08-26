"""
rmip.postprocess
==================
"process the results" flag entry point for RMIP -- new file, unlike the
other repos' ``postprocess()`` (rmip has no existing standalone plotting
module the way MUQ-SAC/VisualAid.py or forward_uq/uq_plots.py do; its report
generation lives in :mod:`rmip.analysis.mechanism_report` instead, driven
today only via ``rmip_cli.py``'s ``report`` subcommand). Same shared CLI
contract as every other repo's postprocess() (see
``src/MUQ-SAC/VisualAid.py``'s module docstring for the full rationale) via
:mod:`mso_io.postprocess_base`.

Reuses :class:`rmip.analysis.mechanism_report.MechanismReport` for the
quality score / mechanism diff (no reimplementation of that scoring logic)
and reads the ``mechanisms``/``species``/``reactions`` tables directly --
these are exactly the per-reaction Arrhenius-parameter and per-species
thermo tables a paper's mechanism appendix needs, and they always exist the
moment a mechanism has been parsed into ``project.db``, well before any
downstream optimization/reduction stage runs -- so this is safe to call
against a run directory the pipeline hasn't finished with yet
(``--intermediate``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

_this_dir_pp = os.path.dirname(os.path.abspath(__file__))
_src_dir_pp = os.path.normpath(os.path.join(_this_dir_pp, ".."))
if _src_dir_pp not in sys.path:
    sys.path.append(_src_dir_pp)
try:
    from mso_io.postprocess_base import (
        PostProcessResult, add_postprocess_args, default_out_dir, discover,
        formats_from_args, write_manifest, write_table,
    )
except ImportError:  # pragma: no cover
    PostProcessResult = None

from rmip.database.schema import get_connection

_RMIP_PATTERNS = {
    "db": "**/*.db",
}

_REACTIONS_SQL = """
SELECT reaction_label, formula_string, kinetics_type, A_si, n, Ea_si,
       A_orig, A_units, Ea_orig, Ea_units, reversible, family, duplicate
FROM reactions WHERE mechanism_id = ?
"""
_SPECIES_SQL = """
SELECT name, formula, mw_kg_mol, canonical_smiles, inchi_key,
       thermo_type, tmin_K, tmid_K, tmax_K, h298_kJ_mol, s298_J_molK
FROM species WHERE mechanism_id = ?
"""


def _pick_db(found_dbs):
    """Prefer a file literally named project.db (this repo's own convention,
    see the gui_stress test fixtures) if there's more than one *.db under
    run_dir; otherwise the first match."""
    for p in found_dbs:
        if p.name == "project.db":
            return p
    return found_dbs[0]


def postprocess(run_dir, out_dir=None, journal=False, csv_only=False,
                 formats=("png",), intermediate=False):
    """Export mechanism/species/reaction tables and a quality-score report
    from an existing (or still-running) RMIP ``project.db``. Never raises on
    missing files. Returns a :class:`mso_io.postprocess_base.PostProcessResult`."""
    if PostProcessResult is None:
        raise ImportError("mso_io.postprocess_base not importable -- is src/ on sys.path?")

    run_dir = os.path.abspath(run_dir)
    out_dir = str(default_out_dir(run_dir, out_dir))
    found = discover(run_dir, _RMIP_PATTERNS)
    result = PostProcessResult(run_dir=run_dir, out_dir=out_dir, partial=bool(intermediate))
    tables_dir = os.path.join(out_dir, "tables")
    figures_dir = os.path.join(out_dir, "figures")

    if not found["db"]:
        result.skipped.append(
            f"no *.db found under {run_dir}"
            + (" (mechanism not yet parsed into a database)" if intermediate else ""))
        write_manifest(result, out_dir)
        return result

    db_path = str(_pick_db(found["db"]))
    conn = get_connection(db_path)
    mechanisms = [dict(r) for r in conn.execute(
        "SELECT id, name, format, filepath, n_species, n_reactions, loaded_at FROM mechanisms"
    ).fetchall()]

    if not mechanisms:
        result.skipped.append(f"{db_path}: no mechanisms row yet (parse in progress)")
        write_manifest(result, out_dir)
        return result

    result.tables.append(write_table(pd.DataFrame(mechanisms), "mechanisms", tables_dir))

    for mech in mechanisms:
        safe_name = str(mech["name"]).replace("/", "_").replace(" ", "_")
        rxn_df = pd.read_sql_query(_REACTIONS_SQL, conn, params=(mech["id"],))
        sp_df = pd.read_sql_query(_SPECIES_SQL, conn, params=(mech["id"],))
        result.tables.append(write_table(rxn_df, f"{safe_name}_reactions", tables_dir))
        result.tables.append(write_table(sp_df, f"{safe_name}_species", tables_dir))

    # -- Quality score (+ diff, if exactly two mechanisms) --
    mech_ids = [m["id"] for m in mechanisms[:2]]
    try:
        from rmip.analysis.mechanism_report import MechanismReport
        rep = MechanismReport(db_path)
        report_result = rep.report(mech_ids)
        quality_rows = [q.to_dict() for q in report_result["quality"].values()]
        result.tables.append(write_table(pd.DataFrame(quality_rows), "quality_score", tables_dir))
        if report_result.get("diff") is not None:
            diff_dict = report_result["diff"].to_dict()
            scalar = {k: v for k, v in diff_dict.items()
                      if not isinstance(v, (list, dict))}
            result.tables.append(write_table(pd.DataFrame([scalar]), "mechanism_diff_summary", tables_dir))
            log_a = diff_dict.get("shared_reaction_log_a_ratio") or {}
            if log_a:
                log_a_df = pd.DataFrame(
                    [{"canonical_key": k, "log10_A_ratio": v} for k, v in log_a.items()])
                result.tables.append(write_table(log_a_df, "mechanism_diff_log_a_ratio", tables_dir))

        if not csv_only:
            try:
                os.makedirs(figures_dir, exist_ok=True)
                html_path = os.path.join(figures_dir, "mechanism_report.html")
                rep.export_html(report_result, html_path)
                result.figures.append(html_path)
            except ImportError as exc:
                result.skipped.append(f"HTML report: {exc} (pip install plotly)")
    except Exception as exc:
        result.skipped.append(f"quality score / diff: {type(exc).__name__}: {exc}")

    if journal:
        # Every table gathered above (mechanism summary, per-mechanism
        # reactions/species, quality score, diff) is exactly what a paper's
        # mechanism appendix / methods section needs -- copy all of it.
        journal_dir = os.path.join(out_dir, "journal_data")
        for t in result.tables:
            df = pd.read_csv(t)
            result.journal_tables.append(
                write_table(df, "journal_" + os.path.splitext(os.path.basename(t))[0], journal_dir))

    write_manifest(result, out_dir)
    return result


def _main():
    import argparse
    parser = argparse.ArgumentParser(
        description="rmip.postprocess: export mechanism/species/reaction "
                     "tables and a quality-score report from an existing "
                     "RMIP project.db.")
    add_postprocess_args(parser)
    args = parser.parse_args()
    if not args.postprocess:
        parser.error("nothing to do -- pass --postprocess RUN_DIR")
    result = postprocess(
        args.postprocess, out_dir=args.out, journal=args.journal,
        csv_only=args.csv_only, formats=formats_from_args(args),
        intermediate=args.intermediate,
    )
    print(f"[rmip.postprocess] {len(result.figures)} figure(s), "
          f"{len(result.tables)} table(s), {len(result.journal_tables)} journal table(s) "
          f"-> {result.out_dir}")
    for note in result.skipped:
        print(f"  skipped: {note}")


if __name__ == "__main__":
    _main()
