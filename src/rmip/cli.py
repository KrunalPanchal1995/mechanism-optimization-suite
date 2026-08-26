"""
rmip/cli.py
===========
Command-line interface for the Reaction Mechanism Intelligence Platform.

Phase 1 commands:  parse, graph, list, summary
Phase 2 commands:  compare, visualize
Phase 3 commands:  sparsity, sensitivity, exp-load, bench, compare-graph

Usage examples
--------------
# Parse a mechanism
python -m rmip.cli parse mechanism.inp \\
    --glossary species_smiles.csv \\
    --thermo   thermo.dat \\
    --db       project.db \\
    --name     "Curran n-heptane 2000"

# Build graph
python -m rmip.cli graph 1 --db project.db --stats

# Compare two mechanisms
python -m rmip.cli compare --mechs 1 2 --db project.db --out comparison/

# Run sparsity analysis
python -m rmip.cli sparsity 1 --db project.db

# Load experimental data
python -m rmip.cli exp-load curran_idt.csv \\
    --db project.db --sim-type IDT --fuel NC7H16 \\
    --name "Curran1998" --P 13.5 --phi 1.0

# Benchmark mechanisms against experimental dataset
python -m rmip.cli bench --mechs 1 2 3 --dataset 1 \\
    --fuel "NC7H16:1.0" --db project.db --out bench_out/

# Full graph-level comparison and union spec
python -m rmip.cli compare-graph --mechs 1 2 3 --db project.db --out graph_cmp/
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

from rmip.config          import RMIPConfig, setup_logging
from rmip.database.schema import init_db, get_mechanisms, get_mechanism_summary
from rmip.database.schema_phase3 import upgrade_db, list_exp_datasets

log = logging.getLogger("rmip.cli")


# ===========================================================================
# Main dispatcher
# ===========================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    setup_logging(args.log_level if hasattr(args, "log_level") else "INFO")

    # M16: never let a handler exception escape as a raw traceback / ambiguous
    # exit status. Report it cleanly and exit non-zero (traceback at DEBUG).
    try:
        rc = args.func(args)
    except KeyboardInterrupt:
        log.error("Interrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        cmd = getattr(args, "func", None)
        cmd_name = getattr(cmd, "__name__", "<command>").replace("_cmd_", "")
        log.error("Command '%s' failed: %s", cmd_name, exc)
        log.debug("Traceback:", exc_info=True)
        return 1
    # A handler returning None is treated as success (exit 0).
    return 0 if rc is None else rc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "python -m rmip.cli",
        description = "RMIP — Reaction Mechanism Intelligence Platform",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = p.add_subparsers(title="commands", metavar="<command>")

    # ---- Phase 1 ----
    _add_parse_cmd(sub)
    _add_graph_cmd(sub)
    _add_list_cmd(sub)
    _add_summary_cmd(sub)

    # ---- Phase 2 ----
    _add_compare_cmd(sub)
    _add_visualize_cmd(sub)
    _add_visualize3d_cmd(sub)

    # ---- Phase 3 ----
    _add_sparsity_cmd(sub)
    _add_sensitivity_cmd(sub)
    _add_exp_load_cmd(sub)
    _add_bench_cmd(sub)
    _add_compare_graph_cmd(sub)
    _add_resolve_glossary_cmd(sub)

    # ---- Phase 3 (new) ----
    _add_fuse_cmd(sub)
    _add_optimize_cmd(sub)
    _add_rank_cmd(sub)
    _add_complete_cmd(sub)
    _add_export_cmd(sub)

    # ---- Phase 4 ----
    _add_unify_plan_cmd(sub)
    _add_unify_build_cmd(sub)
    _add_predict_missing_cmd(sub)

    # ---- Phase 5A ----
    _add_sa_run_cmd(sub)

    # ---- Phase 5B ----
    _add_reduce_cmd(sub)
    _add_reduce_viz_cmd(sub)

    # ---- Zero-RK GSA (ZERORK_GSA_PLAN.md Phase 3) ----
    _add_gsa_run_cmd(sub)

    # ---- Phase 6 (Future.md features 1-7) ----
    _add_reverse_scan_cmd(sub)
    _add_report_cmd(sub)
    _add_plog_surface_cmd(sub)
    _add_ntc_cmd(sub)
    _add_rop_cmd(sub)
    _add_transport_cmd(sub)
    _add_thermo_fit_cmd(sub)
    _add_surrogate_cmd(sub)
    _add_calibrate_cmd(sub)
    _add_b2bdc_cmd(sub)
    _add_csp_cmd(sub)
    _add_lump_cmd(sub)
    _add_error_map_cmd(sub)
    _add_tdac_cmd(sub)
    _add_design_experiments_cmd(sub)
    _add_pareto_cmd(sub)
    _add_family_scan_cmd(sub)
    _add_generate_targeted_cmd(sub)
    _add_import_respecth_cmd(sub)
    _add_build_target_cmd(sub)
    _add_flux_cmd(sub)
    _add_jsr_cmd(sub)
    _add_pfr_cmd(sub)

    # ---- Mechanism health-check (MECHANISM_CHECK_RESEARCH.md) ----
    _add_check_cmd(sub)
    _add_migrate_cmd(sub)
    _add_preflight_cmd(sub)
    _add_idt_sens_cmd(sub)
    _add_thermo_check_cmd(sub)
    _add_import_cmd(sub)
    _add_react_solid_cmd(sub)
    _add_surface_crosscheck_cmd(sub)
    _add_rmg_explain_cmd(sub)
    _add_rmg_lint_cmd(sub)
    _add_rmg_diff_cmd(sub)
    _add_rmg_supervise_cmd(sub)

    # ---- W8b/W9 — regression harness + onboarding ----
    _add_regression_cmd(sub)
    _add_doctor_cmd(sub)
    _add_how_do_i_cmd(sub)

    # ---- Utilities ----
    _add_reactions_cmd(sub)
    _add_reaction_matrix_cmd(sub)

    return p


def _ensure_phase6_schema(db_path: str) -> None:
    """Bootstrap Phase-3 + Phase-6 schema on a DB (idempotent)."""
    from rmip.database.migrations import run_phase3_migrations, run_phase6_migrations

    conn = init_db(db_path)
    upgrade_db(conn)
    run_phase3_migrations(conn)
    run_phase6_migrations(conn)


# ===========================================================================
# Phase 1 commands
# ===========================================================================

def _add_parse_cmd(sub) -> None:
    p = sub.add_parser("parse", help="Parse and store a mechanism file")
    p.add_argument("mech_path",               help="CHEMKIN .inp or Cantera .yaml file")
    p.add_argument("--glossary", "-g",        help="Species SMILES glossary CSV/YAML")
    p.add_argument("--thermo",   "-t",        help="Separate thermo data file (CHEMKIN)")
    p.add_argument("--dict",     "-d",        help="RMG species dictionary .txt (optional, CHEMKIN only)")
    p.add_argument("--db",       required=True, help="SQLite database path")
    p.add_argument("--name",     "-n", default="", help="Human-readable mechanism name")
    p.add_argument("--description",           help="Optional description")
    p.add_argument("--format",   choices=["chemkin","cantera_yaml","cti"],
                   help="Force format (auto-detected if omitted)")
    p.add_argument("--check-duplicates", action="store_true",
                   help="Flag duplicate reactions during parsing")
    p.add_argument("--variants", metavar="FILE",
                   help="Phase-5D species-variant override CSV "
                        "(columns: variant_name, canonical_name, reason)")
    p.add_argument("--allow-pseudo-species", action="store_true",
                   help="Drop undeclared reaction participants (e.g. "
                        "chemiluminescence bookkeeping species like HV) "
                        "with a warning instead of keeping them (mirrors "
                        "ck2yaml --permissive; CHEMKIN parser only, M1)")
    p.set_defaults(func=_cmd_parse)


def _cmd_parse(args) -> int:
    from rmip.core.parser import MechanismParser
    cfg     = RMIPConfig(db_path=args.db)
    parser  = MechanismParser(cfg)
    mech_id = parser.parse(
        mech_path        = args.mech_path,
        glossary_path    = args.glossary,
        thermo_path      = args.thermo,
        dict_path        = args.dict,
        check_duplicates = args.check_duplicates,
        name             = args.name or Path(args.mech_path).stem,
        description      = args.description,
        format           = args.format,
        variants_path    = getattr(args, "variants", None),
        allow_pseudo_species = getattr(args, "allow_pseudo_species", False),
    )
    summary = parser.get_mechanism_summary(mech_id)
    print(f"\n✓ Mechanism stored as id={mech_id}")
    print(f"  Name:         {summary['name']}")
    print(f"  Species:      {summary['n_species']}")
    print(f"  Reactions:    {summary['n_reactions']}")
    print(f"  Glossary hit: {summary['glossary_matched_fraction']:.1%}")
    return 0


# ---- graph ----

def _add_graph_cmd(sub) -> None:
    p = sub.add_parser("graph", help="Build and export the reaction network graph")
    p.add_argument("mech_id", type=int,       help="Mechanism DB id")
    p.add_argument("--db",    required=True,   help="SQLite database path")
    p.add_argument("--output", "-o",           help="Output .graphml path")
    p.add_argument("--json",   "-j",           help="Output .json path")
    p.add_argument("--stats",  action="store_true", help="Print graph statistics")
    p.add_argument("--no-reverse", action="store_true",
                   help="Do not add reverse edges for reversible reactions")
    p.set_defaults(func=_cmd_graph)


def _cmd_graph(args) -> int:
    from rmip.core.graph_builder import GraphBuilder
    gb = GraphBuilder(args.db)
    G  = gb.build_graph(args.mech_id,
                        include_reverse_edges=not args.no_reverse)
    if args.stats:
        stats = gb.get_stats(G)
        print("\nGraph statistics:")
        for k, v in stats.items():
            print(f"  {k:<25} {v}")
    if args.output:
        gb.export_graphml(G, args.output)
        print(f"✓ GraphML → {args.output}")
    if args.json:
        gb.export_json(G, args.json)
        print(f"✓ JSON    → {args.json}")
    return 0


# ---- list ----

def _add_list_cmd(sub) -> None:
    p = sub.add_parser("list", help="List all mechanisms in the database")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.set_defaults(func=_cmd_list)


def _cmd_list(args) -> int:
    conn  = init_db(args.db)
    mechs = get_mechanisms(conn)
    if not mechs:
        print("No mechanisms found in database.")
        return 0
    print(f"\n{'ID':>4}  {'Name':<35}  {'Format':<14}  {'Species':>7}  {'Reactions':>9}  Loaded at")
    print("-" * 90)
    for m in mechs:
        print(f"{m['id']:>4}  {(m['name'] or ''):<35}  {(m['format'] or ''):<14}  "
              f"{(m['n_species'] or 0):>7}  {(m['n_reactions'] or 0):>9}  {m['loaded_at']}")
    return 0


# ---- summary ----

def _add_summary_cmd(sub) -> None:
    p = sub.add_parser("summary", help="Print detailed summary for a mechanism")
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db",    required=True, help="SQLite database path")
    p.add_argument("--json",  action="store_true", help="Output as JSON")
    p.set_defaults(func=_cmd_summary)


def _cmd_summary(args) -> int:
    conn    = init_db(args.db)
    summary = get_mechanism_summary(conn, args.mech_id)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"\nMechanism {args.mech_id}: {summary['name']}")
        print(f"  Format:           {summary['format']}")
        print(f"  File:             {summary['filepath']}")
        print(f"  Species:          {summary['n_species']}")
        print(f"  Reactions:        {summary['n_reactions']}")
        print(f"  Glossary hit:     {summary['glossary_matched_fraction']:.1%}")
        print(f"  Loaded at:        {summary['loaded_at']}")
        print(f"  Kinetics types:")
        for kt, cnt in summary["kinetics_types"].items():
            print(f"    {(kt or 'unknown'):<20} {cnt}")
    return 0


# ===========================================================================
# Mechanism health-check  (MECHANISM_CHECK_RESEARCH.md — LLNL MechChecker /
# CloudFlame / ChemCheck-style diagnostics, plus the Cantera-Users failure
# taxonomy). Works directly on mechanism files; no DB required. Multi-mechanism.
# ===========================================================================

def _add_check_cmd(sub) -> None:
    from rmip.analysis.mechanism_check import ALL_CHECKS
    p = sub.add_parser(
        "check",
        help="Health-check one or more mechanisms (rate/collision/thermo/"
             "balance/duplicate/pdep/completeness/unit diagnostics)")
    p.add_argument("mechanisms", nargs="+",
                   help="One or more mechanism files (.yaml or CHEMKIN .inp/.mech)")
    p.add_argument("--thermo", help="CHEMKIN thermo file (if separate)")
    p.add_argument("--transport", help="CHEMKIN transport file (if separate)")
    p.add_argument("--checks", nargs="+", metavar="NAME",
                   choices=list(ALL_CHECKS),
                   help=f"Subset of checks to run (default: all). "
                        f"Available: {', '.join(ALL_CHECKS)}")
    p.add_argument("--T-grid", nargs="+", type=float, metavar="K",
                   help="Temperatures (K) for the rate/collision sweep")
    p.add_argument("--P-grid", nargs="+", type=float, metavar="ATM",
                   help="Pressures (atm) for the rate/collision sweep")
    p.add_argument("--target-T-high", type=float, default=2500.0,
                   help="Upper simulation temperature for thermo T-range checks")
    p.add_argument("--collision-exceedance", type=float, default=10.0,
                   help="k/collision-limit ratio above which to flag (default 10)")
    p.add_argument("--fix", choices=["none", "auto", "all"], default="none",
                   help="none: report only; auto: apply the obvious (auto) fixes; "
                        "all: apply auto fixes AND every 'recommended' choice fix")
    p.add_argument("--fixed-out", metavar="DIR",
                   help="Directory to write corrected YAML(s) when --fix is set "
                        "(default: alongside each input as <name>.fixed.yaml)")
    p.add_argument("--json", metavar="PATH",
                   help="Write the structured report(s) as JSON")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colour in the text report")
    p.set_defaults(func=_cmd_check)


def _cmd_check(args) -> int:
    import json as _json
    from rmip.analysis.mechanism_check import MechanismChecker
    from rmip.analysis import mechanism_fixer as _fixer

    color = (not args.no_color) and sys.stdout.isatty()
    all_reports = []
    worst_rc = 0

    for mech in args.mechanisms:
        checker = MechanismChecker(
            mech, thermo=args.thermo, transport=args.transport,
            T_grid_K=args.T_grid, P_atm_grid=args.P_grid,
            collision_exceedance=args.collision_exceedance,
            target_T_high=args.target_T_high,
        )
        report = checker.run(checks=args.checks)
        print(report.to_text(color=color))

        # ---- optional repair ----
        if args.fix != "none":
            worst_rc = max(worst_rc, _run_check_fixes(args, checker, report, _fixer))

        all_reports.append(report.to_dict())
        checker.cleanup()
        if not report.ok:
            worst_rc = max(worst_rc, 1)

    if args.json:
        payload = all_reports[0] if len(all_reports) == 1 else all_reports
        with open(args.json, "w") as fh:
            _json.dump(payload, fh, indent=2, default=str)
        print(f"\n✓ JSON report → {args.json}")

    return worst_rc


def _run_check_fixes(args, checker, report, _fixer) -> int:
    """Apply --fix repairs for one mechanism; returns 0 on success, 1 on error."""
    import os
    from rmip.analysis.mechanism_check import MechanismChecker
    if checker.yaml_path is None:
        print("  (cannot fix — no YAML representation was produced)")
        return 1

    fixes = _fixer.collect_auto_fixes(report)
    if args.fix == "all":
        seen = {(f["option_id"], str(f.get("params"))) for f in fixes}
        for f in report.findings:
            if f.auto_fix is not None:
                continue
            rec = next((o for o in f.fix_options if o.recommended), None)
            if rec is not None and (rec.id, str(rec.params)) not in seen:
                fixes.append({"option_id": rec.id, "params": rec.params,
                              "location": f.location})

    if not fixes:
        print("  (no auto-applicable fixes)")
        return 0

    fixer = _fixer.MechanismFixer(checker.yaml_path)
    results = fixer.apply(fixes)

    if args.fixed_out:
        os.makedirs(args.fixed_out, exist_ok=True)
        base = os.path.splitext(os.path.basename(checker.path))[0]
        out = os.path.join(args.fixed_out, base + ".fixed.yaml")
    else:
        base = os.path.splitext(checker.path)[0]
        out = base + ".fixed.yaml"
    fixer.write(out)

    print(f"\n  Applied {sum(1 for r in results if r.status=='applied')} fix(es):")
    for r in results:
        icon = {"applied": "✓", "skipped": "–", "failed": "✗"}.get(r.status, "?")
        print(f"    {icon} [{r.status}] {r.option_id}: {r.message}")
    print(f"  Corrected mechanism → {out}")

    # re-check to prove the repair
    try:
        recheck = MechanismChecker(
            out, T_grid_K=args.T_grid, P_atm_grid=args.P_grid,
            collision_exceedance=args.collision_exceedance).run(checks=args.checks)
        print(f"  Re-check: loaded={recheck.loaded}, "
              f"{recheck.n_errors} error(s), {recheck.n_warnings} warning(s) "
              f"(was {report.n_errors} error(s), {report.n_warnings} warning(s)).")
    except Exception as exc:                               # noqa: BLE001
        print(f"  Re-check failed: {exc}")
        return 1
    return 0


# ===========================================================================
# W1 — Cantera YAML version-migration linter (Cantera google groups/W1…)
# Detects deprecated CTI→YAML keywords (P1: `stoichiometric`→`fixed-stoichiometry`,
# reaction `type: PLOG`→`pressure-dependent-Arrhenius`, etc.) and migrates them.
# ===========================================================================

def _add_migrate_cmd(sub) -> None:
    p = sub.add_parser(
        "migrate",
        help="Migrate deprecated CTI→YAML keywords in a mechanism to the "
             "current Cantera spelling (P1/P29)")
    p.add_argument("mechanism", help="Cantera YAML mechanism file")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--out", metavar="PATH",
                     help="Write the migrated YAML here (default: "
                          "<name>.migrated.yaml)")
    grp.add_argument("--in-place", action="store_true",
                     help="Overwrite the input file (a .bak copy is kept)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report the migrations but write nothing")
    p.add_argument("--json", metavar="PATH", help="Write the change list as JSON")
    p.set_defaults(func=_cmd_migrate)


def _cmd_migrate(args) -> int:
    import json as _json
    import os
    import shutil as _shutil
    from rmip.analysis.yaml_validator import YamlValidator
    from rmip.analysis.mechanism_check import MechanismChecker
    from rmip.analysis import mechanism_fixer as _fixer

    ext = os.path.splitext(args.mechanism)[1].lower()
    if ext not in (".yaml", ".yml"):
        print("migrate operates on Cantera YAML files; convert CHEMKIN first "
              "(rmip preflight / the converter GUI).")
        return 2

    validator = YamlValidator(path=args.mechanism)
    findings = validator.validate()
    deprecations = [f for f in findings if f.category == "deprecated-keyword"]

    print(f"Mechanism: {args.mechanism}")
    if not deprecations:
        print("  No deprecated keywords found. ✓")
        others = [f for f in findings if f.category != "deprecated-keyword"]
        for f in others:
            print(f"  • [{f.severity}] {f.message}  (at {f.location})")
        return 0

    print(f"  {len(deprecations)} deprecated keyword(s):")
    fixes = []
    for f in deprecations:
        opt = f.auto_fix
        print(f"    ✗ {f.message}\n        at: {f.location}")
        if opt is not None:
            fixes.append({"option_id": opt.id, "params": opt.params,
                          "location": f.location})

    if args.dry_run:
        print("\n  --dry-run: no file written.")
        return 1

    fixer = _fixer.MechanismFixer(args.mechanism)
    results = fixer.apply(fixes)

    if args.in_place:
        out = args.mechanism
        _shutil.copyfile(args.mechanism, args.mechanism + ".bak")
    else:
        out = args.out or (os.path.splitext(args.mechanism)[0] + ".migrated.yaml")
    fixer.write(out)

    print(f"\n  Applied {sum(1 for r in results if r.status=='applied')} migration(s):")
    for r in results:
        icon = {"applied": "✓", "skipped": "–", "failed": "✗"}.get(r.status, "?")
        print(f"    {icon} {r.message}")
    print(f"  Migrated mechanism → {out}"
          + ("  (original kept as .bak)" if args.in_place else ""))

    # prove it now loads
    recheck = MechanismChecker(out).run(checks=["schema_deprecation"])
    remaining = [f for f in recheck.findings if f.category == "deprecated-keyword"]
    print(f"  Re-check: loaded={recheck.loaded}, "
          f"{len(remaining)} deprecated keyword(s) remaining.")

    if args.json:
        with open(args.json, "w") as fh:
            _json.dump([r.to_dict() for r in results], fh, indent=2)
        print(f"  ✓ change list → {args.json}")
    return 0 if recheck.loaded else 1


# ===========================================================================
# W2 — CHEMKIN structural preflight linter (Cantera google groups/W2…)
# Line-numbered diagnostics BEFORE ck2yaml: block structure, species case
# consistency (P35), Fortran-exponent/tab/CRLF hazards (P2), NONCON/BULK (P3/P13),
# Ea-unit declaration. --fix writes a cleaned .inp that converts.
# ===========================================================================

def _add_preflight_cmd(sub) -> None:
    p = sub.add_parser(
        "preflight",
        help="Line-numbered CHEMKIN structural preflight before ck2yaml "
             "(P2/P3/P13/P18/P35)")
    p.add_argument("mechanism", help="CHEMKIN .inp/.mech file")
    p.add_argument("--thermo", help="Separate CHEMKIN thermo file")
    p.add_argument("--transport", help="Separate CHEMKIN transport file")
    p.add_argument("--fix", action="store_true",
                   help="Write a cleaned .inp (CRLF/tabs/Fortran exponents)")
    p.add_argument("--fix-case", action="store_true",
                   help="With --fix, also rewrite species-case mismatches in "
                        "reaction equations to the SPECIES-block spelling")
    p.add_argument("--fixed-out", metavar="PATH",
                   help="Where to write the cleaned .inp (default: "
                        "<name>.clean.inp)")
    p.add_argument("--json", metavar="PATH", help="Write the report as JSON")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colour")
    p.set_defaults(func=_cmd_preflight)


def _cmd_preflight(args) -> int:
    import json as _json
    import os
    from rmip.io.chemkin_preflight import ChemkinPreflight

    pf = ChemkinPreflight(args.mechanism, thermo=args.thermo,
                          transport=args.transport)
    report = pf.run()
    color = (not args.no_color) and sys.stdout.isatty()
    print(report.to_text(color=color))

    if args.json:
        with open(args.json, "w") as fh:
            _json.dump(report.to_dict(), fh, indent=2, default=str)
        print(f"✓ JSON report → {args.json}")

    if args.fix:
        out = args.fixed_out or (os.path.splitext(args.mechanism)[0] + ".clean.inp")
        notes = pf.write_clean(out, normalize_case=args.fix_case)
        print(f"\n  Cleaned CHEMKIN → {out}")
        for n in notes:
            print(f"    ✓ {n}")
        if not notes:
            print("    (nothing to normalise)")

    return 0 if report.ok else 1


# ===========================================================================
# W6 — first-class ignition-delay sensitivity (Cantera google groups/W6…, P20)
# ===========================================================================

def _add_idt_sens_cmd(sub) -> None:
    p = sub.add_parser(
        "idt-sens",
        help="Ignition-delay sensitivity d ln(τ)/d ln(k) — brute-force or "
             "CVODES, log-normalised, selectable IDT definition (P20)")
    p.add_argument("mechanism", help="Cantera .yaml (or CHEMKIN — convert first)")
    p.add_argument("--fuel", required=True, help="Fuel species (e.g. H2, CH4)")
    p.add_argument("--oxidizer", default="O2:1.0, N2:3.76",
                   help="Oxidizer spec (default air 'O2:1.0, N2:3.76')")
    p.add_argument("--T", type=float, required=True, help="Initial T [K]")
    p.add_argument("--P", type=float, required=True, help="Pressure [atm]")
    p.add_argument("--phi", type=float, required=True, help="Equivalence ratio")
    p.add_argument("--reactor", choices=["constV", "constP"], default="constV",
                   help="constV (shock-tube IDT, default) or constP")
    p.add_argument("--method", choices=["brute", "cvodes"], default="brute",
                   help="brute (gold-standard) or cvodes (fast)")
    p.add_argument("--idt-def", choices=["dTdt", "threshold", "species"],
                   default="dTdt", help="Ignition-delay definition")
    p.add_argument("--species", help="Species for --idt-def species (e.g. OH)")
    p.add_argument("--threshold-dT", type=float, default=400.0,
                   help="ΔT for the threshold definition / cvodes crossing")
    p.add_argument("--norm", choices=["log", "raw"], default="log",
                   help="log: d ln τ/d ln k (default); raw: dτ/d ln k [s]")
    p.add_argument("--diff", choices=["central", "forward"], default="central",
                   help="Finite-difference scheme (brute only)")
    p.add_argument("--delta", type=float, default=0.05,
                   help="Multiplier perturbation fraction (brute only)")
    p.add_argument("--top", type=int, default=20, help="Show top-N reactions")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel processes for brute-force")
    p.add_argument("--json", metavar="PATH", help="Write full result as JSON")
    p.set_defaults(func=_cmd_idt_sens)


def _cmd_idt_sens(args) -> int:
    from rmip.simulation.idt_sensitivity import IdtSensitivityAnalyzer

    analyzer = IdtSensitivityAnalyzer(
        args.mechanism, fuel=args.fuel, oxidizer=args.oxidizer,
        T_K=args.T, P_atm=args.P, phi=args.phi, reactor=args.reactor,
        idt_definition=args.idt_def, threshold_dT=args.threshold_dT,
        species=args.species)

    if args.idt_def == "species" and not args.species:
        print("--idt-def species requires --species NAME")
        return 2

    try:
        if args.method == "brute":
            result = analyzer.run_brute(delta=args.delta, diff=args.diff,
                                        norm=args.norm, n_workers=args.workers)
        else:
            result = analyzer.run_cvodes(norm=args.norm)
    except RuntimeError as exc:
        print(f"IDT sensitivity failed: {exc}")
        return 1

    print(result.render_text(n=args.top))
    if args.json:
        result.save(args.json)
        print(f"\n✓ JSON → {args.json}")
    return 0


# ===========================================================================
# W4 — thermo-data provenance cross-check (Cantera google groups/W4…, P10/P27)
# ===========================================================================

def _add_thermo_check_cmd(sub) -> None:
    p = sub.add_parser(
        "thermo-check",
        help="Cross-check species thermo (ΔHf/S/Cp at 298 K) against reference "
             "data (built-in ATcT/CODATA subset or a Burcat-style file) — P10/P27")
    p.add_argument("mechanism", help="Cantera .yaml mechanism file")
    p.add_argument("--refs", metavar="THERM.DAT",
                   help="User-supplied CHEMKIN THERMO reference file "
                        "(e.g. Burcat) — overrides the built-in subset")
    p.add_argument("--tol-hf", type=float, default=8.0,
                   help="|ΔHf| tolerance [kJ/mol] (default 8)")
    p.add_argument("--tol-s", type=float, default=10.0,
                   help="|ΔS| tolerance [J/mol/K] (default 10)")
    p.add_argument("--tol-cp", type=float, default=0.08,
                   help="Relative Cp tolerance (default 0.08)")
    p.add_argument("--all", action="store_true",
                   help="Show every matched species, not just outliers")
    p.add_argument("--json", metavar="PATH", help="Write full comparison as JSON")
    p.set_defaults(func=_cmd_thermo_check)


def _cmd_thermo_check(args) -> int:
    import json as _json
    import warnings as _w
    from rmip.analysis.thermo_provenance import (
        ThermoProvenanceChecker, load_chemkin_thermo_refs,
        render_comparison_table,
    )
    import cantera as ct

    refs = None
    if args.refs:
        refs = load_chemkin_thermo_refs(args.refs)
        if not refs:
            print(f"No NASA-7 records could be read from {args.refs}")
            return 2

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        gas = ct.Solution(args.mechanism)
    checker = ThermoProvenanceChecker(refs=refs, tol_hf_kJ=args.tol_hf,
                                      tol_s_J=args.tol_s, tol_cp_rel=args.tol_cp)
    comparisons = checker.compare_gas(gas)
    flagged = [c for c in comparisons if c.flagged]

    print(f"Mechanism: {args.mechanism}")
    print(f"  {len(comparisons)} species matched against "
          f"{'user refs' if refs else 'built-in ATcT/CODATA subset'}; "
          f"{len(flagged)} outlier(s).\n")
    show = comparisons if args.all else flagged
    if show:
        print(render_comparison_table(show))
    elif not args.all:
        print("  No outliers. ✓   (--all shows every matched species)")

    if args.json:
        with open(args.json, "w") as fh:
            _json.dump([c.to_dict() for c in comparisons], fh, indent=2)
        print(f"\n✓ JSON → {args.json}")
    return 1 if flagged else 0


# ===========================================================================
# W3 — foreign-format importers (Cantera google groups/W3…, P5)
# ===========================================================================

def _add_import_cmd(sub) -> None:
    p = sub.add_parser(
        "import",
        help="Import a foreign-format mechanism (OpenSmoke LIQUID/MATERIAL, "
             "CHEMKIN) to Cantera YAML with an explicit assumption report (P5)")
    p.add_argument("mechanism", help="Mechanism file to import")
    p.add_argument("--format", choices=["auto", "opensmoke", "chemkin"],
                   default="auto", help="Input format (default: sniff)")
    p.add_argument("--out", metavar="PATH", help="Output YAML path (default: "
                   "<name>.yaml next to the input)")
    p.add_argument("--thermo", help="Separate CHEMKIN thermo file")
    p.add_argument("--transport", help="Separate CHEMKIN transport file")
    p.add_argument("--gas-only", action="store_true",
                   help="Acknowledge dropping condensed-phase (LIQUID/MATERIAL) "
                        "content and convert the gas-phase subset")
    p.add_argument("--json", metavar="PATH", help="Write the import report JSON")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colour")
    p.set_defaults(func=_cmd_import)


def _cmd_import(args) -> int:
    import json as _json
    import os
    from rmip.io.importers.registry import detect_format, import_mechanism

    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(args.mechanism)
        print(f"Detected format: {fmt}")
    if fmt == "cantera-yaml":
        print("Input is already Cantera YAML — nothing to import "
              "(use 'rmip check' / 'rmip migrate').")
        return 0
    if fmt == "unknown":
        print("Could not identify the mechanism format (not OpenSmoke-flavoured "
              "CHEMKIN, plain CHEMKIN, or Cantera YAML).")
        return 2

    out = args.out or (os.path.splitext(args.mechanism)[0] + ".yaml")
    yaml_path, report = import_mechanism(
        args.mechanism, out, fmt=fmt, thermo=args.thermo,
        transport=args.transport, gas_only=args.gas_only)

    color = (not args.no_color) and sys.stdout.isatty()
    print(report.to_text(color=color))
    if args.json:
        with open(args.json, "w") as fh:
            _json.dump(report.to_dict(), fh, indent=2, default=str)
        print(f"✓ report JSON → {args.json}")
    if yaml_path is None:
        print("\nImport refused — resolve the findings above (condensed-phase "
              "content needs --gas-only to acknowledge the approximation).")
        return 1
    print(f"\n✓ Cantera YAML → {yaml_path}")
    return 0


# ===========================================================================
# W5a — finite bulk-solid reactor (Cantera google groups/W5…, P6/P7/P8/P9)
# ===========================================================================

def _add_react_solid_cmd(sub) -> None:
    p = sub.add_parser(
        "react-solid",
        help="Integrate a depleting bulk solid + surface + gas (finite-solid "
             "reactor; shrinking core) — P6/P7/P8/P9")
    p.add_argument("mechanism", help="Cantera YAML with gas + surface + bulk phases")
    p.add_argument("--gas-phase", help="Gas phase name (default: auto-detect)")
    p.add_argument("--surface-phase", help="Surface phase name (default: auto)")
    p.add_argument("--bulk-phase", help="Bulk solid phase name (default: auto)")
    p.add_argument("--T", type=float, default=1400.0, help="Initial T [K]")
    p.add_argument("--P", type=float, default=1.0, help="Initial pressure [atm]")
    p.add_argument("--X", default="O2:0.21, N2:0.79",
                   help="Initial gas mole fractions")
    p.add_argument("--volume", type=float, default=1.0e-3,
                   help="Gas volume [m^3] (default 1e-3)")
    p.add_argument("--bulk-mass", type=float, default=1.0e-6,
                   help="Initial solid mass [kg] (default 1e-6)")
    p.add_argument("--area", type=float, default=None,
                   help="Initial reactive area [m^2] (default: sphere of "
                        "--bulk-mass at the bulk density)")
    p.add_argument("--area-model", choices=["shrinking-core", "constant"],
                   default="shrinking-core",
                   help="Area vs remaining-mass model")
    p.add_argument("--isothermal", action="store_true",
                   help="Hold T fixed (P9 fallback for fragile exothermic cases)")
    p.add_argument("--t-end", type=float, default=100.0,
                   help="Integration end time [s]")
    p.add_argument("--rtol", type=float, default=1.0e-8)
    p.add_argument("--out", metavar="TRAJ.CSV", help="Write trajectory CSV")
    p.set_defaults(func=_cmd_react_solid)


def _cmd_react_solid(args) -> int:
    from rmip.simulation.finite_solid_reactor import (
        FiniteSolidReactor, IntegrationError, MassBalanceError,
        constant_area, shrinking_core_area)

    model = shrinking_core_area if args.area_model == "shrinking-core" \
        else constant_area
    try:
        reactor = FiniteSolidReactor(
            args.mechanism,
            gas_phase=args.gas_phase, surface_phase=args.surface_phase,
            bulk_phase=args.bulk_phase,
            T0=args.T, P0=args.P * 101325.0, X0=args.X,
            gas_volume_m3=args.volume, bulk_mass_kg=args.bulk_mass,
            area_m2=args.area, isothermal=args.isothermal,
            area_model=model, rtol=args.rtol)
        result = reactor.run(t_end=args.t_end)
    except (MassBalanceError, IntegrationError) as exc:
        print(f"react-solid failed: {exc}")
        return 1
    except ValueError as exc:
        print(f"react-solid setup error: {exc}")
        return 2

    print(result.summary())
    if args.out:
        result.to_csv(args.out)
        print(f"✓ trajectory CSV → {args.out}")
    return 0


# ===========================================================================
# W5b — RMG vs Cantera surface-mechanism cross-check (P21)
# ===========================================================================

def _add_surface_crosscheck_cmd(sub) -> None:
    p = sub.add_parser(
        "surface-crosscheck",
        help="Cross-check a surface mechanism between Cantera and RMG's "
             "surface solver (coverages / rates / Keq) — P21 detector")
    p.add_argument("mechanism", help="Cantera YAML with gas + surface (+ bulk)")
    p.add_argument("--gas-phase", help="Gas phase name (default: auto-detect)")
    p.add_argument("--surface-phase", help="Surface phase name (default: auto)")
    p.add_argument("--T", type=float, default=1000.0, help="Temperature [K]")
    p.add_argument("--P", type=float, default=1.0, help="Pressure [atm]")
    p.add_argument("--X", default="O2:0.21, N2:0.79", help="Gas mole fractions")
    p.add_argument("--rmg-python", default=None,
                   help="Python interpreter of the RMG env (default: the "
                        "reaction_family_kb default)")
    p.add_argument("--rmg-chemkin-gas", metavar="CHEM.INP",
                   help="CHEMKIN gas mechanism for the RMG side (live mode; "
                        "RMG cannot read Cantera YAML)")
    p.add_argument("--rmg-chemkin-surf", metavar="SURF.INP",
                   help="CHEMKIN surface mechanism for the RMG side (live mode)")
    p.add_argument("--rmg-json", metavar="PATH",
                   help="Use a pre-computed RMG-side result JSON instead of "
                        "launching the RMG solver (offline mode)")
    p.add_argument("--tol-keq", type=float, default=0.5,
                   help="|Δ log10 Keq| above which a reaction is flagged")
    p.add_argument("--tol-coverage", type=float, default=0.05,
                   help="Absolute coverage difference flagged (default 0.05)")
    p.add_argument("--json", metavar="PATH", help="Write full report as JSON")
    p.set_defaults(func=_cmd_surface_crosscheck)


def _cmd_surface_crosscheck(args) -> int:
    import json as _json
    from rmip.simulation.surface_crosscheck import SurfaceCrossChecker

    checker = SurfaceCrossChecker(
        args.mechanism, gas_phase=args.gas_phase,
        surface_phase=args.surface_phase,
        T_K=args.T, P_atm=args.P, X=args.X,
        rmg_python=args.rmg_python,
        rmg_chemkin_gas=args.rmg_chemkin_gas,
        rmg_chemkin_surf=args.rmg_chemkin_surf,
        tol_log10_keq=args.tol_keq, tol_coverage=args.tol_coverage)
    try:
        report = checker.run(rmg_json=args.rmg_json)
    except RuntimeError as exc:
        print(f"surface-crosscheck failed: {exc}")
        return 1

    print(report.render_text())
    if args.json:
        with open(args.json, "w") as fh:
            _json.dump(report.to_dict(), fh, indent=2, default=str)
        print(f"✓ JSON → {args.json}")
    return 0 if not report.divergent else 1


# ===========================================================================
# W7 — RMG explainability / supervisor / linter / diff (P15-P17,P26,P30,P39)
# ===========================================================================

def _add_rmg_explain_cmd(sub) -> None:
    p = sub.add_parser(
        "rmg-explain",
        help="Explain a finished RMG run: why each core species was "
             "promoted, and the edge RMG considered but rejected (P17/P26)")
    p.add_argument("run_dir", help="RMG output directory (contains RMG.log)")
    p.add_argument("--top-edge", type=int, default=30,
                   help="Show top-N edge species by flux (default 30)")
    p.add_argument("--json", metavar="PATH", help="Write full report as JSON")
    p.set_defaults(func=_cmd_rmg_explain)


def _cmd_rmg_explain(args) -> int:
    import json as _json
    from rmip.generation.rmg_explainer import RmgRunExplainer

    try:
        explanation = RmgRunExplainer(args.run_dir).explain()
    except FileNotFoundError as exc:
        print(f"rmg-explain: {exc}")
        return 2
    print(explanation.render_text(top_edge=args.top_edge))
    if args.json:
        with open(args.json, "w") as fh:
            _json.dump(explanation.to_dict(), fh, indent=2)
        print(f"\n✓ JSON → {args.json}")
    return 0


def _add_rmg_lint_cmd(sub) -> None:
    p = sub.add_parser(
        "rmg-lint",
        help="Statically lint an RMG input.py before the (expensive) run: "
             "dual-listed / duplicated libraries, tolerance mistakes (P39)")
    p.add_argument("input_py", help="RMG job input file")
    p.add_argument("--json", metavar="PATH", help="Write findings as JSON")
    p.set_defaults(func=_cmd_rmg_lint)


def _cmd_rmg_lint(args) -> int:
    import json as _json
    from rmip.generation.rmg_lint import lint_rmg_input

    findings = lint_rmg_input(args.input_py)
    if not findings:
        print(f"✓ {args.input_py}: no lint findings")
        return 0
    sev_rank = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: sev_rank.get(f.severity, 9))
    for f in findings:
        print(f"[{f.severity.upper():7}] {f.category}: {f.message}")
        if f.location:
            print(f"          at {f.location}")
    if args.json:
        with open(args.json, "w") as fh:
            _json.dump([f.to_dict() for f in findings], fh, indent=2)
        print(f"✓ JSON → {args.json}")
    return 1 if any(f.severity == "error" for f in findings) else 0


def _add_rmg_diff_cmd(sub) -> None:
    p = sub.add_parser(
        "rmg-diff",
        help="Diff two RMG runs (same input, e.g. different RMG versions): "
             "species/reactions added/removed/renamed + rate changes (P15)")
    p.add_argument("run_a", help="First RMG output dir (or chemkin file)")
    p.add_argument("run_b", help="Second RMG output dir (or chemkin file)")
    p.add_argument("--tol-log10-A", type=float, default=0.1,
                   help="|Δlog10 A| above which a rate change is reported")
    p.add_argument("--tol-n", type=float, default=0.05)
    p.add_argument("--tol-Ea", type=float, default=100.0,
                   help="|ΔEa| threshold in the files' Ea units")
    p.add_argument("--json", metavar="PATH", help="Write full diff as JSON")
    p.set_defaults(func=_cmd_rmg_diff)


def _cmd_rmg_diff(args) -> int:
    import json as _json
    from rmip.generation.rmg_diff import reproducibility_diff

    try:
        diff = reproducibility_diff(
            args.run_a, args.run_b, tol_log10_A=args.tol_log10_A,
            tol_n=args.tol_n, tol_Ea=args.tol_Ea)
    except FileNotFoundError as exc:
        print(f"rmg-diff: {exc}")
        return 2
    print(diff.render_text())
    if args.json:
        with open(args.json, "w") as fh:
            _json.dump(diff.to_dict(), fh, indent=2)
        print(f"\n✓ JSON → {args.json}")
    return 0 if diff.identical else 1


def _add_rmg_supervise_cmd(sub) -> None:
    p = sub.add_parser(
        "rmg-supervise",
        help="Run RMG under supervision: wall/memory/stall budgets, "
             "optional tolerance relaxation, partial-result report on "
             "early stop (P30/P16)")
    p.add_argument("input_py", nargs="+",
                   help="RMG input file(s); several run as sequential stages")
    p.add_argument("-o", "--output", required=True,
                   help="Output directory (a batch gets stage_NN subdirs)")
    p.add_argument("--rmg-python", default=None,
                   help="Python interpreter of the RMG env")
    p.add_argument("--rmg-script", default=None,
                   help="Path to rmg.py (default /data/OPEN_CODE/RMG_py/rmg.py)")
    p.add_argument("--mem-gb", type=float, default=None,
                   help="Memory budget for the whole process tree [GB]")
    p.add_argument("--timeout", type=float, default=None,
                   help="Wall-clock budget [s]")
    p.add_argument("--stall-timeout", type=float, default=None,
                   help="Declare a stall after this many seconds without "
                        "log growth")
    p.add_argument("--relax-on-stall", action="store_true",
                   help="Retry a stalled/timed-out run with "
                        "toleranceMoveToCore x10 (bounded)")
    p.set_defaults(func=_cmd_rmg_supervise)


def _cmd_rmg_supervise(args) -> int:
    from rmip.generation.rmg_supervisor import RmgSupervisor

    sup = RmgSupervisor(
        rmg_python=args.rmg_python, rmg_script=args.rmg_script,
        mem_budget_mb=(args.mem_gb * 1024.0 if args.mem_gb else None),
        wall_timeout_s=args.timeout, stall_timeout_s=args.stall_timeout,
        relax_on_stall=args.relax_on_stall)
    try:
        if len(args.input_py) == 1:
            reports = [sup.run(args.input_py[0], args.output)]
        else:
            reports = sup.run_staged(args.input_py, args.output)
    except RuntimeError as exc:
        print(f"rmg-supervise: {exc}")
        return 2
    for r in reports:
        print(r.render_text())
        print()
    return 0 if all(r.ok for r in reports) else 1


# ===========================================================================
# W8b — version-behaviour regression harness (P28/P15)
# ===========================================================================

def _add_regression_cmd(sub) -> None:
    p = sub.add_parser(
        "regression",
        help="Run the canonical IDT/LFS/equilibrium suite and diff against "
             "committed goldens — catches Cantera version-behaviour drift "
             "(W8b, P28)")
    p.add_argument("--env", metavar="PYTHON",
                   help="Python executable of the Cantera env to test "
                        "(default: this interpreter). The env only needs "
                        "cantera installed, not rmip.")
    p.add_argument("--cases", metavar="JSON",
                   help="Custom case-suite JSON ({'cases': [...]}); default: "
                        "the built-in canonical suite")
    p.add_argument("--golden", metavar="JSON",
                   help="Golden file to diff/bless (default: the committed "
                        "goldens next to version_regression.py)")
    p.add_argument("--bless", action="store_true",
                   help="Write this run's outputs as the new golden "
                        "(explicit and audited — never happens silently)")
    p.add_argument("--tol", nargs="+", metavar="TASK=REL",
                   help="Override per-task relative tolerances, e.g. "
                        "idt=0.10 lfs=0.05")
    p.add_argument("--timeout", type=float, default=3600.0,
                   help="Suite wall-clock timeout in seconds (default 3600)")
    p.add_argument("--json", metavar="PATH",
                   help="Write the structured report as JSON")
    p.set_defaults(func=_cmd_regression)


def _cmd_regression(args) -> int:
    import json as _json
    from rmip.analysis import version_regression as vr

    cases = None
    if args.cases:
        with open(args.cases) as fh:
            cases = _json.load(fh)["cases"]

    tolerances = {}
    for spec in args.tol or []:
        task, _, val = spec.partition("=")
        tolerances[task.strip()] = float(val)

    n = len(cases if cases is not None else vr.CANONICAL_SUITE)
    print(f"Running {n} regression case(s) under "
          f"{args.env or sys.executable} …")
    payload = vr.run_suite(env_python=args.env, cases=cases,
                           timeout_s=args.timeout)

    if args.bless:
        path = vr.bless(payload, golden_path=args.golden)
        print(f"✓ Blessed {len(payload['results'])} case(s) → {path}")
        print(f"  (Cantera {payload['meta'].get('cantera_version', '?')})")
        return 0

    report = vr.diff_against_golden(payload, golden_path=args.golden,
                                    tolerances=tolerances or None,
                                    cases=cases)
    print(report.to_text())
    if args.json:
        with open(args.json, "w") as fh:
            _json.dump(report.to_dict(), fh, indent=2, default=str)
        print(f"\n✓ JSON report → {args.json}")
    return 0 if report.ok else 1


# ===========================================================================
# W9 — onboarding: dependency preflight (doctor) + task→recipe router
# ===========================================================================

def _add_doctor_cmd(sub) -> None:
    from rmip.config.preflight import TASKS
    p = sub.add_parser(
        "doctor",
        help="Verify the toolchain a task needs is present BEFORE the job "
             "starts; prints the exact fix for anything missing (W9a, P38)")
    p.add_argument("--task", default="all", choices=sorted(TASKS),
                   help="Which task's prerequisites to check (default: all)")
    p.add_argument("--rmg-input", metavar="INPUT_PY",
                   help="Also statically lint an RMG input.py (duplicate/"
                        "dual-listed libraries, tolerance order) — W9c")
    p.add_argument("--json", metavar="PATH",
                   help="Write the structured report as JSON")
    p.set_defaults(func=_cmd_doctor)


def _cmd_doctor(args) -> int:
    from rmip.config.preflight import check_environment, report_json

    report = check_environment(task=args.task, rmg_input=args.rmg_input)
    print(report.to_text())
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(report_json(report))
        print(f"\n✓ JSON report → {args.json}")
    return 0 if report.ok else 1


def _add_how_do_i_cmd(sub) -> None:
    p = sub.add_parser(
        "how-do-i",
        help='Route a plain-language task to the right command, e.g. '
             'how-do-i "ignition delay sensitivity" (W9b, P36)')
    p.add_argument("query", nargs="+",
                   help="What you want to do, in your own words")
    p.add_argument("--all", action="store_true",
                   help="List every recipe instead of matching")
    p.set_defaults(func=_cmd_how_do_i)


def _cmd_how_do_i(args) -> int:
    from rmip.config.recipes import RECIPES, find_recipes, render_recipes

    query = " ".join(args.query)
    if args.all:
        print(render_recipes(RECIPES))
        return 0
    matches = find_recipes(query)
    print(render_recipes(matches, query=query))
    return 0


# ---- reactions ----

def _add_reactions_cmd(sub) -> None:
    p = sub.add_parser(
        "reactions",
        help="List all reactions with R-labels; export to CSV",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db",  required=True, help="SQLite database path")
    p.add_argument("--out", "-o", default=None,
                   help="Output CSV path (default: reactions_<mech_id>.csv)")
    p.set_defaults(func=_cmd_reactions)


def _cmd_reactions(args) -> int:
    import csv as _csv
    conn = init_db(args.db)
    conn.row_factory = sqlite3.Row

    # Fetch mechanism name for the header
    mech_row = conn.execute(
        "SELECT name FROM mechanisms WHERE id=?", (args.mech_id,)
    ).fetchone()
    if mech_row is None:
        print(f"Mechanism id={args.mech_id} not found.")
        return 1

    rows = conn.execute(
        """
        SELECT reaction_label, formula_string, kinetics_type,
               reversible, duplicate, pressure_dependent, has_third_body,
               A_si, n, Ea_si, A_orig, Ea_orig, A_units, Ea_units,
               canonical_key
        FROM   reactions
        WHERE  mechanism_id=?
        ORDER  BY id
        """,
        (args.mech_id,),
    ).fetchall()

    if not rows:
        print(f"No reactions found for mechanism id={args.mech_id}.")
        return 0

    out_path = args.out or f"reactions_{args.mech_id}.csv"
    fieldnames = [
        "label", "formula_string", "kinetics_type",
        "reversible", "duplicate", "pressure_dependent", "has_third_body",
        "A_si", "n", "Ea_si", "A_orig", "Ea_orig", "A_units", "Ea_units",
        "canonical_key",
    ]

    with open(out_path, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "label":              row["reaction_label"] or "",
                "formula_string":     row["formula_string"],
                "kinetics_type":      row["kinetics_type"] or "arrhenius",
                "reversible":         row["reversible"],
                "duplicate":          row["duplicate"],
                "pressure_dependent": row["pressure_dependent"],
                "has_third_body":     row["has_third_body"],
                "A_si":               row["A_si"],
                "n":                  row["n"],
                "Ea_si":              row["Ea_si"],
                "A_orig":             row["A_orig"],
                "Ea_orig":            row["Ea_orig"],
                "A_units":            row["A_units"] or "",
                "Ea_units":           row["Ea_units"] or "",
                "canonical_key":      row["canonical_key"] or "",
            })

    print(f"\n✓ {len(rows)} reactions for '{mech_row['name']}' → {out_path}")
    print(f"\n{'Label':<8}  {'Formula':<45}  {'Type':<14}  Dup  PD  TB")
    print("-" * 88)
    preview = rows[:15]
    for row in preview:
        dup = "Y" if row["duplicate"]          else "-"
        pd  = "Y" if row["pressure_dependent"] else "-"
        tb  = "Y" if row["has_third_body"]     else "-"
        print(
            f"{(row['reaction_label'] or ''):>8}  "
            f"{row['formula_string']:<45}  "
            f"{(row['kinetics_type'] or 'arrhenius'):<14}  "
            f"{dup:<3}  {pd:<2}  {tb}"
        )
    if len(rows) > len(preview):
        print(f"  … ({len(rows) - len(preview)} more reactions in {out_path})")
    return 0


def _add_reaction_matrix_cmd(sub) -> None:
    p = sub.add_parser(
        "reaction-matrix",
        help="Cross-mechanism reaction presence matrix: which reactions are in which mechanism",
    )
    p.add_argument("--db",  required=True, help="SQLite database path")
    p.add_argument("--mechs", "-m", nargs="*", type=int, default=None,
                   help="Mechanism IDs to include (default: all mechanisms in DB)")
    p.add_argument("--out", "-o", default="reaction_matrix.csv",
                   help="Output CSV path (default: reaction_matrix.csv)")
    p.set_defaults(func=_cmd_reaction_matrix)


def _cmd_reaction_matrix(args) -> int:
    import csv as _csv
    from collections import OrderedDict

    conn = init_db(args.db)
    conn.row_factory = sqlite3.Row

    # Which mechanisms to include
    if args.mechs:
        placeholders = ",".join("?" * len(args.mechs))
        mechs = conn.execute(
            f"SELECT id, name FROM mechanisms WHERE id IN ({placeholders}) ORDER BY id",
            args.mechs,
        ).fetchall()
    else:
        mechs = conn.execute(
            "SELECT id, name FROM mechanisms ORDER BY id"
        ).fetchall()

    if not mechs:
        print("No mechanisms found.")
        return 1

    mech_ids   = [m["id"]   for m in mechs]
    mech_names = {m["id"]: m["name"] for m in mechs}

    # Fetch all reactions for the selected mechanisms
    placeholders = ",".join("?" * len(mech_ids))
    all_rows = conn.execute(
        f"""SELECT mechanism_id, reaction_label, formula_string, canonical_key,
                   kinetics_type, reversible, duplicate, pressure_dependent, has_third_body
            FROM   reactions
            WHERE  mechanism_id IN ({placeholders})
            ORDER  BY mechanism_id, id""",
        mech_ids,
    ).fetchall()

    # Group reactions by identity key.
    # Key = canonical_key if not NULL/empty, else formula_string.
    # If multiple rows from the same mechanism share the same key (DUPLICATE),
    # concatenate their R-labels with "/".
    reaction_map: "OrderedDict[str, dict]" = OrderedDict()

    for row in all_rows:
        ck  = (row["canonical_key"] or "").strip()
        fs  = row["formula_string"]
        key = ck if ck else fs

        if key not in reaction_map:
            reaction_map[key] = {
                "canonical_key":       ck,
                "formula_string":      fs,
                "kinetics_type":       row["kinetics_type"] or "arrhenius",
                "reversible":          row["reversible"],
                "duplicate":           row["duplicate"],
                "pressure_dependent":  row["pressure_dependent"],
                "has_third_body":      row["has_third_body"],
                **{mid: "" for mid in mech_ids},
            }

        mid   = row["mechanism_id"]
        label = row["reaction_label"] or "present"
        if reaction_map[key][mid]:
            reaction_map[key][mid] += f"/{label}"   # DUPLICATE: combine labels
        else:
            reaction_map[key][mid] = label

    # Build CSV
    mech_col_names = [f"{mech_names[mid]} [{mid}]" for mid in mech_ids]
    base_fields    = [
        "canonical_key", "formula_string", "kinetics_type",
        "reversible", "duplicate", "pressure_dependent", "has_third_body",
    ]
    fieldnames = base_fields + mech_col_names

    with open(args.out, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for data in reaction_map.values():
            row_dict = {f: data[f] for f in base_fields}
            for mid, col in zip(mech_ids, mech_col_names):
                row_dict[col] = data[mid]
            writer.writerow(row_dict)

    n_unique = len(reaction_map)
    n_total  = len(all_rows)
    print(f"✓ {n_unique} unique reactions across {len(mechs)} mechanisms → {args.out}")
    if n_total > n_unique:
        print(f"  ({n_total - n_unique} reactions share a canonical key with another "
              f"in the same mechanism — their R-labels are combined with '/')")

    # Quick preview
    print(f"\n{'Formula':<50}  " + "  ".join(f"{m['name'][:12]:<12}" for m in mechs))
    print("-" * (52 + 14 * len(mechs)))
    for i, data in enumerate(reaction_map.values()):
        if i >= 10:
            print(f"  … ({n_unique - 10} more rows in {args.out})")
            break
        cols = "  ".join(f"{data[mid] or '—':<12}" for mid in mech_ids)
        print(f"{data['formula_string']:<50}  {cols}")
    return 0


# ===========================================================================
# Phase 2 commands
# ===========================================================================

def _add_compare_cmd(sub) -> None:
    p = sub.add_parser("compare", help="Compare kinetics/thermo across mechanisms (M6)")
    p.add_argument("--mechs", "-m", nargs="+", type=int, required=True,
                   help="List of mechanism DB ids to compare")
    p.add_argument("--db",   required=True, help="SQLite database path")
    p.add_argument("--out",  default="comparison_out", help="Output directory")
    p.set_defaults(func=_cmd_compare)


def _cmd_compare(args) -> int:
    from rmip.analysis.comparison import ComparisonEngine
    eng    = ComparisonEngine(args.db)
    report = eng.compare(args.mechs)
    eng.save_report(report, args.out)

    sp  = report["species_comparison"]
    rxn = report["reaction_comparison"]
    print(f"\nComparison: mechs {args.mechs}")
    print(f"  Species  — universal: {sp['n_universal']}, "
          f"partial: {sp['n_partial']}, unique: {sp['n_unique']} "
          f"(glossary resolution: {sp['species_resolution_coverage_pct']:.1f}%, "
          f"{sp['n_unresolved_species']} unresolved)")
    print(f"  Reactions— shared: {rxn['n_shared']}, unique: {rxn['n_unique']} "
          f"(resolution: {rxn['reaction_resolution_coverage_pct']:.1f}%, "
          f"{rxn['n_unresolved_reactions']} unresolved)")
    print(f"✓ Report saved to '{args.out}/'")
    return 0


def _add_visualize_cmd(sub) -> None:
    p = sub.add_parser("visualize", help="Launch interactive network visualizer (M3)")
    p.add_argument("mech_id", type=int, nargs="+",
                   help="One or two mechanism DB ids (two → side-by-side view)")
    p.add_argument("--db",    required=True, help="SQLite database path")
    p.add_argument("--port",  type=int, default=8050, help="Port for Dash app")
    p.add_argument("--label", choices=["formula", "id"], default="formula",
                   help="Reaction node label: full formula (default) or R-ID (R1, R2, …)")
    p.add_argument("--union", action="store_true",
                   help="Show union of nodes from all selected mechanisms; nodes absent "
                        "from a mechanism appear as faint ghost nodes with no edges")
    p.add_argument("--export-html", metavar="PATH",
                   help="Export static HTML instead of launching Dash")
    p.add_argument("--export-figure", metavar="PATH",
                   help="Export publication-quality figure (PDF/SVG/PNG) instead of "
                        "launching Dash. Use .pdf or .svg for vector output.")
    p.add_argument("--figure-dpi", type=int, default=300,
                   help="DPI for raster figure export (default 300, ignored for PDF/SVG)")
    # Phase 5A — SA/flux edge highlighting
    p.add_argument("--sa-json", metavar="FILE[,FILE]",
                   help="Sensitivity JSON file (CLAUDE.md §10 format); "
                        "comma-separated pair for dual-panel (one file per mechanism)")
    p.add_argument("--flux-json", metavar="FILE[,FILE]",
                   help="Flux JSON file (optional); comma-separated pair for dual-panel")
    p.add_argument("--threshold", type=float, default=0.1, metavar="FLOAT",
                   help="Importance below this value → grey edges (default 0.1)")
    p.add_argument("--score", choices=["sa", "flux", "importance"], default="importance",
                   help="Score driving edge coloring (default: importance)")
    # Phase 5B — mechanism reduction overlay
    p.add_argument("--reduction-json", metavar="FILE",
                   help="Reduction steps JSON from `rmip reduce` — adds an ε slider "
                        "that dims removed nodes/edges at each threshold")
    p.add_argument("--eps", type=float, default=0.0, metavar="FLOAT",
                   help="Initial ε threshold for reduction overlay (default: 0.0 = full)")
    p.set_defaults(func=_cmd_visualize)


def _cmd_visualize(args) -> int:
    import json as _json
    from rmip.core.graph_builder     import GraphBuilder
    from rmip.analysis.visualization import NetworkVisualizer
    if len(args.mech_id) > 2:
        print("ERROR: visualize accepts at most 2 mechanism IDs (for side-by-side view).")
        return 1
    gb = GraphBuilder(args.db)
    if args.union and len(args.mech_id) > 1:
        graphs = gb.build_union_graphs(args.mech_id)
    else:
        graphs = [gb.build_graph(mid) for mid in args.mech_id]
    titles = [f"RMIP — Mechanism {mid}" for mid in args.mech_id]

    # Phase 5A — attach SA/flux scores so edges pick up importance data
    if getattr(args, "sa_json", None) or getattr(args, "flux_json", None):
        from rmip.analysis.sensitivity import SensitivityFluxIntegrator

        def _load_pair(arg):
            if not arg:
                return [None] * len(graphs)
            files = [f.strip() for f in arg.split(",")]
            loaded = []
            for f in files:
                with open(f) as fh:
                    loaded.append(_json.load(fh))
            while len(loaded) < len(graphs):
                loaded.append(loaded[-1])
            return loaded

        sa_list   = _load_pair(getattr(args, "sa_json", None))
        flux_list = _load_pair(getattr(args, "flux_json", None))
        m8 = SensitivityFluxIntegrator("")   # attach() needs no DB access
        for i, G in enumerate(graphs):
            m8.attach(0, G, sa_list[i], flux_list[i])
        print(f"✓ SA/flux scores attached (score={args.score}, threshold={args.threshold})")

    viz = NetworkVisualizer(graphs, title=titles)
    if args.export_html:
        viz.export_html(args.export_html)
        print(f"✓ Static HTML → {args.export_html}")
    elif args.export_figure:
        viz.export_publication_figure(
            args.export_figure,
            label_mode=args.label,
            dpi=args.figure_dpi,
        )
        print(f"✓ Publication figure → {args.export_figure}")
    else:
        print(f"Launching visualizer on http://localhost:{args.port}")

        # Phase 5B — load reduction steps if provided
        reduction_result = None
        reduction_eps    = getattr(args, "eps", 0.0)
        if getattr(args, "reduction_json", None):
            from rmip.types import ReductionResult
            reduction_result = ReductionResult.load(args.reduction_json)
            print(f"✓ Reduction overlay: {reduction_result.method} "
                  f"({len(reduction_result.steps)} steps), "
                  f"initial ε={reduction_eps:.3f}")

        viz.run(
            port=args.port,
            label_mode=args.label,
            threshold=getattr(args, "threshold", 0.1),
            score=getattr(args, "score", "importance"),
            reduction_result=reduction_result,
            reduction_eps=reduction_eps,
        )
    return 0


def _add_visualize3d_cmd(sub) -> None:
    p = sub.add_parser(
        "visualize3d",
        help="Launch 3D planetary network visualizer (Phase 5C) — species on "
             "concentric spherical orbits by carbon number",
    )
    p.add_argument("mech_id", type=int, nargs="+",
                   help="One or two mechanism DB ids (two → side-by-side view)")
    p.add_argument("--db",    required=True, help="SQLite database path")
    p.add_argument("--port",  type=int, default=8050, help="Port for Dash app")
    p.add_argument("--label", choices=["formula", "id"], default="formula",
                   help="Reaction hover label: full formula (default) or R-ID")
    p.add_argument("--union", action="store_true",
                   help="Show union of nodes from all selected mechanisms; nodes absent "
                        "from a mechanism appear as faint ghost markers")
    p.add_argument("--export-html", metavar="PATH",
                   help="Export self-contained interactive HTML instead of launching Dash")
    p.add_argument("--shell-gap", type=float, default=2.0,
                   help="Radial spacing between carbon-shell rings relative to the "
                        "C0 core radius (default 2.0)")
    p.add_argument("--layer-gap", type=float, default=1.0,
                   help="Vertical spacing between carbon layers in core-diameter "
                        "units (default 1.0; 0 flattens all rings into one plane)")
    p.add_argument("--order", choices=["alpha", "family"], default="alpha",
                   help="Initial species ordering around each ring: alphabetical "
                        "or grouped by chemical family (switchable live in the app)")
    p.set_defaults(func=_cmd_visualize3d)


def _cmd_visualize3d(args) -> int:
    from rmip.core.graph_builder       import GraphBuilder
    from rmip.analysis.visualization3d import NetworkVisualizer3D

    if len(args.mech_id) > 2:
        print("ERROR: visualize3d accepts at most 2 mechanism IDs (for side-by-side view).")
        return 1
    gb = GraphBuilder(args.db)
    # Reverse edges add no visual information in 3D (lines have no arrows) —
    # skip them to halve the edge trace size for large mechanisms.  The 3D view
    # only consumes pos_*_3d, so the expensive spring layout is skipped too.
    if args.union and len(args.mech_id) > 1:
        graphs = gb.build_union_graphs(args.mech_id, include_reverse_edges=False,
                                       include_spring=False,
                                       shell_gap_3d=args.shell_gap,
                                       layer_gap_3d=args.layer_gap)
    else:
        graphs = [gb.build_graph(mid, include_reverse_edges=False,
                                 include_spring=False,
                                 shell_gap_3d=args.shell_gap,
                                 layer_gap_3d=args.layer_gap)
                  for mid in args.mech_id]

    titles = []
    for mid in args.mech_id:
        row = gb._conn.execute(
            "SELECT name FROM mechanisms WHERE id = ?", (mid,)
        ).fetchone()
        titles.append(row["name"] if row else f"Mechanism {mid}")

    viz = NetworkVisualizer3D(graphs, title=titles)
    if args.export_html:
        viz.export_html(args.export_html, label_mode=args.label,
                        order_mode=args.order)
        print(f"✓ Interactive 3D HTML → {args.export_html}")
    else:
        print(f"Launching 3D visualizer on http://localhost:{args.port}")
        viz.run(port=args.port, label_mode=args.label, order_mode=args.order)
    return 0


# ===========================================================================
# Phase 3 commands
# ===========================================================================

def _add_sparsity_cmd(sub) -> None:
    p = sub.add_parser("sparsity",
                       help="Detect dead-end nodes, orphan chains, coverage (M4)")
    p.add_argument("mech_id", type=int,    help="Mechanism DB id")
    p.add_argument("--db",    required=True, help="SQLite database path")
    p.add_argument("--out",   help="Optional JSON output path for full report")
    p.add_argument("--compare", nargs="+", type=int, metavar="MECH_ID",
                   help="Also compare with additional mechanisms")
    p.set_defaults(func=_cmd_sparsity)


def _cmd_sparsity(args) -> int:
    from rmip.core.graph_builder import GraphBuilder
    from rmip.analysis.sparsity  import SparsityDetector

    gb      = GraphBuilder(args.db)
    G       = gb.build_graph(args.mech_id)
    det     = SparsityDetector(args.db)
    report  = det.analyze(args.mech_id, G)

    print(f"\nSparsity report — mech_id={args.mech_id}")
    print(f"  Species:            {report.n_species}")
    print(f"  Reactions:          {report.n_reactions}")
    print(f"  Dead-end sources:   {report.n_dead_end_sources}")
    print(f"  Dead-end sinks:     {report.n_dead_end_sinks}")
    print(f"  Isolated nodes:     {report.n_isolated}")
    print(f"  Orphan reactions:   {len(report.orphan_reaction_nodes)}")
    print(f"  Connected compts:   {report.n_components}")
    print(f"  Coverage fraction:  {report.coverage_fraction:.1%}")

    if args.compare:
        all_ids = [args.mech_id] + args.compare
        cmp     = det.compare_all(all_ids)
        print(f"\nCross-mechanism sparsity comparison: {all_ids}")
        print(f"  Species common to all:   {len(cmp.species_common)}")
        print(f"  Reactions common to all: {len(cmp.reactions_common)}")
        for mid in all_ids:
            print(f"  Exclusive to mech {mid}:  "
                  f"{len(cmp.species_exclusive.get(mid, []))} species, "
                  f"{len(cmp.reactions_exclusive.get(mid, []))} reactions")

    if args.out:
        det.export_report(report, args.out)
        print(f"✓ Sparsity report → {args.out}")

    return 0


def _add_sensitivity_cmd(sub) -> None:
    p = sub.add_parser("sensitivity",
                       help="Attach SA/flux data to graph nodes (M8)")
    p.add_argument("mech_id", type=int,    help="Mechanism DB id")
    p.add_argument("--db",    required=True, help="SQLite database path")
    p.add_argument("--sa-file", required=True,
                   help="JSON file with sensitivity_data dict (see CLAUDE.md §10)")
    p.add_argument("--flux-file",
                   help="JSON file with flux_data dict (optional)")
    p.add_argument("--top",   type=int, default=20,
                   help="Print top-N reactions by importance_score")
    p.add_argument("--out-csv", help="Export importance ranking to CSV")
    p.set_defaults(func=_cmd_sensitivity)


def _cmd_sensitivity(args) -> int:
    import json as _json
    from rmip.core.graph_builder       import GraphBuilder
    from rmip.analysis.sensitivity     import SensitivityFluxIntegrator

    with open(args.sa_file) as fh:
        sa_data = _json.load(fh)
    flux_data = None
    if args.flux_file:
        with open(args.flux_file) as fh:
            flux_data = _json.load(fh)

    gb  = GraphBuilder(args.db)
    G   = gb.build_graph(args.mech_id)
    m8  = SensitivityFluxIntegrator(args.db)
    G   = m8.attach(args.mech_id, G, sa_data, flux_data)
    df  = m8.get_top_reactions(G, n=args.top)

    print(f"\nTop-{args.top} reactions by importance score — mech_id={args.mech_id}")
    print(df.to_string(index=False))

    if args.out_csv:
        m8.export_report(G, args.out_csv)
        print(f"✓ SA report → {args.out_csv}")

    return 0


def _add_sa_run_cmd(sub) -> None:
    p = sub.add_parser("sa-run",
                       help="Run Cantera IDT sensitivity analysis → SA JSON (Phase 5A)")
    p.add_argument("mechanism", help="Cantera .yaml or .cti mechanism file")
    p.add_argument("--T",   type=float, required=True, dest="T_K",  help="Temperature [K]")
    p.add_argument("--P",   type=float, required=True, dest="P_atm", help="Pressure [atm]")
    p.add_argument("--phi", type=float, required=True, help="Equivalence ratio")
    p.add_argument("--fuel", required=True, help="Fuel species name in mechanism")
    p.add_argument("--oxidizer", default="O2", help="Oxidizer species name (default: O2)")
    p.add_argument("--top",  type=int, default=None,
                   help="Keep only top-N reactions by |coefficient| (default: all)")
    p.add_argument("--out",  required=True, metavar="FILE",
                   help="Output JSON path for sensitivity_data (CLAUDE.md §10 format)")
    p.set_defaults(func=_cmd_sa_run)


def _cmd_sa_run(args) -> int:
    from rmip.simulation.sensitivity_runner import CanteraSensitivityRunner

    runner = CanteraSensitivityRunner()
    print(f"Running Cantera IDT SA: T={args.T_K} K, P={args.P_atm} atm, "
          f"phi={args.phi}, fuel={args.fuel} ...")
    sa_data = runner.run_idt(
        mechanism_path=args.mechanism,
        T_K=args.T_K,
        P_atm=args.P_atm,
        phi=args.phi,
        fuel=args.fuel,
        oxidizer=args.oxidizer,
        n_top=args.top,
    )
    runner.save(sa_data, args.out)
    n_rxn = len(sa_data.get("coefficients", {}))
    print(f"✓ {n_rxn} reaction coefficients → {args.out}")
    return 0


def _add_exp_load_cmd(sub) -> None:
    p = sub.add_parser("exp-load",
                       help="Load experimental IDT/LFS data into the database")
    p.add_argument("csv_path",           help="CSV file with experimental data")
    p.add_argument("--db",    required=True, help="SQLite database path")
    p.add_argument("--name",  "-n", required=True, help="Dataset name/label")
    p.add_argument("--sim-type", required=True, choices=["IDT","LFS"],
                   help="Observable type")
    p.add_argument("--fuel",  required=True, help="Fuel species name")
    p.add_argument("--P",     type=float, dest="default_P_atm",
                   help="Default pressure [atm] (used if P_atm column absent)")
    p.add_argument("--phi",   type=float, dest="default_phi",
                   help="Default equivalence ratio (used if phi column absent)")
    p.add_argument("--units", default="s", dest="default_units",
                   help="Default value units (default: 's' for IDT)")
    p.add_argument("--source", help="Optional source/reference string")
    p.set_defaults(func=_cmd_exp_load)


def _cmd_exp_load(args) -> int:
    from rmip.database.schema       import init_db
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.io.exp_data_reader    import ExpDataReader

    conn = init_db(args.db)
    upgrade_db(conn)
    reader = ExpDataReader(args.db)
    ds_id  = reader.load_csv(
        path         = args.csv_path,
        dataset_name = args.name,
        sim_type     = args.sim_type,
        fuel_name    = args.fuel,
        default_P_atm  = args.default_P_atm,
        default_phi    = args.default_phi,
        default_units  = args.default_units,
        source         = args.source,
    )
    print(f"✓ Experimental dataset stored as id={ds_id} ('{args.name}')")

    # Show list of all datasets
    conn   = init_db(args.db)
    upgrade_db(conn)
    datasets = list_exp_datasets(conn)
    print(f"\nAll experimental datasets in '{args.db}':")
    print(f"  {'ID':>4}  {'Name':<30}  {'Type':<6}  {'Fuel':<15}  {'N pts':>6}")
    print("  " + "-" * 65)
    for ds in datasets:
        print(f"  {ds['id']:>4}  {(ds['name'] or ''):<30}  "
              f"{(ds['sim_type'] or ''):<6}  {(ds['fuel_name'] or ''):<15}  "
              f"{(ds['n_points'] or 0):>6}")
    return 0


def _add_bench_cmd(sub) -> None:
    p = sub.add_parser("bench",
                       help="Benchmark mechanisms vs experimental dataset (IDT/LFS)")
    p.add_argument("--mechs", "-m", nargs="+", type=int, required=True,
                   help="Mechanism DB ids to benchmark")
    p.add_argument("--dataset", "-d", type=int, required=True,
                   help="Experimental dataset DB id (from exp-load)")
    p.add_argument("--fuel",   required=True,
                   help="Fuel composition string, e.g. 'NC7H16:1.0'")
    p.add_argument("--db",     required=True, help="SQLite database path")
    p.add_argument("--out",    default="bench_out",
                   help="Output directory for plots and CSV")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers for Cantera simulations")
    p.add_argument("--regime", choices=["LTC","NTC","HTC","HTT","LTT","all"], default="all",
                   help="Regime to show in ranking table. HTT/LTT are used "
                        "instead of LTC/NTC/HTC for mechanisms with a stored "
                        "NTC detection run (see 'rmip ntc').")
    p.set_defaults(func=_cmd_bench)


def _cmd_bench(args) -> int:
    from rmip.analysis.benchmarking import MechanismBenchmarker
    bench = MechanismBenchmarker(args.db)

    print(f"\nBenchmarking mechanisms {args.mechs} against dataset {args.dataset}...")
    scores_df = bench.run_and_score(
        mechanism_ids = args.mechs,
        dataset_id    = args.dataset,
        fuel          = args.fuel,
        n_workers     = args.workers,
    )

    print("\n--- Score summary ---")
    print(scores_df.to_string(index=False))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    bench.print_ranking_table(args.dataset, regime=args.regime)
    bench.plot_IDT_comparison(args.mechs, args.dataset,
                              output_dir=args.out)
    print(f"✓ Plots saved to '{args.out}/'")
    return 0


def _add_compare_graph_cmd(sub) -> None:
    p = sub.add_parser("compare-graph",
                       help="Graph-level comparison: classify all species/reactions "
                            "and build union graph spec (Phase 3 → Phase 4 bridge)")
    p.add_argument("--mechs", "-m", nargs="+", type=int, required=True,
                   help="Mechanism DB ids to compare (≥ 2)")
    p.add_argument("--db",    required=True, help="SQLite database path")
    p.add_argument("--out",   default="graph_cmp_out",
                   help="Output directory for JSON files")
    p.add_argument("--tol-A",  type=float, default=0.01,
                   help="Relative tolerance on A for kinetic identity (default 1%%)")
    p.add_argument("--tol-n",  type=float, default=0.001,
                   help="Absolute tolerance on n (default 0.001)")
    p.add_argument("--tol-Ea", type=float, default=100.0,
                   help="Absolute tolerance on Ea [J/mol] (default 100)")
    p.add_argument("--serve", action="store_true",
                   help="KRUNAL_JULY_23.md point 7 -- after writing the comparison "
                        "JSON, launch the existing Dash NetworkVisualizer on "
                        "localhost so the comparison graph is actually viewable "
                        "(previously compare-graph only wrote JSON, no viewer link)")
    p.add_argument("--port", type=int, default=8050,
                   help="Port for --serve's Dash app (default 8050)")
    p.add_argument("--label", choices=["formula", "id"], default="formula",
                   help="--serve: reaction node label style (see `visualize --help`)")
    p.set_defaults(func=_cmd_compare_graph)


def _cmd_compare_graph(args) -> int:
    from rmip.analysis.graph_comparator import GraphComparator

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cmp    = GraphComparator(args.db,
                             tol_A=args.tol_A, tol_n=args.tol_n, tol_Ea=args.tol_Ea)
    result = cmp.compare(args.mechs)

    summary = result.summary_dict()
    print(f"\nGraph comparison: mechs {args.mechs}")
    print(f"  Species   — universal: {summary['species']['universal']}, "
          f"partial: {summary['species']['partial']}, "
          f"unique: {summary['species']['unique']}")
    print(f"  Reactions — identical: {summary['reactions']['shared_identical']}, "
          f"divergent: {summary['reactions']['shared_divergent']}, "
          f"unique: {summary['reactions']['unique']}")

    # Save full classification JSON
    cls_path = out / "graph_classification.json"
    with open(cls_path, "w") as fh:
        import json as _json
        _json.dump(result.to_dict(), fh, indent=2, default=str)
    print(f"✓ Classification → {cls_path}")

    # Build and save union graph spec
    spec      = cmp.build_union_spec(result)
    spec_path = out / "union_graph_spec.json"
    spec.save_json(str(spec_path))
    print(f"✓ Union graph spec → {spec_path}")
    print(f"  Verbatim reactions (unique):    {len(spec.verbatim_reactions)}")
    print(f"  Copy reactions (identical):     {len(spec.copy_reactions)}")
    print(f"  Fuse reactions (divergent→MSO): {len(spec.fuse_reactions)}")
    print(f"\n  → Hand union_graph_spec.json to M7 FusionEngine (Phase 4)")
    print(f"  → Hand fuse_params in that file to M9 DoEFramework (Phase 4)")

    if getattr(args, "serve", False):
        _serve_comparison_graph(args)

    return 0


def _serve_comparison_graph(args) -> None:
    """KRUNAL_JULY_23.md point 7. union_graph_spec.json itself has no
    species/reaction structural (stoichiometry/participant) data to build a
    network from -- GraphComparator's classification result doesn't carry
    it, only formula strings and canonical keys (see rmip.types.
    ReactionClassification/SpeciesClassification). The structural graph
    still comes from GraphBuilder against the DB (exactly what `rmip
    visualize --union` already does); this just launches that same, already
    -tested viewer for the mechanisms just compared instead of leaving the
    user with JSON files and no way to look at them.
    """
    from rmip.core.graph_builder import GraphBuilder
    from rmip.analysis.visualization import NetworkVisualizer

    view_mechs = args.mechs[:2]
    if len(args.mechs) > 2:
        print(f"\n  Note: --serve shows the first 2 of {len(args.mechs)} compared "
              "mechanisms side-by-side (NetworkVisualizer's panel limit); each "
              "panel still shows the FULL union's species/reactions, with the "
              "other mechanisms' nodes drawn as faint ghost nodes.")

    gb = GraphBuilder(args.db)
    graphs = gb.build_union_graphs(view_mechs)
    titles = [f"RMIP — Mechanism {mid} (compared: {args.mechs})" for mid in view_mechs]
    viz = NetworkVisualizer(graphs, title=titles)
    print(f"\n  → Graph viewer: http://localhost:{args.port}  (Ctrl+C to stop)")
    viz.run(port=args.port, label_mode=args.label)


def _add_resolve_glossary_cmd(sub) -> None:
    p = sub.add_parser(
        "resolve-glossary",
        help="Backfill species inchi_key / reaction canonical_key for "
             "mechanisms parsed without --glossary (KRUNAL_JULY_23.md point "
             "6 -- this is what 'Fused_Mech (fused, 0 sp / 0 rxn)' turned "
             "out to be: mechanisms fused with no glossary applied at parse "
             "time never match across mechanisms, so fusion degenerates "
             "into pure concatenation instead of real dedup)")
    p.add_argument("--mechs", "-m", nargs="+", type=int, required=True,
                   help="Mechanism DB ids to backfill")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--glossary", "-g", required=True,
                   help="Species SMILES glossary CSV/YAML")
    p.set_defaults(func=_cmd_resolve_glossary)


def _cmd_resolve_glossary(args) -> int:
    from rmip.core.glossary_backfill import backfill_glossary

    summary = backfill_glossary(args.db, args.mechs, args.glossary)
    print(f"\nGlossary backfill: mechs {args.mechs}, glossary={args.glossary}")
    for mid, s in summary.items():
        print(f"  mech {mid}:")
        print(f"    species   — resolved {s['species_resolved']}/"
              f"{s['species_unresolved_before']} newly, "
              f"{s['species_still_unresolved']} still unresolved "
              f"(of {s['species_total']} total)")
        print(f"    reactions — resolved {s['reactions_resolved']}/"
              f"{s['reactions_total_unresolved_before']} newly, "
              f"{s['reactions_still_unresolved']} still unresolved")
    print("\n  → Run compare-graph / fuse again to see the improved match rate.")
    return 0


# ===========================================================================
# Phase 3 (new) commands — fuse / optimize / rank / complete / export
# ===========================================================================

# ---------------------------------------------------------------------------
# rmip fuse
# ---------------------------------------------------------------------------

def _add_fuse_cmd(sub) -> None:
    p = sub.add_parser(
        "fuse",
        help="Build a fused mechanism from a prior graph-compare session",
    )
    p.add_argument("--db",         required=True, help="SQLite database path")
    p.add_argument("--mechanisms", "-m", nargs="+", type=int, required=True,
                   help="Source mechanism DB ids (must have been compare-graph'd first)")
    p.add_argument("--name",       required=True, help="Name for the fused mechanism")
    p.add_argument("--description", default="", help="Optional description")
    p.add_argument("--out",        default=None,
                   help="Directory to write mso_bounds.json + union_spec.json "
                        "(default: <name>_fusion/)")
    p.add_argument("--tol-A",  type=float, default=0.01)
    p.add_argument("--tol-n",  type=float, default=0.001)
    p.add_argument("--tol-Ea", type=float, default=100.0)
    p.set_defaults(func=_cmd_fuse)


def _cmd_fuse(args) -> int:
    import json as _json
    from rmip.analysis.graph_comparator import GraphComparator
    from rmip.analysis.fusion import FusionEngine
    from rmip.database.migrations import run_phase3_migrations
    import sqlite3

    out_dir = Path(args.out) if args.out else Path(f"{args.name}_fusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure Phase-3 schema is present
    with sqlite3.connect(args.db) as conn:
        run_phase3_migrations(conn)

    # Graph comparison
    cmp    = GraphComparator(args.db,
                             tol_A=args.tol_A, tol_n=args.tol_n, tol_Ea=args.tol_Ea)
    result = cmp.compare(args.mechanisms)

    summary = result.summary_dict()
    print(f"\nGraph comparison (mechs {args.mechanisms}):")
    print(f"  Species   — universal: {summary['species']['universal']}, "
          f"partial: {summary['species']['partial']}, "
          f"unique: {summary['species']['unique']}")
    print(f"  Reactions — identical: {summary['reactions']['shared_identical']}, "
          f"divergent: {summary['reactions']['shared_divergent']}, "
          f"unique: {summary['reactions']['unique']}")

    # Fusion
    engine       = FusionEngine(args.db)
    fusion_result = engine.fuse(result, name=args.name, description=args.description)

    # Export artefacts (before summary prints, so a downstream crash still
    # leaves the artefacts on disk instead of an orphan committed mechanism).
    spec_path   = out_dir / "union_spec.json"
    bounds_path = out_dir / "mso_bounds.json"
    engine.export_union_spec(fusion_result, str(spec_path))
    engine.export_mso_bounds(fusion_result, str(bounds_path))

    print(f"\nFusion complete → mechanism id={fusion_result.fused_mechanism_id}")
    print(f"  VERBATIM reactions: {fusion_result.n_verbatim}")
    print(f"  COPY reactions:     {fusion_result.n_copy}")
    print(f"  FUSE reactions:     {fusion_result.n_fused}")
    print(f"  Species — universal: {fusion_result.n_species_universal}, "
          f"partial: {fusion_result.n_species_partial}, "
          f"unique: {fusion_result.n_species_unique}")
    print(f"\n✓ Union spec    → {spec_path}")
    print(f"✓ MSO bounds    → {bounds_path}")
    print(f"\n  → Hand mso_bounds.json to `rmip optimize` to run OptimizationTool_v4")
    return 0


# ---------------------------------------------------------------------------
# rmip optimize
# ---------------------------------------------------------------------------

def _add_optimize_cmd(sub) -> None:
    p = sub.add_parser(
        "optimize",
        help="Prepare MSO inputs for a fused mechanism and (optionally) run "
             "OptimizationTool_v4.py, then apply the optimized parameters back to DB",
    )
    p.add_argument("mech_id", type=int, help="Fused mechanism DB id")
    p.add_argument("--db",      required=True, help="SQLite database path")
    p.add_argument("--bounds",  required=True,
                   help="mso_bounds.json from `rmip fuse`")
    p.add_argument("--targets", required=True,
                   help="JSON file with list of target condition dicts "
                        "(keys: T_K, P_atm, phi, fuel, oxidizer, observable, exp_value)")
    p.add_argument("--mso-tool", default=None,
                   help="Path to OptimizationTool_v4.py (if omitted, only prepares JSON "
                        "and prints the command to run manually)")
    p.add_argument("--work-dir", default="mso_work",
                   help="Directory for MSO I/O files (default: mso_work/)")
    p.add_argument("--objective", default="WMRE",
                   choices=["WMRE", "logRMSE", "unbiasedWMRE"],
                   help="Objective function for OptimizationTool_v4 (default: WMRE)")
    p.add_argument("--generations", type=int, default=500,
                   help="GA generations (default: 500)")
    p.add_argument("--population",  type=int, default=50,
                   help="GA population size (default: 50)")
    p.set_defaults(func=_cmd_optimize)


def _cmd_optimize(args) -> int:
    import json as _json
    from rmip.optimization.mso_interface import MSOInterface

    # Load targets
    with open(args.targets) as fh:
        target_conditions = _json.load(fh)
    if not isinstance(target_conditions, list):
        target_conditions = [target_conditions]

    # Load bounds
    with open(args.bounds) as fh:
        bounds_data = _json.load(fh)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    msi = MSOInterface(args.db, work_dir=str(work_dir))

    settings = {
        "objective":          args.objective,
        "ga_num_generations": args.generations,
        "ga_population_size": args.population,
    }

    input_path = msi.prepare(
        fused_mechanism_id=args.mech_id,
        mso_bounds_path=args.bounds,
        target_conditions=target_conditions,
        settings=settings,
    )
    print(f"\n✓ MSO input written → {input_path}")

    if args.mso_tool is None:
        print("\n  OptimizationTool_v4 path not provided — stopping after prepare.")
        print(f"  To run manually:")
        print(f"    python {args.mso_tool or '<path/to/OptimizationTool_v4.py>'} "
              f"--input {input_path}")
        print(f"  Then run:  rmip optimize {args.mech_id} --db {args.db} "
              f"--bounds {args.bounds} --targets {args.targets} "
              f"--mso-tool <path> --work-dir {args.work_dir}")
        return 0

    print(f"\nRunning OptimizationTool_v4 …")
    opt_result = msi.run(tool_path=args.mso_tool, input_path=str(input_path))
    print(f"  Status:    {opt_result.status}")

    if opt_result.status != "success":
        print(f"  ERROR: {opt_result.error_msg}")
        return 1

    obj_str = f"{opt_result.objective_final:.4f}" if opt_result.objective_final is not None else "N/A"
    print(f"  Objective (final): {obj_str}")

    new_id = msi.apply()
    print(f"\n✓ Optimized mechanism saved → DB id={new_id}")
    return 0


# ---------------------------------------------------------------------------
# rmip rank
# ---------------------------------------------------------------------------

def _add_rank_cmd(sub) -> None:
    p = sub.add_parser(
        "rank",
        help="Rank mechanisms by MRE against an experimental dataset",
    )
    p.add_argument("--db",         required=True, help="SQLite database path")
    p.add_argument("--mechanisms", "-m", nargs="+", type=int, required=True,
                   help="Mechanism DB ids to rank")
    p.add_argument("--dataset",    type=int, required=True,
                   help="exp_datasets id to evaluate against")
    p.add_argument("--sim-type",   default="IDT", choices=["IDT", "LFS"],
                   help="Simulation observable type (default: IDT)")
    p.add_argument("--out",        default=None,
                   help="Path to save ranking JSON report (optional)")
    p.set_defaults(func=_cmd_rank)


def _cmd_rank(args) -> int:
    from rmip.simulation.performance_ranker import PerformanceRanker

    ranker = PerformanceRanker(args.db)
    report = ranker.rank(
        mechanism_ids=args.mechanisms,
        dataset_id=args.dataset,
        sim_type=args.sim_type,
    )

    print(report.summary())

    if args.out:
        report.save(args.out)
        print(f"\n✓ Ranking report → {args.out}")

    best = report.best_per_regime()
    print("\nBest mechanism per regime:")
    for regime, score in best.items():
        mech_id = score.mechanism_id if score else None
        print(f"  {regime:<6} → mech id={mech_id}")

    return 0


# ---------------------------------------------------------------------------
# rmip complete
# ---------------------------------------------------------------------------

def _add_complete_cmd(sub) -> None:
    p = sub.add_parser(
        "complete",
        help="Append missing species/reactions from RMG-generated pathways into a mechanism",
    )
    p.add_argument("mech_id", type=int, help="Base mechanism DB id")
    p.add_argument("--db",          required=True, help="SQLite database path")
    p.add_argument("--fuel-smiles", required=True, help="Fuel SMILES string")
    p.add_argument("--fuel-name",   required=True, help="Fuel name (e.g. NC7H16)")
    p.add_argument("--oxidizer",    default="O2", help="Oxidizer species name (default: O2)")
    p.add_argument("--T",           type=float, nargs="+", default=[800.0, 1200.0],
                   help="Temperature(s) for RMG reactor (K)")
    p.add_argument("--P",           type=float, nargs="+", default=[10.0],
                   help="Pressure(s) for RMG reactor (atm)")
    p.add_argument("--phi",         type=float, default=1.0, help="Equivalence ratio")
    p.add_argument("--rmg",         default="rmg.py",
                   help="Path to RMG-Py runner script (default: rmg.py on PATH)")
    p.add_argument("--new-name",    default=None,
                   help="If given, clone the base mechanism before appending "
                        "(non-destructive); otherwise modify in-place")
    p.add_argument("--out",         default="rmg_work",
                   help="Working directory for RMG I/O (default: rmg_work/)")
    p.set_defaults(func=_cmd_complete)


def _cmd_complete(args) -> int:
    from rmip.generation.rmg_interface import RMGInterface

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    iface = RMGInterface(args.db)

    # Stoichiometric fuel-in-air fraction (~0.0585) scaled by phi as a first-order
    # reactor-condition knob for RMG's exploration (RMG uses it only to seed the
    # reaction network, not for a quantitative simulation).
    fuel_mf = min(0.0585 * max(args.phi, 0.01), 0.3)

    print(f"\nGenerating RMG pathways for {args.fuel_name} …")
    rmg_out_dir = iface.generate(
        fuel_smiles=args.fuel_smiles,
        fuel_name=args.fuel_name,
        fuel_mole_frac=fuel_mf,
        T_lo_K=min(args.T),
        T_hi_K=max(args.T),
        P_atm=args.P[0],
        output_dir=str(out_dir),
        rmg_path=args.rmg,
    )

    # RMG writes its CHEMKIN mechanism under <out_dir>/chemkin/.
    chemkin_dir = Path(rmg_out_dir) / "chemkin"
    chem_path = next(
        (chemkin_dir / n for n in ("chem.inp", "chem_annotated.inp")
         if (chemkin_dir / n).is_file()),
        None,
    )
    if chem_path is None:
        print(f"\n✗ RMG produced no CHEMKIN output under {chemkin_dir}", file=sys.stderr)
        return 1
    # RMG embeds thermo in chem.inp; pass a separate therm.dat only if one exists.
    thermo_arg = str(chemkin_dir / "therm.dat") if (chemkin_dir / "therm.dat").is_file() else None

    print(f"  Loading RMG output → {chem_path}")
    rmg_mech_id = iface.load_rmg_output(
        chemkin_path=str(chem_path),
        thermo_path=thermo_arg,
        db_name=f"RMG_{args.fuel_name}",
    )
    print(f"  RMG mechanism loaded → DB id={rmg_mech_id}")

    pathways = iface.find_missing_pathways(args.mech_id, rmg_mech_id)
    print(f"\nMissing pathways vs base mechanism id={args.mech_id}:")
    print(f"  Missing species:   {pathways.n_missing_species}")
    print(f"  Missing reactions: {pathways.n_missing_reactions}")

    if not pathways.missing_inchi_keys and not pathways.missing_canonical_keys:
        print("  ✓ Base mechanism already complete — nothing to append.")
        return 0

    target_id = iface.append_to_mechanism(
        base_mech_id=args.mech_id,
        rmg_mech_id=rmg_mech_id,
        new_name=args.new_name,
    )
    mode = "clone" if args.new_name else "in-place"
    print(f"\n✓ Pathways appended ({mode}) → mechanism DB id={target_id}")
    return 0


# ---------------------------------------------------------------------------
# rmip export
# ---------------------------------------------------------------------------

def _add_export_cmd(sub) -> None:
    p = sub.add_parser(
        "export",
        help="Export a mechanism from the DB to CHEMKIN-II or Cantera YAML format",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id to export")
    p.add_argument("--db",     required=True, help="SQLite database path")
    p.add_argument("--output", required=True,
                   help="Output file path (.inp for CHEMKIN, .yaml for Cantera YAML)")
    p.add_argument("--format", default=None, choices=["chemkin", "yaml"],
                   help="Output format: 'chemkin' (default) or 'yaml' (Cantera YAML)")
    p.add_argument("--thermo", default=None,
                   help="Separate thermo output file (CHEMKIN only)")
    p.add_argument("--no-embed-thermo", action="store_true",
                   help="Write thermo to a separate .therm file alongside --output (CHEMKIN only)")
    p.add_argument("--use-original-units", action="store_true", default=True,
                   help="Use A_orig/Ea_orig from DB if available (CHEMKIN only, default: True)")
    p.add_argument("--ck2yaml",  default=None,
                   help="Path to ck2yaml executable; if given, also convert to "
                        "Cantera YAML after writing CHEMKIN")
    p.set_defaults(func=_cmd_export)


def _cmd_export(args) -> int:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect format from extension if not specified
    fmt = args.format
    if fmt is None:
        fmt = "yaml" if out_path.suffix.lower() in (".yaml", ".yml") else "chemkin"

    if fmt == "yaml":
        from rmip.io.cantera_writer import CanteraYAMLWriter
        writer = CanteraYAMLWriter(args.db)
        writer.write(
            mechanism_id=args.mech_id,
            output_yaml=str(out_path),
            uncertainty_bands=None,
            embed_uncertainty=False,
        )
        print(f"\n✓ Cantera YAML → {out_path}")
        return 0

    # CHEMKIN format (default)
    from rmip.io.chemkin_writer import ChemkinWriter

    thermo_path = args.thermo
    if args.no_embed_thermo and thermo_path is None:
        thermo_path = str(out_path.with_suffix(".therm"))

    writer = ChemkinWriter(args.db, use_original_units=args.use_original_units)
    writer.write(
        mechanism_id=args.mech_id,
        output_path=str(out_path),
        thermo_path=thermo_path,
        include_thermo_in_mech=not args.no_embed_thermo,
    )
    print(f"\n✓ CHEMKIN mechanism → {out_path}")
    if thermo_path:
        print(f"✓ Thermo file      → {thermo_path}")

    if args.ck2yaml:
        yaml_path = str(out_path.with_suffix(".yaml"))
        writer.convert_to_cantera_yaml(
            chemkin_path=str(out_path),
            thermo_path=thermo_path,
            output_yaml=yaml_path,
            ck2yaml_exe=args.ck2yaml,
        )
        print(f"✓ Cantera YAML     → {yaml_path}")

    return 0


# ===========================================================================
# Phase 4 commands
# ===========================================================================

# ---------------------------------------------------------------------------
# rmip unify-plan
# ---------------------------------------------------------------------------

def _add_unify_plan_cmd(sub) -> None:
    p = sub.add_parser(
        "unify-plan",
        help="Classify reactions across mechanisms and emit editable decisions.csv",
    )
    p.add_argument("--db",       required=True, help="SQLite database path")
    p.add_argument("--mechanisms", "-m", nargs="+", type=int, required=True,
                   help="Mechanism DB ids to unify")
    p.add_argument("--mode",     required=True, choices=["anchored", "optimize_all"],
                   help="'anchored' (keep one mechanism as scaffold) or 'optimize_all'")
    p.add_argument("--anchor",   type=int, default=None,
                   help="Anchor mechanism DB id (required when --mode anchored)")
    p.add_argument("--manifest", default="decisions.csv",
                   help="Output path for the decisions CSV (default: decisions.csv)")
    p.set_defaults(func=_cmd_unify_plan)


def _cmd_unify_plan(args) -> int:
    from rmip.analysis.unified_tree import UnifiedTreeBuilder
    builder = UnifiedTreeBuilder(args.db)
    plan = builder.plan(
        mechanism_ids  = args.mechanisms,
        mode           = args.mode,
        anchor_mech_id = args.anchor,
        manifest_path  = args.manifest,
    )

    counts = {"IN_ANCHOR": 0, "PATCH_SINGLE": 0, "PATCH_MULTI": 0,
              "SHARED": 0, "UNIQUE": 0}
    for d in plan.decisions:
        counts[d.status] = counts.get(d.status, 0) + 1

    print(f"\nUnified tree plan ({plan.mode} mode):")
    print(f"  Mechanisms   : {args.mechanisms}")
    if plan.anchor_mech_id is not None:
        print(f"  Anchor       : {plan.anchor_mech_id}")
    print(f"  Total rxns   : {len(plan.decisions)}")
    for status, cnt in counts.items():
        if cnt:
            print(f"  {status:<15} : {cnt}")
    print(f"\n✓ Manifest written → {args.manifest}")
    print("  Review and edit PATCH_MULTI / SHARED rows as needed, then run:")
    print(f"    rmip unify-build --db {args.db} --manifest {args.manifest} --name <name>")
    return 0


# ---------------------------------------------------------------------------
# rmip unify-build
# ---------------------------------------------------------------------------

def _add_unify_build_cmd(sub) -> None:
    p = sub.add_parser(
        "unify-build",
        help="Build the unified mechanism from a (optionally edited) decisions manifest",
    )
    p.add_argument("--db",       required=True, help="SQLite database path")
    p.add_argument("--manifest", default="decisions.csv",
                   help="Path to decisions CSV (default: decisions.csv)")
    p.add_argument("--name",     default="unified",
                   help="Name for the unified mechanism (default: unified)")
    p.add_argument("--description", default="",
                   help="Description for the unified mechanism")
    p.add_argument("--tmin",     type=float, default=500.0,
                   help="Temperature grid minimum (K, default: 500)")
    p.add_argument("--tmax",     type=float, default=2500.0,
                   help="Temperature grid maximum (K, default: 2500)")
    p.add_argument("--dt",       type=float, default=20.0,
                   help="Temperature grid step (K, default: 20)")
    p.add_argument("--pref-atm", type=float, default=1.0,
                   help="Reference pressure for PLOG/falloff bands (atm, default: 1.0)")
    p.add_argument("--use-cantera-falloff", action="store_true",
                   help="Use Cantera to evaluate Troe/Chebyshev contributions (requires Cantera)")
    p.add_argument("--out",      default="unify_out",
                   help="Output directory for CSV/YAML/JSON artefacts (default: unify_out)")
    p.set_defaults(func=_cmd_unify_build)


def _cmd_unify_build(args) -> int:
    from rmip.types import TemperatureGrid
    from rmip.analysis.unified_tree import UnifiedTreeBuilder

    tgrid = TemperatureGrid(T_min=args.tmin, T_max=args.tmax, dT=args.dt)
    builder = UnifiedTreeBuilder(args.db)

    result = builder.build(
        manifest_path          = args.manifest,
        name                   = args.name,
        description            = args.description,
        tgrid                  = tgrid,
        out_dir                = args.out,
        P_ref_atm              = args.pref_atm,
        use_cantera_for_falloff = args.use_cantera_falloff,
    )

    print(f"\n{result.summary()}")
    print(f"\n✓ Unified mechanism → DB id={result.unified_mechanism_id}")
    print(f"✓ Uncertainty bands  → {result.bands_csv_path}")
    print(f"✓ Summary CSV        → {result.summary_csv_path}")
    if result.yaml_path:
        print(f"✓ Cantera YAML       → {result.yaml_path}")
    print(f"✓ MSO bounds         → {result.mso_bounds_path}")
    print(f"✓ Provenance         → {result.provenance_path}")
    print(f"\n  → Run `rmip optimize {result.unified_mechanism_id} --bounds {result.mso_bounds_path} ...`")
    return 0


# ---------------------------------------------------------------------------
# rmip predict-missing
# ---------------------------------------------------------------------------

def _add_predict_missing_cmd(sub) -> None:
    p = sub.add_parser(
        "predict-missing",
        help="Identify species/reactions missing from the unified mechanism → RMG seed",
    )
    p.add_argument("--db",       required=True, help="SQLite database path")
    p.add_argument("--unified",  type=int, required=True,
                   help="Unified mechanism DB id")
    p.add_argument("--mechanisms", "-m", nargs="+", type=int, required=True,
                   help="Contributing mechanism DB ids used to build the unified mechanism")
    p.add_argument("--out",      default="rmg_seed",
                   help="Output directory for missing_report.json and seed.json (default: rmg_seed)")
    p.set_defaults(func=_cmd_predict_missing)


def _cmd_predict_missing(args) -> int:
    from rmip.generation.missing_species_predictor import MissingSpeciesPredictor

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = MissingSpeciesPredictor(args.db)
    report    = predictor.predict(
        unified_mech_id       = args.unified,
        contributing_mech_ids = args.mechanisms,
    )

    # Write missing_report.json
    report_path = str(out_dir / "missing_report.json")
    import json as _json
    with open(report_path, "w") as fh:
        _json.dump(report.to_dict(), fh, indent=2)

    # Write RMG seed
    seed_path = str(out_dir / "rmg_seed.json")
    seed = predictor.to_rmg_seed(report, seed_path)

    print(f"\nMissing species report for unified mech id={args.unified}:")
    print(f"  Cross-mech gap species   : {len(report.gap_species)}")
    print(f"  Cross-mech gap reactions : {len(report.gap_reactions)}")
    print(f"  Dead-end species         : {len(report.dead_end_species)}")
    print(f"  Orphan-reaction species  : {len(report.orphan_species)}")
    print(f"  RMG seed SMILES          : {len(seed.seed_smiles)}")
    print(f"  RMG must-include SMILES  : {len(seed.must_include_smiles)}")
    print(f"\n✓ Missing report → {report_path}")
    print(f"✓ RMG seed       → {seed_path}")
    print(f"\n  → Run: rmip complete {args.unified} --db {args.db} \\")
    print(f"         --fuel-smiles <SMILES> --fuel-name <NAME> --rmg <path>")
    return 0


# ---------------------------------------------------------------------------
# rmip reduce   (Phase 5B)
# ---------------------------------------------------------------------------

def _add_reduce_cmd(sub) -> None:
    p = sub.add_parser(
        "reduce",
        help="DRG/DRGEP/PFA/DRGPSA/DRGSA mechanism reduction sweep → JSON (Phase 5B)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id to reduce")
    p.add_argument("--db",      required=True, help="SQLite database path")
    p.add_argument("--method",  default="DRG",
                   choices=["DRG", "DRGEP", "PFA", "DRGPSA", "DRGSA"],
                   help="Reduction algorithm (default: DRG)")
    p.add_argument("--target",  required=True,
                   help='Comma-separated seed species (e.g. "H2,O2,H2O")')
    p.add_argument("--eps-max", type=float, default=0.5,
                   help="Maximum ε threshold (default: 0.5)")
    p.add_argument("--eps-steps", type=int, default=51,
                   help="Number of ε steps in [0, eps-max] (default: 51)")
    p.add_argument("--out",     default=None,
                   help="Output JSON path for reduction steps "
                        "(default: <method>_<mech_id>_steps.json)")
    p.add_argument("--no-db-save", action="store_true",
                   help="Do not insert a row in the reduction_runs table")
    p.add_argument("--mech-path",  default=None,
                   help="CHEMKIN/YAML file path — required for Cantera rate evaluation; "
                        "if omitted, falls back to flux_score on graph or unweighted DRG")
    p.add_argument("--T",  type=float, default=None, dest="T_K",
                   help="Temperature [K] for Cantera rate evaluation")
    p.add_argument("--P",  type=float, default=None, dest="P_atm",
                   help="Pressure [atm] for Cantera rate evaluation")
    p.add_argument("--phi", type=float, default=None,
                   help="Equivalence ratio for Cantera rate evaluation")
    p.add_argument("--fuel", default=None,
                   help="Fuel species name for Cantera rate evaluation")
    p.add_argument("--sa-json", default=None, dest="sa_json",
                   help="Pre-computed SA JSON file (for DRGPSA/DRGSA; "
                        "format: CanteraSensitivityRunner output)")
    p.add_argument("--condition", action="append", default=None,
                   metavar="T_K,P_atm,phi,fuel",
                   help="Repeatable: one experimental condition as "
                        "'T_K,P_atm,phi,fuel' (e.g. '1000,1.0,1.0,H2:1.0'). "
                        "When given one or more times, runs the "
                        "multi-condition union-reduce workflow instead of "
                        "the single-condition path below: the reducer is "
                        "run once per --condition and the retained "
                        "species/reactions are unioned at each matching ε "
                        "(see rmip.reduction.multi_condition). All "
                        "conditions use the standard air oxidizer "
                        "'O2:1.0, N2:3.76'. Requires --mech-path for real "
                        "Cantera rate evaluation per condition (otherwise "
                        "every condition degrades to the same unweighted "
                        "fallback and the union is a no-op).")
    p.set_defaults(func=_cmd_reduce)


def _cmd_reduce(args) -> int:
    from rmip.core.graph_builder import GraphBuilder
    from rmip.database.migrations import _table_exists, run_phase5b_migrations
    import sqlite3

    # Ensure reduction_runs table exists
    with sqlite3.connect(args.db) as conn:
        if not _table_exists(conn, "reduction_runs"):
            run_phase5b_migrations(conn)

    # Build graph (with positions so flux_score check works; without spring for speed)
    gb = GraphBuilder(args.db)
    G  = gb.build_graph(args.mech_id, include_reverse_edges=False, include_spring=False)

    # Parse target species
    target_species = [s.strip() for s in args.target.split(",") if s.strip()]

    # Build epsilon range
    n     = max(2, args.eps_steps)
    eps_max = max(0.0, args.eps_max)
    eps_range = [round(i * eps_max / (n - 1), 8) for i in range(n)]

    # --condition (repeatable): multi-condition union-reduce workflow
    # (FEATURE_GAPS_FROM_MONOLITH.md ranked candidate #5). Parsed here, but
    # dispatched in its own branch below -- when --condition is never given,
    # the single-condition path underneath runs completely unchanged.
    conditions_multi = None
    if getattr(args, "condition", None):
        from rmip.types import SimCondition as _SimConditionMulti
        conditions_multi = []
        for c_str in args.condition:
            c_parts = [p.strip() for p in c_str.split(",")]
            if len(c_parts) != 4:
                print(f"--condition must be 'T_K,P_atm,phi,fuel', got: {c_str!r}")
                return 1
            c_T, c_P, c_phi, c_fuel = c_parts
            conditions_multi.append(_SimConditionMulti(
                T_K=float(c_T), P_atm=float(c_P), phi=float(c_phi), fuel=c_fuel,
                oxidizer="O2:1.0, N2:3.76",
            ))

    if conditions_multi:
        from rmip.reduction.multi_condition import reduce_multi_condition

        sa_json = getattr(args, "sa_json", None)
        print(f"Running {args.method} MULTI-CONDITION reduction on mech id={args.mech_id} "
              f"with {len(target_species)} seed species, {n} ε steps, "
              f"{len(conditions_multi)} condition(s) ...")
        if sa_json:
            print(f"  SA data: {sa_json}")

        result = reduce_multi_condition(
            mech_id        = args.mech_id,
            db_path        = args.db,
            G              = G,
            target_species = target_species,
            conditions     = conditions_multi,
            method         = args.method,
            epsilon_range  = eps_range,
            mech_path      = args.mech_path,
            sa_json_path   = sa_json,
            save_db        = not args.no_db_save,
        )
    else:
        # Build condition for Cantera (if all params provided)
        condition = None
        if all(x is not None for x in [args.mech_path, args.T_K, args.P_atm, args.phi, args.fuel]):
            from rmip.types import SimCondition
            condition = SimCondition(
                T_K      = args.T_K,
                P_atm    = args.P_atm,
                phi      = args.phi,
                fuel     = args.fuel,
                oxidizer = "O2:1.0, N2:3.76",
            )

        # Choose reducer
        sa_json = getattr(args, "sa_json", None)
        if args.method == "DRG":
            from rmip.reduction.drg import DRGReducer
            reducer = DRGReducer(args.db)
        elif args.method == "DRGEP":
            from rmip.reduction.drgep import DRGEPReducer
            reducer = DRGEPReducer(args.db)
        elif args.method == "PFA":
            from rmip.reduction.pfa import PFAReducer
            reducer = PFAReducer(args.db)
        elif args.method == "DRGPSA":
            from rmip.reduction.drgpsa import DRGPSAReducer
            reducer = DRGPSAReducer(args.db, sa_json_path=sa_json)
        else:  # DRGSA
            from rmip.reduction.drgpsa import DRGSAReducer
            reducer = DRGSAReducer(args.db, sa_json_path=sa_json)

        print(f"Running {args.method} reduction on mech id={args.mech_id} "
              f"with {len(target_species)} seed species, {n} ε steps ...")
        if sa_json:
            print(f"  SA data: {sa_json}")

        result = reducer.reduce(
            mech_id        = args.mech_id,
            G              = G,
            target_species = target_species,
            epsilon_range  = eps_range,
            condition      = condition,
            mech_path      = args.mech_path,
            save_db        = not args.no_db_save,
        )

    out_path = args.out or f"{args.method.lower()}_{args.mech_id}_steps.json"
    result.save(out_path)

    step0, stepN = result.steps[0], result.steps[-1]
    print(f"\n{args.method} reduction complete:")
    print(f"  Full mechanism : {step0.n_species} species / {step0.n_reactions} reactions")
    print(f"  ε = {stepN.threshold:.3f} retained : {stepN.n_species} species / "
          f"{stepN.n_reactions} reactions")
    print(f"\n  Steps saved → {out_path}")
    if result.run_id:
        print(f"  DB reduction_runs id = {result.run_id}")
    print(f"\n  Visualize with:")
    print(f"    python rmip_cli.py visualize {args.mech_id} --db {args.db} "
          f"--reduction-json {out_path}")
    print(f"    python rmip_cli.py reduce-viz {args.mech_id} --db {args.db} "
          f"--reduction-json {out_path}")
    return 0


# ---------------------------------------------------------------------------
# rmip gsa-run   (ZERORK_GSA_PLAN.md Phase 3 — Sobol/Morris via SALib + Zero-RK)
# ---------------------------------------------------------------------------

def _add_gsa_run_cmd(sub) -> None:
    p = sub.add_parser(
        "gsa-run",
        help="Sobol/Morris GSA over reaction A-factors via SALib + Zero-RK's "
             "perturbAFactorGSA_mpi.x (ZERORK_GSA_PLAN.md Phase 3)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id to screen")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--method", default="sobol", choices=["sobol", "morris"],
                   help="GSA method (default: sobol)")
    p.add_argument("--n-samples", type=int, default=256, dest="n_samples",
                   help="SALib base sample count N (Sobol: total runs = "
                        "N*(D+2); Morris: total runs = N*(D+1); D = number "
                        "of reactions screened) (default: 256)")
    p.add_argument("--reactions", default="all",
                   help='Comma-separated reaction DB ids to screen, or '
                        '"all" for every reaction in the mechanism '
                        '(default: all)')
    p.add_argument("--factor-bound", type=float, default=2.0, dest="factor_bound",
                   help="Symmetric log-uniform A-factor multiplier bound "
                        "(2.0 -> each reaction's A-factor sampled in "
                        "[A/2, 2A]) (default: 2.0)")
    p.add_argument("--zerork-install-prefix", required=True,
                   dest="zerork_install_prefix",
                   help="Zero-RK install prefix (must contain bin/perturbAFactorGSA_mpi.x)")
    p.add_argument("--T", type=float, required=True, dest="T_K", help="Temperature [K]")
    p.add_argument("--P", type=float, required=True, dest="P_atm", help="Pressure [atm]")
    p.add_argument("--phi", type=float, required=True, help="Equivalence ratio")
    p.add_argument("--fuel", required=True, help="Fuel composition (e.g. \"H2:1.0\")")
    p.add_argument("--oxidizer", default="O2:1.0, N2:3.76",
                   help='Oxidizer composition (default: "O2:1.0, N2:3.76")')
    p.add_argument("--idt-temp-delta", type=float, default=400.0, dest="idt_temp_delta",
                   help="Ignition-delay temperature-rise threshold [K] (default: 400.0)")
    p.add_argument("--n-mpi-ranks", type=int, default=4, dest="n_mpi_ranks",
                   help="MPI ranks for perturbAFactorGSA_mpi.x (default: 4)")
    p.add_argument("--seed", type=int, default=None, help="SALib RNG seed (default: none)")
    p.add_argument("--work-dir", default=None, dest="work_dir",
                   help="Working directory for generated files "
                        "(default: gsa_work_<mech_id>_<method>)")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: <method>_<mech_id>_gsa.json)")
    p.add_argument("--no-db-save", action="store_true",
                   help="Do not insert rows in sensitivity_runs/sobol_indices/design_matrices")
    p.add_argument("--timeout", type=float, default=None,
                   help="Timeout [s] for the Zero-RK MPI batch run (default: none)")
    p.set_defaults(func=_cmd_gsa_run)


def _cmd_gsa_run(args) -> int:
    from rmip.simulation.zerork_gsa import ZerorkGSARunner
    from rmip.types import GSARunConfig

    reaction_ids = None
    if args.reactions.strip().lower() != "all":
        reaction_ids = [int(s.strip()) for s in args.reactions.split(",") if s.strip()]

    config = GSARunConfig(
        mechanism_id=args.mech_id, method=args.method, n_samples=args.n_samples,
        reaction_ids=reaction_ids, factor_bound=args.factor_bound,
        T_K=args.T_K, P_atm=args.P_atm, phi=args.phi, fuel=args.fuel,
        oxidizer=args.oxidizer, idt_temp_delta=args.idt_temp_delta,
        n_mpi_ranks=args.n_mpi_ranks, seed=args.seed,
    )
    work_dir = args.work_dir or f"gsa_work_{args.mech_id}_{args.method}"
    n_reactions = len(reaction_ids) if reaction_ids else "all"
    print(f"Running {args.method} GSA on mech id={args.mech_id}, "
          f"{n_reactions} reaction(s) screened, base N={args.n_samples} ...")

    runner = ZerorkGSARunner(args.db, args.zerork_install_prefix)
    result = runner.run_screening(
        config, work_dir=work_dir, timeout=args.timeout, save_db=not args.no_db_save,
    )

    out_path = args.out or f"{args.method}_{args.mech_id}_gsa.json"
    result.save(out_path)

    print(f"\n{args.method} GSA complete:")
    print(f"  Parameters screened : {len(result.parameter_names)}")
    print(f"  Total samples run   : {result.n_samples} "
          f"({result.n_failed_samples} failed/non-ignition, mean-imputed)")
    top_key = "S1" if args.method == "sobol" else "mu_star"
    top_values = getattr(result, top_key) or {}
    ranked = sorted(top_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print(f"  Top {len(ranked)} by |{top_key}|:")
    for name, value in ranked:
        print(f"    {value:+.4g}  {name}")
    print(f"\n  Results saved → {out_path}")
    if result.run_id:
        print(f"  DB sensitivity_runs id = {result.run_id}")
    return 0


# ---------------------------------------------------------------------------
# rmip reduce-viz   (Phase 5B-extended — coefficient & trim visualization)
# ---------------------------------------------------------------------------

def _add_reduce_viz_cmd(sub) -> None:
    p = sub.add_parser(
        "reduce-viz",
        help="Interactive coefficient graph + trim animation for a reduction result (Phase 5B)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id that was reduced")
    p.add_argument("--db",             required=True,  help="SQLite database path")
    p.add_argument("--reduction-json", required=True,  dest="reduction_json",
                   help="JSON file produced by `rmip reduce`")
    p.add_argument("--mode",           default="both",
                   choices=["coeff", "trim", "both"],
                   help="coeff=coefficient network only, trim=trim animation only, "
                        "both=combined (default: both)")
    p.add_argument("--port",           type=int, default=8055,
                   help="Dash server port (default: 8055; trim uses port+1 if mode=both)")
    p.add_argument("--export-html",    default=None, dest="export_html",
                   help="If given, export static HTML snapshots to this path "
                        "instead of launching the Dash server")
    p.add_argument("--threshold",      type=float, default=0.05,
                   help="Initial r_AB display threshold for network views (default: 0.05)")
    p.set_defaults(func=_cmd_reduce_viz)


def _cmd_reduce_viz(args) -> int:
    from rmip.types import ReductionResult
    from rmip.core.graph_builder import GraphBuilder

    print(f"Loading reduction result from {args.reduction_json} …")
    result = ReductionResult.load(args.reduction_json)

    print(f"Building mechanism graph for mech_id={args.mech_id} …")
    gb = GraphBuilder(args.db)
    G  = gb.build_graph(args.mech_id, include_reverse_edges=False,
                        include_spring=True, include_3d=True)

    mode = args.mode

    # ── Coefficient viz ────────────────────────────────────────────────────
    if mode in ("coeff", "both"):
        if result.r_AB is None:
            print("WARNING: reduction result has no r_AB matrix — "
                  "re-run `reduce` with a DRG/DRGPSA method to generate it.")
        else:
            from rmip.reduction.coefficient_viz import CoefficientVisualizer
            coeff_viz = CoefficientVisualizer(result, G, threshold_default=args.threshold)

            if args.export_html:
                from pathlib import Path
                base = Path(args.export_html).with_suffix("")
                coeff_path = str(base) + "_coeff.html"
                coeff_viz.export_html(coeff_path)
                print(f"Coefficient viz exported → {coeff_path}")
            elif mode == "both":
                import multiprocessing
                proc = multiprocessing.Process(
                    target=coeff_viz.run,
                    kwargs={"port": args.port, "debug": False},
                    daemon=True,
                )
                proc.start()
                print(f"Coefficient viz running at http://localhost:{args.port}")
            else:
                print(f"Launching coefficient visualizer at http://localhost:{args.port}")
                coeff_viz.run(port=args.port)
                return 0

    # ── Trim viz ───────────────────────────────────────────────────────────
    if mode in ("trim", "both"):
        from rmip.reduction.trim_viz import TrimVisualizer
        trim_viz = TrimVisualizer(result, G)
        trim_port = args.port + 1 if mode == "both" else args.port

        if args.export_html:
            from pathlib import Path
            base = Path(args.export_html).with_suffix("")
            trim_path = str(base) + "_trim.html"
            trim_viz.export_html(trim_path)
            print(f"Trim viz exported → {trim_path}")
        else:
            print(f"Launching trim visualizer at http://localhost:{trim_port}")
            trim_viz.run(port=trim_port)

    return 0


# ===========================================================================
# Phase 6 commands (Future.md features 1-7)
# ===========================================================================

_SOLVER_ALIASES = {"cantera": "cantera", "chemkin": "CHEMKIN_PRO", "flamemaster": "FlameMaster"}


def _add_reverse_scan_cmd(sub) -> None:
    p = sub.add_parser(
        "reverse-scan",
        help="Flag reactions whose Keq-implied reverse rate exceeds the "
             "collision limit (Future.md pt. 1)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id to scan")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--t-grid", default=None,
                   help="Comma-separated T grid [K] (default: 600,900,1200,1500,1800,2200)")
    p.add_argument("--threshold", type=float, default=1000.0,
                   help="Flag when k_rev/collision_limit exceeds this factor (default: 1000)")
    p.add_argument("--out", default=None, help="Output JSON path (default: reverse_scan_<id>.json)")
    p.set_defaults(func=_cmd_reverse_scan)


def _cmd_reverse_scan(args) -> int:
    from rmip.analysis.reverse_rate_scanner import ReverseRateScanner

    _ensure_phase6_schema(args.db)
    t_grid = [float(x) for x in args.t_grid.split(",")] if args.t_grid else None
    scanner = ReverseRateScanner(args.db, T_grid_K=t_grid, exceedance_threshold=args.threshold)
    flags = scanner.scan(args.mech_id)
    scanner.store(args.mech_id, flags)

    flagged = [f for f in flags if f.flagged]
    print(f"\nReverse-rate scan — mechanism_id={args.mech_id}")
    print(f"  {len(flags)} (reaction, T) pairs evaluated, {len(flagged)} flagged "
          f"(k_rev/collision_limit > {args.threshold:g})")
    if flagged:
        print(f"\n  {'Reaction':<40} {'T [K]':>8} {'k_rev':>12} {'exceedance':>12}")
        print("  " + "-" * 76)
        for f in sorted(flagged, key=lambda x: -x.exceedance_factor)[:20]:
            print(f"  {f.formula_string[:40]:<40} {f.T_K:>8.0f} {f.k_rev:>12.3e} {f.exceedance_factor:>12.1f}")

    out_path = args.out or f"reverse_scan_{args.mech_id}.json"
    with open(out_path, "w") as fh:
        json.dump([f.to_dict() for f in flags], fh, indent=2)
    print(f"\n  Full results saved -> {out_path}")
    return 0


def _add_report_cmd(sub) -> None:
    p = sub.add_parser(
        "report",
        help="Mechanism quality score, and diff when two mechanisms given "
             "(Future.md pt. 2+4, merged)",
    )
    p.add_argument("mech_ids", type=int, nargs="+", help="1 or 2 mechanism DB ids")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--atct-ref", default=None,
                   help="Optional local ATcT/JANAF reference JSON: {inchi_key_or_name: Hf298_kJ_mol}")
    p.add_argument("--run-ntc", action="store_true",
                   help="Also run an NTC sweep per mechanism to populate ntc_coverage_score "
                        "(requires --fuel)")
    p.add_argument("--fuel", default=None, help="Fuel composition string for --run-ntc")
    p.add_argument("--phi", type=float, default=1.0)
    p.add_argument("--pressure", type=float, default=1.0, dest="P_atm")
    p.add_argument("--t-min", type=float, default=650.0)
    p.add_argument("--t-max", type=float, default=1300.0)
    p.add_argument("--n-points", type=int, default=15)
    p.add_argument("--out", default=None, help="HTML report path (default: mechanism_report.html)")
    p.set_defaults(func=_cmd_report)


def _cmd_report(args) -> int:
    from rmip.analysis.mechanism_report import MechanismReport

    if len(args.mech_ids) not in (1, 2):
        print("Error: report accepts 1 or 2 mechanism ids")
        return 1
    _ensure_phase6_schema(args.db)

    ntc_reports = {}
    if args.run_ntc:
        if not args.fuel:
            print("Error: --run-ntc requires --fuel")
            return 1
        from rmip.analysis.ntc_detector import NTCDetector
        for mid in args.mech_ids:
            det = NTCDetector(mid, args.db)
            rep = det.detect(fuel=args.fuel, phi=args.phi, P_atm=args.P_atm,
                              T_min=args.t_min, T_max=args.t_max, n_points=args.n_points)
            det.store(rep)
            ntc_reports[mid] = rep
            print(f"  NTC sweep mech_id={mid}: has_ntc={rep.has_ntc}, "
                  f"onset={rep.T_NTC_onset}")

    rep_engine = MechanismReport(args.db, atct_reference_path=args.atct_ref)
    result = rep_engine.report(args.mech_ids, ntc_reports=ntc_reports)

    print(f"\nMechanism Report — ids {args.mech_ids}")
    for mid, q in result["quality"].items():
        print(f"\n  [{q.mechanism_name}] (id={mid})")
        print(f"    Composite score      : {q.composite_score:.1f} / 100")
        print(f"    SMILES resolution    : {q.smiles_resolution_pct:.1f}%")
        print(f"    Thermo consistency   : {q.thermo_consistency_score:.3f}")
        print(f"    Rate physical bounds : {q.rate_bounds_score:.3f} "
              f"({q.n_reverse_rate_flags} flagged)")
        if q.hf_agreement_score is not None:
            print(f"    Hf agreement (ATcT)  : {q.hf_agreement_score:.3f}")
        if q.ntc_coverage_score is not None:
            print(f"    NTC coverage         : {q.ntc_coverage_score:.3f}")

    diff = result.get("diff")
    if diff is not None:
        print(f"\n  Diff: {diff.mechanism_name_a} -> {diff.mechanism_name_b}")
        print(f"    Species  added={len(diff.species_added)} removed={len(diff.species_removed)} "
              f"shared={diff.n_species_shared}")
        print(f"    Reactions added={len(diff.reactions_added)} removed={len(diff.reactions_removed)} "
              f"shared={diff.n_reactions_shared}")

    out_path = args.out or "mechanism_report.html"
    rep_engine.export_html(result, out_path)
    print(f"\n  HTML report -> {out_path}")
    return 0


def _add_plog_surface_cmd(sub) -> None:
    p = sub.add_parser(
        "plog-surface",
        help="Interactive log k(T,P) surface for a PLOG reaction (Future.md pt. 5)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("reaction_id", type=int, help="Reaction DB id (must be kinetics_type='plog')")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--t-min", type=float, default=500.0)
    p.add_argument("--t-max", type=float, default=2500.0)
    p.add_argument("--dt", type=float, default=100.0)
    p.add_argument("--p-grid", default=None,
                   help="Comma-separated pressure grid [atm] "
                        "(default: 0.1,0.3,1,3,10,30,100)")
    p.add_argument("--export-html", default=None,
                   help="Output HTML path (default: plog_surface_<rxn_id>.html)")
    p.set_defaults(func=_cmd_plog_surface)


def _cmd_plog_surface(args) -> int:
    from rmip.analysis.plog_surface import PlogSurfaceBuilder

    _ensure_phase6_schema(args.db)
    t_grid = list(_frange(args.t_min, args.t_max, args.dt))
    p_grid = [float(x) for x in args.p_grid.split(",")] if args.p_grid else None

    builder = PlogSurfaceBuilder(args.db, T_grid_K=t_grid, P_grid_atm=p_grid)
    surface = builder.build(args.mech_id, args.reaction_id)

    print(f"\nPLOG surface — {surface.formula_string} (reaction_id={args.reaction_id})")
    print(f"  Grid: {len(surface.T_grid_K)} T points x {len(surface.P_grid_atm)} P points")
    print(f"  High-P limit ill-defined: {surface.high_p_limit_ill_defined}")
    if surface.non_monotonic_T:
        print(f"  Non-monotonic k(P) at {len(surface.non_monotonic_T)} T point(s)")
    for w in surface.warnings:
        print(f"  WARNING: {w}")

    out_path = args.export_html or f"plog_surface_{args.reaction_id}.html"
    builder.export_html(surface, out_path)
    print(f"\n  Interactive surface -> {out_path}")
    return 0


def _add_ntc_cmd(sub) -> None:
    p = sub.add_parser(
        "ntc",
        help="Detect NTC onset from a phi-fixed IDT sweep; split targets "
             "HTT/LTT (Future.md pt. 7, replaces fixed LTC/NTC/HTC thresholds)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--fuel", required=True, help='Fuel composition string, e.g. "NC7H16:1.0"')
    p.add_argument("--oxidizer", default="O2:1.0, N2:3.76")
    p.add_argument("--phi", type=float, default=1.0)
    p.add_argument("--pressure", type=float, default=1.0, dest="P_atm")
    p.add_argument("--t-min", type=float, default=650.0)
    p.add_argument("--t-max", type=float, default=1300.0)
    p.add_argument("--n-points", type=int, default=25)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tag-dataset", type=int, default=None,
                   help="exp_datasets id whose points should be tagged HTT/LTT "
                        "using the detected onset")
    p.add_argument("--export-html", default=None,
                   help="Output HTML path (default: ntc_<mech_id>.html)")
    p.set_defaults(func=_cmd_ntc)


def _cmd_ntc(args) -> int:
    from rmip.analysis.ntc_detector import NTCDetector

    _ensure_phase6_schema(args.db)
    det = NTCDetector(args.mech_id, args.db)
    report = det.detect(
        fuel=args.fuel, phi=args.phi, P_atm=args.P_atm, oxidizer=args.oxidizer,
        T_min=args.t_min, T_max=args.t_max, n_points=args.n_points, n_workers=args.workers,
    )
    run_id = det.store(report)

    print(f"\nNTC detection — mechanism_id={args.mech_id}, fuel={args.fuel}, phi={args.phi}")
    print(f"  has_ntc: {report.has_ntc}")
    if report.has_ntc:
        print(f"  T_NTC_onset : {report.T_NTC_onset:.1f} K")
        print(f"  T_NTC_end   : {report.T_NTC_end} K" if report.T_NTC_end else "  T_NTC_end   : (not found within grid)")
        print(f"  IDT_ratio_NTC: {report.IDT_ratio_NTC}")
        print(f"  HTT points (T > onset): {len(report.htt_points_K)}")
        print(f"  LTT points (T < onset): {len(report.ltt_points_K)}")
        print(f"\n  This onset now REPLACES the fixed 850/1000 K LTC/NTC/HTC split for "
              f"this mechanism in 'rmip bench'/'rmip rank' (HTT/LTT regimes).")
    else:
        print("  No NTC region detected in the swept range — "
              "'rmip bench'/'rmip rank' will keep using fixed LTC/NTC/HTC thresholds.")
    print(f"  Stored as ntc_runs id={run_id}")

    if args.tag_dataset is not None:
        n = det.tag_exp_dataset(args.tag_dataset, report)
        print(f"  Tagged {n} points in exp_dataset id={args.tag_dataset} as HTT/LTT")

    out_path = args.export_html or f"ntc_{args.mech_id}.html"
    det.export_html(report, out_path)
    print(f"\n  Interactive plot -> {out_path}")
    return 0


def _add_rop_cmd(sub) -> None:
    p = sub.add_parser(
        "rop",
        help="First-class in-process Cantera rate-of-production / "
             "reaction-path-flux diagnostic at one (T,P,phi) condition "
             "(Future.md pt. 3, native — no MUQ-SAC subprocess)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--fuel", required=True, help='Fuel composition, e.g. "NC7H16:1.0"')
    p.add_argument("--oxidizer", default="O2:1.0, N2:3.76")
    p.add_argument("--T", type=float, required=True, dest="T_K", help="Temperature [K]")
    p.add_argument("--P", type=float, required=True, dest="P_atm", help="Pressure [atm]")
    p.add_argument("--phi", type=float, required=True, help="Equivalence ratio")
    p.add_argument("--element", default="C", choices=["C", "H", "O"],
                   help="Element to trace for the flux (Sankey) view, default C")
    p.add_argument("--at", default="ignition",
                   help="Evaluation state: ignition | initial | equilibrium | "
                        "time:<seconds> (default: ignition)")
    p.add_argument("--top", type=int, default=15,
                   help="Print the top-N reactions by |net ROP| (default 15)")
    p.add_argument("--export", default=None,
                   help="Write the full ROP + element-flux result to this JSON path")
    p.add_argument("--dry-run", action="store_true",
                   help="Load the mechanism and validate args, but do not persist")
    p.set_defaults(func=_cmd_rop)


def _cmd_rop(args) -> int:
    from rmip.simulation.rop_runner import ROPRunner

    _ensure_phase6_schema(args.db)
    runner = ROPRunner(args.mech_id, args.db)
    result = runner.run_single(
        T_K=args.T_K, P_atm=args.P_atm, phi=args.phi,
        fuel=args.fuel, oxidizer=args.oxidizer,
        element=args.element, at=args.at,
    )

    print(f"\nROP diagnostic — mechanism_id={args.mech_id}, fuel={args.fuel}, "
          f"element={args.element}, at={args.at}")
    print(f"  converged: {result.converged}")
    if result.error_msg:
        print(f"  error: {result.error_msg}")
        return 1
    print(f"  reactions with ROP data: {len(result.rop_by_reaction)}")
    print(f"  {args.element}-flux Sankey edges: {len(result.element_flux)}")

    top = sorted(result.rop_by_reaction.items(),
                 key=lambda kv: abs(kv[1]), reverse=True)[:args.top]
    if top:
        print(f"\n  Top {len(top)} reactions by |net rate of progress| [kmol/m3/s]:")
        for key, rate in top:
            print(f"    {rate:+.4e}   {key}")

    if args.export:
        import json
        payload = {
            **result.to_dict(),
            "rop_by_reaction": result.rop_by_reaction,
            "element_flux":    result.element_flux,
        }
        with open(args.export, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  Full result -> {args.export}")

    if not args.dry_run:
        from rmip.database.schema import init_db
        from rmip.database.schema_phase6 import insert_flux_run
        conn = init_db(args.db)
        run_id = insert_flux_run(
            conn, mechanism_id=result.mechanism_id, solver=result.solver,
            T_K=result.condition.T_K, P_atm=result.condition.P_atm,
            phi=result.condition.phi, fuel_composition=result.condition.fuel,
            element=result.element, rop_by_reaction=result.rop_by_reaction,
            element_flux=result.element_flux, converged=result.converged,
            cpu_time_s=result.cpu_time_s, error_msg=result.error_msg,
        )
        print(f"\n  Stored as flux_runs id={run_id} (solver=cantera)")
    return 0


def _add_transport_cmd(sub) -> None:
    p = sub.add_parser(
        "transport",
        help="Report Lennard-Jones transport coverage, estimate missing species "
             "from SMILES, and export a CHEMKIN transport file (Future.md pt. 4)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--transport-file", default=None,
                   help="Existing CHEMKIN transport file to seed known data")
    p.add_argument("--estimate", action="store_true",
                   help="Estimate transport for species missing it, from their SMILES")
    p.add_argument("--export-chemkin", default=None,
                   help="Write a complete CHEMKIN transport file to this path")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not write estimated transport back to the DB")
    p.set_defaults(func=_cmd_transport)


def _cmd_transport(args) -> int:
    from rmip.database.schema import get_species_for_mechanism
    from rmip.io.transport_io import (
        TransportData, parse_chemkin_transport, write_chemkin_transport,
    )

    _ensure_phase6_schema(args.db)

    conn = init_db(args.db)
    species = get_species_for_mechanism(conn, args.mech_id)
    if not species:
        print(f"No species found for mechanism_id={args.mech_id}")
        return 1

    # 1. Seed known transport from an explicit CHEMKIN file …
    known: dict = {}
    if args.transport_file:
        known.update(parse_chemkin_transport(args.transport_file))
    # … and/or from the mechanism's own Cantera representation.
    try:
        from rmip.simulation.base import MechanismLoader
        sol = MechanismLoader(args.db).load(args.mech_id)
        for sp in sol.species():
            td = sp.input_data.get("transport")
            if td and sp.name not in known:
                known[sp.name] = TransportData.from_cantera_dict(td)
    except Exception as exc:  # cantera missing, no transport in YAML, etc.
        logger.debug("Could not read transport from Cantera solution: %s", exc)

    resolved: dict = {}
    n_known = n_estimated = n_missing = 0
    est_rows = []
    for sp in species:
        name = sp["name"]
        smiles = sp["canonical_smiles"] or sp["source_smiles"]
        if name in known:
            resolved[name] = known[name]
            n_known += 1
            continue
        if args.estimate and smiles:
            from rmip.analysis.transport_estimator import estimate_transport
            td = estimate_transport(smiles, name=name)
            if td is not None:
                resolved[name] = td
                est_rows.append((sp["id"], td))
                n_estimated += 1
                continue
        n_missing += 1

    print(f"\nTransport coverage — mechanism_id={args.mech_id} "
          f"({len(species)} species)")
    print(f"  known (from file/Cantera): {n_known}")
    print(f"  estimated from SMILES:     {n_estimated}")
    print(f"  still missing:             {n_missing}")

    # Persist estimated transport back to the DB (unless dry-run).
    if est_rows and not args.dry_run:
        for sp_id, td in est_rows:
            conn.execute(
                "UPDATE species SET transport_geometry=?, well_depth_K=?, "
                "diameter_A=?, dipole_D=?, polarizability_A3=?, rot_relax=?, "
                "transport_source=? WHERE id=?",
                (td.geometry, td.well_depth_K, td.diameter_A, td.dipole_D,
                 td.polarizability_A3, td.rot_relax, td.source, sp_id),
            )
        conn.commit()
        print(f"  → wrote {len(est_rows)} estimated transport rows to the DB")
    conn.close()

    if args.export_chemkin:
        text = write_chemkin_transport(
            resolved,
            header=f"Transport for mechanism_id={args.mech_id} "
                   f"({n_known} known, {n_estimated} estimated)",
        )
        with open(args.export_chemkin, "w") as fh:
            fh.write(text)
        print(f"  CHEMKIN transport → {args.export_chemkin} "
              f"({len(resolved)} species)")
    return 0


def _add_thermo_fit_cmd(sub) -> None:
    p = sub.add_parser(
        "thermo-fit",
        help="Scan NASA-7 thermo for Tmid discontinuities and refit species to "
             "continuity-consistent polynomials (Future.md pt. 3)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--tol", type=float, default=0.01,
                   help="Relative Cp/H/S jump at Tmid above which a species is "
                        "deemed inconsistent (default: 0.01 = 1%%)")
    p.add_argument("--repair", action="store_true",
                   help="Refit inconsistent species to continuous polynomials")
    p.add_argument("--dry-run", action="store_true",
                   help="With --repair, compute but do NOT persist to the DB")
    p.add_argument("--verify", action="store_true",
                   help="Run the reverse-rate scanner before/after repair to "
                        "measure the reduction in collision-limit-exceeding "
                        "derived reverse rates")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: thermo_fit_<id>.json)")
    p.set_defaults(func=_cmd_thermo_fit)


def _cmd_thermo_fit(args) -> int:
    from rmip.analysis.thermo_fitter import ThermoRepairer

    _ensure_phase6_schema(args.db)
    repairer = ThermoRepairer(args.db, tol=args.tol)

    all_species = repairer.scan(args.mech_id)
    if not all_species:
        print(f"No species with full NASA-7 thermo found for "
              f"mechanism_id={args.mech_id}")
        repairer.close()
        return 1

    inconsistent = [s for s in all_species if s.worst_jump_before > args.tol]
    print(f"\nThermo consistency — mechanism_id={args.mech_id} "
          f"({len(all_species)} species with NASA-7)")
    print(f"  inconsistent at Tmid (worst jump > {args.tol:g}): "
          f"{len(inconsistent)}")

    if inconsistent:
        print(f"\n  {'Species':<20} {'Tmid':>7} {'Cp jump':>10} "
              f"{'H jump':>10} {'S jump':>10}")
        print("  " + "-" * 60)
        for s in sorted(inconsistent, key=lambda x: -x.worst_jump_before)[:20]:
            j = s.jumps_before
            print(f"  {s.name[:20]:<20} {s.tmid_K:>7.0f} "
                  f"{j['cp']:>10.2e} {j['h']:>10.2e} {j['s']:>10.2e}")

    flags_before = flags_after = None
    if args.verify:
        n_flag, n_eval = repairer.count_reverse_flags(args.mech_id)
        flags_before = {"flagged": n_flag, "evaluated": n_eval}
        print(f"\n  Reverse-rate flags BEFORE repair: {n_flag}/{n_eval}")

    results = []
    if args.repair:
        results = repairer.repair(args.mech_id, dry_run=args.dry_run)
        n_ok = sum(1 for r in results if r.repaired)
        n_persisted = sum(1 for r in results if r.persisted)
        print(f"\n  Repaired {n_ok} species"
              + (f", persisted {n_persisted} to DB" if not args.dry_run
                 else " (dry-run, nothing written)"))
        for r in results:
            if r.repaired and r.jumps_after:
                after = max(r.jumps_after.values())
                print(f"    {r.name[:20]:<20} worst jump "
                      f"{r.worst_jump_before:.2e} → {after:.2e} "
                      f"(Cp RMS {r.cp_rms_error:.1e}, "
                      f"H298 {r.h298_kJ_mol_before:.2f}→"
                      f"{r.h298_kJ_mol_after:.2f} kJ/mol)")
            elif r.note:
                print(f"    {r.name[:20]:<20} {r.note}")

        if args.verify and not args.dry_run and n_persisted:
            n_flag, n_eval = repairer.count_reverse_flags(args.mech_id)
            flags_after = {"flagged": n_flag, "evaluated": n_eval}
            delta = flags_before["flagged"] - n_flag
            print(f"\n  Reverse-rate flags AFTER repair:  {n_flag}/{n_eval} "
                  f"({'-' if delta >= 0 else '+'}{abs(delta)})")

    repairer.close()

    out_path = args.out or f"thermo_fit_{args.mech_id}.json"
    report = {
        "mechanism_id": args.mech_id,
        "tol": args.tol,
        "n_species_nasa7": len(all_species),
        "n_inconsistent": len(inconsistent),
        "scan": [s.to_dict() for s in all_species],
        "repaired": [r.to_dict() for r in results],
        "reverse_flags_before": flags_before,
        "reverse_flags_after": flags_after,
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  Full report saved → {out_path}")
    return 0


def _add_surrogate_cmd(sub) -> None:
    p = sub.add_parser(
        "surrogate",
        help="Build a gPC polynomial-chaos IDT surrogate over selected reaction "
             "A-factors; reports analytic Sobol indices + active subspace "
             "(Future.md pt. 7)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--reactions", required=True,
                   help="Comma-separated 0-based Cantera reaction indices to "
                        "treat as uncertain (e.g. '0,3,12')")
    p.add_argument("--fuel", required=True, help="Fuel composition, e.g. 'NC7H16:1.0'")
    p.add_argument("--oxidizer", default="O2:1.0, N2:3.76")
    p.add_argument("--temperature", type=float, default=1000.0, dest="T_K")
    p.add_argument("--pressure", type=float, default=1.0, dest="P_atm", help="[atm]")
    p.add_argument("--phi", type=float, default=1.0)
    p.add_argument("--factor", type=float, default=2.0,
                   help="A-factor uncertainty bound (uniform on [1/f, f]); default 2")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--order", type=int, default=3, help="PCE polynomial order")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--t-end", type=float, default=10.0, help="IDT integration horizon [s]")
    p.add_argument("--chaospy-python", default=None,
                   help="Override the chaospy venv python (default: "
                        "/home/krithika/.venvs/chaospy_gpc/bin/python3)")
    p.add_argument("--work-dir", default=None,
                   help="Scratch dir for sampling/QoI files (default: surrogate_<id>/)")
    p.add_argument("--out", default=None, help="Output JSON path (default: surrogate_<id>.json)")
    p.add_argument("--no-db", action="store_true", help="Do not persist to the DB")
    p.set_defaults(func=_cmd_surrogate)


def _cmd_surrogate(args) -> int:
    from rmip.analysis.surrogate import PCESurrogateBuilder

    try:
        reaction_indices = [int(x) for x in args.reactions.split(",") if x.strip() != ""]
    except ValueError:
        print("--reactions must be a comma-separated list of integers")
        return 2
    if not reaction_indices:
        print("--reactions selected no reactions")
        return 2

    work_dir = args.work_dir or f"surrogate_{args.mech_id}"
    builder = PCESurrogateBuilder(args.db, chaospy_python=args.chaospy_python)
    result = builder.build_for_mechanism(
        args.mech_id, reaction_indices,
        T_K=args.T_K, P_atm=args.P_atm, phi=args.phi,
        fuel=args.fuel, oxidizer=args.oxidizer,
        work_dir=work_dir, factor_bound=args.factor,
        n_samples=args.n_samples, order=args.order, seed=args.seed,
        t_end_s=args.t_end, save_db=not args.no_db,
    )

    print(f"\ngPC IDT surrogate — mechanism_id={args.mech_id}")
    print(f"  T={args.T_K:g} K, P={args.P_atm:g} atm, phi={args.phi:g}, fuel={args.fuel}")
    print(f"  {len(reaction_indices)} uncertain reactions, factor={args.factor:g}, "
          f"order={args.order}, {result.n_samples} samples "
          f"({result.n_failed_samples} failed)")
    print(f"  PCE terms: {result.n_terms}   in-sample R²: {result.r2_insample:.4f}")
    print(f"  log10(IDT) mean={result.qoi_mean:.3f}  variance={result.qoi_variance:.3e}")

    print(f"\n  {'Reaction':<44} {'S1':>8} {'ST':>8} {'activity':>10}")
    print("  " + "-" * 72)
    for name, st in result.ranked_parameters():
        s1 = result.sobol_main.get(name, 0.0)
        act = result.as_activity_scores.get(name, 0.0)
        print(f"  {name[:44]:<44} {s1:>8.3f} {st:>8.3f} {act:>10.3e}")

    print(f"\n  Active subspace: {result.as_n_active} active direction(s) of "
          f"{len(reaction_indices)}")
    evals = result.as_eigenvalues
    if evals:
        top = ", ".join(f"{v:.2e}" for v in evals[:min(5, len(evals))])
        print(f"    eigenvalues (desc): {top}"
              + (" …" if len(evals) > 5 else ""))
    if result.run_id:
        print(f"\n  Persisted surrogate_runs.id={result.run_id}")

    out_path = args.out or f"surrogate_{args.mech_id}.json"
    result.save(out_path)
    print(f"  Full result saved → {out_path}")
    return 0


def _add_calibrate_cmd(sub) -> None:
    p = sub.add_parser(
        "calibrate",
        help="Bayesian / MUM-PCE calibration: infer the posterior on reaction "
             "A-factors from experimental targets, using stored gPC surrogates "
             "as the cheap forward model (Future.md pt. 6)",
    )
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--target", action="append", required=True, dest="targets",
                   help="A target as 'surrogate_run_id,observed,sigma[,name]' "
                        "(repeatable). 'observed'/'sigma' are in the surrogate's "
                        "output space, i.e. log10(IDT) by default.")
    p.add_argument("--prior-factor", type=float, default=None,
                   help="Prior A-factor uncertainty (σ_prior = ln(f)/2 per param). "
                        "Default: taken from the first surrogate's factor_bound.")
    p.add_argument("--prior-sigma", type=float, default=None,
                   help="Explicit prior σ in log-multiplier units (overrides --prior-factor)")
    p.add_argument("--bounds-sigma", type=float, default=2.0,
                   help="MAP search box half-width in units of σ_prior (default 2)")
    p.add_argument("--mcmc", action="store_true",
                   help="Also run an emcee full-posterior sample (uses the venv)")
    p.add_argument("--mcmc-steps", type=int, default=4000)
    p.add_argument("--mcmc-burn", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--work-dir", default=None, help="Scratch dir for MCMC files")
    p.add_argument("--out", default=None, help="Output JSON path (default: calibration.json)")
    p.add_argument("--no-db", action="store_true", help="Do not persist to the DB")
    p.set_defaults(func=_cmd_calibrate)


def _cmd_calibrate(args) -> int:
    import math as _math
    from rmip.analysis.bayesian_calibration import (
        MUMPCECalibrator, load_surrogate_target, surrogate_parameter_names,
        insert_calibration_run,
    )

    # Parse targets: 'run_id,observed,sigma[,name]'
    parsed = []
    for spec in args.targets:
        parts = [s.strip() for s in spec.split(",")]
        if len(parts) < 3:
            print(f"bad --target {spec!r}: need 'run_id,observed,sigma[,name]'")
            return 2
        rid, obs, sig = int(parts[0]), float(parts[1]), float(parts[2])
        name = parts[3] if len(parts) > 3 else None
        parsed.append((rid, obs, sig, name))

    run_ids = [p[0] for p in parsed]
    try:
        targets = [load_surrogate_target(args.db, rid, obs, sig, name)
                   for rid, obs, sig, name in parsed]
    except (KeyError, ValueError) as exc:
        print(f"Error loading surrogate target: {exc}")
        return 1

    param_names = surrogate_parameter_names(args.db, run_ids[0])

    # Prior sigma: explicit, or from --prior-factor, or the surrogate's factor_bound.
    d = len(param_names)
    if args.prior_sigma is not None:
        prior_sigma = args.prior_sigma
    else:
        factor = args.prior_factor
        if factor is None:
            import sqlite3
            conn = sqlite3.connect(args.db)
            row = conn.execute("SELECT factor_bound FROM surrogate_runs WHERE id=?",
                               (run_ids[0],)).fetchone()
            conn.close()
            factor = (row[0] if row and row[0] else 2.0)
        prior_sigma = _math.log(factor) / 2.0

    # Fetch the source mechanism id (for persistence) from the first surrogate.
    import sqlite3
    conn = sqlite3.connect(args.db)
    mrow = conn.execute("SELECT mechanism_id FROM surrogate_runs WHERE id=?",
                        (run_ids[0],)).fetchone()
    conn.close()
    mechanism_id = mrow[0] if mrow else None

    calibrator = MUMPCECalibrator(targets, param_names, prior_sigma,
                                  bounds_sigma=args.bounds_sigma)
    result = calibrator.calibrate()
    result.mechanism_id = mechanism_id

    print(f"\nMUM-PCE calibration — {result.n_targets} target(s), "
          f"{d} parameter(s)")
    print(f"  converged: {result.converged}   "
          f"χ²={result.chi2:.4g}   reduced χ²={result.reduced_chi2:.4g}")
    print(f"\n  {'Target':<22} {'observed':>10} {'prior→pred':>12} {'MAP→pred':>12}")
    print("  " + "-" * 58)
    for i, name in enumerate(result.target_names):
        print(f"  {name[:22]:<22} {result.observed[i]:>10.4f} "
              f"{result.predicted_prior[i]:>12.4f} {result.predicted_map[i]:>12.4f}")

    print(f"\n  {'Parameter':<40} {'MAP':>8} {'σ_post':>8} {'reduc%':>7}")
    print("  " + "-" * 66)
    for name, red in result.ranked_uncertainty_reduction():
        i = result.parameter_names.index(name)
        print(f"  {name[:40]:<40} {result.posterior_mean[i]:>8.3f} "
              f"{result.posterior_sigma[i]:>8.3f} {100*red:>6.1f}%")

    if args.mcmc:
        work = args.work_dir or "calibration_mcmc"
        try:
            mc = calibrator.run_mcmc(work, n_steps=args.mcmc_steps,
                                     n_burn=args.mcmc_burn, seed=args.seed)
            result.mcmc_mean = mc["mean"]
            result.mcmc_sigma = mc["sigma"]
            result.mcmc_n_samples = mc["n_samples"]
            print(f"\n  emcee posterior: {mc['n_samples']} samples "
                  f"(acceptance {mc['acceptance_fraction']:.2f})")
            print(f"    MCMC mean:  {[round(v,3) for v in mc['mean']]}")
            print(f"    MCMC σ:     {[round(v,3) for v in mc['sigma']]}")
        except RuntimeError as exc:
            print(f"\n  [MCMC skipped] {exc}")

    if not args.no_db:
        result.run_id = insert_calibration_run(args.db, result, mechanism_id, run_ids)
        if result.run_id:
            print(f"\n  Persisted calibration_runs.id={result.run_id}")

    out_path = args.out or "calibration.json"
    result.save(out_path)
    print(f"  Full result saved → {out_path}")
    return 0


def _add_b2bdc_cmd(sub) -> None:
    p = sub.add_parser(
        "b2bdc",
        help="Bound-to-Bound Data Collaboration: dataset consistency measure + "
             "feasible-set parameter/prediction bounds over stored gPC "
             "surrogates and experimental bounds (Future.md pt. 5)",
    )
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--constraint", action="append", required=True, dest="constraints",
                   help="An experiment as 'surrogate_run_id,lower,upper[,name]' "
                        "(repeatable); bounds in the surrogate's output space "
                        "(log10(IDT) by default)")
    p.add_argument("--predict", action="append", default=None, dest="predicts",
                   help="A prediction QOI as 'surrogate_run_id[,name]' (repeatable): "
                        "report its range over the feasible set")
    p.add_argument("--factor", type=float, default=None,
                   help="Prior parameter box as A-factor f → x∈[-ln f, ln f]. "
                        "Default: the first surrogate's stored factor_bound.")
    p.add_argument("--n-starts", type=int, default=12,
                   help="Multi-start count for the local optimiser (default 12)")
    p.add_argument("--outer", action="store_true",
                   help="Compute rigorous SDP OUTER bounds (S-procedure relaxation "
                        "via the cvxpy venv) instead of the local inner bounds; "
                        "requires order-2 (quadratic) surrogates")
    p.add_argument("--cvxpy-python", default=None, dest="cvxpy_python",
                   help="Python interpreter of the cvxpy venv (for --outer)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="Output JSON path (default: b2bdc.json)")
    p.add_argument("--no-db", action="store_true", help="Do not persist to the DB")
    p.set_defaults(func=_cmd_b2bdc)


def _cmd_b2bdc(args) -> int:
    import math as _math
    import sqlite3
    from rmip.analysis.b2bdc import B2BDCDataset, load_constraint, insert_b2bdc_run
    from rmip.analysis.bayesian_calibration import (
        PCEPoly, load_surrogate_target, surrogate_parameter_names,
    )

    parsed = []
    for spec in args.constraints:
        parts = [s.strip() for s in spec.split(",")]
        if len(parts) < 3:
            print(f"bad --constraint {spec!r}: need 'run_id,lower,upper[,name]'")
            return 2
        parsed.append((int(parts[0]), float(parts[1]), float(parts[2]),
                       parts[3] if len(parts) > 3 else None))

    run_ids = [p[0] for p in parsed]
    try:
        constraints = [load_constraint(args.db, rid, lo, hi, name)
                       for rid, lo, hi, name in parsed]
        param_names = surrogate_parameter_names(args.db, run_ids[0])
    except (KeyError, ValueError) as exc:
        print(f"Error loading constraint: {exc}")
        return 1

    d = len(param_names)
    factor = args.factor
    if factor is None:
        conn = sqlite3.connect(args.db)
        row = conn.execute("SELECT factor_bound, mechanism_id FROM surrogate_runs "
                           "WHERE id=?", (run_ids[0],)).fetchone()
        conn.close()
        factor = (row[0] if row and row[0] else 2.0)
        mechanism_id = row[1] if row else None
    else:
        conn = sqlite3.connect(args.db)
        row = conn.execute("SELECT mechanism_id FROM surrogate_runs WHERE id=?",
                           (run_ids[0],)).fetchone()
        conn.close()
        mechanism_id = row[0] if row else None

    box = B2BDCDataset.box_from_factor(factor, d)
    dataset = B2BDCDataset(constraints, param_names, box)

    predictions = []
    if args.predicts:
        for spec in args.predicts:
            parts = [s.strip() for s in spec.split(",")]
            rid = int(parts[0])
            pname = parts[1] if len(parts) > 1 else f"surrogate_{rid}"
            t = load_surrogate_target(args.db, rid, 0.0, 1.0, pname)
            predictions.append((pname, t.surrogate))

    if args.outer:
        try:
            result = dataset.analyze_outer(
                predictions=predictions, mechanism_id=mechanism_id,
                cvxpy_python=args.cvxpy_python, work_dir="b2bdc_sdp")
        except (ValueError, RuntimeError) as exc:
            print(f"B2BDC outer bound failed: {exc}")
            return 1
    else:
        result = dataset.analyze(predictions=predictions, n_starts=args.n_starts,
                                 seed=args.seed, mechanism_id=mechanism_id)

    print(f"\nB2BDC dataset — {result.n_constraints} experiment(s), "
          f"{d} parameter(s)   [{result.bound_type} bounds]")
    verdict = "CONSISTENT" if result.consistent else "INCONSISTENT"
    print(f"  consistency measure C = {result.consistency_measure:.4f}  → {verdict}")
    if result.consistent:
        print(f"  (feasible set non-empty with normalised margin {result.consistency_measure:.3f})")
    else:
        print("  (no parameter set reproduces all experiments within the prior box)")

    print(f"\n  {'Experiment':<26} {'margin@feasible':>16}")
    print("  " + "-" * 44)
    for name, m in zip(result.constraint_names, result.constraint_margins):
        print(f"  {name[:26]:<26} {m:>16.3f}")

    if result.parameter_bounds:
        print(f"\n  {'Parameter':<40} {'posterior [lo, hi]':>22} {'reduc%':>7}")
        print("  " + "-" * 72)
        for name in result.parameter_names:
            lo, hi = result.parameter_bounds[name]
            red = result.parameter_reduction.get(name, 0.0)
            print(f"  {name[:40]:<40} [{lo:>8.3f},{hi:>8.3f}] {100*red:>6.1f}%")

    if result.predictions:
        print(f"\n  {'Prediction QOI':<30} {'feasible range [lo, hi]':>26}")
        print("  " + "-" * 58)
        for pr in result.predictions:
            print(f"  {pr['name'][:30]:<30} [{pr['lower']:>10.4f}, {pr['upper']:>10.4f}]")

    for note in result.notes:
        print(f"\n  NOTE: {note}")

    if not args.no_db:
        result.run_id = insert_b2bdc_run(args.db, result, run_ids)
        if result.run_id:
            print(f"\n  Persisted b2bdc_runs.id={result.run_id}")

    out_path = args.out or "b2bdc.json"
    result.save(out_path)
    print(f"  Full result saved → {out_path}")
    return 0


def _add_csp_cmd(sub) -> None:
    p = sub.add_parser(
        "csp",
        help="Computational Singular Perturbation: chemical-Jacobian timescale "
             "analysis + QSS-species identification at a condition — the "
             "principled second stage of reduction (FEATURE_ROADMAP #10)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--fuel", required=True, help='Fuel spec, e.g. "H2:1.0"')
    p.add_argument("--oxidizer", default="O2:1.0, N2:3.76")
    p.add_argument("--T", type=float, required=True, dest="T_K", help="Temperature [K]")
    p.add_argument("--P", type=float, default=1.0, dest="P_atm", help="Pressure [atm]")
    p.add_argument("--phi", type=float, default=1.0, help="Equivalence ratio")
    p.add_argument("--at", default="ignition",
                   help="Eval state: ignition|initial|equilibrium|time:<s> "
                        "(default: ignition — where chemistry is most active)")
    p.add_argument("--tau", type=float, default=None,
                   help="Fast/slow timescale split [s] (default: auto, largest "
                        "spectral gap)")
    p.add_argument("--max-qss", type=int, default=None,
                   help="Cap the QSS candidate list to the N shortest-lived")
    p.add_argument("--exclude", default=None,
                   help="Comma-separated species never reported as QSS "
                        "(fuel/oxidiser already excluded)")
    p.add_argument("--export-yaml", default=None, dest="export_yaml",
                   help="Also materialise the QSSA-skeletal reduced mechanism "
                        "(QSS species removed) to this Cantera YAML path "
                        "(requires the mechanism parsed into the DB)")
    p.add_argument("--out", default=None, help="Output JSON path (default: csp_<id>.json)")
    p.add_argument("--no-db", action="store_true", help="Do not persist to the DB")
    p.set_defaults(func=_cmd_csp)


def _cmd_csp(args) -> int:
    from rmip.reduction.csp import CSPAnalyzer

    exclude = ([s.strip() for s in args.exclude.split(",") if s.strip()]
               if args.exclude else None)
    analyzer = CSPAnalyzer(args.mech_id, args.db)
    try:
        result = analyzer.analyze(
            T_K=args.T_K, P_atm=args.P_atm, phi=args.phi, fuel=args.fuel,
            oxidizer=args.oxidizer, at=args.at, tau_threshold=args.tau,
            max_qss=args.max_qss, exclude=exclude, save_db=not args.no_db,
        )
    except Exception as exc:
        print(f"CSP analysis failed: {exc}")
        return 1

    print(f"\nCSP timescale analysis — mech {args.mech_id} @ {result.eval_state} "
          f"(T={result.state_T_K:.1f} K, {args.P_atm} atm, φ={args.phi})")
    jac = "analytic" if result.jacobian_analytic else "finite-difference"
    print(f"  {result.n_species} species / {result.n_reactions} reactions   "
          f"[{jac} Jacobian]")
    print(f"  timescales: fastest {result.fastest_tau:.2e} s  →  "
          f"slowest {result.slowest_tau:.2e} s")
    print(f"  τ* = {result.tau_threshold:.2e} s  →  "
          f"{result.n_exhausted_modes} exhausted (fast) mode(s)")

    if result.qss_candidates:
        print(f"\n  {len(result.qss_candidates)} QSS candidate species "
              f"(lifetime < τ*, ranked shortest-lived first):")
        print(f"\n  {'Species':<12} {'lifetime[s]':>12} {'defect':>8} "
              f"{'mode τ[s]':>12} {'pointer':>8}")
        print("  " + "-" * 56)
        for q in result.qss_candidates:
            mt = f"{q['pointer_tau']:.2e}" if q["pointer_tau"] is not None else "   —"
            pv = f"{q['pointer_val']:.2f}" if q["pointer_val"] is not None else "  —"
            print(f"  {q['species'][:12]:<12} {q['lifetime']:>12.2e} "
                  f"{q['defect']:>8.3f} {mt:>12} {pv:>8}")
        print(f"\n  → up to {result.n_qss} species could be placed in QSS "
              f"(≈{result.n_species}→{result.n_species - result.n_qss} after "
              f"elimination)")
    else:
        print("\n  No species has a lifetime below τ* — no QSS candidates here.")

    for note in result.notes:
        print(f"\n  NOTE: {note}")
    if result.run_id:
        print(f"\n  Persisted csp_runs.id={result.run_id}")

    if args.export_yaml:
        from rmip.reduction.csp import export_qssa
        try:
            nid = export_qssa(args.db, result, args.export_yaml)
            print(f"  QSSA-skeletal mechanism → {args.export_yaml} (new mech id {nid})")
        except ValueError as exc:
            print(f"  [export skipped] {exc}")

    out_path = args.out or f"csp_{args.mech_id}.json"
    result.save(out_path)
    print(f"  Full result saved → {out_path}")
    return 0


def _add_lump_cmd(sub) -> None:
    p = sub.add_parser(
        "lump",
        help="A-priori thermodynamic chemical/isomer lumping proposals: merge "
             "thermochemically-similar isomers into pseudo-species "
             "(FEATURE_ROADMAP #10)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--dHf-tol", type=float, default=10.0, dest="dHf_tol",
                   help="Max ΔH_f° spread within a lump [kJ/mol] (default 10)")
    p.add_argument("--dS-tol", type=float, default=20.0, dest="dS_tol",
                   help="Max S° spread within a lump [J/mol/K] (default 20)")
    p.add_argument("--min-atoms", type=int, default=4, dest="min_atoms",
                   help="Skip isomer classes with fewer than N atoms (default 4)")
    p.add_argument("--exclude", default=None,
                   help="Comma-separated species never lumped")
    p.add_argument("--export-yaml", default=None, dest="export_yaml",
                   help="Also materialise the lumped mechanism (isomer members "
                        "collapsed onto representatives) to this Cantera YAML path "
                        "(requires the mechanism parsed into the DB; rates are not "
                        "re-fitted — a documented follow-on)")
    p.add_argument("--out", default=None, help="Output JSON path (default: lump_<id>.json)")
    p.add_argument("--no-db", action="store_true", help="Do not persist to the DB")
    p.set_defaults(func=_cmd_lump)


def _cmd_lump(args) -> int:
    from rmip.reduction.lumping import ChemicalLumper

    exclude = ([s.strip() for s in args.exclude.split(",") if s.strip()]
               if args.exclude else None)
    lumper = ChemicalLumper(args.db)
    try:
        result = lumper.propose(
            args.mech_id, dHf_tol_kJ=args.dHf_tol, dS_tol=args.dS_tol,
            min_atoms=args.min_atoms, exclude=exclude, save_db=not args.no_db,
        )
    except Exception as exc:
        print(f"Lumping failed: {exc}")
        return 1

    print(f"\nA-priori thermodynamic lumping — mech {args.mech_id}")
    print(f"  tolerances: |ΔHf| ≤ {result.dHf_tol_kJ} kJ/mol, "
          f"|ΔS| ≤ {result.dS_tol} J/mol/K, min atoms {result.min_atoms}")
    print(f"  {result.n_lumps} lump(s) → −{result.n_species_removed} species "
          f"({result.n_species_full} → {result.n_species_lumped})")

    if result.groups:
        print(f"\n  {'Formula':<12} {'Members':<44} {'ΔHf[kJ]':>8} {'ΔS':>7}")
        print("  " + "-" * 74)
        for g in result.groups:
            members = ", ".join(g["members"])
            print(f"  {g['formula'][:12]:<12} {members[:44]:<44} "
                  f"{g['dHf_spread_kJ']:>8.2f} {g['dS_spread']:>7.2f}")
            print(f"  {'':<12} → lumped into '{g['representative']}' "
                  f"(most stable)")
    for note in result.notes:
        print(f"\n  NOTE: {note}")
    if result.run_id:
        print(f"\n  Persisted lumping_runs.id={result.run_id}")

    if args.export_yaml:
        from rmip.reduction.lumping import export_lumped
        try:
            nid = export_lumped(args.db, result, args.export_yaml)
            print(f"  Lumped mechanism → {args.export_yaml} (new mech id {nid})")
        except ValueError as exc:
            print(f"  [export skipped] {exc}")

    out_path = args.out or f"lump_{args.mech_id}.json"
    result.save(out_path)
    print(f"  Full result saved → {out_path}")
    return 0


def _add_error_map_cmd(sub) -> None:
    p = sub.add_parser(
        "error-map",
        help="Skeletal-vs-detailed error atlas: map a reduced mechanism's IDT/LFS "
             "error against the detailed one over a (T,P,phi) grid — where the "
             "skeletal model holds and where it breaks (FEATURE_ROADMAP #11a)",
    )
    p.add_argument("detailed_id", type=int, help="Detailed (reference) mechanism id")
    p.add_argument("skeletal_id", type=int, help="Reduced/skeletal mechanism id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--sim-type", default="IDT", choices=["IDT", "LFS", "idt", "lfs"],
                   dest="sim_type", help="Quantity of interest (default IDT)")
    p.add_argument("--fuel", required=True, help='Fuel spec, e.g. "H2:1.0"')
    p.add_argument("--oxidizer", default="O2:1.0, N2:3.76")
    p.add_argument("--grid", required=True,
                   help="Grid axes, e.g. 'T=800:1600:9,P=1:40:3,phi=0.5:1.5:3' "
                        "(a:b:n linspace, or a,b,c explicit; missing axes default)")
    p.add_argument("--tol", type=float, default=0.30,
                   help="Relative-error threshold for a 'trustworthy' cell (default 0.30)")
    p.add_argument("--workers", type=int, default=1, help="Parallel sim workers")
    p.add_argument("--out", default=None, help="Output JSON path (default: error_map.json)")
    p.add_argument("--no-db", action="store_true", help="Do not persist to the DB")
    p.set_defaults(func=_cmd_error_map)


def _cmd_error_map(args) -> int:
    from rmip.analysis.error_map import ErrorMapper, parse_grid

    sim_type = args.sim_type.upper()
    default_T = 300.0 if sim_type == "LFS" else 1000.0
    conditions = parse_grid(args.grid, fuel=args.fuel, oxidizer=args.oxidizer,
                            default_T=default_T)
    try:
        result = ErrorMapper(args.db).map_error(
            args.detailed_id, args.skeletal_id, sim_type, conditions,
            tolerance=args.tol, n_workers=args.workers, save_db=not args.no_db)
    except Exception as exc:
        print(f"Error-map failed: {exc}")
        return 1

    print(f"\nSkeletal-vs-detailed error atlas ({sim_type}) — "
          f"detailed {args.detailed_id} vs skeletal {args.skeletal_id}")
    print(f"  {result.n_conditions} grid cells, {result.n_evaluated} evaluated "
          f"(both converged)")
    if result.n_evaluated:
        print(f"  max relative error : {result.max_rel_err:.3g}")
        print(f"  mean relative error: {result.mean_rel_err:.3g}")
        print(f"  max log10 error    : {result.max_log_err:.3g}")
        print(f"  within tol {result.tolerance:.2f}: "
              f"{100 * result.frac_within_tol:.0f}% of evaluated cells")
        w = result.worst_cell
        print(f"  worst cell: T={w['T_K']:.0f}K P={w['P_atm']:.3g}atm "
              f"phi={w['phi']:.2f}  rel_err={w['rel_err']:.3g}")
    else:
        print("  No cell had both mechanisms converge — check the grid/fuel.")
    for note in result.notes:
        print(f"\n  NOTE: {note}")
    if result.run_id:
        print(f"\n  Persisted error_map_runs.id={result.run_id}")

    out_path = args.out or "error_map.json"
    result.save(out_path)
    print(f"  Full result saved → {out_path}")
    return 0


def _add_tdac_cmd(sub) -> None:
    p = sub.add_parser(
        "tdac",
        help="Tabulated dynamic adaptive chemistry: integrate a batch reactor with "
             "on-the-fly DRGEP reduction + state tabulation — reports IDT accuracy "
             "vs the full mechanism and the active-species / hit-rate efficiency "
             "(FEATURE_ROADMAP #11b)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--fuel", required=True, help='Fuel spec, e.g. "H2:1.0"')
    p.add_argument("--oxidizer", default="O2:1.0, N2:3.76")
    p.add_argument("--T", type=float, required=True, dest="T_K", help="Temperature [K]")
    p.add_argument("--P", type=float, default=1.0, dest="P_atm", help="Pressure [atm]")
    p.add_argument("--phi", type=float, default=1.0, help="Equivalence ratio")
    p.add_argument("--target-species", default=None, dest="target_species",
                   help="Comma-separated DRGEP targets kept always active "
                        "(default: fuel + O2)")
    p.add_argument("--eps", type=float, default=1e-2,
                   help="DRGEP threshold (smaller ⇒ larger active set ⇒ more "
                        "accurate; default 0.01)")
    p.add_argument("--update-every", type=int, default=10, dest="update_every",
                   help="Recompute the active set every N macro steps (default 10)")
    p.add_argument("--bin-T", type=float, default=50.0, dest="bin_T",
                   help="Tabulation temperature bin width [K] (default 50)")
    p.add_argument("--t-end", type=float, default=10.0, dest="t_end",
                   help="Integration end time [s] (default 10)")
    p.add_argument("--out", default=None, help="Output JSON path (default: tdac_<id>.json)")
    p.add_argument("--no-db", action="store_true", help="Do not persist to the DB")
    p.set_defaults(func=_cmd_tdac)


def _cmd_tdac(args) -> int:
    from rmip.simulation.tdac_runner import TDACRunner

    targets = ([s.strip() for s in args.target_species.split(",") if s.strip()]
               if args.target_species else None)
    runner = TDACRunner(args.mech_id, args.db, t_end_s=args.t_end)
    try:
        res = runner.run_single(
            T_K=args.T_K, P_atm=args.P_atm, phi=args.phi, fuel=args.fuel,
            oxidizer=args.oxidizer, target_species=targets, eps=args.eps,
            update_every=args.update_every, bin_T=args.bin_T,
            save_db=not args.no_db)
    except Exception as exc:
        print(f"TDAC run failed: {exc}")
        return 1

    print(f"\nTDAC batch reactor — mech {args.mech_id} @ {args.T_K:.0f}K "
          f"{args.P_atm}atm φ={args.phi} ({args.fuel})")
    print(f"  full mechanism: {res.n_species_full} species / {res.n_reactions_full} reactions")
    print(f"  DRGEP targets: {', '.join(res.target_species)}   eps={res.eps}")
    idt = f"{res.idt_s:.4e} s" if res.idt_s else "no ignition"
    print(f"  TDAC IDT: {idt}   (converged={res.converged})")
    if res.mean_active_species is not None:
        print(f"  active species: mean {res.mean_active_species:.1f}/"
              f"{res.n_species_full} ({100*res.mean_active_fraction:.0f}%), "
              f"range {res.min_active_species}–{res.max_active_species}")
        print(f"  tabulation: {res.n_updates} updates, "
              f"{res.n_cache_hits} hits / {res.n_cache_builds} builds "
              f"→ hit-rate {res.hit_rate:.2f}")
    for note in res.notes:
        print(f"\n  NOTE: {note}")
    if res.run_id:
        print(f"\n  Persisted tdac_runs.id={res.run_id}")

    out_path = args.out or f"tdac_{args.mech_id}.json"
    res.save(out_path)
    print(f"  Full result saved → {out_path}")
    return 0


def _add_design_experiments_cmd(sub) -> None:
    p = sub.add_parser(
        "design-experiments",
        help="Optimal experiment design: rank/select the most informative next "
             "experiments (by A-/D-optimal Fisher information) over stored gPC "
             "surrogates (FEATURE_ROADMAP #8)",
    )
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--candidate", action="append", required=True, dest="candidates",
                   help="A candidate as 'surrogate_run_id[,sigma]' (repeatable); "
                        "sigma is the measurement noise in the surrogate's output "
                        "space (default 1.0)")
    p.add_argument("--criterion", default="D", choices=["D", "A", "d", "a"],
                   help="D-optimality (uncertainty-ellipsoid volume) or "
                        "A-optimality (average variance). Default D")
    p.add_argument("--select", type=int, default=None, dest="n_select",
                   help="How many experiments to select (default: all, i.e. a full ranking)")
    p.add_argument("--prior-factor", type=float, default=None,
                   help="Prior A-factor uncertainty (σ=ln f/2). Default: first "
                        "surrogate's factor_bound")
    p.add_argument("--prior-sigma", type=float, default=None,
                   help="Explicit prior σ (overrides --prior-factor)")
    p.add_argument("--out", default=None, help="Output JSON path (default: doe.json)")
    p.add_argument("--no-db", action="store_true", help="Do not persist to the DB")
    p.set_defaults(func=_cmd_design_experiments)


def _cmd_design_experiments(args) -> int:
    from rmip.analysis.experiment_design import ExperimentDesigner

    run_ids, sigmas = [], []
    for spec in args.candidates:
        parts = [s.strip() for s in spec.split(",")]
        run_ids.append(int(parts[0]))
        sigmas.append(float(parts[1]) if len(parts) > 1 else 1.0)

    try:
        res = ExperimentDesigner(args.db).design(
            run_ids, sigmas=sigmas, prior_sigma=args.prior_sigma,
            prior_factor=args.prior_factor, n_select=args.n_select,
            criterion=args.criterion, save_db=not args.no_db)
    except (KeyError, ValueError) as exc:
        print(f"design-experiments failed: {exc}")
        return 1

    print(f"\nOptimal experiment design ({res.criterion}-optimal) — "
          f"{res.n_candidates} candidate(s), {len(res.parameter_names)} parameter(s)")
    print(f"  trace(cov): {res.trace_cov_prior:.4g} → {res.trace_cov_post:.4g}   "
          f"logdet(precision): {res.logdet_prior:.4g} → {res.logdet_post:.4g}")
    print(f"\n  Selected ({res.n_selected}, in order):")
    print(f"  {'#':<3} {'run_id':>7} {'experiment':<24} {'info gain':>12}")
    print("  " + "-" * 50)
    for s in res.selected:
        print(f"  {s['rank']:<3} {s['run_id']:>7} {s['name'][:24]:<24} {s['gain']:>12.4g}")

    print(f"\n  Full first-step ranking:")
    for r in res.ranked_candidates:
        print(f"    run {r['run_id']:>4}  {r['name'][:28]:<28}  gain {r['first_step_gain']:.4g}")

    print(f"\n  {'Parameter':<40} {'σ_post':>8} {'reduc%':>7}")
    print("  " + "-" * 58)
    for name, ps, red in zip(res.parameter_names, res.posterior_sigma,
                             res.uncertainty_reduction):
        print(f"  {name[:40]:<40} {ps:>8.3f} {100*red:>6.1f}%")

    if res.run_id:
        print(f"\n  Persisted doe_runs.id={res.run_id}")
    out_path = args.out or "doe.json"
    res.save(out_path)
    print(f"  Full result saved → {out_path}")
    return 0


def _add_pareto_cmd(sub) -> None:
    p = sub.add_parser(
        "pareto",
        help="Multi-objective (NSGA-II) mechanism optimization: the Pareto front "
             "of reaction A-factor sets trading off several surrogate-based "
             "objectives (FEATURE_ROADMAP #9)",
    )
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--objective", action="append", required=True, dest="objectives",
                   help="An objective as 'surrogate_run_id,target[,name]' "
                        "(repeatable); minimises |surrogate(theta) - target|")
    p.add_argument("--factor", type=float, default=None,
                   help="Prior A-factor box f → theta∈[-ln f, ln f]. Default: the "
                        "first surrogate's factor_bound")
    p.add_argument("--generations", type=int, default=50, dest="n_gen")
    p.add_argument("--pop", type=int, default=40, dest="pop_size")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="Output JSON path (default: pareto.json)")
    p.add_argument("--no-db", action="store_true", help="Do not persist to the DB")
    p.set_defaults(func=_cmd_pareto)


def _cmd_pareto(args) -> int:
    from rmip.analysis.pareto import ParetoOptimizer

    objectives, names = [], []
    for spec in args.objectives:
        parts = [s.strip() for s in spec.split(",")]
        if len(parts) < 2:
            print(f"bad --objective {spec!r}: need 'run_id,target[,name]'")
            return 2
        objectives.append((int(parts[0]), float(parts[1])))
        names.append(parts[2] if len(parts) > 2 else f"surrogate_{parts[0]}")

    try:
        res = ParetoOptimizer(args.db).optimize(
            objectives, factor=args.factor, objective_names=names,
            pop_size=args.pop_size, n_gen=args.n_gen, seed=args.seed,
            save_db=not args.no_db)
    except (KeyError, ValueError) as exc:
        print(f"pareto failed: {exc}")
        return 1

    print(f"\nMulti-objective Pareto (NSGA-II) — {len(res.objective_names)} "
          f"objectives, {res.n_generations} generations, pop {res.pop_size}")
    print(f"  {res.n_front} non-dominated solution(s)"
          + (f"   hypervolume={res.hypervolume:.4g}" if res.hypervolume is not None else ""))
    print(f"\n  {'#':<3} " + " ".join(f"{n[:12]:>12}" for n in res.objective_names))
    print("  " + "-" * (4 + 13 * len(res.objective_names)))
    for i, sol in enumerate(res.front[:20]):
        print(f"  {i+1:<3} " + " ".join(f"{v:>12.4g}" for v in sol["objectives"]))
    if len(res.front) > 20:
        print(f"  … and {len(res.front) - 20} more")
    for note in res.notes:
        print(f"\n  NOTE: {note}")
    if res.run_id:
        print(f"\n  Persisted pareto_runs.id={res.run_id}")

    out_path = args.out or "pareto.json"
    res.save(out_path)
    print(f"  Full result saved → {out_path}")
    return 0


def _add_family_scan_cmd(sub) -> None:
    p = sub.add_parser(
        "family-scan",
        help="Reaction-family completeness scan: which reactions/species the "
             "kinetic-theory families admit for this mechanism that it is missing "
             "(targeted RMG, FEATURE_ROADMAP #2)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--families", default=None,
                   help="Comma-separated family subset (default: all)")
    p.add_argument("--rmg-python", default=None, dest="rmg_python",
                   help="Python interpreter of the RMG conda env")
    p.add_argument("--rmg-database", default=None, dest="rmg_database",
                   help="RMG-database path (else RMG_DATABASE_PATH / RMG default)")
    p.add_argument("--out", default=None, help="Output JSON path (default: family_scan.json)")
    p.add_argument("--no-db", action="store_true", help="Do not persist to the DB")
    p.set_defaults(func=_cmd_family_scan)


def _cmd_family_scan(args) -> int:
    from rmip.generation.reaction_family_kb import ReactionFamilyKB

    fams = ([s.strip() for s in args.families.split(",") if s.strip()]
            if args.families else None)
    kb = ReactionFamilyKB(args.db, rmg_python=args.rmg_python,
                          rmg_database=args.rmg_database)
    try:
        rep = kb.family_scan(args.mech_id, only_families=fams, save_db=not args.no_db)
    except (RuntimeError, KeyError) as exc:
        print(f"family-scan failed: {exc}")
        return 1

    print(f"\nReaction-family completeness — mechanism {args.mech_id} "
          f"({rep.n_species} species)")
    print(f"  {rep.n_enumerated} reactions enumerated over "
          f"{len(rep.families_seen)} families")
    print(f"  missing reactions (family-admissible, absent): {rep.n_missing_reactions}")
    print(f"  missing species (from kinetic theory): {rep.n_missing_species}")
    if rep.missing_reactions:
        print(f"\n  {'kind':<5} {'family':<22} reaction")
        print("  " + "-" * 60)
        for r in rep.missing_reactions[:25]:
            eq = " + ".join(r["reactants"]) + " <=> " + " + ".join(r["products"])
            print(f"  {r.get('kind','?'):<5} {str(r.get('family'))[:22]:<22} {eq[:40]}")
        if rep.n_missing_reactions > 25:
            print(f"  … and {rep.n_missing_reactions - 25} more")
    if rep.missing_species:
        print(f"\n  missing species: "
              + ", ".join(s["smiles"] for s in rep.missing_species[:20]))
    for note in rep.notes:
        print(f"\n  NOTE: {note}")
    if rep.run_id:
        print(f"\n  Persisted family_scan_runs.id={rep.run_id}")
    out_path = args.out or "family_scan.json"
    rep.save(out_path)
    print(f"  Full result saved → {out_path}")
    return 0


def _add_generate_targeted_cmd(sub) -> None:
    p = sub.add_parser(
        "generate-targeted",
        help="Generate one generation of reactions for given seed species (SMILES) "
             "and graft them into the mechanism; --reactions-only adds no new "
             "species (targeted RMG, FEATURE_ROADMAP #2)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id to extend")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--species", required=True,
                   help="Comma-separated seed species SMILES to add chemistry for")
    p.add_argument("--reactions-only", action="store_true", dest="reactions_only",
                   help="Keep only reactions whose participants already exist "
                        "(add no new species)")
    p.add_argument("--families", default=None, help="Comma-separated family subset")
    p.add_argument("--new-name", default=None, dest="new_name",
                   help="Create a new mechanism with this name instead of "
                        "mutating the base in place")
    p.add_argument("--rmg-python", default=None, dest="rmg_python")
    p.add_argument("--rmg-database", default=None, dest="rmg_database")
    p.set_defaults(func=_cmd_generate_targeted)


def _cmd_generate_targeted(args) -> int:
    from rmip.generation.reaction_family_kb import ReactionFamilyKB

    seeds = [s.strip() for s in args.species.split(",") if s.strip()]
    fams = ([s.strip() for s in args.families.split(",") if s.strip()]
            if args.families else None)
    kb = ReactionFamilyKB(args.db, rmg_python=args.rmg_python,
                          rmg_database=args.rmg_database)
    try:
        new_id = kb.generate_targeted(
            args.mech_id, seeds, reactions_only=args.reactions_only,
            new_name=args.new_name, only_families=fams)
    except (RuntimeError, KeyError) as exc:
        print(f"generate-targeted failed: {exc}")
        return 1
    print(f"\nTargeted generation grafted into mechanism {new_id} "
          f"(seeds: {', '.join(seeds)})")
    if new_id != args.mech_id:
        print(f"  (new mechanism id {new_id} = base {args.mech_id} + generated chemistry)")
    return 0


def _add_import_respecth_cmd(sub) -> None:
    p = sub.add_parser(
        "import-respecth",
        help="Ingest ReSpecTh/PrIMe (RKD XML) experimental data — bridges both "
             "the legacy .target pipeline and the rmip DB (Future.md pt. 2)",
    )
    p.add_argument("xml_path", help="A ReSpecTh .xml file or a directory of them")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--fuel", default="NH3",
                   help="Comma-separated fuel species, e.g. 'NH3,H2' (default NH3)")
    p.add_argument("--solver", default="cantera",
                   choices=["cantera", "CHEMKIN_PRO", "FlameMaster"],
                   help="Solver tag written into every .add block")
    p.add_argument("--out-dir", default=None,
                   help="Where to write legacy .target/.add/.bib/.csv artefacts")
    p.add_argument("--dataset-prefix", default=None,
                   help="Prefix for DB dataset names (disambiguate re-imports)")
    p.add_argument("--no-db", action="store_true",
                   help="Only produce the legacy artefacts; skip DB loading")
    p.set_defaults(func=_cmd_import_respecth)


def _cmd_import_respecth(args) -> int:
    _ensure_phase6_schema(args.db)

    fuel_list = [f.strip() for f in args.fuel.split(",") if f.strip()]
    # xml2csv keys fuels by single letters a, b, c, … in declaration order.
    input_fuel = {chr(ord("a") + i): f for i, f in enumerate(fuel_list)}

    if args.no_db:
        from rmip.io.respecth_reader import convert_respecth
        summary = convert_respecth(
            args.xml_path, args.out_dir,
            fuel_list=fuel_list, input_fuel=input_fuel, solver=args.solver,
        )
        print(f"\nReSpecTh conversion (legacy artefacts only) — {args.xml_path}")
        print(f"  files ok: {summary.get('n_files_ok')}, "
              f"skipped: {summary.get('n_files_skipped')}, rows: {summary.get('n_rows')}")
        for k in ("target_path", "add_path", "bib_path", "csv_path"):
            if summary.get(k):
                print(f"  {k}: {summary[k]}")
        for w in summary.get("warnings", []) or []:
            print(f"  ⚠ {w}")
        return 0

    from rmip.io.exp_data_reader import ExpDataReader
    reader = ExpDataReader(args.db)
    result = reader.load_respecth_xml(
        args.xml_path, out_dir=args.out_dir,
        fuel_list=fuel_list, input_fuel=input_fuel, solver=args.solver,
        dataset_prefix=args.dataset_prefix,
    )
    conv = result["conversion"]
    print(f"\nReSpecTh import — {args.xml_path}")
    print(f"  files ok: {conv.get('n_files_ok')}, "
          f"skipped: {conv.get('n_files_skipped')}, rows: {conv.get('n_rows')}")
    print(f"  legacy .target: {result.get('target_path')}")
    print(f"  DB datasets created: {len(result['dataset_ids'])} "
          f"(ids {result['dataset_ids']})")
    print(f"  DB experimental points: {result['n_points']}")
    for w in conv.get("warnings", []) or []:
        print(f"  ⚠ {w}")
    return 0


def _add_build_target_cmd(sub) -> None:
    p = sub.add_parser(
        "build-target",
        help="Build a .target/.add/target.opt file set for a single JSR/PFR "
             "condition from a base target.opt (Future.md pt. 3 — "
             "MuqsacBridge.build_target_files; GUI: rmip_suite_gui.py's "
             "Target Builder tab)",
    )
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--base-target-opt", required=True,
                   help="Existing target.opt supplying Bin/Locations/Stats/etc. "
                        "to derive the new one from")
    p.add_argument("--run-script", default="src/MUQ-SAC/run_script.py",
                   help="Path to src/MUQ-SAC/run_script.py (only recorded — "
                        "this command never invokes the pipeline itself)")
    p.add_argument("--target-type", choices=["JSR", "PFR"], required=True,
                   help="build_target_files() only supports JSR/PFR — other "
                        "target types have no builder yet")
    p.add_argument("--solver", required=True, choices=list(_SOLVER_ALIASES.keys()),
                   help="Solver tag written into the .add block")
    p.add_argument("--fuel", required=True,
                   help='Fuel composition string, e.g. "NC7H16:1.0" or a '
                        'multi-species blend "NC7H16:0.7, CH4:0.3"')
    p.add_argument("--oxidizer", default="O2:1.0, N2:3.76",
                   help='Oxidizer composition string (default: "O2:1.0, N2:3.76")')
    p.add_argument("--T", type=float, required=True, dest="T_K", help="Temperature [K]")
    p.add_argument("--P", type=float, required=True, dest="P_atm", help="Pressure [atm]")
    p.add_argument("--phi", type=float, default=1.0, help="Equivalence ratio")
    p.add_argument("--work-dir", required=True,
                   help="Directory to write rmip.target/rmip.add/target.opt into")
    p.add_argument("--dataset-id", default=None, dest="dataset_id",
                   help="Dataset id written into the .target/.add files "
                        "(default: auto-generated)")
    p.add_argument("--species", default=None,
                   help="Species recorded in the .add block's 'species' key "
                        "(default: the first fuel species)")
    p.add_argument("--residence-time", type=float, default=1.0, dest="residence_time_s",
                   help="Residence time [s] written into the .add block")
    p.add_argument("--max-sim-time", type=float, default=10.0, dest="max_sim_time_s",
                   help="MaxSimulationTime written into the .add block")
    p.add_argument("--pfr-length", type=float, default=0.1, help="PFR reactor length [m]")
    p.add_argument("--pfr-diameter", type=float, default=0.02, help="PFR reactor diameter [m]")
    p.add_argument("--pfr-mdot", type=float, default=1e-3, help="PFR mass flow rate [kg/s]")
    p.add_argument("--pfr-n-points", type=int, default=200, help="PFR axial resolution")
    p.add_argument("--flux", action="store_true",
                   help="Also enable reaction-flux analysis in the .add block")
    p.add_argument("--flux-element", default="C", help="Element to trace if --flux is set")
    p.set_defaults(func=_cmd_build_target)


def _cmd_build_target(args) -> int:
    from rmip.simulation.muqsac_bridge import MuqsacBridge
    from rmip.types import SimCondition

    condition = SimCondition(T_K=args.T_K, P_atm=args.P_atm, phi=args.phi,
                             fuel=args.fuel, oxidizer=args.oxidizer)
    bridge = MuqsacBridge(args.db, args.run_script)

    extra_add: dict = {
        "residenceTime": args.residence_time_s,
        "MaxSimulationTime": args.max_sim_time_s,
    }
    if args.species:
        extra_add["species"] = args.species
    if args.target_type == "PFR":
        extra_add.update({
            "pfr_length": args.pfr_length, "pfr_diameter": args.pfr_diameter,
            "pfr_mdot": args.pfr_mdot, "pfr_n_points": args.pfr_n_points,
        })

    target_opt_path = bridge.build_target_files(
        base_target_opt=args.base_target_opt, mechanism_id=args.mech_id,
        target_type=args.target_type, solver=_SOLVER_ALIASES[args.solver],
        condition=condition, work_dir=args.work_dir, extra_add=extra_add,
        flux_analysis=args.flux, flux_element=args.flux_element,
        dataset_id=args.dataset_id,
    )
    work_dir = Path(args.work_dir)
    print(f"\nBuilt {args.target_type} target files for mechanism_id={args.mech_id}:")
    print(f"  .target    : {work_dir / 'rmip.target'}")
    print(f"  .add       : {work_dir / 'rmip.add'}")
    print(f"  target.opt : {target_opt_path}")
    return 0


def _add_flux_cmd(sub) -> None:
    p = sub.add_parser(
        "flux",
        help="Reaction-pathway flux/ROP analysis via the MUQ-SAC pipeline, "
             "any of the 3 solvers (Future.md pt. 3)",
    )
    _add_muqsac_bridge_args(p, reactor=True)
    p.add_argument("--element", default="C", help="Element to trace (C|H|O), default C")
    p.set_defaults(func=_cmd_flux)


def _cmd_flux(args) -> int:
    result = _run_muqsac_bridge_case(args, target_type=args.target_type, flux_analysis=True)
    if result is None:
        return 0 if args.dry_run else 1
    from rmip.simulation.muqsac_bridge import MuqsacBridge
    bridge = MuqsacBridge(args.db, args.run_script)
    flux_result = bridge.parse_flux_output(
        args.work_dir, args.mech_id, _SOLVER_ALIASES[args.solver], result["condition"],
        element=args.element,
    )
    if not args.dry_run:
        bridge.store_flux_result(flux_result)
    print(f"\nFlux analysis — mechanism_id={args.mech_id}, solver={args.solver}, "
          f"element={args.element}")
    print(f"  converged: {flux_result.converged}")
    print(f"  reactions with ROP data: {len(flux_result.rop_by_reaction)}")
    print(f"  Sankey-ready edges: {len(flux_result.element_flux)}")
    # M16: an unconverged flux run must exit non-zero.
    return 0 if flux_result.converged else 1


def _add_jsr_cmd(sub) -> None:
    p = sub.add_parser(
        "jsr",
        help="JSR (perfectly-stirred reactor) via the MUQ-SAC pipeline, "
             "any of the 3 solvers (Future.md pt. 6)",
    )
    _add_muqsac_bridge_args(p, reactor=False)
    p.add_argument("--residence-time", type=float, default=1.0, dest="residence_time_s")
    p.add_argument("--max-sim-time", type=float, default=10.0, dest="max_sim_time_s")
    p.add_argument("--flux", action="store_true", help="Also enable reaction-flux analysis")
    p.add_argument("--flux-element", default="C")
    p.set_defaults(func=_cmd_jsr, target_type="JSR")


def _cmd_jsr(args) -> int:
    from rmip.simulation.jsr_runner import JSRRunner
    from rmip.types import SimCondition

    _ensure_phase6_schema(args.db)
    cond = SimCondition(T_K=args.T_K, P_atm=args.P_atm, phi=args.phi,
                        fuel=args.fuel, oxidizer=args.oxidizer)
    runner = JSRRunner(args.mech_id, args.db, args.base_target_opt, args.run_script)

    if args.dry_run:
        from rmip.simulation.muqsac_bridge import MuqsacBridge
        bridge = MuqsacBridge(args.db, args.run_script)
        target_opt = bridge.build_target_files(
            base_target_opt=args.base_target_opt, mechanism_id=args.mech_id,
            target_type="JSR", solver=_SOLVER_ALIASES[args.solver], condition=cond,
            work_dir=args.work_dir,
            extra_add={"residenceTime": args.residence_time_s,
                       "MaxSimulationTime": args.max_sim_time_s, "species": args.species},
            flux_analysis=args.flux, flux_element=args.flux_element,
        )
        print(f"Dry run — files written, pipeline NOT invoked:")
        print(f"  {target_opt}")
        print(f"  Would run: python3 {args.run_script} {target_opt}")
        return 0

    result = runner.run_single(
        cond, solver=_SOLVER_ALIASES[args.solver], residence_time_s=args.residence_time_s,
        work_dir=args.work_dir, max_simulation_time_s=args.max_sim_time_s,
        flux_analysis=args.flux, flux_element=args.flux_element,
        extra_add={"species": args.species} if args.species else None,
        run_timeout=args.timeout,
    )
    print(f"\nJSR — mechanism_id={args.mech_id}, solver={args.solver}")
    print(f"  converged: {result.converged}")
    if result.error_msg:
        print(f"  error: {result.error_msg}")
    print(f"  species tracked in output: {len(result.species_mole_fractions)}")
    # M16: a failed/unconverged run must exit non-zero, not silently return 0.
    return 0 if result.converged else 1


def _add_pfr_cmd(sub) -> None:
    p = sub.add_parser(
        "pfr",
        help="Plug-Flow Reactor via the MUQ-SAC pipeline, any of the 3 "
             "solvers (Future.md pt. 6)",
    )
    _add_muqsac_bridge_args(p, reactor=False)
    p.add_argument("--pfr-length", type=float, default=0.1, help="Reactor length [m]")
    p.add_argument("--pfr-diameter", type=float, default=0.02, help="Reactor diameter [m]")
    p.add_argument("--pfr-mdot", type=float, default=1e-3, help="Mass flow rate [kg/s]")
    p.add_argument("--pfr-n-points", type=int, default=200, help="Axial resolution")
    p.add_argument("--flux", action="store_true", help="Also enable reaction-flux analysis")
    p.add_argument("--flux-element", default="C")
    p.set_defaults(func=_cmd_pfr, target_type="PFR")


def _cmd_pfr(args) -> int:
    from rmip.simulation.pfr_runner import PFRRunner
    from rmip.types import SimCondition

    _ensure_phase6_schema(args.db)
    cond = SimCondition(T_K=args.T_K, P_atm=args.P_atm, phi=args.phi,
                        fuel=args.fuel, oxidizer=args.oxidizer)
    runner = PFRRunner(args.mech_id, args.db, args.base_target_opt, args.run_script)

    if args.dry_run:
        from rmip.simulation.muqsac_bridge import MuqsacBridge
        bridge = MuqsacBridge(args.db, args.run_script)
        target_opt = bridge.build_target_files(
            base_target_opt=args.base_target_opt, mechanism_id=args.mech_id,
            target_type="PFR", solver=_SOLVER_ALIASES[args.solver], condition=cond,
            work_dir=args.work_dir,
            extra_add={"pfr_length": args.pfr_length, "pfr_diameter": args.pfr_diameter,
                       "pfr_mdot": args.pfr_mdot, "pfr_n_points": args.pfr_n_points,
                       "species": args.species},
            flux_analysis=args.flux, flux_element=args.flux_element,
        )
        print(f"Dry run — files written, pipeline NOT invoked:")
        print(f"  {target_opt}")
        print(f"  Would run: python3 {args.run_script} {target_opt}")
        return 0

    result = runner.run_single(
        cond, solver=_SOLVER_ALIASES[args.solver], work_dir=args.work_dir,
        pfr_length=args.pfr_length, pfr_diameter=args.pfr_diameter,
        pfr_mdot=args.pfr_mdot, pfr_n_points=args.pfr_n_points,
        flux_analysis=args.flux, flux_element=args.flux_element,
        extra_add={"species": args.species} if args.species else None,
        run_timeout=args.timeout,
    )
    print(f"\nPFR — mechanism_id={args.mech_id}, solver={args.solver}")
    print(f"  converged: {result.converged}")
    if result.error_msg:
        print(f"  error: {result.error_msg}")
    print(f"  residence_time_s: {result.residence_time_s}")
    print(f"  species tracked in output: {len(result.species_mole_fractions)}")
    # M16: a failed/unconverged run must exit non-zero, not silently return 0.
    return 0 if result.converged else 1


def _add_muqsac_bridge_args(p, reactor: bool) -> None:
    """Shared arguments for jsr/pfr/flux commands (all drive MuqsacBridge)."""
    p.add_argument("mech_id", type=int, help="Mechanism DB id")
    p.add_argument("--db", required=True, help="SQLite database path")
    p.add_argument("--solver", required=True, choices=list(_SOLVER_ALIASES.keys()),
                   help="Solver backend")
    p.add_argument("--fuel", required=True, help='Fuel composition string, e.g. "NC7H16:1.0"')
    p.add_argument("--oxidizer", default="O2:1.0, N2:3.76")
    p.add_argument("--T", type=float, required=True, dest="T_K", help="Temperature [K]")
    p.add_argument("--P", type=float, required=True, dest="P_atm", help="Pressure [atm]")
    p.add_argument("--phi", type=float, default=1.0)
    p.add_argument("--species", default=None,
                   help="Species whose mole fraction is tracked as the scalar result "
                        "(default: the first fuel species)")
    p.add_argument("--base-target-opt", required=True,
                   help="Existing target.opt supplying Bin/Locations/Stats/etc. "
                        "(e.g. from generate_target_opt.py)")
    p.add_argument("--run-script", default="src/MUQ-SAC/run_script.py",
                   help="Path to src/MUQ-SAC/run_script.py")
    p.add_argument("--work-dir", required=True,
                   help="Directory for the generated .target/.add/target.opt and pipeline output/")
    p.add_argument("--timeout", type=float, default=None, help="Subprocess timeout [s]")
    p.add_argument("--dry-run", action="store_true",
                   help="Write .target/.add/target.opt only — do NOT invoke run_script.py "
                       "(inspect the generated files / command before a real run)")
    if reactor:
        p.add_argument("--target-type", choices=["JSR", "PFR"], required=True,
                        help="Which reactor deck to generate for the flux analysis")


def _run_muqsac_bridge_case(args, target_type: str, flux_analysis: bool):
    """Shared build+run for the `flux` command (jsr/pfr have their own runners)."""
    from rmip.simulation.muqsac_bridge import MuqsacBridge
    from rmip.types import SimCondition

    _ensure_phase6_schema(args.db)
    cond = SimCondition(T_K=args.T_K, P_atm=args.P_atm, phi=args.phi,
                        fuel=args.fuel, oxidizer=args.oxidizer)
    bridge = MuqsacBridge(args.db, args.run_script)
    extra_add = {"species": args.species} if args.species else None
    target_opt = bridge.build_target_files(
        base_target_opt=args.base_target_opt, mechanism_id=args.mech_id,
        target_type=target_type, solver=_SOLVER_ALIASES[args.solver], condition=cond,
        work_dir=args.work_dir, extra_add=extra_add,
        flux_analysis=flux_analysis, flux_element=getattr(args, "element", "C"),
    )
    if args.dry_run:
        print(f"Dry run — files written, pipeline NOT invoked:")
        print(f"  {target_opt}")
        print(f"  Would run: python3 {args.run_script} {target_opt}")
        return None

    proc = bridge.run(target_opt, timeout=args.timeout)
    if proc.returncode != 0:
        print(f"WARNING: MUQ-SAC pipeline exited with code {proc.returncode}")
        if proc.stderr:
            print(proc.stderr[-2000:])
    return {"condition": cond, "target_opt": target_opt, "proc": proc}


def _frange(start: float, stop: float, step: float):
    n = int(round((stop - start) / step)) + 1
    for i in range(n):
        yield start + i * step


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())
