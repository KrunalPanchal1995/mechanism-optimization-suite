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
    return args.func(args)


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

    # ---- Phase 3 ----
    _add_sparsity_cmd(sub)
    _add_sensitivity_cmd(sub)
    _add_exp_load_cmd(sub)
    _add_bench_cmd(sub)
    _add_compare_graph_cmd(sub)

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

    # ---- Utilities ----
    _add_reactions_cmd(sub)
    _add_reaction_matrix_cmd(sub)

    return p


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
          f"partial: {sp['n_partial']}, unique: {sp['n_unique']}")
    print(f"  Reactions— shared: {rxn['n_shared']}, unique: {rxn['n_unique']}")
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
        viz.run(
            port=args.port,
            label_mode=args.label,
            threshold=getattr(args, "threshold", 0.1),
            score=getattr(args, "score", "importance"),
        )
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
    p.add_argument("--regime", choices=["LTC","NTC","HTC","all"], default="all",
                   help="Regime to show in ranking table")
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

    print(f"\nFusion complete → mechanism id={fusion_result.fused_mechanism_id}")
    print(f"  VERBATIM reactions: {fusion_result.n_verbatim}")
    print(f"  COPY reactions:     {fusion_result.n_copy}")
    print(f"  FUSE reactions:     {fusion_result.n_fused}")
    print(f"  Species:            {fusion_result.n_species}")

    # Export artefacts
    spec_path   = out_dir / "union_spec.json"
    bounds_path = out_dir / "mso_bounds.json"
    engine.export_union_spec(fusion_result, str(spec_path))
    engine.export_mso_bounds(fusion_result, str(bounds_path))
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
    print(f"  Objective: {opt_result.objective_initial:.4f} → "
          f"{opt_result.objective_final:.4f}")

    if opt_result.status != "success":
        print(f"  ERROR: {opt_result.error_message}")
        return 1

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
    for regime, mech_id in best.items():
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

    iface = RMGInterface(args.db, work_dir=str(out_dir))

    print(f"\nGenerating RMG pathways for {args.fuel_name} …")
    rmg_mech_id = iface.generate(
        fuel_smiles=args.fuel_smiles,
        fuel_name=args.fuel_name,
        oxidizer=args.oxidizer,
        T_list=args.T,
        P_list=args.P,
        phi=args.phi,
        rmg_exe=args.rmg,
    )
    print(f"  RMG mechanism loaded → DB id={rmg_mech_id}")

    pathways = iface.find_missing_pathways(args.mech_id, rmg_mech_id)
    print(f"\nMissing pathways vs base mechanism id={args.mech_id}:")
    print(f"  Missing species:   {len(pathways.missing_species)}")
    print(f"  Missing reactions: {len(pathways.missing_reactions)}")

    if not pathways.missing_species and not pathways.missing_reactions:
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


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())
