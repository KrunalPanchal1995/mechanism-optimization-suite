#!/usr/bin/env python3
"""
graph_sa_run.py — Sensitivity graph analysis driver.

Reads a MUQ-SAC / MSO ``target.opt`` config plus the sensitivity pickle it
produced (``sens_parameters.pkl``) and renders a graph-based analysis of the
*sensitive reactions* and *sensitive species* on the mechanism's reaction
network — reusing the RMIP graph engine.

Usage
-----
    python graph_sa_run.py DATASET/METHYL_BUTANOATE_2026/target.opt
    python graph_sa_run.py path/to/target.opt --out outputs/graph_sa --top 25

Outputs (written to ``--out``)
------------------------------
    sensitive_subgraph.png       compact, labelled sensitive-chemistry graph
    full_graph_sensitive.png     whole mechanism with sensitive parts highlighted
    sensitive_reactions.csv      ranked reactions (max score + per-target columns)
    sensitive_species.csv        ranked species (aggregated score)

Prerequisite
------------
The SA pickle must already exist next to ``target.opt`` (produced by
``src/MUQ-SAC/run_script.py``).  This driver does not run Cantera/simulations.
"""

import os
import sys
import csv
import argparse
from pathlib import Path

# networkx's spring_layout reaches for ``scipy.sparse`` via ``import scipy as sp``;
# on some scipy builds the submodule is not auto-loaded, so import it explicitly
# here to populate the attribute before any layout call.
import scipy.sparse  # noqa: F401

# ── Expose the RMIP package (lives at src/rmip, imports itself as ``rmip``) ──
# Append — never insert at 0 — so ``src/io`` does not shadow the stdlib ``io``.
_ROOT = Path(__file__).resolve().parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from rmip.config import RMIPConfig
from rmip.core.parser import MechanismParser
from rmip.core.graph_builder import GraphBuilder
from rmip.database.schema import get_connection
from rmip.analysis.sensitivity import SensitivityFluxIntegrator

from graph_sa.sa_bridge import (
    load_target_opt, load_sa_pickle, aggregate_reaction_scores,
    match_scores_to_db, species_sensitivity, to_sensitivity_data,
    build_sensitive_subgraph,
)
from graph_sa import plots


def _get_or_parse_mechanism(db_path: str, mech_path: str, name: str) -> int:
    """Return the DB id of *mech_path*, parsing it into *db_path* if needed."""
    import sqlite3 as _sqlite3
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT id FROM mechanisms WHERE name=?", (name,)).fetchone()
    except _sqlite3.OperationalError:
        row = None  # fresh DB — schema not created yet
    if row:
        print(f"  Mechanism '{name}' already in DB (id={row['id']}) — reusing.")
        conn.close()
        return row["id"]
    conn.close()
    print(f"  Parsing mechanism {mech_path} → {db_path} ...")
    cfg = RMIPConfig(db_path=db_path, log_level="WARNING")
    parser = MechanismParser(cfg)
    mech_id = parser.parse(mech_path, name=name)
    print(f"  Parsed as mechanism id={mech_id}.")
    return mech_id


def _write_reactions_csv(path, matched, per_target):
    target_ids = sorted({t for f in per_target for t in per_target[f]},
                        key=lambda x: int(x) if str(x).isdigit() else x)
    rows = sorted(matched.items(), key=lambda kv: kv[1]["score"], reverse=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "db_id", "formula_string", "sa_formula",
                    "score_norm"] + [f"target_{t}" for t in target_ids])
        for i, (db_fs, info) in enumerate(rows, 1):
            per = per_target.get(info["sa_formula"], {})
            w.writerow([i, info["db_id"], db_fs, info["sa_formula"],
                        f"{info['score']:.4f}"]
                       + [f"{per.get(t, 0.0):.4g}" for t in target_ids])


def _write_species_csv(path, species):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "name", "formula", "carbon_number",
                    "score_norm", "n_sensitive_reactions"])
        for i, s in enumerate(species, 1):
            w.writerow([i, s["name"], s["formula"], s["carbon_number"],
                        f"{s['score']:.4f}", s["n_reactions"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sensitivity graph analysis (graph_sa)")
    ap.add_argument("target_opt", help="Path to target.opt config")
    ap.add_argument("--out", default="outputs/graph_sa", help="Output directory")
    ap.add_argument("--db", default=None,
                    help="SQLite DB path (default: <out>/graph_sa.db)")
    ap.add_argument("--sa-pickle", default=None,
                    help="Override SA pickle path (default: next to target.opt)")
    ap.add_argument("--method", choices=["max", "mean"], default="max",
                    help="Aggregate |coeff| across targets (default: max)")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="Sensitivity below this is greyed (default: 0.1)")
    ap.add_argument("--top", type=int, default=25,
                    help="Rows printed to stdout (default: 25)")
    ap.add_argument("--no-full-graph", action="store_true",
                    help="Skip the (slow) full-mechanism figure")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    db_path = args.db or os.path.join(out_dir, "graph_sa.db")

    print("═" * 70)
    print("  Sensitivity Graph Analysis")
    print("═" * 70)

    # 1. Parse config
    info = load_target_opt(args.target_opt)
    mech_path = info["mechanism"]
    sa_pickle = args.sa_pickle or info["sa_pickle"]
    print(f"  Config       : {info['config_path']}")
    print(f"  Mechanism    : {mech_path}")
    print(f"  SA pickle    : {sa_pickle}")
    print(f"  Analysis type: {info['analysis_type']}")

    if not mech_path or not os.path.exists(mech_path):
        print(f"\nERROR: mechanism file not found: {mech_path}")
        return 2
    if not os.path.exists(sa_pickle):
        print(f"\nERROR: SA pickle not found: {sa_pickle}")
        return 2

    # 2. Mechanism → DB
    name = os.path.splitext(os.path.basename(mech_path))[0]
    mech_id = _get_or_parse_mechanism(db_path, mech_path, name)

    # 3. Load + aggregate SA, match to DB
    sa = load_sa_pickle(sa_pickle)
    norm, raw, per_target = aggregate_reaction_scores(sa, method=args.method)
    print(f"\n  SA targets              : {len(sa)}")
    print(f"  Distinct SA reactions   : {len(norm)}")

    conn = get_connection(db_path)
    matched, unmatched = match_scores_to_db(conn, mech_id, norm)
    print(f"  Matched to mechanism    : {len(matched)}")
    print(f"  Unmatched (skipped)     : {len(unmatched)}")
    if unmatched:
        for f in unmatched[:10]:
            print(f"      · {f}")
        if len(unmatched) > 10:
            print(f"      · ... {len(unmatched) - 10} more")

    species = species_sensitivity(conn, mech_id, matched)

    # 4. Build graph + attach scores
    gb = GraphBuilder(db_path)
    G = gb.build_graph(mech_id)
    m8 = SensitivityFluxIntegrator(db_path)
    m8.attach(mech_id, G, to_sensitivity_data(matched), None)
    subG = build_sensitive_subgraph(G, matched)

    # 5. Render + export
    sp_score_map = {s["name"]: s["score"] for s in species}
    sub_png = os.path.join(out_dir, "sensitive_subgraph.png")
    plots.render_subgraph(subG, sub_png, species_scores=sp_score_map,
                          threshold=args.threshold,
                          title=f"{name} — sensitive reactions & species")
    print(f"\n  ✓ {sub_png}")

    if not args.no_full_graph:
        full_png = os.path.join(out_dir, "full_graph_sensitive.png")
        plots.render_full_highlighted(G, full_png, threshold=args.threshold,
                                      title=f"{name} — full mechanism")
        print(f"  ✓ {full_png}")

    rxn_csv = os.path.join(out_dir, "sensitive_reactions.csv")
    sp_csv = os.path.join(out_dir, "sensitive_species.csv")
    _write_reactions_csv(rxn_csv, matched, per_target)
    _write_species_csv(sp_csv, species)
    print(f"  ✓ {rxn_csv}")
    print(f"  ✓ {sp_csv}")

    # 6. stdout ranking
    print("\n" + "─" * 70)
    print(f"  Top {args.top} sensitive reactions (max |coeff| across targets)")
    print("─" * 70)
    for i, (db_fs, d) in enumerate(
            sorted(matched.items(), key=lambda kv: kv[1]["score"], reverse=True)[:args.top], 1):
        print(f"  {i:2d}. {d['score']:.3f}  {db_fs}")

    print("\n" + "─" * 70)
    print(f"  Top {args.top} sensitive species")
    print("─" * 70)
    for i, s in enumerate(species[:args.top], 1):
        print(f"  {i:2d}. {s['score']:.3f}  {s['name']:<14s} "
              f"(C{s['carbon_number']}, in {s['n_reactions']} sensitive rxns)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
