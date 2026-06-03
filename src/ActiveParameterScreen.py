#!/usr/bin/env python3
"""
ActiveParameterScreen.py

Main driver for the Active Parameter Framework.

Pipeline:
    SA files --> parser --> screening --> results CSVs
    classification CSV (user-supplied) --> XML / plots / LaTeX
    mechanism YAML --> filtered mechanism (only if requested)

All imports and call signatures match the modules as actually written.
"""

import argparse
import os

import pandas as pd

from modules.interfaces import load_classification_file
from modules import sensitivity_parser as sp
from modules import mechanism_utils as mu
from modules.screening import run_screening
from modules.xml_generator import generate_xml_files
from modules.plotting import generate_group_plots
from modules.latex_tables import generate_latex_tables
from modules.filtered_mechanism import generate_filtered_mechanism


def get_args():
    p = argparse.ArgumentParser(description="Active Parameter Framework")
    p.add_argument("--sens-dir", default=".")
    p.add_argument("--mechanism", default="mechanism.yaml")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--carbon", type=int, default=0)
    p.add_argument("--classification-file", default=None,
                   help="User-supplied CSV with Reaction_ID,Reaction,Group,f")
    p.add_argument("--fuel", default="Fuel")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--generate-xml", action="store_true")
    p.add_argument("--generate-plots", action="store_true")
    p.add_argument("--per-case-plots", action="store_true",
                   help="Also emit one plot per (case, temperature), "
                        "with the case id in the filename.")
    p.add_argument("--generate-latex", action="store_true")
    p.add_argument("--longtable", action="store_true")
    p.add_argument("--filtered-mechanism", action="store_true")
    p.add_argument("--compare-temperatures", nargs="+", type=float,
                   default=None)
    p.add_argument("--compare-colors", nargs="+", default=None,
                   help="Optional colors for the comparison plot, one per "
                        "temperature (e.g. --compare-colors red blue green "
                        "or hex like '#1f77b4'). Cycled if fewer than the "
                        "number of temperatures. Defaults to matplotlib's "
                        "color cycle.")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n=== Active Parameter Screening ===\n")

    # 1. Parse SA files (single pass -> both canonical forms)
    detailed_records = sp.build_detailed_records(args.sens_dir)
    sens_db = sp.build_sensitivity_database(args.sens_dir)
    sp.summarize_database(sens_db)

    # 2. Mechanism lookup + carbon map
    mech = mu.load_mechanism(args.mechanism)
    lookup = mu.build_reaction_lookup(mech)
    carbon_map = mu.build_species_carbon_map(mech)

    # 3. Screen
    result = run_screening(
        detailed_records, lookup, carbon_map,
        threshold=args.threshold, carbon_threshold=args.carbon,
    )

    pd.DataFrame(result.detailed_rows).to_csv(
        os.path.join(args.results_dir, "active_reactions_detailed.csv"),
        index=False)
    pd.DataFrame(result.master_rows).to_csv(
        os.path.join(args.results_dir, "master_active_reactions.csv"),
        index=False)
    pd.DataFrame(result.species_rows).to_csv(
        os.path.join(args.results_dir, "active_species.csv"), index=False)
    pd.DataFrame(result.summary_rows).to_csv(
        os.path.join(args.results_dir, "active_reaction_summary.csv"),
        index=False)
    if result.mismatch_log:
        with open(os.path.join(args.results_dir,
                               "reaction_id_mismatch.log"), "w") as f:
            f.writelines(result.mismatch_log)
        print(f"\n[WARN] {len(result.mismatch_log)} SA/YAML mismatches "
              f"-> reaction_id_mismatch.log")

    print(f"\nScreening complete. {len(result.master_rows)} active reactions.")

    # 4. Classification-driven outputs
    needs_class = (args.generate_xml or args.generate_plots
                   or args.generate_latex)
    if needs_class:
        if not args.classification_file:
            raise SystemExit(
                "XML/plots/LaTeX require --classification-file "
                "(Reaction_ID,Reaction,Group,f[,data_type,Reaction_Type,"
                "RxnCount]).")
        cdf = load_classification_file(args.classification_file)

        if args.generate_xml:
            print("\nGenerating XML...")
            generate_xml_files(cdf, os.path.join(args.results_dir, "xml"),
                               reaction_lookup=lookup)

        if args.generate_plots:
            print("\nGenerating plots...")
            case_db = None
            if args.per_case_plots:
                case_db = sp.build_case_database(args.sens_dir)
            generate_group_plots(
                cdf, sens_db, os.path.join(args.results_dir, "plots"),
                threshold=args.threshold,
                compare_temperatures=args.compare_temperatures,
                case_db=case_db,
                compare_colors=args.compare_colors,
            )

        if args.generate_latex:
            print("\nGenerating LaTeX tables...")
            generate_latex_tables(
                cdf, os.path.join(args.results_dir, "latex"),
                fuel_name=args.fuel, longtable=args.longtable,
            )

    # 5. Filtered mechanism (only if requested)
    if args.filtered_mechanism:
        print("\nGenerating filtered mechanism...")
        master = pd.DataFrame(result.master_rows)
        generate_filtered_mechanism(
            mech,
            reaction_ids=master["Reaction_ID"].tolist(),
            outfile=os.path.join(args.results_dir, "filtered_mechanism.yaml"),
        )

    print("\nDone. Results in", args.results_dir)


if __name__ == "__main__":
    main()
