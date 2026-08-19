"""
For each flagged band-invalid row (Case 3 row 609, Case 0 row 1702), plots
EVERY reaction in the case's identifiable layout -- not just the single
worst-offending one violation_detail reports -- showing the physical band,
background samples from each of the sampler's three generator classes
(A/B/C), and the row's own curve, colored by whichever class it matches.

Plotting itself is delegated to ResponseSurface.RowGeneratorDiagnostics
(the production class); this script only assembles the case's
layout/blocks and calls it.

Zero new simulations. Generator-class attribution is an exact
nearest-neighbor match against DesignMatrix.csv (see
RowGeneratorDiagnostics' docstring) -- confirmed 2026-07-31 both flagged
rows match Class B at ~1e-9 distance, i.e. exactly.

TARGETS below are kept VERBATIM from test_build/plot_row_generator_diagnostics.py.
Only plumbing changed: the hardcoded absolute sys.path insert is replaced
with `_bootstrap`, loaders come from case_data, D/OUT come from config.
Additionally writes results_dir("tenfold")/row_generator_diagnostics.json.

Run: python3.9 plot_row_generator_diagnostics.py [--output-dir DIR] [--results-dir DIR]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # validation/
import config
config.set_single_threaded_blas()   # MUST run before `import numpy` below

import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

import _bootstrap
from IdentifiablePRS import build_layout_blocks
from ResponseSurface import RowGeneratorDiagnostics
import case_data

TARGETS = [
    dict(case=3, row=609, worst_rxn="nc7h16 + ho2 => c7h15-4 + h2o2"),
    dict(case=0, row=1702, worst_rxn="nc7h16 + ho2 => c7h15-2 + h2o2"),
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None, help="Figure output dir (default: config.figures_dir('tenfold'))")
    ap.add_argument("--results-dir", default=None, help="Metrics JSON dir (default: config.results_dir('tenfold'))")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir
    D = data_dir if data_dir is not None else config.DATA_DIR
    out_figs = args.output_dir or config.figures_dir("tenfold")
    out_results = args.results_dir or config.results_dir("tenfold")
    os.makedirs(out_figs, exist_ok=True)
    os.makedirs(out_results, exist_ok=True)

    rxns, sigma_of_rxn = case_data.load_sigma_and_rxns(data_dir=data_dir)
    Temp = case_data.load_temps(data_dir=data_dir)
    rxn_index = {r: j for j, r in enumerate(rxns)}

    results = []
    for tgt in TARGETS:
        case, row, worst_rxn = tgt["case"], tgt["row"], tgt["worst_rxn"]
        X0, y0, sel = case_data.load_case(case, data_dir=data_dir)
        layout = case_data.make_layout(sel, rxns)
        Tc = Temp[case]
        T_arr = np.array([Tc * 0.85, Tc, Tc * 1.25])
        blocks = build_layout_blocks(layout, lambda r: sigma_of_rxn[r], T_arr, svd_tol=1e-6)

        diag = RowGeneratorDiagnostics(
            case=case, blocks=blocks, rxn_index=rxn_index,
            design_matrix_raw_csv_path=f"{D}DM_FOR_PARTIAL_PRS/{case}/DesignMatrix.csv",
            class_pool_dir=f"{D}DM_FOR_PARTIAL_PRS/{case}/",
            T_min=Tc * 0.85, T_max=Tc * 1.25)

        out_path, generator_by_rxn = diag.plot_row(row, X0, location=out_figs)

        cls, pool_idx, dist = diag.identify_generator(row, worst_rxn)
        n_fail = sum(1 for b in blocks
                     if float((np.abs(X0[row, b.col_slice] @ b.W.T) / b.fp).max()) > 1.0)
        print(f"Case {case} row {row}: worst-offending reaction '{worst_rxn}' "
              f"matched generator Class {cls.upper() if cls else '?'} (dist={dist:.2e}); "
              f"{n_fail}/{len(blocks)} reactions band-invalid for this row")
        print(f"  per-reaction generators: {generator_by_rxn}")
        print(f"  wrote {out_path}")
        results.append({
            "case": case, "row": row, "worst_rxn": worst_rxn,
            "matched_class": cls, "match_distance": float(dist),
            "n_reactions_band_invalid": n_fail, "n_reactions_total": len(blocks),
            "generator_by_rxn": generator_by_rxn, "figure_path": out_path,
        })

    out_path = os.path.join(out_results, "row_generator_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump({"targets": results}, f, indent=2)
    print(f"\nresults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
