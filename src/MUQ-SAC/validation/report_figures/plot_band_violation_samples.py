"""
Arrhenius-space plot of REAL design-matrix rows (the actual ResponseSurface
PRS input, DM_FOR_PARTIAL_PRS/<case>/pDesignMatrix.csv) for the two
band-invalid rows named explicitly in report_10fold_validation/main.tex's
diagnosis section:
    Case 3, row 609, reaction "nc7h16 + ho2 => c7h15-4 + h2o2", ratio 26.99
    Case 0, row 1702, reaction "nc7h16 + ho2 => c7h15-2 + h2o2", ratio 27.55
plus ~15 band-valid rows of the SAME case+reaction for visual contrast, using
VisualAid.SampleViolationPlotter.

Zero new simulations -- reads the same cached design matrices / Sigma
matrices the 10-fold CV audit used.

TARGETS below are kept VERBATIM from test_build/plot_band_violation_samples.py.
The only changes here are plumbing: the two hardcoded absolute
`sys.path.insert(0, ...)` calls (one for test_build/, one for src/MUQ-SAC/)
collapse to one `_bootstrap` call, loaders come from case_data, and D/OUT
come from config. Additionally writes
results_dir("tenfold")/band_violation_samples.json.

Run: python3.9 plot_band_violation_samples.py [--output-dir DIR] [--results-dir DIR]
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

import _bootstrap
from IdentifiablePRS import build_layout_blocks
import VisualAid as VA
import case_data

TARGETS = [
    dict(case=3, row=609, rxn="nc7h16 + ho2 => c7h15-4 + h2o2"),
    dict(case=0, row=1702, rxn="nc7h16 + ho2 => c7h15-2 + h2o2"),
]


def prep_case(case, rxns, sigma_of_rxn, Temp, data_dir=None):
    X0, y0, sel = case_data.load_case(case, data_dir=data_dir)
    Xo, yo = case_data.load_outlier_pool(case, data_dir=data_dir)
    X = np.vstack([X0, Xo]) if len(yo) > 0 else X0
    layout = case_data.make_layout(sel, rxns)
    Tc = Temp[case]
    T_arr = np.array([Tc * 0.85, Tc, Tc * 1.25])
    blocks = build_layout_blocks(layout, lambda r: sigma_of_rxn[r], T_arr, svd_tol=1e-6)
    return X, blocks, Tc


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

    results = []
    for tgt in TARGETS:
        case, target_row, rxn = tgt["case"], tgt["row"], tgt["rxn"]
        X, blocks, Tc = prep_case(case, rxns, sigma_of_rxn, Temp, data_dir=data_dir)
        block = next(b for b in blocks if b.rxn == rxn)

        Z = X[:, block.col_slice]                       # (N, m) raw zeta_r for this reaction
        dk_at_Tarr = Z @ block.W.T                        # (N, nT) at the 3 sampler-check temperatures
        ratio_rxn = (np.abs(dk_at_Tarr) / block.fp[None, :]).max(axis=1)  # per-row, THIS reaction only

        valid_rows = np.where(ratio_rxn <= 1.0)[0]
        invalid_rows = np.where(ratio_rxn > 1.0)[0]
        assert target_row in invalid_rows, (
            f"expected row {target_row} to be band-invalid for {rxn}, "
            f"got ratio={ratio_rxn[target_row]:.2f}")

        rng = np.random.default_rng(0)
        show_valid = rng.choice(valid_rows, size=min(15, len(valid_rows)), replace=False)

        nominal_path = f"{D}Uncertainty_Matrices/{rxn}/nominal.csv"
        P0 = VA.SampleViolationPlotter.load_nominal(nominal_path)
        plotter = VA.SampleViolationPlotter(rxn, P0, block.L_r, block.indices,
                                             T_min=Tc * 0.85, T_max=Tc * 1.25)

        valid_samples = [(Z[i], int(i), float(ratio_rxn[i])) for i in show_valid]
        invalid_samples = [(Z[target_row], int(target_row), float(ratio_rxn[target_row]))]

        out_path = plotter.plot(valid_samples, invalid_samples, location=out_figs, case=case)
        pct_invalid = 100 * len(invalid_rows) / len(ratio_rxn)
        print(f"Case {case} row {target_row} ({rxn}): ratio={ratio_rxn[target_row]:.2f}, "
              f"{len(invalid_rows)}/{len(ratio_rxn)} rows band-invalid for this reaction "
              f"({pct_invalid:.1f}%)")
        print(f"  wrote {out_path}")
        results.append({
            "case": case, "row": target_row, "rxn": rxn,
            "ratio": float(ratio_rxn[target_row]),
            "n_invalid_rxn": int(len(invalid_rows)), "n_total_rxn": int(len(ratio_rxn)),
            "pct_invalid_rxn": pct_invalid, "figure_path": out_path,
        })

    out_path = os.path.join(out_results, "band_violation_samples.json")
    with open(out_path, "w") as f:
        json.dump({"targets": results}, f, indent=2)
    print(f"\nresults written to {out_path}")
    return results


if __name__ == "__main__":
    main()
