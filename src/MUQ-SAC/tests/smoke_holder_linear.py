#!/usr/bin/env python3.9
"""Smoke test for ResponseEncoder's holder_linear (per-sample p-Holder)
variant + regression roundtrips for power/rank.

Run from the MUQ-SAC dir:  python3.9 tests/smoke_holder_linear.py
Reads existing simulation data (no Chemkin, no writes into study dirs).
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase10_prs import ResponseEncoder  # noqa: E402

PILOT_LST = "/data/KRUNAL/pilot_5case_case0/Opt/Data/Simulations/sim_data_case-0.lst"
TCC_LST = ("/data/KRUNAL/MULTI-STAGE_OPTIMIZATION/TEST_CORRELATIONS_CORRECTED/"
           "Opt/Data/Simulations/sim_data_case-0.lst")


def load_y(lst_path):
    """sim_data_case-N.lst rows are 'row_idx <tab> ln_tau_us'."""
    y = []
    with open(lst_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                y.append(float(parts[1]))
    return np.asarray(y, dtype=float)


def check_roundtrip(enc, y, tag, tol=1e-9):
    z = enc.encode(y)
    y_rt = enc.decode(z)
    err = float(np.max(np.abs(y_rt - y)))
    assert err < tol, f"{tag}: roundtrip max err {err:.3e} >= {tol}"
    print(f"[{tag}] n={len(y)}  roundtrip max err = {err:.3e}  OK")


def check_holder_linear(y, tag):
    enc = ResponseEncoder(variant="holder_linear", p_lo=0.02, criteria_y=2.0)
    enc.fit(y)
    n = len(y)

    # 1. p bounds: all per-sample p_i inside [p_lo, 1]
    assert enc.p_values is not None and len(enc.p_values) == n
    assert np.all(enc.p_values >= 0.02 - 1e-9), \
        f"{tag}: p_i below p_lo ({enc.p_values.min():.6f})"
    assert np.all(enc.p_values <= 1.0 + 1e-9), \
        f"{tag}: p_i above 1 ({enc.p_values.max():.6f})"

    # 2. exact roundtrip at fit points
    check_roundtrip(enc, y, f"{tag}/holder_linear")

    # 3. tables strictly increasing, no NaN
    assert np.all(np.diff(enc.table_u) > 0), f"{tag}: table_u not increasing"
    assert np.all(np.diff(enc.table_z) > 0), f"{tag}: table_z not increasing"
    z = enc.encode(y)
    assert not np.any(np.isnan(z)), f"{tag}: NaN in encode"

    # 4. out-of-range z clamps (no NaN) to the table endpoints
    z_out = np.array([enc.z_min - 10.0, enc.z_max + 10.0])
    y_out = enc.decode(z_out)
    assert not np.any(np.isnan(y_out)), f"{tag}: NaN in out-of-range decode"
    assert np.allclose(y_out, [enc.table_u[0], enc.table_u[-1]], atol=1e-12), \
        f"{tag}: out-of-range decode did not clamp to table endpoints"

    # 5. save -> load -> identical decode
    with tempfile.TemporaryDirectory() as enc_dir:
        enc.save(enc_dir, 0)
        files = sorted(os.listdir(enc_dir))
        assert f"p_values_case-0.npy" in files, f"{tag}: p_values not saved"
        assert f"holder_table_case-0.npy" in files, f"{tag}: table not saved"
        assert f"encoder_case-0_holder_linear.json" in files, f"{tag}: json not saved"
        enc2 = ResponseEncoder.load(enc_dir, 0)
        err = float(np.max(np.abs(enc2.decode(enc2.encode(y)) - y)))
        assert err < 1e-9, f"{tag}: reload roundtrip err {err:.3e}"

    st = enc.stats
    print(f"[{tag}/holder_linear] p in [{st['p_min']:.4f}, {st['p_max']:.4f}]"
          f"  z in [{st['z_min']:.4f}, {st['z_max']:.4f}]"
          f"  linear_r2={st['linear_r2']:.6f}"
          f"  rms_residual={st['rms_residual']:.4f}"
          f"  n_clipped_plo={st['n_clipped_plo']}"
          f"  n_clipped_p1={st['n_clipped_p1']}")
    return enc


def check_regressions(y, tag):
    for variant, kw in (("power", {"P": 0.1}), ("rank", {})):
        enc = ResponseEncoder(variant=variant, criteria_y=2.0, **kw)
        enc.fit(y)
        check_roundtrip(enc, y, f"{tag}/{variant}")


def main():
    for path, tag in ((PILOT_LST, "pilot"), (TCC_LST, "tcc")):
        if not os.path.isfile(path):
            print(f"[{tag}] {path} missing -- skipped")
            continue
        y = load_y(path)
        print(f"[{tag}] loaded {len(y)} responses, y in "
              f"[{y.min():.4f}, {y.max():.4f}]")
        check_holder_linear(y, tag)
        check_regressions(y, tag)
    print("\nSMOKE TEST: all asserts passed")


if __name__ == "__main__":
    main()
