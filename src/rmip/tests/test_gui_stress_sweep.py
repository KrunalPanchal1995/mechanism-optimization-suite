"""
rmip/tests/test_gui_stress_sweep.py
====================================
GUI_STRESS_TEST_PLAN.md executed: an in-process worker-function run for all
8 standalone launcher GUI apps against the checked-in dummy fixture set
(src/rmip/tests/fixtures/gui_stress/, built by make_gui_stress_fixtures.py).

"In-process" per the plan's locked decision: every test here calls the same
worker function / underlying CLI module a button click would invoke, NOT a
literal Tk widget click on a live display (the shared X display `:52.0`
hazard flagged 2026-07-17 is still a live risk). Where an app only shells
out to a subprocess/PTY (brute_force_sa, nominal_sim, forward_uq,
mso_compare, rmip_suite), the test invokes the underlying CLI/module
directly via `subprocess.run` -- the exact command the GUI's PtyTerminal
would run, just without the PTY/terminal-emulation layer.

Fixtures are session-scoped (built once, reused by every app's test) so
results are directly comparable and each test only pays its own app's cost.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gui_stress"


@pytest.fixture(scope="module")
def fixtures_dir(tmp_path_factory):
    """Fresh copy of the checked-in gui_stress fixtures per test module run,
    so tests that write into the fixture dir (DB mutations, generated
    outputs) never dirty the checked-in copy."""
    dst = tmp_path_factory.mktemp("gui_stress_fixtures")
    shutil.copytree(_FIXTURES, dst, dirs_exist_ok=True)
    return dst


# =============================================================================
# App 1 -- mechanism_converter_gui.py
# =============================================================================

def test_mechanism_converter_golden_path(fixtures_dir, tmp_path):
    """convert() in-process: CHEMKIN -> Cantera YAML + RMIP DB, golden path
    (mech_A, real params, no corruption)."""
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO))
    from mechanism_converter_gui import convert

    out_dir = tmp_path / "golden"
    log_lines = []
    errs = convert(
        str(fixtures_dir / "mech_A.inp"),
        str(fixtures_dir / "thermo_A.dat"),
        "",
        str(out_dir),
        make_yaml=True, make_pre=False, make_db=True,
        scanman="", db_path=str(out_dir / "out.db"),
        glossary=str(fixtures_dir / "glossary.csv"),
        permissive=True, no_validate=False, fix_exp=True, fix_crlf=True,
        log=log_lines.append,
    )
    assert errs == [], errs

    # Content checks, not just exit code.
    import cantera as ct
    yaml_path = out_dir / "mech_A.yaml"
    assert yaml_path.is_file()
    gas = ct.Solution(str(yaml_path))
    assert gas.n_species == 9
    assert gas.n_reactions == 8

    import sqlite3
    conn = sqlite3.connect(out_dir / "out.db")
    n_mechs = conn.execute("SELECT COUNT(*) FROM mechanisms").fetchone()[0]
    assert n_mechs == 1
    row = conn.execute(
        "SELECT n_species, n_reactions FROM mechanisms WHERE name='mech_A'"
    ).fetchone()
    assert row == (9, 8)


def test_mechanism_converter_pre_with_transport(fixtures_dir, tmp_path):
    """convert() .pre path via real ScanMan -- needs both thermo and
    transport; verifies content (species/thermo coverage), not just that
    the subprocess exited 0."""
    scanman = "/home/krithika/FlameMaster/Bin/bin/ScanMan"
    if not Path(scanman).exists():
        pytest.skip("ScanMan binary not present on this machine")

    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO))
    from mechanism_converter_gui import convert

    # Ad hoc transport file, mech_A spellings, real Connaire-2004 LJ params
    # (MECHANISM/Connaire/h2_v1a_tran.txt), keyed by identity.
    trans_raw = {
        "argon_a": "0   136.500     3.330     0.000     0.000     0.000",
        "oxo":     "0    80.000     2.750     0.000     0.000     0.000",
        "dioxygenium": "1   107.400     3.458     0.000     1.600     3.800",
        "hydroxyl":    "1    80.000     2.750     0.000     0.000     0.000",
        "aqua":        "2   572.400     2.605     1.844     0.000     4.000",
        "peroxidum":   "2   107.400     3.458     0.000     0.000     3.800",
        "protium":     "0   145.000     2.050     0.000     0.000     0.000",
        "diprotium":   "1    38.000     2.920     0.000     0.790   280.000",
        "hydroperox":  "2   107.400     3.458     0.000     0.000     1.000",
    }
    trans_path = tmp_path / "trans_A.txt"
    trans_path.write_text("\n".join(f"{k:<18} {v}" for k, v in trans_raw.items()) + "\n")

    out_dir = tmp_path / "pre_out"
    log_lines = []
    errs = convert(
        str(fixtures_dir / "mech_A.inp"),
        str(fixtures_dir / "thermo_A.dat"),
        str(trans_path),
        str(out_dir),
        make_yaml=False, make_pre=True, make_db=False,
        scanman=scanman, db_path="", glossary="",
        permissive=True, no_validate=False, fix_exp=False, fix_crlf=False,
        log=log_lines.append,
    )
    assert errs == [], errs
    pre_path = out_dir / "mech_A.pre"
    assert pre_path.is_file()
    assert pre_path.stat().st_size > 0        # .pre is FlameMaster's binary format
    assert "Thermodynamic data found for 9/9" in "\n".join(log_lines)


def test_mechanism_converter_autofix_exponent_and_crlf(fixtures_dir, tmp_path):
    """Deliberately corrupt one Arrhenius A-factor to the Fortran no-E
    exponent form and switch to CRLF line endings, then confirm the
    auto-fix path (fix_fortran_exponents / preprocess_copy) repairs it and
    ck2yaml succeeds on the repaired copy -- not just the clean-input path."""
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO))
    from mechanism_converter_gui import convert, fix_fortran_exponents

    src_text = (fixtures_dir / "mech_A.inp").read_text()
    assert "3.161E+04" in src_text
    broken = src_text.replace("3.161E+04", "3.161+04")

    fixed_text, n_fixes = fix_fortran_exponents(broken)
    assert n_fixes == 1
    assert "3.161E+04" in fixed_text

    broken_crlf = broken.replace("\n", "\r\n")
    broken_path = tmp_path / "mech_A_broken.inp"
    broken_path.write_bytes(broken_crlf.encode())

    out_dir = tmp_path / "broken_out"
    log_lines = []
    errs = convert(
        str(broken_path), str(fixtures_dir / "thermo_A.dat"), "",
        str(out_dir),
        make_yaml=True, make_pre=False, make_db=False,
        scanman="", db_path="", glossary="",
        permissive=True, no_validate=False, fix_exp=True, fix_crlf=True,
        log=log_lines.append,
    )
    assert errs == [], errs
    joined = "\n".join(log_lines)
    assert "CRLF -> LF" in joined
    assert "Fortran exponent(s) repaired" in joined

    import cantera as ct
    yaml_path = out_dir / "mech_A_broken.yaml"
    assert yaml_path.is_file()
    gas = ct.Solution(str(yaml_path))
    assert gas.n_species == 9
    assert gas.n_reactions == 8


def test_mechanism_converter_self_contained_thermo_not_doubled(fixtures_dir, tmp_path):
    """Real-mechanism regression (found 2026-07-17 stress-testing
    mechanism_converter_gui against the real Keromnes/Li/Davis fused
    mechanism, which -- like many real fused/merged mechanism exports --
    is a single self-contained .inp with its own inline THERMO ALL block):
    passing that SAME file as both `inp` and `thermo` (a natural mistake for
    any self-contained mechanism, since the GUI has a separate thermo field
    the user must decide whether to fill in) made ck2yaml re-parse the
    REACTIONS section a second time while processing the "additional
    thermo" file, silently doubling every reaction in the output YAML --
    112 reactions instead of 56 on the real mechanism, and outright
    crashing with "Undeclared duplicate reaction" when two of the doubled
    reactions happened to collide. The small dummy fixtures never caught
    this because they always use a separate thermo file (mech_A.inp +
    thermo_A.dat) -- never a self-contained one.

    Repro here: concatenate mech_A.inp + thermo_A.dat into one
    self-contained file (inline THERMO block ahead of REACTIONS, same
    shape as the real fused mechanism) and pass it as both `inp` and
    `thermo`. Fixed in `convert()`: when the resolved thermo path equals
    the resolved input path, the redundant --thermo argument is dropped
    before invoking ck2yaml.
    """
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO))
    from mechanism_converter_gui import convert

    mech_text = (fixtures_dir / "mech_A.inp").read_text()
    thermo_text = (fixtures_dir / "thermo_A.dat").read_text()
    # Splice the THERMO block in ahead of REACTIONS, mirroring real
    # self-contained CHEMKIN exports (SPECIES -> THERMO ALL -> REACTIONS).
    head, _, tail = mech_text.partition("REACTIONS")
    self_contained = head + thermo_text + "\n" + "REACTIONS" + tail
    combined_path = tmp_path / "self_contained.inp"
    combined_path.write_text(self_contained)

    out_dir = tmp_path / "self_contained_out"
    log_lines = []
    errs = convert(
        str(combined_path), str(combined_path), "",
        str(out_dir),
        make_yaml=True, make_pre=False, make_db=False,
        scanman="", db_path="", glossary=str(fixtures_dir / "glossary.csv"),
        permissive=True, no_validate=False, fix_exp=True, fix_crlf=True,
        log=log_lines.append,
    )
    assert errs == [], errs
    assert any("redundant --thermo" in l for l in log_lines), log_lines

    import cantera as ct
    yaml_path = out_dir / "self_contained.yaml"
    assert yaml_path.is_file()
    gas = ct.Solution(str(yaml_path))
    assert gas.n_species == 9
    assert gas.n_reactions == 8   # NOT 16 -- the doubling bug's signature


# =============================================================================
# App 2 -- brute_force_sa_gui.py
# =============================================================================

def _write_gui_stress_target_opt(fixtures_dir: Path, out: Path) -> Path:
    """Copy the fixture set's target.opt, repointing every absolute path
    that was baked in at fixture-build time (pointing at the checked-in
    src/rmip/tests/fixtures/gui_stress/) to this test's own tmp_path copy."""
    text = (fixtures_dir / "target.opt").read_text()
    text = text.replace(str(_FIXTURES), str(fixtures_dir))
    dst = out / "target.opt"
    dst.write_text(text)
    return dst


def test_brute_force_sa_b12_and_mono_fuel_fix_no_longer_crash(fixtures_dir, tmp_path):
    """B12 (doubled relative path), the Mono-fuel-dict crash, and M30 (bug.md
    -- a ZeroDivisionError in the `n`/`Ea`-parameter SA-coefficient step for
    any reaction restricted to `factor`-type/A-only uncertainty, which this
    fixture's mechanism uses for every reaction) must all stay fixed: running
    sens_3_params.py against the real target.opt shape must get all the way
    through target loading, uncertainty selection, design-matrix generation,
    real Cantera ignition-delay simulation, and SA-coefficient computation
    for all three Arrhenius parameters (A, n, Ea).

    M30's root cause: DesignMatrix2_0.py's getSA_3P_samples() (the one-at-a-
    time brute-force SA sweep) derived its per-reaction sample index via
    _get_rxn_param_indices(), a helper meant for a *different* caller
    (partial_PRS_system.py's real joint MUQ-SAC PRS/GSA path) where each
    reaction's per-reaction chunk width legitimately varies with its
    activeParameters length. getSA_3P_samples always passes a uniform
    3-per-reaction [A,n,Ea] mask (independently for each of param_type
    A/n/Ea), so for any reaction whose activeParameters length != 3 (any
    `factor`-type reaction, restricted to A-only), the chunk read silently
    misaligned and collapsed to idx=(0,) regardless of the requested
    param_type -- so "n" and "Ea" were never actually sampled for such
    reactions, their delta_n/delta_Ea came out exactly 0.0, and
    normalized_n = 1.0/delta_n_val crashed. Fixed by mapping param_type
    directly to its fixed Arrhenius-parameter index instead of going through
    the variable-width helper.
    """
    target_opt = _write_gui_stress_target_opt(fixtures_dir, tmp_path)
    r = subprocess.run(
        [sys.executable, str(_REPO / "src" / "Sensitivity_Analysis" / "sens_3_params.py"),
         str(target_opt)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=280,
    )
    combined = r.stdout + r.stderr
    assert "doubled" not in combined.lower()
    assert "could not convert string to float" not in combined   # the Mono-fuel bug
    assert "ZeroDivisionError" not in combined                   # M30
    assert "SECTION 14" in combined                    # reached SA-coefficient computation
    assert "η₀ (nominal IDT)" in combined or "nominal IDT" in combined  # real IDT was computed
    assert "3-PARAMETER SENSITIVITY ANALYSIS COMPLETE" in combined
    assert "SA computed for 3 reactions." in combined
    # The script's own deliberate end-of-run sentinel (pre-existing, not part
    # of M30) -- a clean run always ends by raising this.
    assert "3-PARAM SA DONE" in combined


# =============================================================================
# App 3 -- nominal_sim_gui.py
# =============================================================================

def _write_nominal_sim_config(fixtures_dir: Path, out: Path) -> Path:
    cfg = f"""\
project:
  name:        GUI_Stress_Nominal
  working_dir: {out}
  fuel:        H2

muqsac_src_dir: {_REPO / "src" / "MUQ-SAC"}
base_opt_file:  {fixtures_dir / "target.opt"}

simulations:
  nominal_htc:
    mechanism:        {fixtures_dir / "mech_std.yaml"}
    targets:          {fixtures_dir / "targets" / "gui_stress.target"}
    targets_count:    2
    uncertainty_data: {_REPO / "src" / "rmip" / "tests" / "fixtures" / "h2_o2_crosscheck_unsrt.xml"}
    thermo_file:      {fixtures_dir / "thermo_std.dat"}
    trans_file:       {fixtures_dir / "trans_std.txt"}
    output_dir:       ./SIM_RUNS/nominal_htc

merge:
  htc_output_dir: ./MERGE_OUTPUT/HTC
  ltc_output_dir: ./MERGE_OUTPUT/LTC

error_metrics:
  htc_output_dir:       ./ERROR_METRICS/HTC
  ltc_output_dir:       ./ERROR_METRICS/LTC
  uncertainty_fraction: 0.10

plots:
  htc_output_dir: ./PLOTS/HTC
  ltc_output_dir: ./PLOTS/LTC
  plot_per_phi:   true
  plot_combined:  true
  mode:           nominal_only
"""
    dst = out / "mso_config.yaml"
    dst.write_text(cfg)
    return dst


def test_nominal_sim_load_run_labels(fixtures_dir, tmp_path):
    """_load_run_labels() (the in-process call nominal_sim_gui's UI makes
    to populate the run-label dropdown) against a real config."""
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO))
    from nominal_sim_gui import _load_run_labels

    cfg_path = _write_nominal_sim_config(fixtures_dir, tmp_path)
    labels = _load_run_labels(str(cfg_path))
    assert labels == ["nominal_htc"]


def test_nominal_sim_full_run_b14(fixtures_dir, tmp_path):
    """B14 (nominal_sim_gui 'Run' always crashed, no cwd/PYTHONPATH) --
    re-verify by running the exact runpy bootstrap nominal_sim_gui._run_sim
    uses, end to end, and check the produced dataset CSV content (not just
    exit code)."""
    cfg_path = _write_nominal_sim_config(fixtures_dir, tmp_path)
    bootstrap = (
        "import sys, runpy; "
        f"sys.path.insert(0, {str(_REPO / 'src')!r}); "
        "runpy.run_module('mso_compare.run_nominal_sim', run_name='__main__')"
    )
    r = subprocess.run(
        [sys.executable, "-c", bootstrap,
         "--config", str(cfg_path), "--run", "nominal_htc"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    combined = r.stdout + r.stderr
    assert "Status: done" in combined, combined[-2000:]

    csv_path = tmp_path / "SIM_RUNS" / "nominal_htc" / "Plot" / "Dataset" / "Tig" / "dataset_001.csv"
    assert csv_path.is_file()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 2
    nominals = [float(row["Nominal"]) for row in rows]
    assert all(n > 0 for n in nominals)
    # Lower T (1350K, row 2) must give a LONGER ignition delay than higher T
    # (1450K, row 1) -- basic Arrhenius sanity check on the real computed values.
    assert float(rows[0]["T"]) == pytest.approx(1450.0)
    assert float(rows[1]["T"]) == pytest.approx(1350.0)
    assert nominals[1] > nominals[0]


# =============================================================================
# App 4 -- mean_value_fusion_gui.py (2nd pass -- Part 43 covered this already;
# this pass adds the divergent-kinetics FUSE/mean-averaging path, which
# Part 43's identical-across-mechanisms fixture never exercised, and
# confirms the "priority-pick fusion mode" finding below).
# =============================================================================

def test_mean_value_fusion_golden_path(fixtures_dir, tmp_path):
    """_run_fusion() in-process against the 3-mechanism dummy DB: R1/R2/R3
    (byte-identical kinetics across mechanisms) must classify as COPY, not
    FUSE, and the union must be exactly 20 distinct reactions / 9 species,
    matching the graph-comparison structure verified when the fixtures were
    built (9 universal species, 3 identical groups, 17 unique)."""
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO))
    from mean_value_fusion_gui import _run_fusion, _list_mechanisms

    db_path = str(fixtures_dir / "project.db")
    mechs = _list_mechanisms(db_path)
    assert [m[1] for m in mechs] == ["mech_A", "mech_B", "mech_C"]

    out_dir = tmp_path / "fusion_out"
    summary = _run_fusion(db_path, [1, 2, 3], 0.01, 0.001, 100.0,
                           "gui_stress_fused", str(out_dir), True)
    assert summary["n_verbatim"] == 17
    assert summary["n_copy"] == 3
    assert summary["n_fuse"] == 0          # identical kinetics -> COPY, not FUSE
    assert summary["n_species"] == 9
    assert summary["balanced"] is True
    assert summary["violations"] == []

    import cantera as ct
    gas = ct.Solution(str(out_dir / "gui_stress_fused.yaml"))
    assert gas.n_species == 9
    assert gas.n_reactions == 20


def test_mean_value_fusion_divergent_kinetics_geometric_mean(fixtures_dir, tmp_path):
    """When the same canonical reaction has DIVERGENT A-factors across
    mechanisms (beyond tol_A), _run_fusion() must classify it as FUSE and
    compute the geometric mean of A -- the code path Part 43's
    identical-kinetics-only fixture never touched. Perturb R1
    (H+O2<=>O+OH) in mech_B's DB row by 1.5x and verify the fused A matches
    geomean(A, 1.5A, A) = A * 1.5**(1/3) to high precision."""
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO))
    from mean_value_fusion_gui import _run_fusion
    import sqlite3

    db_path = str(tmp_path / "project.db")
    shutil.copy(fixtures_dir / "project.db", db_path)

    conn = sqlite3.connect(db_path)
    # id=9 is R1 (H+O2<=>O+OH, canonical_key 'O=O,[H]>>[OH],[O]') in mech_B
    # (mechanism_id=2) -- established by inspecting the fixture DB when this
    # test was written.
    r1_canonical_key, a_si_orig = conn.execute(
        "SELECT canonical_key, A_si FROM reactions WHERE id=9"
    ).fetchone()
    conn.execute("UPDATE reactions SET A_si=? WHERE id=9", (a_si_orig * 1.5,))
    conn.commit()

    out_dir = tmp_path / "fusion_out_divergent"
    summary = _run_fusion(db_path, [1, 2, 3], 0.01, 0.001, 100.0,
                           "gui_stress_fused_divergent", str(out_dir), True)
    assert summary["n_fuse"] == 1
    assert summary["n_copy"] == 2          # R2/R3 still identical -> still COPY
    assert summary["balanced"] is True

    # Query the fused mechanism's own DB row directly (species spellings
    # differ across mech_A/B/C by design, so matching by Cantera equation
    # string against a canonical "H+O2<=>O+OH" text would never match --
    # the DB's canonical_key is the reliable identity to join on).
    fused_A_si, = conn.execute(
        "SELECT A_si FROM reactions WHERE mechanism_id=? AND canonical_key=?",
        (summary["fused_mech_id"], r1_canonical_key),
    ).fetchone()
    conn.close()
    expected = a_si_orig * (1.5 ** (1 / 3))
    assert fused_A_si == pytest.approx(expected, rel=1e-6)


# =============================================================================
# App 5 -- uncertainty_generator_gui.py (2nd pass -- Part 43 only added the
# "Generate" path; this pass adds the Generate -> Collapse round-trip).
# =============================================================================

def test_uncertainty_generator_roundtrip(fixtures_dir, tmp_path):
    """generate_uncertainty_xml() -> compute_symmetric_bounds() round-trip
    against mech_std.yaml: the XML the Generate path writes must be
    correctly re-read by the Collapse path, with no per-reaction errors and
    physically sane numbers (A_nominal must match the mechanism's real
    Arrhenius A; A_lb < A_nominal < A_ub).

    Also locks in a real bug found+fixed this session: generate_uncertainty_
    xml() always emitted a blank <class> element. MechManipulator2_0.py's
    doPerturbation() dispatches on that value (Elementary/PLOG/Falloff/...)
    and SILENTLY no-ops perturbation for any reaction whose classification
    doesn't match one of its branches -- blank matches none of them. Fixed
    by defaulting <class> to "Elementary" (overridable via
    rxn["classification"]).
    """
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO / "src" / "MUQ-SAC"))
    sys.path.insert(0, str(_REPO))
    from uncertainty_generator_gui import generate_uncertainty_xml, compute_symmetric_bounds

    reactions = [
        {"name": "H + O2 <=> O + OH", "type": "pressure_independent",
         "perturbation_type": "factor", "data_mode": "constant;array",
         "temperatures": [500.0, 1500.0, 2500.0], "uncertainties": [0.40, 0.25, 0.15]},
        {"name": "O + H2 <=> H + OH", "type": "pressure_independent",
         "perturbation_type": "factor", "data_mode": "constant;array",
         "temperatures": [500.0, 1500.0, 2500.0], "uncertainties": [0.35, 0.24, 0.16]},
    ]
    xml_path = tmp_path / "generated_unsrt.xml"
    xml_text = generate_uncertainty_xml(reactions, str(xml_path))
    assert "<class>Elementary</class>" in xml_text
    assert xml_path.is_file()

    mech_std = fixtures_dir / "mech_std.yaml"
    results = compute_symmetric_bounds(str(xml_path), str(mech_std))
    assert len(results) == 2
    for r in results:
        assert "error" not in r, r
        assert r["A_lb"] < r["A_nominal"] < r["A_ub"]
        assert r["factor"] > 1.0

    # R1's real A in mech_std.inp is 9.841E+13 (FFCM-1 value baked into the
    # fixture) -- getNominal() must read it back correctly, not some
    # unrelated/default value.
    assert results[0]["A_nominal"] == pytest.approx(9.841e13, rel=1e-3)


def test_uncertainty_generator_headless_cli_generate_and_compute(fixtures_dir, tmp_path):
    """KRUNAL_JULY_23.md PySide6 conversion: the --generate-xml/--compute-bounds
    headless CLI modes mso_studio.core.uncertainty shells out to. Same
    round-trip as test_uncertainty_generator_roundtrip above, but through
    the actual subprocess argv contract instead of an in-process call."""
    import json
    reactions = [
        {"name": "H + O2 <=> O + OH", "type": "pressure_independent",
         "perturbation_type": "factor", "data_mode": "constant;array",
         "temperatures": [500.0, 1500.0, 2500.0], "uncertainties": [0.40, 0.25, 0.15]},
    ]
    reactions_json = tmp_path / "reactions.json"
    reactions_json.write_text(json.dumps(reactions))
    xml_path = tmp_path / "generated.xml"

    proc = subprocess.run(
        [sys.executable, str(_REPO / "uncertainty_generator_gui.py"), "--generate-xml",
         "--reactions-json", str(reactions_json), "--output", str(xml_path)],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert xml_path.is_file()

    out_csv = tmp_path / "bounds.csv"
    proc2 = subprocess.run(
        [sys.executable, str(_REPO / "uncertainty_generator_gui.py"), "--compute-bounds",
         "--xml", str(xml_path), "--mechanism", str(fixtures_dir / "mech_std.yaml"),
         "--output-csv", str(out_csv)],
        capture_output=True, text=True, timeout=60)
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr
    assert "n_ok=1" in proc2.stdout
    assert "n_err=0" in proc2.stdout
    assert out_csv.is_file()


def test_uncertainty_generator_headless_cli_reports_errors_via_exit_code(tmp_path):
    import json
    reactions = [{"name": "NotARealReaction<=>Nope", "type": "pressure_independent",
                 "temperatures": [500.0, 2500.0], "uncertainties": [0.3, 0.3]}]
    reactions_json = tmp_path / "reactions.json"
    reactions_json.write_text(json.dumps(reactions))
    xml_path = tmp_path / "generated.xml"
    subprocess.run(
        [sys.executable, str(_REPO / "uncertainty_generator_gui.py"), "--generate-xml",
         "--reactions-json", str(reactions_json), "--output", str(xml_path)],
        capture_output=True, text=True, timeout=30, check=True)

    fake_mech = tmp_path / "empty.yaml"
    fake_mech.write_text("phases: []\n")
    proc = subprocess.run(
        [sys.executable, str(_REPO / "uncertainty_generator_gui.py"), "--compute-bounds",
         "--xml", str(xml_path), "--mechanism", str(fake_mech),
         "--output-csv", str(tmp_path / "bounds.csv")],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 1
    assert "ERROR" in proc.stdout


# =============================================================================
# App 6 -- forward_uq_gui.py. B13 ("Run" can essentially never succeed --
# doubled path + hard-required Opt/ dirs) fixed this session; verified in
# two stages: (1) the doubled-path bug is gone -- the script now reaches its
# legitimate "Opt/ not found" guard instead of crashing on the config load;
# (2) with a synthesized minimal PRS cache (bootstrap_forward_uq_cache.py),
# a full 5-step run completes and produces real statistics.
# =============================================================================

def test_forward_uq_b13_reaches_opt_guard_not_doubled_path(fixtures_dir, tmp_path):
    """Before B13's fix, forward_uq_gui._run_uq's cmd (relative
    target_opt_path + relative --output_dir, both built from a cwd the
    child is ALSO launched into) doubled both paths, and forward_uq.py died
    with FileNotFoundError trying to open its own config. After the fix
    (out_dir resolved to absolute up front), it must load the config
    cleanly and reach the legitimate, unrelated "Opt/ directory not found"
    guard (forward_uq.py:373-381) -- proof the path bug is gone, not proof
    the run is complete (that's the next test)."""
    target_opt = _write_gui_stress_target_opt(fixtures_dir, tmp_path)
    r = subprocess.run(
        [sys.executable, str(_REPO / "forward_uq.py"), str(target_opt),
         "--n_samples", "512", "--output_dir", str(tmp_path / "results")],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
    )
    combined = r.stdout + r.stderr
    assert "FileNotFoundError" not in combined
    assert "Loaded 2 target cases" in combined
    assert "Parameter space: n_full = 3" in combined
    assert "Opt/ directory not found" in combined   # legitimate, expected guard


def test_forward_uq_full_run_with_synthetic_prs_cache(fixtures_dir, tmp_path):
    """With the synthesized minimal Opt/ + DM_FOR_PARTIAL_PRS/ cache in
    place, forward_uq.py must complete all 5 steps and write real,
    non-empty forward_uq_statistics.csv + sobol_indices.csv."""
    sys.path.insert(0, str(_REPO / "src"))
    sys.path.insert(0, str(_REPO / "src" / "MUQ-SAC"))
    sys.path.insert(0, str(_REPO))
    sys.path.insert(0, str(_FIXTURES))
    import importlib
    bootstrap = importlib.import_module("bootstrap_forward_uq_cache")

    target_opt = _write_gui_stress_target_opt(fixtures_dir, tmp_path)
    bootstrap.build(
        tmp_path,
        str(_REPO / "src" / "rmip" / "tests" / "fixtures" / "h2_o2_crosscheck_unsrt.xml"),
        str(fixtures_dir / "mech_std.yaml"),
        n_cases=2,
    )

    r = subprocess.run(
        [sys.executable, str(_REPO / "forward_uq.py"), str(target_opt),
         "--n_samples", "512", "--output_dir", "results"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=90,
    )
    combined = r.stdout + r.stderr
    assert "Forward UQ complete" in combined, combined[-3000:]

    stats_path = tmp_path / "forward_uq_statistics.csv"
    assert stats_path.is_file()
    rows = list(csv.DictReader(stats_path.open()))
    assert len(rows) == 2
    for row in rows:
        assert float(row["eta_mean"]) > 0
        assert float(row["phys_p50"]) > 0
        assert row["case_id"] in ("case-0", "case-1")

    assert (tmp_path / "sobol_indices.csv").is_file()
    for plot_name in ("parity_forward_uq.pdf", "coverage_summary.pdf", "sobol_heatmap.pdf"):
        assert (tmp_path / "results" / plot_name).is_file()


# =============================================================================
# App 7 -- mso_compare_gui.py. Never tested before. Deliberately runs 2
# simulation entries (nominal_htc + nominal_ltc) so Phase 1 dispatches
# >=2 CONCURRENT ThreadPoolExecutor workers -- the shape M24 (bug.md,
# pipeline.py's ThreadPoolExecutor + per-worker os.chdir() CWD race) needs
# to have any chance of firing. Don't dodge concurrency in the test setup.
# =============================================================================

def _write_mso_pipeline_config(fixtures_dir: Path, out: Path) -> Path:
    cfg = f"""\
project:
  name:        GUI_Stress_MSO
  working_dir: {out}
  fuel:        H2

muqsac_src_dir: {_REPO / "src" / "MUQ-SAC"}
base_opt_file:  {fixtures_dir / "target.opt"}

simulations:
  nominal_htc:
    mechanism:        {fixtures_dir / "mech_std.yaml"}
    targets:          {fixtures_dir / "targets" / "gui_stress.target"}
    targets_count:    2
    uncertainty_data: {_REPO / "src" / "rmip" / "tests" / "fixtures" / "h2_o2_crosscheck_unsrt.xml"}
    thermo_file:      {fixtures_dir / "thermo_std.dat"}
    trans_file:       {fixtures_dir / "trans_std.txt"}
    output_dir:       ./SIM_RUNS/nominal_htc

  nominal_ltc:
    mechanism:        {fixtures_dir / "mech_std.yaml"}
    targets:          {fixtures_dir / "targets" / "gui_stress.target"}
    targets_count:    2
    uncertainty_data: {_REPO / "src" / "rmip" / "tests" / "fixtures" / "h2_o2_crosscheck_unsrt.xml"}
    thermo_file:      {fixtures_dir / "thermo_std.dat"}
    trans_file:       {fixtures_dir / "trans_std.txt"}
    output_dir:       ./SIM_RUNS/nominal_ltc

merge:
  htc_output_dir: ./MERGE_OUTPUT/HTC
  ltc_output_dir: ./MERGE_OUTPUT/LTC

error_metrics:
  htc_output_dir:       ./ERROR_METRICS/HTC
  ltc_output_dir:       ./ERROR_METRICS/LTC
  uncertainty_fraction: 0.10

plots:
  htc_output_dir: ./PLOTS/HTC
  ltc_output_dir: ./PLOTS/LTC
  plot_per_phi:   true
  plot_combined:  true
  mode:           nominal_only
"""
    dst = out / "mso_pipeline_config.yaml"
    dst.write_text(cfg)
    return dst


def test_mso_compare_concurrent_phase1_no_cross_contamination(fixtures_dir, tmp_path):
    """run_mso_pipeline.py --regime both with 2 simulation entries: Phase 1
    dispatches both concurrently (mso_compare/pipeline.py:170's
    ThreadPoolExecutor). Both must complete and land their OWN correct
    dataset CSV in their OWN output_dir -- no missing/misplaced files, no
    cross-written content (the M24 failure mode)."""
    cfg_path = _write_mso_pipeline_config(fixtures_dir, tmp_path)
    r = subprocess.run(
        [sys.executable, str(_REPO / "src" / "run_mso_pipeline.py"),
         "--config", str(cfg_path), "--regime", "both"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=180,
    )
    combined = r.stdout + r.stderr
    assert "status: done" in combined or "[status: done]" in combined
    assert "nominal_htc  [status: done]" in combined
    assert "nominal_ltc  [status: done]" in combined

    for label in ("nominal_htc", "nominal_ltc"):
        csv_path = tmp_path / "SIM_RUNS" / label / "Plot" / "Dataset" / "Tig" / "dataset_001.csv"
        assert csv_path.is_file(), f"{label} missing its own dataset CSV"
        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 2
        nominals = [float(row["Nominal"]) for row in rows]
        assert all(n > 0 for n in nominals)
        # Both entries use identical inputs -- must both independently
        # reproduce the SAME correct physics (962/1325 us, per App 3), not
        # some corrupted/zeroed/cross-written value.
        assert nominals[0] == pytest.approx(962.34, rel=1e-3)
        assert nominals[1] == pytest.approx(1324.88, rel=1e-3)


# =============================================================================
# App 8 -- rmip_suite_gui.py, all 4 internal tabs. Biggest surface, never
# tested; GSA and Active-Param-Selection had ZERO GUI coverage ever.
# =============================================================================

def test_rmip_suite_tab1_compare_graph(fixtures_dir):
    """GraphComparisonTab's command: `rmip_cli.py compare-graph`. Already
    the exact command used while building/verifying the fixture set itself
    (9 universal species, 3 identical reaction groups, 17 unique) -- this
    test locks that verification in permanently."""
    r = subprocess.run(
        [sys.executable, str(_REPO / "rmip_cli.py"), "compare-graph",
         "--mechs", "1", "2", "3", "--db", str(fixtures_dir / "project.db"),
         "--out", str(fixtures_dir.parent / "graph_cmp_out")],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "universal: 9" in r.stdout
    assert "identical: 3" in r.stdout
    assert "unique: 17" in r.stdout


def test_rmip_suite_tab2_reduce(fixtures_dir, tmp_path):
    """ReductionTab's command: `rmip_cli.py reduce`. DRG on mech_A (id=1)
    must correctly identify the inert third-body-only species (AR, never a
    true reactant/product) as removable at threshold 0."""
    out_json = tmp_path / "reduce.json"
    r = subprocess.run(
        [sys.executable, str(_REPO / "rmip_cli.py"), "reduce", "1",
         "--db", str(fixtures_dir / "project.db"), "--method", "DRG",
         "--target", "diprotium,dioxygenium,aqua",
         "--eps-max", "0.5", "--eps-steps", "5", "--out", str(out_json)],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert out_json.is_file()
    import json
    data = json.loads(out_json.read_text())
    step0 = data["steps"][0]
    assert step0["threshold"] == 0.0
    assert "argon_a" in step0["removed_species"]
    assert step0["n_species"] == 8   # 9 declared - inert AR


def test_rmip_suite_tab2_reduce_multi_condition(fixtures_dir, tmp_path):
    """ReductionTab's multi-condition row editor (Phase-3 follow-on Part D,
    FEATURE_GAPS_FROM_MONOLITH.md ranked candidate #5). In-process per this
    file's convention (no synthetic Tk clicks): the condition Treeview's
    backing data structure (self._cond_rows, populated the same way
    self._add_cond_row() does) is set directly, then _build_command() is
    called -- exactly the same "button click -> _build_command() -> argv"
    path a live click would take, minus the click itself."""
    import tkinter as tk
    from tkinter import ttk
    sys.path.insert(0, str(_REPO))
    import rmip_suite_gui as gui_app

    root = tk.Tk()
    try:
        nb = ttk.Notebook(root)
        db_var = tk.StringVar(value=str(fixtures_dir / "project.db"))
        tab = gui_app.ReductionTab(nb, "Mechanism Reduction", db_var)
        tab.mech_id_var.set("1")
        tab.target_var.set("diprotium,dioxygenium,aqua")
        tab.eps_max_var.set("0.5")
        tab.eps_steps_var.set("5")

        out_json = tmp_path / "reduce_multi.json"
        tab.out_var.set(str(out_json))

        # Populate the condition rows the same way _add_cond_row() would,
        # via its own entry StringVars (not a synthetic widget click).
        tab._cond_T_var.set("1000"); tab._cond_P_var.set("1.0")
        tab._cond_phi_var.set("1.0"); tab._cond_fuel_var.set("diprotium:1.0")
        tab._add_cond_row()
        tab._cond_T_var.set("1600"); tab._cond_P_var.set("5.0")
        tab._cond_phi_var.set("0.7"); tab._cond_fuel_var.set("diprotium:1.0")
        tab._add_cond_row()

        assert len(tab._cond_rows) == 2
        cmd = tab._build_command()
    finally:
        root.destroy()

    assert cmd.count("--condition") == 2
    assert "1000.0,1.0,1.0,diprotium:1.0" in cmd
    assert "1600.0,5.0,0.7,diprotium:1.0" in cmd

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MULTI-CONDITION" in r.stdout
    assert out_json.is_file()
    import json
    data = json.loads(out_json.read_text())
    step0 = data["steps"][0]
    # Seed species always retained regardless of condition.
    assert {"diprotium", "dioxygenium", "aqua"} <= set(step0["retained_species"])


def test_rmip_suite_tab2_reduce_no_conditions_unchanged(fixtures_dir, tmp_path):
    """Empty condition tree -> _build_command() emits no --condition flags
    at all, i.e. the exact same argv as the plain single-condition test
    above (test_rmip_suite_tab2_reduce) -- locks in that ReductionTab's
    existing behaviour is unchanged when no rows are added."""
    import tkinter as tk
    from tkinter import ttk
    sys.path.insert(0, str(_REPO))
    import rmip_suite_gui as gui_app

    root = tk.Tk()
    try:
        nb = ttk.Notebook(root)
        db_var = tk.StringVar(value=str(fixtures_dir / "project.db"))
        tab = gui_app.ReductionTab(nb, "Mechanism Reduction", db_var)
        tab.mech_id_var.set("1")
        tab.target_var.set("diprotium,dioxygenium,aqua")
        assert tab._cond_rows == []
        cmd = tab._build_command()
    finally:
        root.destroy()

    assert "--condition" not in cmd


def test_rmip_suite_target_builder_tab_build_command(tmp_path):
    """TargetBuilderTab (Phase-3 follow-on Part A) -- _build_command()
    assembles `rmip build-target` argv from the mech/target-type/solver/
    condition fields plus the fuel-blend row editor, for a given set of
    field values (in-process, no live Tk clicks)."""
    import tkinter as tk
    from tkinter import ttk
    sys.path.insert(0, str(_REPO))
    import rmip_suite_gui as gui_app

    root = tk.Tk()
    try:
        nb = ttk.Notebook(root)
        db_var = tk.StringVar(value="project.db")
        tab = gui_app.TargetBuilderTab(nb, "Target Builder", db_var)
        tab.mech_id_var.set("1")
        tab.base_opt_var.set("target.opt")
        tab.work_dir_var.set("outputs/tb_test")
        tab.T_var.set("1200"); tab.P_var.set("2.0"); tab.phi_var.set("1.0")

        tab._fuel_sp_var.set("NC7H16")
        tab._fuel_frac_var.set("1.0")
        tab._add_fuel_row()

        cmd = tab._build_command()
    finally:
        root.destroy()

    assert cmd[2] == "build-target"
    assert "1" in cmd
    assert "--target-type" in cmd and cmd[cmd.index("--target-type") + 1] == "JSR"
    assert "--fuel" in cmd and cmd[cmd.index("--fuel") + 1] == "NC7H16:1.0"
    assert "--T" in cmd and cmd[cmd.index("--T") + 1] == "1200"
    assert "--work-dir" in cmd and cmd[cmd.index("--work-dir") + 1] == "outputs/tb_test"


def test_rmip_suite_target_builder_tab_requires_fuel_row(tmp_path):
    import tkinter as tk
    from tkinter import ttk
    sys.path.insert(0, str(_REPO))
    import rmip_suite_gui as gui_app

    root = tk.Tk()
    try:
        nb = ttk.Notebook(root)
        db_var = tk.StringVar(value="project.db")
        tab = gui_app.TargetBuilderTab(nb, "Target Builder", db_var)
        tab.mech_id_var.set("1")
        tab.base_opt_var.set("target.opt")
        tab.work_dir_var.set("outputs/tb_test")
        with pytest.raises(ValueError):
            tab._build_command()
    finally:
        root.destroy()


def test_rmip_suite_respecth_import_tab_build_command(tmp_path):
    """ReSpecThImportTab (Phase-3 follow-on Part C) -- _build_command()
    wraps the EXISTING `rmip import-respecth` subcommand; for a given set
    of field values, produces the expected argv."""
    import tkinter as tk
    from tkinter import ttk
    sys.path.insert(0, str(_REPO))
    import rmip_suite_gui as gui_app

    root = tk.Tk()
    try:
        nb = ttk.Notebook(root)
        db_var = tk.StringVar(value="project.db")
        tab = gui_app.ReSpecThImportTab(nb, "ReSpecTh Import", db_var)
        tab.xml_path_var.set("/data/xml_datasets")
        tab.fuel_var.set("NH3,H2")
        tab.solver_var.set("cantera")
        tab.out_dir_var.set("outputs/respecth")
        tab.dataset_prefix_var.set("batch1")
        tab.no_db_var.set(True)

        cmd = tab._build_command()
    finally:
        root.destroy()

    assert cmd[2] == "import-respecth"
    assert "/data/xml_datasets" in cmd
    assert "--fuel" in cmd and cmd[cmd.index("--fuel") + 1] == "NH3,H2"
    assert "--out-dir" in cmd and cmd[cmd.index("--out-dir") + 1] == "outputs/respecth"
    assert "--dataset-prefix" in cmd and cmd[cmd.index("--dataset-prefix") + 1] == "batch1"
    assert "--no-db" in cmd


def test_rmip_suite_tab3_gsa_run(fixtures_dir, tmp_path):
    """GSATab's command: `rmip_cli.py gsa-run` -- real Sobol GSA via SALib +
    Zero-RK's perturbAFactorGSA_mpi.x. Never had any GUI-adjacent coverage
    before this sweep."""
    zerork_prefix = "/home/krithika/zero-rk/build2/inst_dir"
    if not (Path(zerork_prefix) / "bin" / "perturbAFactorGSA_mpi.x").exists():
        pytest.skip("Zero-RK GSA binary not present on this machine")

    out_json = tmp_path / "gsa.json"
    r = subprocess.run(
        [sys.executable, str(_REPO / "rmip_cli.py"), "gsa-run", "1",
         "--db", str(fixtures_dir / "project.db"), "--method", "sobol",
         "--n-samples", "2", "--reactions", "all",
         "--zerork-install-prefix", zerork_prefix,
         "--T", "1450.0", "--P", "2.2", "--phi", "1.0",
         "--fuel", "diprotium:1.0", "--oxidizer", "dioxygenium:0.5, argon_a:1.985",
         "--n-mpi-ranks", "2", "--out", str(out_json),
         "--work-dir", str(tmp_path / "gsa_work")],
        capture_output=True, text=True, timeout=90,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GSA complete" in r.stdout
    assert out_json.is_file()
    import json
    data = json.loads(out_json.read_text())
    assert data["n_samples"] == 20          # N=2, D=8 reactions -> N*(D+2)
    assert len(data["S1"]) == 8
    assert data["n_failed_samples"] == 0


def test_rmip_suite_tab4_active_param_selection(fixtures_dir, tmp_path):
    """ActiveParamSelectionTab's command: src/ActiveParameterScreen.py.
    Never had any GUI-adjacent coverage before this sweep. Builds 2
    synthetic SA .txt files (the format modules/sensitivity_parser.py
    expects -- distinct from sens_3_params.py's own CSV output) and
    verifies threshold screening, reaction-type classification, and
    species extraction against a real ck2yaml'd mech_A.yaml."""
    ck2yaml = shutil.which("ck2yaml")
    if not ck2yaml:
        pytest.skip("ck2yaml not on PATH")

    mech_yaml = tmp_path / "mech_A.yaml"
    r = subprocess.run(
        [ck2yaml, f"--input={fixtures_dir / 'mech_A.inp'}",
         f"--thermo={fixtures_dir / 'thermo_A.dat'}",
         f"--output={mech_yaml}", "--permissive"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    import yaml as _yaml
    mech = _yaml.safe_load(mech_yaml.read_text())
    equations = [rxn["equation"] for rxn in mech["reactions"]]
    assert len(equations) == 8

    sens_dir = tmp_path / "sens_dir"
    sens_dir.mkdir()
    case_sensitivities = {
        "case_0_T1450": [0.85, 0.60, 0.02, 0.10, 0.95, 0.01, 0.03, 0.005],
        "case_1_T1350": [0.80, 0.55, 0.03, 0.12, 0.90, 0.02, 0.04, 0.006],
    }
    for fname, vals in case_sensitivities.items():
        lines = ["Sensitivity\tReaction_ID\tReaction"]
        for i, (v, eq) in enumerate(zip(vals, equations)):
            lines.append(f"{v}\t{i + 1}\t{eq}")
        (sens_dir / f"{fname}.txt").write_text("\n".join(lines) + "\n")

    results_dir = tmp_path / "apsel_results"
    r = subprocess.run(
        [sys.executable, str(_REPO / "src" / "ActiveParameterScreen.py"),
         "--sens-dir", str(sens_dir), "--mechanism", str(mech_yaml),
         "--threshold", "0.05", "--carbon", "0", "--fuel", "diprotium",
         "--results-dir", str(results_dir)],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "4 active reactions" in r.stdout

    rows = list(csv.DictReader((results_dir / "master_active_reactions.csv").open()))
    assert len(rows) == 4
    by_id = {int(row["Reaction_ID"]): row for row in rows}
    # Reaction 5 (falloff, max sensitivity 0.95 in case_0) must be #1.
    assert by_id[5]["Reaction_Type"] == "Falloff"
    assert float(by_id[5]["Maximum_Abs_Sensitivity"]) == pytest.approx(0.95)
    assert by_id[4]["Reaction_Type"] == "ThirdBody"
    # Reactions below threshold 0.05 (3, 6, 7, 8) must be excluded.
    assert set(by_id.keys()) == {1, 2, 4, 5}

    species_rows = list(csv.DictReader((results_dir / "active_species.csv").open()))
    species_names = {row["Species"] for row in species_rows}
    assert species_names == {"protium", "dioxygenium", "oxo", "hydroxyl",
                              "diprotium", "hydroperox"}
    for row in species_rows:
        assert int(row["Carbon_Number"]) == 0   # pure H/O/Ar chemistry
