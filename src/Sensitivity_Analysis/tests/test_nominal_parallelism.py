#!/usr/bin/env python3.9
"""No-license proof that the NOMINAL simulation path is case-wise parallel.

make_nominal_dir_in_parallel must launch one simulation per case (75 cases x
1 sample each, the unperturbed mechanism) in a SINGLE pool of
``Counts.parallel_threads`` workers, so all case runs start concurrently --
exactly like the mso_compare nominal runner -- instead of the old
``make_dir_in_parallel`` behavior, which loops cases serially (fine for the
multiply/divide stages with ~20 samples per case, fatal for nominal's
75 x 1 layout).

The test mocks every external dependency (mechanism parser, yamlwriter,
make_input_file.create_input_file, thread cap) and runs the REAL
simulation_manager2_0.SM.make_nominal_dir_in_parallel in a temp directory
with 75 fake cases. The fake ``run`` scripts timestamp their own start, so
the test measures launch concurrency directly. No Chemkin, no FlameMaster,
no licenses.

Run:
    python3.9 -m pytest tests/test_nominal_parallelism.py -v
    (from Sensitivity_Analysis/, or any cwd -- paths resolve from __file__)
"""
import multiprocessing
import os
import sys
import time
import types

import pytest

_SA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SA_DIR not in sys.path:
    sys.path.insert(0, _SA_DIR)

# Monkeypatches must reach the forked pool workers (simulation_manager2_0
# dispatches run_map / run_executable_files_ via multiprocessing.Pool), so
# pin the fork start method BEFORE importing the module.
multiprocessing.set_start_method("fork", force=True)

import simulation_manager2_0 as sim2k  # noqa: E402
import make_input_file as _mif          # noqa: E402


N_CASES = 75
N_THREADS = 70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeParser:
    """Replaces MechanismParser.Parser -- SM.__init__ only deep-copies .mech."""

    def __init__(self, path):
        self.mech = {"phases": [{"species": []}], "species": {}, "reactions": []}


class _FakeYamlWriter:
    """Replaces the compiled yamlwriter extension (no .so, no file format)."""

    @staticmethod
    def dump_to_yaml(location, filename, mech_dict):
        out = os.path.join(location, filename)
        with open(out, "w") as fh:
            fh.write("fake: true\n")
        return out


def _setup(tmp_path, monkeypatch, start_log=None):
    """Build the mocked world and return (sm_kwargs, tmp_path, start_log)."""
    # Deterministic thread cap: pins min(parallel_threads, cap) = N_THREADS.
    monkeypatch.setattr(sim2k, "_thread_cap", lambda: 200)
    monkeypatch.setattr(sim2k, "Parser", _FakeParser)
    monkeypatch.setattr(sim2k, "yamlwriter", _FakeYamlWriter)

    if start_log is None:
        start_log = tmp_path / "fake_run_starts.txt"
    start_log_path = str(start_log)

    def fake_create_input_file(case, opt_dict, target, mech_file=None, solver=None):
        # The run script records its own wall-clock start (evidence that the
        # POOL launched cases concurrently), sleeps briefly, then fabricates
        # an output/ so the flow's post-conditions hold.
        _run = (
            "#!/bin/bash\n"
            "echo \"$(date +%s.%N 2>/dev/null || date +%s)\" >> " + start_log_path + "\n"
            "sleep 2\n"
            "mkdir -p output; echo done > output/tau.out\n"
        )
        return ("", "#!/bin/bash\nexit 0", _run, "")

    monkeypatch.setattr(_mif, "create_input_file", fake_create_input_file)

    pre_file = tmp_path / "initial_pre.txt"
    pre_file.write_text("x\n")
    opt = {
        "Stats":     {"PRS_type": "Full", "Order_of_PRS": 1},
        "Inputs":    {"fileType": "chemkin"},
        "Counts":    {"parallel_threads": N_THREADS},
        "Locations": {"mechanism": "unused", "Initial_pre_file": str(pre_file)},
        "Bin":       {"yaml_writer": str(tmp_path)},
    }
    targets = [types.SimpleNamespace() for _ in range(N_CASES)]
    return opt, targets, tmp_path, start_log


def _run_nominal(opt, targets, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sm = sim2k.SM(targets, opt, {}, [])
    t0 = time.time()
    locs = sm.make_nominal_dir_in_parallel()
    wall = time.time() - t0
    return locs, wall


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_nominal_cases_run_concurrently(tmp_path, monkeypatch):
    """Phase-1 core proof: 75 case runs all launch in one pool, together.

    This test passes BEFORE any diagnostic additions -- it measures the
    concurrency of the current merged make_nominal_dir_in_parallel via the
    fake run scripts' own timestamps (no case_start_times.txt involved).
    """
    assert multiprocessing.get_start_method() == "fork"
    opt, targets, tmp_path, start_log = _setup(tmp_path, monkeypatch)

    locs, wall = _run_nominal(opt, targets, tmp_path, monkeypatch)

    # -- artifact completeness -------------------------------------------
    assert len(locs) == N_CASES
    loc_lines = [l for l in (tmp_path / "locations").read_text().splitlines() if l.strip()]
    prg_lines = [l for l in (tmp_path / "progress").read_text().splitlines() if l.strip()]
    assert len(loc_lines) == N_CASES
    assert len(prg_lines) == N_CASES
    for i in range(N_CASES):
        case_dir = tmp_path / f"case-{i}"
        assert case_dir.is_dir(), f"missing {case_dir}"
        assert (case_dir / "output").is_dir(), f"missing output/ in {case_dir}"
        assert (case_dir / "run").exists()

    # -- concurrency (the point of the test) ------------------------------
    starts = sorted(float(l) for l in start_log.read_text().splitlines() if l.strip())
    assert len(starts) == N_CASES
    spread = starts[-1] - starts[0]
    within_5s = sum(1 for s in starts if s - starts[0] <= 5.0)
    # A serial implementation would put ~2 starts inside 5 s (75 x 2 s sleep).
    assert within_5s >= 65, (
        f"only {within_5s}/{N_CASES} case runs started within 5s of the first "
        f"-- nominal launch is NOT case-parallel (spread={spread:.1f}s)"
    )
    assert spread < 8.0, f"case-run start spread too wide: {spread:.1f}s"
    # 75 x 2 s serial = 150 s; parallel = one pool cycle + fork overhead.
    assert wall < 45, f"wall time {wall:.1f}s looks serial"


def test_nominal_start_log_and_diagnostics(tmp_path, monkeypatch, capsys):
    """Post-change diagnostics: case_start_times.txt + printed pool evidence."""
    opt, targets, tmp_path, _ = _setup(tmp_path, monkeypatch)
    _run_nominal(opt, targets, tmp_path, monkeypatch)

    start_log = tmp_path / "case_start_times.txt"
    assert start_log.exists(), "case_start_times.txt missing (run worker logging)"
    starts = sorted(float(l.split("\t")[0])
                    for l in start_log.read_text().splitlines() if l.strip())
    assert len(starts) == N_CASES

    out = capsys.readouterr().out
    assert f"pool of {N_THREADS} workers" in out, "pool-size launch line missing"
    assert "Launch evidence:" in out, "launch-evidence summary missing"
    assert f"{N_CASES}/{N_CASES} case runs started" in out


def test_nominal_rerun_existing_dirs_no_fileexists(tmp_path, monkeypatch):
    """Re-run regression: pre-existing complete case dirs must not crash.

    Old behavior: getNominalDirectoryList never recognized existing case
    dirs (existence check used the wrong name), so run_generate_dir's
    os.mkdir raised FileExistsError in every pool worker on re-entry.
    """
    opt, targets, tmp_path, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    for i in range(3):
        (tmp_path / f"case-{i}" / "output").mkdir(parents=True)
        (tmp_path / f"case-{i}" / "output" / "tau.out").write_text("old\n")

    locs, _ = _run_nominal(opt, targets, tmp_path, monkeypatch)

    assert len(locs) == N_CASES
    for i in range(3):
        assert (tmp_path / f"case-{i}" / "output" / "tau.out").exists()
