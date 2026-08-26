"""
rmip/tests/test_simulation_manager_yaml_write.py
=================================================
Regression coverage for the brute-force SA `mechanism.yaml` silent-write
bug (bug.md, ~line 65: "IndexError: index 0 ... size 0" -- the compiled
`yamlwriter.dump_to_yaml` extension silently no-ops on real-scale
mechanisms inside a forked multiprocessing.Pool worker, with no Python
exception anywhere, so `run_map` sails on into `data_management.py::run_sim`
which several steps later crashes with a confusing IndexError instead of
the real "mechanism.yaml never got written" problem).

`src/Sensitivity_Analysis/simulation_manager2_0.py::_write_mechanism_yaml_verified`
wraps the one call site in `run_map` with verify+retry+pure-Python-fallback
logic. These tests exercise that helper directly (monkeypatching the
compiled `yamlwriter` module's `dump_to_yaml` to reproduce the exact silent
no-op failure mode) rather than the full brute-force SA pipeline, which
needs a real target.opt + uncertainty-XML fixture set to drive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


# Bare top-level module names simulation_manager2_0.py imports that ALSO
# exist, under the exact same name but different content, in src/MUQ-SAC/
# (and src/MUQ-SAC_EXTENDED/). Since none of this is a real Python package
# (no relative imports, just flat sys.path directories), Python's sys.modules
# cache is keyed on the bare name alone -- so importing this module here and
# leaving these cached would make any OTHER test that expects to import its
# own (e.g. MUQ-SAC's) copy of `data_management`/`make_input_file`/etc.
# silently receive the wrong one for the rest of the pytest session. Must
# save/pop before import and fully restore after, not just pop our own
# top-level module.
_COLLIDING_MODULES = [
    "simulation_manager2_0", "yamlwriter", "parallel_yaml_writer",
    "Uncertainty", "make_input_file", "MechManipulator2_0",
    "data_management", "MechanismParser",
]


@pytest.fixture()
def sm(monkeypatch):
    """Fresh import of src/Sensitivity_Analysis/simulation_manager2_0.py,
    with its sys.path entry inserted FIRST -- MUQ-SAC and MUQ-SAC_EXTENDED
    each ship a byte-similar but distinct top-level module of the exact same
    name, so we must both order sys.path correctly and drop any
    already-cached copy before importing, then restore whatever was cached
    before us so we don't leak our copies into later tests."""
    sa_dir = str(_REPO / "src" / "Sensitivity_Analysis")
    monkeypatch.syspath_prepend(sa_dir)
    saved = {name: sys.modules.pop(name, None) for name in _COLLIDING_MODULES}
    try:
        import simulation_manager2_0 as module
        assert module.__file__ == str(_REPO / "src" / "Sensitivity_Analysis" / "simulation_manager2_0.py")
        yield module
    finally:
        for name in _COLLIDING_MODULES:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


_MECH = {
    "species": ["H2", "O2", "H2O", "OH", "H", "O", "N2"],
    "reactions": [
        {"equation": "H2 + O2 <=> OH + OH", "rate-constant": {"A": 1.7e13, "b": 0.0, "Ea": 47780.0}},
        {"equation": "H + O2 <=> OH + O", "rate-constant": {"A": 2.65e16, "b": -0.6707, "Ea": 17041.0}},
    ],
}


def test_fallback_produces_valid_yaml_when_yamlwriter_silently_no_ops(sm, tmp_path, monkeypatch):
    """Reproduce the exact bug: the compiled extension returns normally but
    writes nothing. _write_mechanism_yaml_verified must fall back to a
    pure-Python dump and still hand back a valid, non-empty, re-loadable
    mechanism.yaml -- never letting a caller silently proceed with no file
    on disk."""

    def silent_no_op(location, filename, data):
        return None  # exactly yamlwriter.cpp's silent-failure behavior

    monkeypatch.setattr(sm.yamlwriter, "dump_to_yaml", silent_no_op)

    out_path = sm._write_mechanism_yaml_verified(
        str(tmp_path), "mechanism.yaml", _MECH, max_retries=2, retry_delay_s=0.01
    )

    assert out_path == str(tmp_path / "mechanism.yaml")
    p = Path(out_path)
    assert p.exists()
    assert p.stat().st_size > 0

    loaded = sm.yaml.safe_load(p.read_text())
    assert loaded["species"] == _MECH["species"]
    assert len(loaded["reactions"]) == 2


def test_raises_runtime_error_when_both_writer_and_fallback_fail(sm, tmp_path, monkeypatch):
    """If the compiled extension AND the pure-Python fallback both fail, the
    helper must raise a clear RuntimeError -- not let execution continue
    into the downstream code that produces the confusing IndexError, and
    not let some other unrelated exception type leak out instead."""

    def silent_no_op(location, filename, data):
        return None

    def broken_dump(*args, **kwargs):
        raise OSError("simulated: fallback YAML dump target is unwritable")

    monkeypatch.setattr(sm.yamlwriter, "dump_to_yaml", silent_no_op)
    monkeypatch.setattr(sm.yaml, "dump", broken_dump)

    with pytest.raises(RuntimeError) as excinfo:
        sm._write_mechanism_yaml_verified(
            str(tmp_path), "mechanism.yaml", _MECH, max_retries=2, retry_delay_s=0.01
        )

    msg = str(excinfo.value)
    assert str(tmp_path / "mechanism.yaml") in msg
    assert not (tmp_path / "mechanism.yaml").exists() or (tmp_path / "mechanism.yaml").stat().st_size == 0


def test_happy_path_uses_real_yamlwriter_extension(sm, tmp_path):
    """Sanity check: when the compiled extension actually works (the normal
    case), the helper returns the same path yamlwriter itself wrote, without
    needing the fallback at all."""
    out_path = sm._write_mechanism_yaml_verified(str(tmp_path), "mechanism.yaml", _MECH)
    p = Path(out_path)
    assert p.exists() and p.stat().st_size > 0


def test_run_map_3_is_the_real_production_write_path_and_now_protected(sm, tmp_path, monkeypatch):
    """2026-07-23 investigation (bug.md): traced sens_3_params.py's actual
    brute-force SA call chain precisely --
    SM.getPerturbedMechLocation -> Worker.do_job_map_create_3 -> run_map_3
    -- and found it calls yamlwriter.dump_to_yaml directly, NOT through
    _write_mechanism_yaml_verified (that wrapper was scoped to `run_map`,
    which turned out to be unused by this pipeline: make_dir_in_parallel,
    the method sens_3_params.py actually calls for per-case simulation
    directories, dispatches to run_map_opt, which never touches
    mechanism.yaml at all). Fixed by routing run_map_3 through the same
    verified wrapper. This test proves run_map_3 now survives the exact
    silent-no-op failure mode that motivated the wrapper in the first
    place, instead of leaving a missing mechanism_<n>.yaml file with no
    exception raised anywhere."""

    def silent_no_op(location, filename, data):
        return None  # exactly yamlwriter.cpp's silent-failure behavior

    monkeypatch.setattr(sm.yamlwriter, "dump_to_yaml", silent_no_op)

    # run_map_3's params tuple: (mech_dict, location, row_index)
    result = sm.run_map_3((_MECH, str(tmp_path), 7), total=1)

    assert result == (_MECH, 1)
    out_path = tmp_path / "mechanism_7.yaml"
    assert out_path.exists() and out_path.stat().st_size > 0
    loaded = sm.yaml.safe_load(out_path.read_text())
    assert loaded["species"] == _MECH["species"]


def test_run_map_3_happy_path_real_extension(sm, tmp_path):
    """Sanity check with the real compiled extension (no monkeypatching):
    run_map_3 still produces the correctly-named mechanism_<row>.yaml file
    it always did on the success path -- the fix must not change on-disk
    behavior when nothing is actually failing."""
    result = sm.run_map_3((_MECH, str(tmp_path), 3), total=5)
    assert result == (_MECH, 5)
    out_path = tmp_path / "mechanism_3.yaml"
    assert out_path.exists() and out_path.stat().st_size > 0
