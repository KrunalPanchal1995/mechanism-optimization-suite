"""Regression tests for rmip.simulation.subprocess_util."""
from rmip.simulation.subprocess_util import (
    rmg_subprocess_env,
    sanitized_subprocess_env,
)


def test_strips_foreign_ld_library_path_entries(monkeypatch):
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        "/home/user/Desktop/ANSYS/ansys_inc/v231/tp/stdc++:/usr/local/lib",
    )
    env = sanitized_subprocess_env()
    assert env["LD_LIBRARY_PATH"] == "/usr/local/lib"


def test_drops_ld_library_path_when_only_foreign_entries(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ansys_inc/v231/tp/stdc++")
    env = sanitized_subprocess_env()
    assert "LD_LIBRARY_PATH" not in env


def test_leaves_clean_ld_library_path_untouched(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/lib:/opt/conda/lib")
    env = sanitized_subprocess_env()
    assert env["LD_LIBRARY_PATH"] == "/usr/local/lib:/opt/conda/lib"


def test_leaves_ld_preload_untouched(monkeypatch):
    # 2026-08-06: sanitized_subprocess_env must NOT strip LD_PRELOAD — on this
    # machine it's what steers Zero-RK's MPI binaries away from a broken
    # system-wide ld.so.cache default for libstdc++.so.6 (see module
    # docstring). An earlier attempt to strip it broke
    # test_zerork_gsa.py's real-binary tests.
    monkeypatch.setenv("LD_PRELOAD", "libstdc++.so.6.0.28")
    env = sanitized_subprocess_env()
    assert env["LD_PRELOAD"] == "libstdc++.so.6.0.28"


def test_extra_overrides_win_over_sanitized_copy(monkeypatch):
    monkeypatch.setenv("NUMEXPR_MAX_THREADS", "8")
    env = sanitized_subprocess_env(extra={"NUMEXPR_MAX_THREADS": "1"})
    assert env["NUMEXPR_MAX_THREADS"] == "1"


def test_rmg_subprocess_env_unsets_ld_preload(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "libstdc++.so.6.0.28")
    env = rmg_subprocess_env("/home/user/anaconda3/envs/rmg_env/bin/python")
    assert "LD_PRELOAD" not in env


def test_rmg_subprocess_env_puts_rmg_env_lib_first_on_ld_library_path(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/lib")
    env = rmg_subprocess_env("/home/user/anaconda3/envs/rmg_env/bin/python")
    entries = env["LD_LIBRARY_PATH"].split(":")
    assert entries[0] == "/home/user/anaconda3/envs/rmg_env/lib"
    assert "/usr/local/lib" in entries


def test_rmg_subprocess_env_works_with_no_existing_ld_library_path(monkeypatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    env = rmg_subprocess_env("/home/user/anaconda3/envs/rmg_env/bin/python")
    assert env["LD_LIBRARY_PATH"] == "/home/user/anaconda3/envs/rmg_env/lib"
