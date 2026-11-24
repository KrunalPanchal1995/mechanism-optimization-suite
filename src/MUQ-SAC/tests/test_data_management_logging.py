#!/usr/bin/env python3.9
"""Verify data_management.py's print->log + progress-bar conversion.

Contract (user requirement, 2026-08-29):
  * the terminal shows ONLY the tqdm progress bar of the re-run pool;
  * every diagnostic (info, warnings, errors, tracebacks) goes to
    data_management.log;
  * errors surface solely as the bar's postfix failure count (✗ N) --
    never as extra terminal lines.

No Chemkin, no FlameMaster, no licenses: run_sim is monkeypatched and the
pool runs trivial fake tasks.

Run:
    python3.9 -m pytest tests/test_data_management_logging.py -v
    (from MUQ-SAC/, or any cwd -- paths resolve from __file__)
"""
import logging
import multiprocessing
import os
import sys
import time

import pytest

_MUQSAC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MUQSAC_DIR not in sys.path:
    sys.path.insert(0, _MUQSAC_DIR)

# Pin fork BEFORE importing data_management: the monkeypatched run_sim must
# reach the forked pool workers.
multiprocessing.set_start_method("fork", force=True)


def _fake_run_sim(index, case, input_, caseID, path, total):
    """Module-level (picklable) stand-in for the pool task.

    Emits one log record from the FORKED worker: pool.terminate() SIGTERMs
    workers without running logging shutdown, so only the per-record-flush
    FileHandler makes this line durable."""
    logging.getLogger("data_management").info("fake run_sim: index=%s path=%s", index, path)
    time.sleep(0.2)
    return (index, 1.0, 2.0, path, total)


def _fake_run_sim_fail(index, case, input_, caseID, path, total):
    raise RuntimeError("boom")


def _repoint_logger(tmp_path):
    """data_management imports once per session; re-point its logger at this
    test's own tmp dir (the module's `if not logger.handlers:` guard keeps the
    first test's file otherwise)."""
    import data_management as dm
    dm.logger.handlers.clear()
    fh = logging.FileHandler(str(tmp_path / "data_management.log"), mode="a")
    fh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    dm.logger.addHandler(fh)
    return dm


def test_worker_bar_and_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_MGMT_LOG", str(tmp_path / "data_management.log"))
    dm = _repoint_logger(tmp_path)
    monkeypatch.setattr(dm, "run_sim", _fake_run_sim)
    monkeypatch.setenv("DATA_MGMT_LOG", str(tmp_path / "data_management.log"))
    import data_management as dm
    monkeypatch.setattr(dm, "run_sim", _fake_run_sim)

    dm.logger.error("a test error lands in the log file only")

    w = dm.Worker(2)
    eta, ETA, paths = w.do_job_async(
        [f"/x/case-0/{i}/run" for i in range(6)],
        object(), {"Counts": {"parallel_threads": 2}}, 0,
    )
    assert len(eta) == 6 and len(paths) == 6

    out = capsys.readouterr().out
    # terminal = the bar only
    assert "Re-running simulations" in out
    assert "sim" in out
    # old percent ticker gone, errors never printed to the terminal
    assert "is complete" not in out
    assert "a test error lands in the log file only" not in out

    log_text = (tmp_path / "data_management.log").read_text()
    assert "a test error lands in the log file only" in log_text
    # forked-worker records survive pool.terminate() thanks to per-record flush
    assert "fake run_sim: index=3" in log_text


class _NonTTYStream:
    """Simulates the real pipeline: stdout is a pipe, isatty() -> False.

    tqdm auto-disables its display on such a stream unless disable=False --
    which would silently kill the bar exactly where the user wants it
    (SA_tool.out / run_log.txt tee). Regression guard for that flag.
    """
    def __init__(self):
        self.buf = []

    def write(self, s):
        self.buf.append(s)

    def flush(self):
        pass

    def isatty(self):
        return False


def test_bar_renders_on_non_tty_stdout(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_MGMT_LOG", str(tmp_path / "data_management.log"))
    dm = _repoint_logger(tmp_path)
    monkeypatch.setattr(dm, "run_sim", _fake_run_sim)

    stream = _NonTTYStream()
    monkeypatch.setattr(dm.sys, "stdout", stream)

    w = dm.Worker(2)
    eta, ETA, paths = w.do_job_async(
        [f"/x/case-0/{i}/run" for i in range(4)],
        object(), {"Counts": {"parallel_threads": 2}}, 0,
    )
    assert len(eta) == 4
    rendered = "".join(stream.buf)
    assert "Re-running simulations" in rendered   # bar did NOT auto-disable


def test_worker_bar_counts_errors_on_postfix_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_MGMT_LOG", str(tmp_path / "data_management.log"))
    dm = _repoint_logger(tmp_path)
    monkeypatch.setattr(dm, "run_sim", _fake_run_sim_fail)

    w = dm.Worker(2)
    eta, ETA, paths = w.do_job_async(
        [f"/x/case-0/{i}/run" for i in range(3)],
        object(), {"Counts": {"parallel_threads": 2}}, 0,
    )
    assert eta == [] and paths == []          # failed tasks yield no results

    out = capsys.readouterr().out
    # failure count on the bar postfix, glyph-safe
    assert ("✗" in out) or ("[FAIL]" in out)
    # and no extra terminal line for the error itself
    assert "pool worker failed" not in out

    log_text = (tmp_path / "data_management.log").read_text()
    assert "pool worker failed" in log_text
