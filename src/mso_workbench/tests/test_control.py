"""control.py: the file-based pause protocol's read/write primitives."""
import json
import os

import pytest

from mso_workbench import control


def test_no_request_or_state_initially(tmp_path):
    assert control.read_request(tmp_path) is None
    assert control.read_state(tmp_path) is None
    assert control.heartbeat_age_s(tmp_path) is None


def test_write_then_read_request_round_trips(tmp_path):
    control.write_request(tmp_path, control.ACTION_PAUSE, at="sampling.pre_design_matrix")
    req = control.read_request(tmp_path)
    assert req["action"] == control.ACTION_PAUSE
    assert req["at"] == "sampling.pre_design_matrix"
    assert "ts" in req


def test_clear_request_removes_it(tmp_path):
    control.write_request(tmp_path, control.ACTION_PAUSE)
    control.clear_request(tmp_path)
    assert control.read_request(tmp_path) is None


def test_clear_request_on_nonexistent_is_a_noop(tmp_path):
    control.clear_request(tmp_path)  # must not raise


def test_write_then_read_state_round_trips(tmp_path):
    control.write_state(tmp_path, control.STATUS_PAUSED, "sampling.pre_design_matrix", pid=1234)
    state = control.read_state(tmp_path)
    assert state["status"] == control.STATUS_PAUSED
    assert state["stage"] == "sampling.pre_design_matrix"
    assert state["pid"] == 1234


def test_write_state_defaults_pid_to_current_process(tmp_path):
    control.write_state(tmp_path, control.STATUS_RUNNING, "config.loaded")
    assert control.read_state(tmp_path)["pid"] == os.getpid()


def test_malformed_request_file_reads_as_none(tmp_path):
    path = control.control_dir(tmp_path) / control.REQUEST_FILE
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json")
    assert control.read_request(tmp_path) is None


def test_touch_heartbeat_updates_age(tmp_path):
    control.touch_heartbeat(tmp_path)
    age = control.heartbeat_age_s(tmp_path)
    assert age is not None
    assert age >= 0
    assert age < 5  # sanity: just touched


def test_two_run_dirs_do_not_cross_contaminate(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    control.write_request(run_a, control.ACTION_PAUSE, at="sampling.pre_design_matrix")
    assert control.read_request(run_b) is None
    assert control.read_request(run_a) is not None


def test_is_enabled_true_by_default(monkeypatch):
    monkeypatch.delenv("MSO_WORKBENCH", raising=False)
    assert control.is_enabled()


def test_is_enabled_false_when_explicitly_zero(monkeypatch):
    monkeypatch.setenv("MSO_WORKBENCH", "0")
    assert not control.is_enabled()


def test_is_enabled_true_for_any_other_value(monkeypatch):
    monkeypatch.setenv("MSO_WORKBENCH", "1")
    assert control.is_enabled()


def test_atomic_write_leaves_no_tmp_file_behind(tmp_path):
    control.write_request(tmp_path, control.ACTION_PAUSE)
    leftover = list(control.control_dir(tmp_path).glob("*.tmp*"))
    assert leftover == []
