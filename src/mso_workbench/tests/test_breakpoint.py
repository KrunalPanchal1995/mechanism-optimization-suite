"""
breakpoint(): pause/resume/step against a synthetic pipeline -- a small
function that calls breakpoint() at a few named stages in sequence, run on
a background thread while the test acts as the "Workbench UI" writing
requests and observing state transitions. Every wait has a hard timeout so
a real bug (breakpoint() blocking forever) fails the test instead of
hanging the suite.
"""
import threading
import time

import pytest

from mso_workbench import breakpoint as wb
from mso_workbench import control

TIMEOUT_S = 5.0
_STAGES = ["stage_a", "stage_b", "stage_c"]


def _run_synthetic_pipeline(run_dir, log: list, extra_locals=None):
    for stage in _STAGES:
        wb.breakpoint(stage, locals_=(extra_locals or {}), source=("synthetic.py", "run"), run_dir=str(run_dir))
        log.append(stage)


def _wait_until(predicate, timeout=TIMEOUT_S):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_no_request_runs_straight_through(tmp_path):
    log = []
    _run_synthetic_pipeline(tmp_path, log)
    assert log == _STAGES
    state = control.read_state(tmp_path)
    assert state["status"] == control.STATUS_RUNNING
    assert state["stage"] == "stage_c"


def test_workbench_disabled_is_a_true_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("MSO_WORKBENCH", "0")
    control.write_request(tmp_path, control.ACTION_PAUSE, at="stage_b")  # should be ignored entirely
    log = []
    _run_synthetic_pipeline(tmp_path, log)
    assert log == _STAGES
    assert control.read_state(tmp_path) is None  # never touched at all


def test_pause_at_specific_stage_blocks_until_resumed(tmp_path):
    control.write_request(tmp_path, control.ACTION_PAUSE, at="stage_b")
    log = []
    thread = threading.Thread(target=_run_synthetic_pipeline, args=(tmp_path, log))
    thread.start()

    assert _wait_until(lambda: log == ["stage_a"]), "should stop right after stage_a, before stage_b runs"
    assert _wait_until(lambda: (control.read_state(tmp_path) or {}).get("status") == control.STATUS_PAUSED)
    assert control.read_state(tmp_path)["stage"] == "stage_b"

    control.write_request(tmp_path, control.ACTION_RESUME)
    thread.join(timeout=TIMEOUT_S)
    assert not thread.is_alive(), "breakpoint() never unblocked after resume"
    assert log == _STAGES


def test_pause_at_next_stops_at_the_first_boundary(tmp_path):
    control.write_request(tmp_path, control.ACTION_PAUSE, at=control.AT_NEXT)
    log = []
    thread = threading.Thread(target=_run_synthetic_pipeline, args=(tmp_path, log))
    thread.start()

    assert _wait_until(lambda: (control.read_state(tmp_path) or {}).get("status") == control.STATUS_PAUSED)
    assert control.read_state(tmp_path)["stage"] == "stage_a"
    assert log == []  # paused before even appending stage_a

    control.write_request(tmp_path, control.ACTION_RESUME)
    thread.join(timeout=TIMEOUT_S)
    assert log == _STAGES


def test_step_pauses_once_then_rearms_for_the_next_boundary(tmp_path):
    control.write_request(tmp_path, control.ACTION_STEP, at=control.AT_NEXT)
    log = []
    thread = threading.Thread(target=_run_synthetic_pipeline, args=(tmp_path, log))
    thread.start()

    assert _wait_until(lambda: (control.read_state(tmp_path) or {}).get("status") == control.STATUS_PAUSED)
    assert control.read_state(tmp_path)["stage"] == "stage_a"

    # Step again -> advances to stage_b then pauses again there.
    control.write_request(tmp_path, control.ACTION_STEP, at=control.AT_NEXT)
    assert _wait_until(lambda: log == ["stage_a"])
    assert _wait_until(
        lambda: (control.read_state(tmp_path) or {}).get("stage") == "stage_b"
        and (control.read_state(tmp_path) or {}).get("status") == control.STATUS_PAUSED
    )

    control.write_request(tmp_path, control.ACTION_RESUME)
    thread.join(timeout=TIMEOUT_S)
    assert not thread.is_alive()
    assert log == _STAGES


def test_pause_writes_a_snapshot_with_the_given_locals(tmp_path):
    control.write_request(tmp_path, control.ACTION_PAUSE, at="stage_b")
    log = []
    thread = threading.Thread(
        target=_run_synthetic_pipeline, args=(tmp_path, log, {"sample_count": 42, "seed": 7}),
    )
    thread.start()
    assert _wait_until(lambda: (control.read_state(tmp_path) or {}).get("status") == control.STATUS_PAUSED)

    from mso_workbench.snapshot import list_snapshots, load_manifest, load_namespace
    snaps = list_snapshots(tmp_path)
    assert len(snaps) == 1
    manifest = load_manifest(snaps[0])
    assert manifest["stage"] == "stage_b"
    assert manifest["source_ref"] == {"file": "synthetic.py", "function": "run"}
    namespace = load_namespace(snaps[0])
    assert namespace == {"sample_count": 42, "seed": 7}

    control.write_request(tmp_path, control.ACTION_RESUME)
    thread.join(timeout=TIMEOUT_S)


def test_clearing_request_externally_resumes_a_paused_breakpoint(tmp_path):
    """A UI might just delete the request file rather than write an
    explicit resume action -- breakpoint() must treat "no request" while
    paused the same as an explicit resume, not hang forever."""
    control.write_request(tmp_path, control.ACTION_PAUSE, at="stage_a")
    log = []
    thread = threading.Thread(target=_run_synthetic_pipeline, args=(tmp_path, log))
    thread.start()
    assert _wait_until(lambda: (control.read_state(tmp_path) or {}).get("status") == control.STATUS_PAUSED)

    control.clear_request(tmp_path)
    thread.join(timeout=TIMEOUT_S)
    assert not thread.is_alive()
    assert log == _STAGES
