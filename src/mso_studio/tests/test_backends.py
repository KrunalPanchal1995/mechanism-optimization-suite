from pathlib import Path

from mso_studio.core.backends import (
    BackendError,
    run_backend,
    sanitized_subprocess_env,
    science_command,
)


def test_sanitized_env_strips_foreign_libpath(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ansys_inc/lib:/usr/local/lib")
    env = sanitized_subprocess_env()
    assert "/usr/local/lib" in env["LD_LIBRARY_PATH"]
    assert "ansys_inc" not in env["LD_LIBRARY_PATH"]


def test_sanitized_env_drops_key_when_all_foreign(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ansys_inc/lib")
    env = sanitized_subprocess_env()
    assert "LD_LIBRARY_PATH" not in env


def test_science_command_resolves_relative_to_src():
    cmd = science_command("run_mso_pipeline.py", "--config", "x.yaml")
    assert cmd[0].endswith("python3.9") or "python" in cmd[0]
    assert cmd[1].endswith("src/run_mso_pipeline.py")
    assert cmd[2:] == ["--config", "x.yaml"]


def test_run_backend_captures_stdout_and_status():
    lines = []
    result = run_backend(["python3", "-c", "print('hello'); print('world')"], on_line=lines.append)
    assert result.status == "done"
    assert result.return_code == 0
    assert lines == ["hello", "world"]


def test_run_backend_failure_status_and_error_message():
    result = run_backend(["python3", "-c", "import sys; print('boom'); sys.exit(3)"])
    assert result.status == "failed"
    assert result.return_code == 3
    exc = BackendError(result.command, result.return_code, result.log_path, result.tail)
    assert "exited with code 3" in str(exc)
    assert "boom" in str(exc)


def test_run_backend_writes_log_file(tmp_path):
    log_path = tmp_path / "run.log"
    run_backend(["python3", "-c", "print('line1')"], log_path=log_path)
    assert log_path.exists()
    assert "line1" in log_path.read_text()
