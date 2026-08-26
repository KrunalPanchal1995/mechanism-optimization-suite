import json
from pathlib import Path

import pytest

from mso_studio.core.runrecord import RunConfig, RunRecord, new_record, sha256_of_file


def test_runconfig_roundtrip():
    cfg = RunConfig(module="nominal_sim", label="nominal_htc", inputs={"mechanism": "a.yaml"})
    data = cfg.to_dict()
    restored = RunConfig.from_dict(data)
    assert restored == cfg


def test_runrecord_roundtrip(tmp_path):
    cfg = RunConfig(module="nominal_sim", label="nominal_htc")
    record = new_record(cfg)
    record.results["n_datasets"] = 3
    record.mark_finished("done", return_code=0)

    saved_path = record.save(tmp_path)
    assert saved_path.exists()

    loaded = RunRecord.load(saved_path)
    assert loaded.config == cfg
    assert loaded.results["n_datasets"] == 3
    assert loaded.status == "done"
    assert loaded.wall_time_s is not None


def test_runrecord_latest(tmp_path, monkeypatch):
    monkeypatch.setattr("mso_studio.core.runrecord.runs_root", lambda: tmp_path)

    cfg1 = RunConfig(module="nominal_sim", label="first")
    r1 = new_record(cfg1)
    r1.mark_finished("done")
    r1.save()

    cfg2 = RunConfig(module="nominal_sim", label="second")
    r2 = new_record(cfg2)
    r2.mark_finished("done")
    saved2 = r2.save()
    # Ensure r2's file is strictly newer on disk (mtime resolution safety).
    import os
    import time
    time.sleep(0.01)
    os.utime(saved2, None)

    latest = RunRecord.latest(module="nominal_sim")
    assert latest is not None
    assert latest.config.label == "second"


def test_sha256_of_missing_file_is_none(tmp_path):
    assert sha256_of_file(tmp_path / "nope.txt") is None


def test_sha256_of_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    digest = sha256_of_file(f)
    assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
