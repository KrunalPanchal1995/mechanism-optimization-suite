"""
rmip/tests/test_fusion_writer_load_guard.py
=============================================
Regression coverage for the fusion-writer load guard (Phase 2, bug.md):
``ChemkinWriter.convert_to_cantera_yaml``'s new ``verify_loads=True`` default
attempts a real ``cantera.Solution()`` load of the ck2yaml output and raises
if it fails, instead of silently returning a path to a broken YAML.

Note: the installed Cantera's ck2yaml (3.0.1) turns out to ALREADY
self-validate via an internal ``Solution()`` call and exits non-zero on the
exact "Undeclared duplicate reactions" defect this guard targets -- so on
this machine, the pre-existing ``proc.returncode != 0`` check already catches
the real Keromnes/Li/Davis repro before ``verify_loads`` even runs (see
``TestRealRepro`` below, which covers this end-to-end regardless of which
check fires). The ``TestVerifyLoadsFlag`` class isolates and tests the new
code path directly (via a stubbed ck2yaml that exits 0 with broken output),
independent of ck2yaml's own version-dependent internal behavior -- this is
what actually exercises the new logic and is real defense-in-depth for
Cantera versions / defect classes ck2yaml's own check doesn't cover.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("cantera")

from rmip.io.chemkin_writer import ChemkinWriter  # noqa: E402

_FIXTURE_DB = (
    Path(__file__).resolve().parent
    / "fixtures" / "fused_mechanisms" / "fusion_stress_stale_keys.db"
)

# A minimal, genuinely Cantera-invalid YAML (undeclared duplicate reactions --
# same species/mechanism as the real repro, but hand-written so this class of
# test doesn't depend on ck2yaml/a real DB at all).
_BROKEN_YAML = """\
description: broken -- undeclared duplicate reaction (H2 dissociation)
phases:
- name: gas
  thermo: ideal-gas
  elements: [H]
  species: [H, H2]
  kinetics: gas
  reactions: all
species:
- name: H
  composition: {H: 1}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
           data: [[2.5, 0.0, 0.0, 0.0, 0.0, 25473.66, -0.44668285],
                   [2.5, 0.0, 0.0, 0.0, 0.0, 25473.66, -0.44668285]]}
- name: H2
  composition: {H: 2}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
           data: [[3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10,
                    2.00255376e-14, -950.158922, -3.20502331],
                   [3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10,
                    2.00255376e-14, -950.158922, -3.20502331]]}
reactions:
- equation: H2 + M <=> 2 H + M
  type: three-body
  rate-constant: {A: 4.577e+19, b: -1.4, Ea: 104390.0}
- equation: 2 H + H2 <=> 2 H2
  rate-constant: {A: 4.577e+19, b: -1.4, Ea: 104390.0}
"""


class TestVerifyLoadsFlag:
    """Isolates convert_to_cantera_yaml's new verify_loads logic with a
    stubbed ck2yaml (always exits 0), independent of any real ck2yaml
    version's own internal validation behavior."""

    def _stub_ck2yaml(self, tmp_path, broken: bool) -> str:
        """A trivial shell script masquerading as ck2yaml: parses
        --output=<path> and writes either the broken or a valid fixture
        YAML there, always exiting 0."""
        content = _BROKEN_YAML if broken else _VALID_YAML
        script = tmp_path / "fake_ck2yaml.py"
        script.write_text(
            "import sys\n"
            "out = [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--output=')][0]\n"
            f"open(out, 'w').write({content!r})\n"
        )
        return str(script)

    def test_verify_loads_true_raises_on_broken_output(self, tmp_path, monkeypatch):
        import sys

        import subprocess as _subprocess
        real_run = _subprocess.run
        fake = self._stub_ck2yaml(tmp_path, broken=True)
        monkeypatch.setattr(
            "rmip.io.chemkin_writer.subprocess.run",
            lambda cmd, **kw: real_run([sys.executable, fake] + cmd[1:], **kw),
        )
        writer = ChemkinWriter(":memory:")
        with pytest.raises(RuntimeError, match="does not load in Cantera"):
            writer.convert_to_cantera_yaml(
                str(tmp_path / "in.inp"),
                output_yaml=str(tmp_path / "out.yaml"),
                verify_loads=True,
            )
        # ck2yaml itself "succeeded" -- the broken file IS on disk, this
        # guard's whole point is to not let that pass as a silent success.
        assert (tmp_path / "out.yaml").exists()

    def test_verify_loads_false_bypasses_the_check(self, tmp_path, monkeypatch):
        import sys
        import subprocess as _subprocess

        real_run = _subprocess.run
        fake = self._stub_ck2yaml(tmp_path, broken=True)
        monkeypatch.setattr(
            "rmip.io.chemkin_writer.subprocess.run",
            lambda cmd, **kw: real_run([sys.executable, fake] + cmd[1:], **kw),
        )
        writer = ChemkinWriter(":memory:")
        yaml_path = writer.convert_to_cantera_yaml(
            str(tmp_path / "in.inp"),
            output_yaml=str(tmp_path / "out.yaml"),
            verify_loads=False,
        )
        assert Path(yaml_path).exists() and Path(yaml_path).stat().st_size > 0

    def test_verify_loads_true_passes_on_valid_output(self, tmp_path, monkeypatch):
        import sys
        import subprocess as _subprocess

        real_run = _subprocess.run
        fake = self._stub_ck2yaml(tmp_path, broken=False)
        monkeypatch.setattr(
            "rmip.io.chemkin_writer.subprocess.run",
            lambda cmd, **kw: real_run([sys.executable, fake] + cmd[1:], **kw),
        )
        writer = ChemkinWriter(":memory:")
        yaml_path = writer.convert_to_cantera_yaml(
            str(tmp_path / "in.inp"),
            output_yaml=str(tmp_path / "out.yaml"),
            verify_loads=True,
        )
        assert Path(yaml_path).exists()


_VALID_YAML = """\
description: valid, loadable mechanism
phases:
- name: gas
  thermo: ideal-gas
  elements: [H]
  species: [H, H2]
  kinetics: gas
  reactions: all
species:
- name: H
  composition: {H: 1}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
           data: [[2.5, 0.0, 0.0, 0.0, 0.0, 25473.66, -0.44668285],
                   [2.5, 0.0, 0.0, 0.0, 0.0, 25473.66, -0.44668285]]}
- name: H2
  composition: {H: 2}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 6000.0],
           data: [[3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10,
                    2.00255376e-14, -950.158922, -3.20502331],
                   [3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10,
                    2.00255376e-14, -950.158922, -3.20502331]]}
reactions:
- equation: H2 + M <=> 2 H + M
  type: three-body
  rate-constant: {A: 4.577e+19, b: -1.4, Ea: 104390.0}
"""


@pytest.mark.skipif(shutil.which("ck2yaml") is None, reason="ck2yaml not on PATH")
class TestRealRepro:
    """End-to-end against the real, checked-in fusion_stress_stale_keys.db
    fixture -- a copy of this suite's own pre-canonical-key-rebuild-
    migration fusion DB (Keromnes/Li/Davis, mechanism ids 1/2/3), the exact
    DB that produced the broken Fused_mean_KLD.yaml documented in bug.md."""

    def test_stale_keys_write_cantera_now_raises(self, tmp_path):
        """Without the canonical-key-rebuild migration, re-fusing the real
        pre-fix DB with write_cantera=True must raise somewhere in the
        pipeline instead of silently succeeding -- whether ck2yaml's own
        internal validation catches it (this machine's Cantera version) or
        this suite's verify_loads guard does (older/different versions)."""
        from rmip.stubs import FusionEngine
        from rmip.analysis.graph_comparator import GraphComparator

        db_path = str(tmp_path / "stale.db")
        shutil.copy(_FIXTURE_DB, db_path)

        cmp = GraphComparator(db_path)
        result = cmp.compare([1, 2, 3])
        spec = cmp.build_union_spec(result)

        engine = FusionEngine(spec, db_path)
        out_dir = tmp_path / "out"
        with pytest.raises(RuntimeError):
            engine.fuse("Fused_stale_repro", output_dir=str(out_dir), write_cantera=True)

    def test_migrated_keys_write_cantera_succeeds_and_loads(self, tmp_path):
        """The same DB, but with the canonical-key-rebuild migration applied
        first -- write_cantera=True must succeed and produce a real,
        loadable mechanism (17 species / 56 reactions, matching the known-
        good Fused_mean_KLD_v2.yaml fixture)."""
        import cantera as ct

        from rmip.database.migrations import run_canonical_key_rebuild_migration
        from rmip.stubs import FusionEngine
        from rmip.analysis.graph_comparator import GraphComparator

        db_path = str(tmp_path / "migrated.db")
        shutil.copy(_FIXTURE_DB, db_path)

        conn = sqlite3.connect(db_path)
        run_canonical_key_rebuild_migration(conn)
        conn.close()

        cmp = GraphComparator(db_path)
        result = cmp.compare([1, 2, 3])
        spec = cmp.build_union_spec(result)

        engine = FusionEngine(spec, db_path)
        out_dir = tmp_path / "out"
        engine.fuse("Fused_migrated_repro", output_dir=str(out_dir), write_cantera=True)

        yaml_path = out_dir / "Fused_migrated_repro.yaml"
        assert yaml_path.exists()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gas = ct.Solution(str(yaml_path))
        assert gas.n_species == 17
        assert gas.n_reactions == 56
