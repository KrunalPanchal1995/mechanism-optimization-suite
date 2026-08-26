"""
Tests for W1 — the Cantera YAML schema validator & migration linter.

Covers `Cantera google groups/W1_yaml_validator_migration.md`: detection of
deprecated CTI→YAML keywords (P1/P29), the pure-dict `migrate()`, the
`migrate_keyword` fixer handler, integration into the `rmip check` battery, and
the `rmip migrate` CLI.

Run:
    cd /data/MECHANISM_OPTIMIZATION_SUITE
    python -m pytest src/rmip/tests/test_yaml_validator.py -v
"""
import os
import sys
import warnings

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

warnings.filterwarnings("ignore")
pytest.importorskip("cantera")

from rmip.analysis.yaml_validator import (             # noqa: E402
    YamlValidator, KNOWN_RENAMES, RenameRule,
)
from rmip.analysis import mechanism_fixer as fixer_mod  # noqa: E402
from rmip.analysis.mechanism_check import MechanismChecker, ALL_CHECKS  # noqa: E402
from rmip.tests.fixtures._make_broken_mech import build_deprecated_dict  # noqa: E402


@pytest.fixture()
def dep_yaml(tmp_path):
    """Materialise the deprecated-keyword fixture to a temp file."""
    import yaml
    p = tmp_path / "broken_deprecated.yaml"
    with open(p, "w") as fh:
        yaml.safe_dump(build_deprecated_dict(), fh, sort_keys=False)
    return str(p)


class TestDetection:
    def test_detects_all_three_deprecations(self, dep_yaml):
        findings = YamlValidator(path=dep_yaml).validate()
        deps = [f for f in findings if f.category == "deprecated-keyword"]
        got = {f.details["deprecated"] for f in deps}
        assert got == {"ideal_gas", "three_body", "PLOG"}
        # every deprecation offers an auto migrate fix
        assert all(f.auto_fix is not None
                   and f.auto_fix.id == "migrate_keyword" for f in deps)

    def test_current_names_map_to_valid_replacements(self, dep_yaml):
        findings = YamlValidator(path=dep_yaml).validate()
        by_dep = {f.details["deprecated"]: f.details["current"]
                  for f in findings if f.category == "deprecated-keyword"}
        assert by_dep["ideal_gas"] == "ideal-gas"
        assert by_dep["three_body"] == "three-body"
        assert by_dep["PLOG"] == "pressure-dependent-Arrhenius"

    def test_clean_mechanism_has_no_deprecations(self):
        import cantera as ct
        gri = os.path.join(ct.__path__[0], "data", "gri30.yaml")
        findings = YamlValidator(path=gri).validate()
        assert not [f for f in findings if f.category == "deprecated-keyword"]

    def test_unrecognized_type_is_info_not_error(self, tmp_path):
        import cantera as ct, yaml
        base = yaml.safe_load(open(os.path.join(ct.__path__[0], "data", "h2o2.yaml")))
        base["reactions"].append({"equation": "H2O2 <=> 2 OH",
                                  "type": "totally-made-up-type",
                                  "rate-constant": {"A": 1e12, "b": 0, "Ea": 4e4}})
        p = tmp_path / "weird.yaml"
        yaml.safe_dump(base, open(p, "w"))
        findings = YamlValidator(path=str(p)).validate()
        weird = [f for f in findings if f.category == "unrecognized-type"]
        assert len(weird) == 1
        assert weird[0].severity == "info"        # never a hard error

    def test_rename_index_is_consistent(self):
        assert all(isinstance(r, RenameRule) for r in KNOWN_RENAMES)
        # no rule maps a token to itself
        assert all(r.deprecated != r.current for r in KNOWN_RENAMES)


class TestMigration:
    def test_migrate_dict_produces_loadable_mechanism(self, dep_yaml, tmp_path):
        import cantera as ct, yaml
        data, changes = YamlValidator(path=dep_yaml).migrate()
        assert len(changes) == 3
        out = tmp_path / "migrated.yaml"
        yaml.safe_dump(data, open(out, "w"), sort_keys=False)
        gas = ct.Solution(str(out))               # must not raise
        assert gas.n_reactions > 0

    def test_fixer_migrate_keyword_makes_it_loadable(self, dep_yaml, tmp_path):
        rep = MechanismChecker(dep_yaml).run(checks=["schema_deprecation"])
        fixes = fixer_mod.collect_auto_fixes(rep)
        assert len(fixes) == 3 and all(f["option_id"] == "migrate_keyword"
                                       for f in fixes)
        fixer = fixer_mod.MechanismFixer(dep_yaml)
        results = fixer.apply(fixes)
        assert all(r.status == "applied" for r in results)
        out = str(tmp_path / "fixed.yaml")
        fixer.write(out)
        rep2 = MechanismChecker(out).run(checks=["schema_deprecation"])
        assert rep2.loaded
        assert not [f for f in rep2.findings
                    if f.category == "deprecated-keyword"]


class TestIntegration:
    def test_schema_deprecation_registered_and_runs_preload(self, dep_yaml):
        assert "schema_deprecation" in ALL_CHECKS
        # the fixture does not load, yet the check still fires (raw-YAML scan)
        rep = MechanismChecker(dep_yaml).run()
        assert rep.loaded is False
        assert any(f.check == "schema_deprecation" for f in rep.findings)

    def test_migrate_cli_end_to_end(self, dep_yaml, tmp_path):
        from rmip.cli import main
        out = tmp_path / "cli_migrated.yaml"
        rc = main(["migrate", dep_yaml, "--out", str(out)])
        assert rc == 0                            # 0 = migrated file loads
        assert out.exists()
        import cantera as ct
        ct.Solution(str(out))                     # loads

    def test_migrate_cli_dry_run_writes_nothing(self, dep_yaml, tmp_path):
        from rmip.cli import main
        out = tmp_path / "should_not_exist.yaml"
        rc = main(["migrate", dep_yaml, "--out", str(out), "--dry-run"])
        assert rc == 1                            # deprecations present, nothing written
        assert not out.exists()
