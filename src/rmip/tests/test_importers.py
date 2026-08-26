"""
Tests for W3 — the foreign-format importer framework (OpenSmoke → Cantera YAML).

Covers `Cantera google groups/W3_foreign_format_importers.md` (P5): format
sniffing/routing, the refuse-without-acknowledgement contract for condensed
content, the gas-only import (drops LIQUID/MATERIAL species + their reactions,
reports every drop, converts + loads), the plain-CHEMKIN path, and the
`rmip import` CLI.

Run:
    cd /data/MECHANISM_OPTIMIZATION_SUITE
    python -m pytest src/rmip/tests/test_importers.py -v
"""
import os
import shutil
import sys
import warnings

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

warnings.filterwarnings("ignore")
ct = pytest.importorskip("cantera")
pytestmark = pytest.mark.skipif(shutil.which("ck2yaml") is None,
                                reason="ck2yaml not available")

from rmip.io.importers.registry import detect_format, import_mechanism  # noqa: E402
from rmip.io.importers.opensmoke import OpenSmokeImporter               # noqa: E402
from rmip.tests.fixtures.chemkin._make_opensmoke_fixture import build_text  # noqa: E402

_H2O2 = os.path.join(ct.__path__[0], "data", "h2o2.yaml")
_LI = os.path.join(_ROOT, "MECHANISM/Li_2015/chem_Li2015_wetCO_mech.inp")


@pytest.fixture(scope="module")
def os_fixture(tmp_path_factory):
    p = tmp_path_factory.mktemp("osfx") / "opensmoke_pe.inp"
    p.write_text(build_text())
    return str(p)


class TestSniffRouting:
    def test_opensmoke_detected(self, os_fixture):
        assert detect_format(os_fixture) == "opensmoke"
        assert OpenSmokeImporter.sniff(os_fixture)

    def test_plain_chemkin_detected(self):
        if not os.path.exists(_LI):
            pytest.skip("Li_2015 not present")
        assert detect_format(_LI) == "chemkin"
        assert not OpenSmokeImporter.sniff(_LI)

    def test_yaml_detected(self):
        assert detect_format(_H2O2) == "cantera-yaml"


class TestRefusalContract:
    def test_import_refused_without_gas_only(self, os_fixture, tmp_path):
        out = str(tmp_path / "pe.yaml")
        yaml_path, report = import_mechanism(os_fixture, out, fmt="opensmoke")
        assert yaml_path is None
        assert not os.path.exists(out)
        errs = [f for f in report.findings
                if f.category == "unrepresentable-content"]
        assert len(errs) == 1
        assert "PE" in errs[0].details["condensed_species"]
        # every unrepresentable construct is named
        cats = {f.category for f in report.findings}
        assert {"material", "condensed-species"} <= cats


class TestGasOnlyImport:
    def test_gas_only_converts_and_loads(self, os_fixture, tmp_path):
        out = str(tmp_path / "pe.yaml")
        yaml_path, report = import_mechanism(os_fixture, out, fmt="opensmoke",
                                             gas_only=True)
        assert yaml_path == out and os.path.exists(out)
        gas = ct.Solution(out)
        assert "PE" not in gas.species_names
        assert gas.n_species == 7 and gas.n_reactions == 3
        # the dropped heterogeneous reaction is reported, with its equation
        drops = [f for f in report.findings if f.category == "dropped-reaction"]
        assert len(drops) == 1 and "PE" in drops[0].details["equation"]
        assert report.meta["dropped_species"] == ["PE"]

    def test_strip_preserves_gas_reactions(self, os_fixture):
        imp = OpenSmokeImporter()
        text = open(os_fixture).read()
        gas_text, condensed, _ = imp.strip_to_gas(text)
        assert condensed == {"PE"}
        assert "H+O2=O+OH" in gas_text
        code = "\n".join(ln.split("!", 1)[0] for ln in gas_text.split("\n"))
        assert "LIQUID" not in code.upper() and "MATERIAL" not in code.upper()
        # dropped reaction is commented, not silently deleted
        assert "dropped heterogeneous" in gas_text


class TestChemkinPath:
    def test_plain_chemkin_imports(self, tmp_path):
        if not os.path.exists(_LI):
            pytest.skip("Li_2015 not present")
        out = str(tmp_path / "li.yaml")
        yaml_path, report = import_mechanism(_LI, out, fmt="chemkin")
        assert yaml_path == out
        gas = ct.Solution(out)
        assert gas.n_reactions > 0
        assert any(f.category == "converted" for f in report.findings)


class TestCli:
    def test_cli_refuses_then_accepts(self, os_fixture, tmp_path):
        from rmip.cli import main
        out = str(tmp_path / "pe.yaml")
        assert main(["import", os_fixture, "--out", out]) == 1
        assert main(["import", os_fixture, "--out", out, "--gas-only"]) == 0
        assert os.path.exists(out)

    def test_cli_yaml_passthrough(self):
        from rmip.cli import main
        assert main(["import", _H2O2]) == 0
