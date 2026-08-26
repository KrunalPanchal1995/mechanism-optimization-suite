"""
Tests for W2 — the CHEMKIN structural preflight linter.

Covers `Cantera google groups/W2_chemkin_preflight_converter.md`: line-numbered
detection of the broken fixtures (P2/P3/P18/P35), no false positives on real
mechanisms, the `--fix` clean path (which converts via ck2yaml), reaction-scoped
species-case normalisation, the `rmip preflight` CLI, and the converter-GUI
preflight gate.

Run:
    cd /data/MECHANISM_OPTIMIZATION_SUITE
    python -m pytest src/rmip/tests/test_chemkin_preflight.py -v
"""
import os
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from rmip.io.chemkin_preflight import ChemkinPreflight, preflight   # noqa: E402
from rmip.tests.fixtures.chemkin._make_chemkin_fixtures import (     # noqa: E402
    write_all, FIXTURES,
)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("chemkin_fx")
    write_all(str(d))
    return str(d)


def _cats(report):
    return {f.category for f in report.findings}


class TestDetection:
    def test_case_mismatch(self, fx):
        rep = preflight(os.path.join(fx, "case_mismatch.inp"))
        hits = [f for f in rep.findings if f.category == "species-case-mismatch"]
        assert hits and hits[0].details["used"] == "Ar"
        assert hits[0].details["declared"] == "AR"
        assert hits[0].details["line"] == 10
        assert hits[0].auto_fix is not None

    def test_fortran_exponent(self, fx):
        rep = preflight(os.path.join(fx, "fortran_exp.inp"))
        hits = [f for f in rep.findings if f.category == "fortran-exponent"]
        assert hits and hits[0].details["line"] == 8
        assert rep.n_errors >= 1

    def test_noncon_flagged_as_warning(self, fx):
        rep = preflight(os.path.join(fx, "noncon.inp"))
        hits = [f for f in rep.findings if f.category == "noncon"]
        assert hits and hits[0].severity == "warning"

    def test_missing_end(self, fx):
        rep = preflight(os.path.join(fx, "no_end.inp"))
        hits = [f for f in rep.findings if f.category == "missing-end"]
        assert hits and rep.n_errors >= 1

    def test_tab_char(self, fx):
        rep = preflight(os.path.join(fx, "tab_delim.inp"))
        hits = [f for f in rep.findings if f.category == "tab-char"]
        assert hits and hits[0].details["line"] == 8

    def test_undeclared_species_has_auto_fix(self, fx):
        rep = preflight(os.path.join(fx, "undeclared_species.inp"))
        hits = [f for f in rep.findings if f.category == "undeclared-species"]
        assert hits and hits[0].details["token"] == "HO2X"
        assert hits[0].fix_options
        assert hits[0].fix_options[0].id == "insert_missing_species"

    def test_pseudo_species_gets_no_auto_fix(self, fx):
        # HV (chemiluminescence bookkeeping) must NOT be silently added as a
        # real species with no thermo data -- see _PSEUDO_SPECIES.
        body = ("ELEMENTS\nH O\nEND\nSPECIES\nH2 O2 H2O OH H\nEND\n"
                "REACTIONS\nOH+H=H2O+HV        1.0E13 0.0 0.0\nEND\n")
        p = os.path.join(fx, "_pseudo_species_hv.inp")
        with open(p, "w") as fh:
            fh.write(body)
        rep = preflight(p)
        hits = [f for f in rep.findings if f.category == "undeclared-species"]
        assert hits and hits[0].details["token"] == "HV"
        assert not hits[0].fix_options

    def test_hyphenated_species_name_not_flagged_as_fortran_exponent(self, fx):
        # KRUNAL_JULY_23.md / KONNOV_2019 regression: a real ring-locant
        # species name must never be mistaken for a broken exponent.
        rep = preflight(os.path.join(fx, "species_hyphen_locant.inp"))
        assert not any(f.category == "fortran-exponent" for f in rep.findings)
        assert not any(f.category == "undeclared-species" for f in rep.findings)

    def test_all_fixtures_produce_some_finding(self, fx):
        for name in FIXTURES:
            rep = preflight(os.path.join(fx, name))
            assert rep.findings, f"{name} produced no findings"


class TestNoFalsePositives:
    @pytest.mark.parametrize("mech", [
        "MECHANISM/Li_2015/chem_Li2015_wetCO_mech.inp",
        "MECHANISM/GRI3/chem_gri30_wetCO.inp",
    ])
    def test_real_mechanisms_have_no_errors(self, mech):
        path = os.path.join(_ROOT, mech)
        if not os.path.exists(path):
            pytest.skip(f"{mech} not present")
        rep = preflight(path)
        assert rep.n_errors == 0, [f.message for f in rep.findings
                                   if f.severity == "error"]


class TestFixPath:
    def test_clean_normalizes_fortran_and_writes_file(self, fx, tmp_path):
        pf = ChemkinPreflight(os.path.join(fx, "fortran_exp.inp"))
        pf.run()
        out = str(tmp_path / "clean.inp")
        notes = pf.write_clean(out)
        assert any("Fortran exponent" in n for n in notes)
        assert "E+14" in open(out).read() or "E14" in open(out).read()

    def test_case_fix_scoped_to_reactions(self, fx, tmp_path):
        pf = ChemkinPreflight(os.path.join(fx, "case_mismatch.inp"))
        pf.run()
        out = str(tmp_path / "clean.inp")
        pf.write_clean(out, normalize_case=True)
        txt = open(out).read()
        # the reaction now uses AR, and no stray 'Ar' remains
        assert "H+H+AR=H2+AR" in txt
        assert "Ar" not in txt

    def test_undeclared_species_auto_inserted(self, fx, tmp_path):
        pf = ChemkinPreflight(os.path.join(fx, "undeclared_species.inp"))
        pf.run()
        out = str(tmp_path / "clean.inp")
        notes = pf.write_clean(out)
        assert any("undeclared species" in n and "HO2X" in n for n in notes)
        txt = open(out).read()
        # inserted as its own SPECIES-block line, before that block's END
        species_block = txt.split("SPECIES", 1)[1].split("END", 1)[0]
        assert "HO2X" in species_block.split()
        # re-running preflight on the cleaned file finds no undeclared-species
        # finding for HO2X any more
        rep2 = preflight(out)
        assert not any(f.category == "undeclared-species"
                       and f.details.get("token") == "HO2X"
                       for f in rep2.findings)

    def test_undeclared_species_insertion_is_idempotent_and_deduped(self, fx, tmp_path):
        # same reaction repeats a missing species on two lines -> only one
        # SPECIES-block entry should be added
        body = ("ELEMENTS\nC H O\nEND\nSPECIES\nCH4 O2 CO2 H2O\nEND\n"
                "REACTIONS\n"
                "CH4+O2=CO2+H2O+BAR   1.0E13 0.0 0.0\n"
                "H2O+BAR=CH4+O2       1.0E12 0.0 0.0\n"
                "END\n")
        p = tmp_path / "dup_undeclared.inp"
        p.write_text(body)
        pf = ChemkinPreflight(str(p))
        pf.run()
        out = str(tmp_path / "clean.inp")
        pf.write_clean(out)
        txt = open(out).read()
        species_block = txt.split("SPECIES", 1)[1].split("END", 1)[0]
        assert species_block.split().count("BAR") == 1

    def test_species_name_with_hyphen_survives_clean_text(self, fx, tmp_path):
        # KRUNAL_JULY_23.md / KONNOV_2019 regression (see fixture docstring):
        # the old regex corrupted "C2H4O1-2" -> "C2H4O1E-2" whenever it stood
        # alone on a SPECIES declaration line.
        pf = ChemkinPreflight(os.path.join(fx, "species_hyphen_locant.inp"))
        pf.run()
        out = str(tmp_path / "clean.inp")
        notes = pf.write_clean(out)
        assert not any("Fortran exponent" in n for n in notes)
        txt = open(out).read()
        assert "C2H4O1-2" in txt
        assert "C2H4O1E-2" not in txt

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(
            _ROOT, "MECHANISM/KONNOV_2019_AMMONIA_HYDROGEN/128-957.inp")),
        reason="KONNOV_2019_AMMONIA_HYDROGEN mechanism not present")
    def test_konnov_2019_species_names_not_corrupted(self, tmp_path):
        """KRUNAL_JULY_23.md: fix_fortran_exponents()/'-fix' used to rewrite
        the real ring-locant species 'C2H4O1-2'/'C2H3O1-2' to
        'C2H4O1E-2'/'C2H3O1E-2' in the SPECIES block, desyncing them from
        their (correctly-spelled) reaction usages and making ck2yaml report
        the reaction usage as an unparsable token. Confirm the cleaned file
        keeps both names intact everywhere."""
        src = os.path.join(_ROOT,
                            "MECHANISM/KONNOV_2019_AMMONIA_HYDROGEN/128-957.inp")
        pf = ChemkinPreflight(src)
        pf.run()
        out = str(tmp_path / "clean.inp")
        pf.write_clean(out)
        txt = open(out).read()
        for name in ("C2H4O1-2", "C2H3O1-2"):
            assert name in txt
            assert name.replace("-", "E-") not in txt

    def test_case_fix_leaves_real_file_content_unchanged(self, tmp_path):
        src = os.path.join(_ROOT,
                           "MECHANISM/Li_2015/chem_Li2015_wetCO_mech.inp")
        if not os.path.exists(src):
            pytest.skip("Li_2015 not present")
        pf = ChemkinPreflight(src)
        pf.run()
        out = str(tmp_path / "clean.inp")
        pf.write_clean(out, normalize_case=True)
        a = open(src, encoding="utf-8", errors="replace").read().replace("\r", "")
        b = open(out).read()
        # only whitespace/CRLF differences allowed — content identical
        assert a.split() == b.split()

    @pytest.mark.skipif(shutil.which("ck2yaml") is None, reason="ck2yaml absent")
    def test_cleaned_injected_mechanism_converts(self, tmp_path):
        """Inject a Fortran exponent into a real mechanism → fix → ck2yaml."""
        import re
        src = os.path.join(_ROOT,
                           "MECHANISM/Li_2015/chem_Li2015_wetCO_mech.inp")
        if not os.path.exists(src):
            pytest.skip("Li_2015 not present")
        lines = open(src, encoding="utf-8", errors="replace").read().split("\n")
        inr = False
        for i, ln in enumerate(lines):
            if ln.strip().upper().startswith("REACTIONS"):
                inr = True
                continue
            if inr and re.search(r"=.*\dE\+?\d", ln) and not ln.lstrip().startswith("!"):
                lines[i] = re.sub(r"E\+?(\d)", r"+\1", ln, count=1)
                break
        broken = tmp_path / "broken.inp"
        broken.write_text("\n".join(lines))
        rep = preflight(str(broken))
        assert any(f.category == "fortran-exponent" for f in rep.findings)
        clean = tmp_path / "clean.inp"
        ChemkinPreflight(str(broken)).write_clean(str(clean))
        yaml_out = tmp_path / "clean.yaml"
        r = subprocess.run(
            ["ck2yaml", f"--input={clean}", f"--output={yaml_out}", "--permissive"],
            capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and yaml_out.exists(), r.stderr
        import cantera as ct
        ct.Solution(str(yaml_out))


class TestCli:
    def test_preflight_cli_exit_code(self, fx):
        from rmip.cli import main
        assert main(["preflight", os.path.join(fx, "fortran_exp.inp")]) == 1
        # a clean real mechanism → 0
        li = os.path.join(_ROOT, "MECHANISM/Li_2015/chem_Li2015_wetCO_mech.inp")
        if os.path.exists(li):
            assert main(["preflight", li]) == 0

    def test_preflight_cli_fix_writes_clean(self, fx, tmp_path):
        from rmip.cli import main
        out = tmp_path / "clean.inp"
        main(["preflight", os.path.join(fx, "fortran_exp.inp"),
              "--fix", "--fixed-out", str(out)])
        assert out.exists()


class TestConverterRunStatusExport:
    """KRUNAL_JULY_23.md point 1: 'Add option of exporting the status of the
    run from any GUI into a txt file'. Real tk.Tk() root, no synthetic
    clicks -- same convention as test_gui_stress_sweep.py's widget checks."""

    def test_build_run_status_reflects_last_run(self, tmp_path):
        tk = pytest.importorskip("tkinter")
        import mechanism_converter_gui as conv

        root = tk.Tk()
        try:
            app = conv.ConverterApp(root)
            app._last_vals = {
                "inp": "/data/MECHANISM/KONNOV/128-957.inp",
                "thermo": "/data/MECHANISM/KONNOV/128-957.dat",
                "trans": "", "scanman": "/usr/bin/ScanMan",
                "db_path": "/data/outputs/project.db",
                "glossary": "/data/species_smiles.csv",
                "out_dir": "/data/outputs/converted",
            }
            app._last_flags = {"make_yaml": True, "fix_exp": True, "permissive": True}
            app._last_status = "failed"
            app._last_errors = ["ck2yaml failed (exit 1)"]
            app._out.insert("1.0", "ERROR: Unable to parse ...\n")

            report = app._build_run_status()
            assert report.tool == "CHEMKIN Mechanism Converter"
            assert report.inputs["mechanism (.inp/.mech)"] == \
                "/data/MECHANISM/KONNOV/128-957.inp"
            assert report.output_dir == "/data/outputs/converted"
            assert report.options["fix_exp"] is True
            assert report.status == "failed"
            assert report.summary == "ck2yaml failed (exit 1)"
            assert "Unable to parse" in report.log

            out = tmp_path / "status.txt"
            report.write_txt(str(out))
            text = out.read_text()
            assert "128-957.inp" in text and "ck2yaml failed" in text
        finally:
            root.destroy()

    def test_export_button_present_in_ui(self):
        tk = pytest.importorskip("tkinter")
        import mechanism_converter_gui as conv

        def _find_texts(widget):
            texts = []
            if "text" in widget.keys():
                texts.append(widget.cget("text"))
            for child in widget.winfo_children():
                texts.extend(_find_texts(child))
            return texts

        root = tk.Tk()
        try:
            app = conv.ConverterApp(root)
            all_texts = _find_texts(root)
            assert "Export run status…" in all_texts
        finally:
            root.destroy()


class TestConverterHeadlessCli:
    """The --convert headless CLI mode (KRUNAL_JULY_23.md PySide6 port):
    mso_studio.core.converter shells out to this instead of importing
    rmip/cantera into the PySide6 venv, so this argv contract must actually
    run convert() end-to-end and report success/failure via exit code."""

    def test_convert_cli_succeeds_on_clean_mechanism(self, tmp_path):
        li = os.path.join(_ROOT, "MECHANISM/Li_2015/chem_Li2015_wetCO_mech.inp")
        if not os.path.exists(li):
            pytest.skip("Li_2015 not present")
        proc = subprocess.run(
            [sys.executable, os.path.join(_ROOT, "mechanism_converter_gui.py"),
             "--convert", "--input", li, "--out-dir", str(tmp_path),
             "--make-yaml", "--permissive", "--no-validate"],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (tmp_path / "chem_Li2015_wetCO_mech.yaml").exists()

    def test_convert_cli_reports_failure_via_exit_code(self, fx, tmp_path):
        proc = subprocess.run(
            [sys.executable, os.path.join(_ROOT, "mechanism_converter_gui.py"),
             "--convert", "--input", os.path.join(fx, "case_mismatch.inp"),
             "--out-dir", str(tmp_path), "--make-yaml"],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 1
        assert "ERROR" in proc.stdout

    def test_convert_cli_requires_input(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(_ROOT, "mechanism_converter_gui.py"), "--convert"],
            capture_output=True, text=True, timeout=30)
        assert proc.returncode == 1
        assert "--input" in proc.stdout


class TestConverterGate:
    def test_convert_blocks_on_preflight_error_unless_permissive(self, fx, tmp_path):
        import mechanism_converter_gui as conv
        msgs = []
        errs = conv.convert(
            os.path.join(fx, "case_mismatch.inp"), "", "", str(tmp_path),
            make_yaml=True, make_pre=False, make_db=False,
            scanman="", db_path="", glossary="",
            permissive=False, no_validate=True, fix_exp=False, fix_crlf=False,
            log=msgs.append)
        assert any("preflight" in e.lower() for e in errs)

    def test_convert_auto_inserts_undeclared_species(self, fx, tmp_path):
        """KRUNAL_JULY_23.md point 2: convert() must patch a genuinely
        undeclared reaction participant into the SPECIES block on its own,
        no checkbox/user input required, before ck2yaml runs."""
        import mechanism_converter_gui as conv
        msgs = []
        conv.convert(
            os.path.join(fx, "undeclared_species.inp"), "", "", str(tmp_path),
            make_yaml=False, make_pre=False, make_db=False,
            scanman="", db_path="", glossary="",
            permissive=True, no_validate=True, fix_exp=True, fix_crlf=True,
            log=msgs.append)
        assert any("HO2X" in m and "undeclared species" in m for m in msgs)
        fixed = tmp_path / "fixed_undeclared_species.inp"
        assert fixed.exists()
        species_block = fixed.read_text().split("SPECIES", 1)[1].split("END", 1)[0]
        assert "HO2X" in species_block.split()

    def test_convert_konnov_2019_no_longer_corrupts_species_names(self, tmp_path):
        """KRUNAL_JULY_23.md's exact repro: converting KONNOV_2019 used to
        fail with `Unexpected token "C2H4O1-2+"` because the (now-fixed)
        Fortran-exponent regex corrupted the SPECIES-block declaration.
        convert() must no longer do that."""
        import mechanism_converter_gui as conv
        src = os.path.join(_ROOT,
                            "MECHANISM/KONNOV_2019_AMMONIA_HYDROGEN/128-957.inp")
        if not os.path.exists(src):
            pytest.skip("KONNOV_2019_AMMONIA_HYDROGEN not present")
        msgs = []
        conv.convert(
            src, "", "", str(tmp_path),
            make_yaml=False, make_pre=False, make_db=False,
            scanman="", db_path="", glossary="",
            permissive=True, no_validate=True, fix_exp=True, fix_crlf=True,
            log=msgs.append)
        assert not any("C2H4O1E-2" in m or "C2H3O1E-2" in m for m in msgs)
        # KONNOV_2019's .inp needs no genuine Fortran-exponent/CRLF fix at
        # all (the old bug's "2 exponent(s) repaired" were both this same
        # species-name corruption) -- so preprocess_copy() may leave the
        # file untouched and never write a "fixed_" copy. Check whichever
        # the pipeline actually used.
        fixed = tmp_path / "fixed_128-957.inp"
        txt = fixed.read_text() if fixed.exists() else open(src).read()
        assert "C2H4O1-2" in txt and "C2H4O1E-2" not in txt
        assert "C2H3O1-2" in txt and "C2H3O1E-2" not in txt
