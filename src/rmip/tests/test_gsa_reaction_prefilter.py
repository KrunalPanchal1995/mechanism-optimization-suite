"""
rmip/tests/test_gsa_reaction_prefilter.py
==========================================
Benchmark/regression tests for the opt-in GSA reaction-prefilter wired into
``src/MUQ-SAC/run_script.py`` + ``src/MUQ-SAC/DesignMatrix.py``
(``MarkDownFiles/ZERORK_GSA_PLAN.md`` Phase 4's previously-deferred item).
See ``src/rmip/simulation/gsa_reaction_prefilter.py`` for the module under
test.

Why this benchmark is done at the DesignMatrix level, not a literal
run_script.py end-to-end run
--------------------------------------------------------------------------
A literal end-to-end ``run_script.py`` run needs a licensed Chemkin-Pro /
FlameMaster install plus MUQ-SAC's own compiled ``samap``/``jpdap``
binaries -- none of which exist in this environment (confirmed with KP:
accepted as a known gap, not something this suite works around). What CAN
be, and is, exercised for real here: ``build_unsrt_subset()`` (real
``Uncertainty.py`` XML parsing) + real ``DesignMatrix`` construction/
sampling (real ``qmc_sampling.py`` machinery) -- pure Python + Cantera, no
FlameMaster/samap/jpdap anywhere in the call path -- against the SAME real
3-reaction H2/O2 fixture and GRI-3.0 mechanism (``mechanism_id=8`` in
``outputs/project.db``) that
``gsa_prs_crosscheck.py``/``test_gsa_prs_crosscheck.py`` already validated.
Real, converged S1/ST values for that exact fixture (from
``ZERORK_GSA_PLAN.md`` lines ~641-652, "AGREE" verdict, n_design=40/
n_direct_base=512): ``H+O2<=>O+OH`` dominant at S1_prs=0.9992/
S1_direct=0.9991; ``O+H2<=>H+OH`` near-zero at 0.0002/0.0002;
``OH+H2<=>H+H2O`` near-zero at 0.0006/0.0008. Used verbatim below.

``run_script.py`` restructuring note
--------------------------------------
``run_script.py`` has NO ``if __name__ == "__main__":`` guard -- its entire
body (argv parsing, file I/O, ``exit()`` on missing args, and everything
downstream) executes unconditionally at import time. This is pre-existing,
zero-test-coverage script structure and this change deliberately does NOT
restructure it (the risk of re-indenting ~1000 lines of untested production
code to make the whole module import-safe far outweighs the benefit here).
Instead, ``_maybe_apply_gsa_prefilter`` is extracted from the file's SOURCE
TEXT via the ``ast`` module (see ``_extract_top_level_function`` below),
compiled, and exec'd in an isolated namespace -- this exercises the exact
function body that ships in ``run_script.py`` as a genuine unit test,
without ever executing the script's side-effecting top-level code. The only
change made to ``run_script.py`` itself is purely additive: the new function
definition (placed before the argv-parsing block) and a single-line call
site right after ``unsrt_data`` is loaded/unpickled -- zero existing
statements were altered or reordered.

MANUAL end-to-end verification checklist (for whoever eventually has the
licensed toolchain)
--------------------------------------------------------------------------
1. Pick a small, working ``target.opt`` that already runs ``run_script.py``
   successfully end-to-end (e.g. ``DATASET/H2_O2_TEST/target.opt``).
2. Run it once with NO ``GSA_Prefilter`` key (today's default) and keep its
   full output tree (``Opt/``, ``DesignMatrix.csv``, ``Optimized_mech.yaml``,
   ``unsrt.pkl``, ...) as a baseline.
3. Run it again from a clean directory, byte-for-byte the same
   ``target.opt``/inputs, and confirm the new run's output tree is IDENTICAL
   to the baseline -- this is the "opt-in, default OFF, provably a no-op"
   guarantee this test file can only verify at the DesignMatrix/unsrt_data
   level; a real run is the final confirmation.
4. Add a ``GSA_Prefilter`` block (see the schema example in
   ``MarkDownFiles/ZERORK_GSA_PLAN.md``'s Phase 4 section) pointing at a
   real ``rmip gsa-run`` JSON result for the same mechanism, with a
   threshold/``top_n`` that retains a strict, known subset of reactions.
5. Re-run ``run_script.py`` with that config from a clean directory and
   confirm: (a) ``ACTIVE_RXN_LIST.csv``/``rxn_list_a_fact.csv`` only list
   the retained reactions; (b) ``Optimized_mech.yaml`` has non-NaN,
   physically sane Arrhenius parameters for the retained reactions and
   UNCHANGED (nominal) parameters for every dropped reaction (they were
   never in the optimization); (c) the run completes without new
   errors/exceptions relative to step 2's disabled baseline.
"""
from __future__ import annotations

import ast
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MUQSAC_SRC = _REPO_ROOT / "src" / "MUQ-SAC"
_RUN_SCRIPT_PATH = _MUQSAC_SRC / "run_script.py"
_FIXTURE_XML = _REPO_ROOT / "src" / "rmip" / "tests" / "fixtures" / "h2_o2_crosscheck_unsrt.xml"
_GUI_STRESS_YAML = (
    _REPO_ROOT / "src" / "rmip" / "tests" / "fixtures" / "gui_stress" / "mech_std.yaml"
)
_PROJECT_DB = _REPO_ROOT / "outputs" / "project.db"
_ZERORK_INSTALL = "/home/krithika/zero-rk/build2/inst_dir"
_CK2YAML = "/home/krithika/anaconda3/bin/ck2yaml"

# Real, converged values from ZERORK_GSA_PLAN.md's Phase 4 cross-check
# (n_design=40, n_direct_base=512, verdict AGREE) -- see module docstring.
_REAL_S1 = {
    "H + O2 <=> O + OH": 0.9992,     # dominant
    "O + H2 <=> H + OH": 0.0002,     # near-zero
    "OH + H2 <=> H + H2O": 0.0006,   # near-zero
}
_REAL_ST = {
    "H + O2 <=> O + OH": 0.9991,
    "O + H2 <=> H + OH": 0.0002,
    "OH + H2 <=> H + H2O": 0.0008,
}
_DOMINANT_RXN = "H + O2 <=> O + OH"


def _ensure_muqsac_importable() -> None:
    for p in (str(_REPO_ROOT / "src"), str(_MUQSAC_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _muqsac_importable() -> bool:
    try:
        _ensure_muqsac_importable()
        import Uncertainty  # noqa: F401
        import MechanismParser  # noqa: F401
        import MechManipulator2_0  # noqa: F401
        import DesignMatrix  # noqa: F401
        return True
    except Exception:
        return False


def _zerork_export_available() -> bool:
    return (Path(_ZERORK_INSTALL) / "bin").exists()


def _ck2yaml_available() -> bool:
    return shutil.which("ck2yaml") is not None or Path(_CK2YAML).exists()


_needs_muqsac = pytest.mark.skipif(
    not _muqsac_importable(), reason="src/MUQ-SAC modules not importable",
)
_needs_project_db = pytest.mark.skipif(
    not _PROJECT_DB.exists(), reason="outputs/project.db not present",
)
_needs_zerork_export = pytest.mark.skipif(
    not _zerork_export_available(), reason="Zero-RK install dir not present",
)
_needs_ck2yaml = pytest.mark.skipif(
    not _ck2yaml_available(), reason="ck2yaml not available",
)


def _cantera_importable() -> bool:
    try:
        import cantera  # noqa: F401
        return True
    except Exception:
        return False


_needs_cantera = pytest.mark.skipif(
    not _cantera_importable(), reason="cantera not importable",
)


# ─────────────────────────────────────────────────────────────────────────────
# ast-extraction helper: pull one top-level function's exact source out of
# run_script.py without executing the rest of that file's module-level body
# (see module docstring's "run_script.py restructuring note").
# ─────────────────────────────────────────────────────────────────────────────

def _extract_top_level_function(source_path: Path, func_name: str):
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            segment = ast.get_source_segment(source, node)
            ns = {"__file__": str(source_path)}
            exec(compile(segment, f"<extracted {func_name}>", "exec"), ns)
            return ns[func_name]
    raise AssertionError(f"{func_name!r} not found as a top-level function in {source_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit tests for the new rmip.simulation.gsa_reaction_prefilter module
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyGsaPrefilterNoOp:
    def test_disabled_config_is_identity_empty_dict(self):
        from rmip.simulation.gsa_reaction_prefilter import apply_gsa_prefilter

        unsrt = {"rxn1": object(), "rxn2": object()}
        out = apply_gsa_prefilter(unsrt, {})
        assert out is unsrt  # same object, not just an equal dict

    def test_disabled_config_is_identity_none_like(self):
        from rmip.simulation.gsa_reaction_prefilter import apply_gsa_prefilter

        unsrt = {"rxn1": object()}
        # optInputs.get("GSA_Prefilter") or {} in run_script.py can also
        # produce an explicit {} from a None value -- exercise both shapes
        # apply_gsa_prefilter itself might see.
        assert apply_gsa_prefilter(unsrt, None) is unsrt

    def test_enabled_but_explicitly_disabled_is_identity(self):
        from rmip.simulation.gsa_reaction_prefilter import apply_gsa_prefilter

        unsrt = {"rxn1": object()}
        cfg = {"enabled": False, "gsa_result_json": "/nonexistent/path.json"}
        # Must short-circuit BEFORE ever opening gsa_result_json.
        assert apply_gsa_prefilter(unsrt, cfg) is unsrt


class TestApplyGsaPrefilterEnabled:
    def _write_gsa_json(self, tmp_path):
        from rmip.types import SobolResult

        names = ["R1:rxnA", "R2:rxnB", "R3:rxnC"]
        result = SobolResult(
            mechanism_id=1, method="sobol", parameter_names=names, n_samples=64,
            S1={names[0]: 0.9, names[1]: 0.02, names[2]: 0.001},
            S1_conf={n: 0.001 for n in names},
            ST={names[0]: 0.91, names[1]: 0.03, names[2]: 0.002},
            ST_conf={n: 0.001 for n in names},
        )
        path = tmp_path / "sobol_1_gsa.json"
        result.save(str(path))
        return path

    def test_enabled_filters_correctly_threshold(self, tmp_path):
        from rmip.simulation.gsa_reaction_prefilter import apply_gsa_prefilter

        gsa_path = self._write_gsa_json(tmp_path)
        unsrt = {"rxnA": "A", "rxnB": "B", "rxnC": "C"}
        cfg = {"gsa_result_json": str(gsa_path), "metric": "ST", "threshold": 0.05}
        filtered = apply_gsa_prefilter(unsrt, cfg)
        assert set(filtered.keys()) == {"rxnA"}
        assert filtered is not unsrt

    def test_enabled_filters_correctly_top_n(self, tmp_path):
        from rmip.simulation.gsa_reaction_prefilter import apply_gsa_prefilter

        gsa_path = self._write_gsa_json(tmp_path)
        unsrt = {"rxnA": "A", "rxnB": "B", "rxnC": "C"}
        cfg = {"gsa_result_json": str(gsa_path), "metric": "S1", "top_n": 2}
        filtered = apply_gsa_prefilter(unsrt, cfg)
        assert set(filtered.keys()) == {"rxnA", "rxnB"}

    def test_key_mismatch_raises_value_error_not_empty_dict(self, tmp_path):
        from rmip.simulation.gsa_reaction_prefilter import apply_gsa_prefilter

        gsa_path = self._write_gsa_json(tmp_path)
        unsrt = {"totally different reaction key": "X"}
        cfg = {"gsa_result_json": str(gsa_path), "metric": "S1", "threshold": 0.01}
        with pytest.raises(ValueError, match="zero overlap"):
            apply_gsa_prefilter(unsrt, cfg)

    def test_threshold_too_high_returns_empty_not_raise(self, tmp_path):
        """A legitimate (if unusual) all-excluded outcome -- overlap exists,
        just nothing clears the bar -- is NOT a format mismatch and must not
        raise; see apply_gsa_prefilter's docstring for the distinction."""
        from rmip.simulation.gsa_reaction_prefilter import apply_gsa_prefilter

        gsa_path = self._write_gsa_json(tmp_path)
        unsrt = {"rxnA": "A", "rxnB": "B", "rxnC": "C"}
        cfg = {"gsa_result_json": str(gsa_path), "metric": "S1", "threshold": 0.999}
        filtered = apply_gsa_prefilter(unsrt, cfg)
        assert filtered == {}


class TestSelectGsaImportantReactions:
    def _write(self, tmp_path, values, metric="S1"):
        from rmip.types import SobolResult

        names = list(values.keys())
        kwargs = {metric: dict(values)}
        result = SobolResult(
            mechanism_id=1, method="sobol" if metric != "mu_star" else "morris",
            parameter_names=names, n_samples=8, **kwargs,
        )
        path = tmp_path / "gsa.json"
        result.save(str(path))
        return path

    def test_both_threshold_and_top_n_raises(self, tmp_path):
        from rmip.simulation.gsa_reaction_prefilter import select_gsa_important_reactions

        path = self._write(tmp_path, {"R1:a": 0.5, "R2:b": 0.1})
        with pytest.raises(ValueError, match="only one of"):
            select_gsa_important_reactions(str(path), metric="S1", threshold=0.2, top_n=1)

    def test_top_n_alone_does_not_trip_mutual_exclusion(self, tmp_path):
        """Regression guard: threshold's default must be None, not the
        default 0.01 value, or passing top_n alone would spuriously raise."""
        from rmip.simulation.gsa_reaction_prefilter import select_gsa_important_reactions

        path = self._write(tmp_path, {"R1:a": 0.5, "R2:b": 0.1, "R3:c": 0.05})
        result = select_gsa_important_reactions(str(path), metric="S1", top_n=1)
        assert result == {"a"}

    def test_metric_variants_S1_ST_mu_star(self, tmp_path):
        from rmip.simulation.gsa_reaction_prefilter import select_gsa_important_reactions

        path_s1 = self._write(tmp_path, {"R1:a": 0.8, "R2:b": 0.01}, metric="S1")
        assert select_gsa_important_reactions(str(path_s1), metric="S1", threshold=0.1) == {"a"}

        path_st = self._write(tmp_path, {"R1:a": 0.8, "R2:b": 0.01}, metric="ST")
        assert select_gsa_important_reactions(str(path_st), metric="ST", threshold=0.1) == {"a"}

        path_mu = self._write(tmp_path, {"R1:a": 5.0, "R2:b": 0.01}, metric="mu_star")
        assert select_gsa_important_reactions(str(path_mu), metric="mu_star", threshold=1.0) == {"a"}

    def test_wrong_metric_for_method_raises(self, tmp_path):
        """A Morris result has no S1/ST populated -- asking for S1 must fail
        loudly, not return an empty set silently."""
        from rmip.simulation.gsa_reaction_prefilter import select_gsa_important_reactions

        path = self._write(tmp_path, {"R1:a": 5.0}, metric="mu_star")
        with pytest.raises(ValueError, match="no 'S1' values|no S1 values"):
            select_gsa_important_reactions(str(path), metric="S1", threshold=0.1)

    def test_invalid_metric_name_raises(self, tmp_path):
        from rmip.simulation.gsa_reaction_prefilter import select_gsa_important_reactions

        path = self._write(tmp_path, {"R1:a": 0.5})
        with pytest.raises(ValueError, match="metric must be one of"):
            select_gsa_important_reactions(str(path), metric="bogus", threshold=0.1)

    def test_label_prefix_stripped(self, tmp_path):
        from rmip.simulation.gsa_reaction_prefilter import select_gsa_important_reactions

        path = self._write(tmp_path, {"R12:H + O2 <=> O + OH": 0.9})
        assert select_gsa_important_reactions(str(path), metric="S1", threshold=0.1) == {
            "H + O2 <=> O + OH"
        }


@_needs_cantera
class TestBuildReactionIdMaps:
    def test_db_mode_vs_file_mode_identical(self, tmp_path):
        """DB-mode and file-mode must agree for a mechanism whose DB rows
        were inserted in Solution.reactions() file order starting at id=1
        (see gsa_reaction_prefilter.py's module docstring for why this
        equivalence isn't guaranteed in general -- DB ids are a global
        autoincrement across every mechanism in a real database)."""
        import cantera as ct
        from rmip.simulation.gsa_reaction_prefilter import build_reaction_id_maps

        if not _GUI_STRESS_YAML.exists():
            pytest.skip("gui_stress mech_std.yaml fixture not present")

        sol = ct.Solution(str(_GUI_STRESS_YAML))
        db_path = tmp_path / "tiny_reactions.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE reactions (id INTEGER PRIMARY KEY, "
            "mechanism_id INTEGER, formula_string TEXT)"
        )
        for i, rxn in enumerate(sol.reactions()):
            conn.execute(
                "INSERT INTO reactions (id, mechanism_id, formula_string) VALUES (?,?,?)",
                (i + 1, 1, rxn.equation),
            )
        conn.commit()
        conn.close()

        db_maps = build_reaction_id_maps(db_path=str(db_path), mechanism_id=1)
        file_maps = build_reaction_id_maps(mechanism_path=str(_GUI_STRESS_YAML))

        assert db_maps.id_of_formula == file_maps.id_of_formula
        assert db_maps.formula_of_id == file_maps.formula_of_id
        assert len(db_maps.id_of_formula) == sol.n_reactions

    def test_neither_arg_combo_raises(self):
        from rmip.simulation.gsa_reaction_prefilter import build_reaction_id_maps

        with pytest.raises(ValueError):
            build_reaction_id_maps()

    def test_both_arg_combos_raises(self, tmp_path):
        from rmip.simulation.gsa_reaction_prefilter import build_reaction_id_maps

        db_path = tmp_path / "x.db"
        sqlite3.connect(str(db_path)).close()
        with pytest.raises(ValueError):
            build_reaction_id_maps(
                db_path=str(db_path), mechanism_id=1, mechanism_path=str(_GUI_STRESS_YAML),
            )

    def test_partial_db_args_raises(self, tmp_path):
        from rmip.simulation.gsa_reaction_prefilter import build_reaction_id_maps

        db_path = tmp_path / "x.db"
        sqlite3.connect(str(db_path)).close()
        with pytest.raises(ValueError):
            build_reaction_id_maps(db_path=str(db_path))
        with pytest.raises(ValueError):
            build_reaction_id_maps(mechanism_id=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. run_script.py's wiring: the extracted function must be a provable no-op
#    when GSA_Prefilter is absent, and must never trigger the rmip import in
#    that case.
# ─────────────────────────────────────────────────────────────────────────────

class TestRunScriptGsaPrefilterHook:
    def test_function_is_a_top_level_def_in_run_script(self):
        assert _RUN_SCRIPT_PATH.exists()
        fn = _extract_top_level_function(_RUN_SCRIPT_PATH, "_maybe_apply_gsa_prefilter")
        assert callable(fn)

    def test_noop_when_absent_returns_identity(self):
        fn = _extract_top_level_function(_RUN_SCRIPT_PATH, "_maybe_apply_gsa_prefilter")
        unsrt = {"H + O2 <=> O + OH": object()}
        out = fn({}, unsrt)
        assert out is unsrt

    def test_noop_when_absent_import_never_triggered(self):
        """Forcibly clear any cached import of the prefilter module first
        (an EARLIER test in this same session may have legitimately
        imported it directly), then confirm calling the extracted function
        with an empty/absent GSA_Prefilter config does NOT re-trigger that
        import -- i.e. the early `if not gsa_cfg: return unsrt_data` really
        does run before the `from rmip.simulation.gsa_reaction_prefilter
        import apply_gsa_prefilter` line, not just look like it does."""
        fn = _extract_top_level_function(_RUN_SCRIPT_PATH, "_maybe_apply_gsa_prefilter")

        mod_name = "rmip.simulation.gsa_reaction_prefilter"
        saved = sys.modules.pop(mod_name, None)
        try:
            unsrt = {"rxn": object()}
            fn({"GSA_Prefilter": None}, unsrt)
            fn({}, unsrt)
            assert mod_name not in sys.modules
        finally:
            if saved is not None:
                sys.modules[mod_name] = saved

    def test_no_target_opt_in_repo_sets_gsa_prefilter(self):
        """Every existing target.opt-shaped config in this repo has zero
        `GSA_Prefilter` keys -- proves the wiring is a no-op for every
        config that exists today, not just synthetic ones in this file."""
        hits = []
        for pattern in ("*.opt", "*target*.yaml", "*target*.yml"):
            for path in _REPO_ROOT.rglob(pattern):
                if "version_history" in path.parts:
                    continue
                try:
                    text = path.read_text(errors="ignore")
                except OSError:
                    continue
                if "GSA_Prefilter" in text:
                    hits.append(str(path))
        assert hits == [], f"unexpected GSA_Prefilter key(s) found in: {hits}"


# ─────────────────────────────────────────────────────────────────────────────
# 3/4. DesignMatrix-level benchmark: real build_unsrt_subset() + real
#      DesignMatrix, against the SAME real 3-reaction fixture and GRI-3.0
#      mechanism gsa_prs_crosscheck.py/test_gsa_prs_crosscheck.py validated.
# ─────────────────────────────────────────────────────────────────────────────

@_needs_muqsac
@_needs_project_db
@_needs_zerork_export
@_needs_ck2yaml
class TestDesignMatrixBenchmark:
    @pytest.fixture(scope="class")
    def real_unsrt_subset(self, tmp_path_factory):
        """Real build_unsrt_subset() output: 3 reactions, keyed by bare
        Cantera equation strings, against a real GRI-3.0 (mechanism_id=8)
        Cantera YAML freshly exported from outputs/project.db. No
        FlameMaster/samap/jpdap anywhere in this call path -- pure DB
        export + ck2yaml + Uncertainty.py XML parsing."""
        import rmip.simulation.gsa_prs_crosscheck as crosscheck

        with sqlite3.connect(str(_PROJECT_DB)) as conn:
            has_mech8 = conn.execute(
                "SELECT 1 FROM reactions WHERE mechanism_id=8 LIMIT 1"
            ).fetchone()
        if not has_mech8:
            pytest.skip("mechanism_id=8 (GRI-3.0) not present in outputs/project.db")

        work = tmp_path_factory.mktemp("gsa_prefilter_dm_bench")
        nominal_yaml = crosscheck.prepare_nominal_cantera_yaml(
            str(_PROJECT_DB), 8, str(work / "nominal"), ck2yaml_exe=_CK2YAML,
        )
        unsrt = crosscheck.build_unsrt_subset(str(_FIXTURE_XML), str(nominal_yaml))
        assert set(unsrt.keys()) == set(_REAL_S1.keys())
        return unsrt

    def _build_dm(self, unsrt_subset, work_dir, n_design=16, seed=42):
        import DesignMatrix as DM

        import os
        cwd0 = os.getcwd()
        os.chdir(work_dir)
        try:
            dm = DM.DesignMatrix(
                unsrt_subset, "A-facto", n_design, ind=len(unsrt_subset),
                method="sobol", seed=seed,
            )
            samples = dm.getSamples()
        finally:
            os.chdir(cwd0)
        return dm, samples

    # ── Test 3: baseline unchanged when prefilter absent ──────────────────
    def test_baseline_unchanged_when_prefilter_absent(self, real_unsrt_subset, tmp_path):
        """Confirms _maybe_apply_gsa_prefilter's no-op path feeds
        DesignMatrix an UNCHANGED dict, and that DesignMatrix's own
        behaviour (n_reactions_in_design, retained reaction-key set, and --
        confirmed empirically deterministic below -- the actual sampled
        matrix) is exactly what you get calling DesignMatrix directly with
        zero prefilter code ever imported.

        Determinism check: DesignMatrix(..., method="sobol", seed=42) was
        empirically verified (see this task's dry run) to reproduce a
        bit-identical sample matrix across two independent instantiations
        with the same seed -- scipy's qmc.Sobol(seed=int) is deterministic
        for a fixed integer seed. This test re-verifies that here rather
        than assuming it, and asserts full numeric equality since it holds.
        """
        fn = _extract_top_level_function(_RUN_SCRIPT_PATH, "_maybe_apply_gsa_prefilter")
        passed_through = fn({}, real_unsrt_subset)
        assert passed_through is real_unsrt_subset

        work_a = tmp_path / "a"
        work_b = tmp_path / "b"
        work_a.mkdir()
        work_b.mkdir()

        dm_a, X_a = self._build_dm(passed_through, work_a)
        dm_b, X_b = self._build_dm(real_unsrt_subset, work_b)

        assert dm_a.n_reactions_in_design == 3
        assert dm_b.n_reactions_in_design == 3
        assert set(dm_a.unsrt.keys()) == set(real_unsrt_subset.keys())
        assert np.array_equal(X_a, X_b), (
            "DesignMatrix output diverged between the no-op-prefiltered path "
            "and the direct (never-imported-prefilter-code) path -- the "
            "no-op guarantee is broken."
        )

    # ── Test 4: prefiltered restricts reactions sanely ────────────────────
    def test_prefiltered_restricts_reactions_sanely(self, real_unsrt_subset, tmp_path):
        """Uses the REAL S1 values from ZERORK_GSA_PLAN.md's Phase 4
        cross-check (module docstring _REAL_S1) with a threshold that
        cleanly separates the dominant reaction (~0.999) from the two
        near-zero ones (~0.0002-0.0006): 0.01 sits squarely between them.

        NOTE on what is and isn't asserted about the surviving reaction's
        OWN sampled values: DesignMatrix's joint low-discrepancy sampling
        (qmc_sampling.joint_uniform_block, method='sobol') generates ONE
        joint (n_samples, n_reactions) block per call -- every reaction's
        column is drawn from the SAME shared Sobol engine, so a reaction's
        column literally differs depending on how many OTHER reactions are
        sampled alongside it (confirmed empirically before writing this
        test: column 0 of a 3-dim joint Sobol block is NOT the same
        sequence as column 0 of a 1-dim block, even with an identical
        seed). So this test does NOT assert the filtered DM's one column
        equals the unfiltered DM's corresponding column -- that would be a
        false claim about this architecture. What IS verified: (a) the
        prefilter reduces the reaction/column count exactly as expected
        (1 vs 3); (b) the correct (dominant) reaction is the one retained;
        (c) the filtered DesignMatrix's own shape is well-formed and in
        [-1, 1]; (d) the filtered path is itself internally deterministic
        (same seed -> bit-identical output), proving the prefilter doesn't
        introduce nondeterminism even though its numeric values necessarily
        differ from the unfiltered run.
        """
        from rmip.simulation.gsa_reaction_prefilter import apply_gsa_prefilter
        from rmip.types import SobolResult

        names = [f"R{i+1}:{rxn}" for i, rxn in enumerate(real_unsrt_subset.keys())]
        s1_by_name = {names[i]: _REAL_S1[rxn] for i, rxn in enumerate(real_unsrt_subset.keys())}
        st_by_name = {names[i]: _REAL_ST[rxn] for i, rxn in enumerate(real_unsrt_subset.keys())}
        result = SobolResult(
            mechanism_id=8, method="sobol", parameter_names=names, n_samples=512,
            S1=s1_by_name, ST=st_by_name,
        )
        gsa_json_path = tmp_path / "sobol_8_gsa.json"
        result.save(str(gsa_json_path))

        cfg = {"gsa_result_json": str(gsa_json_path), "metric": "S1", "threshold": 0.01}
        filtered = apply_gsa_prefilter(real_unsrt_subset, cfg)
        assert set(filtered.keys()) == {_DOMINANT_RXN}
        assert len(filtered) == 1

        work_filtered = tmp_path / "filtered"
        work_filtered.mkdir()
        dm_f1, X_f1 = self._build_dm(filtered, work_filtered)
        assert dm_f1.n_reactions_in_design == 1
        assert X_f1.shape[1] == 1
        assert X_f1.min() >= -1.0 - 1e-9
        assert X_f1.max() <= 1.0 + 1e-9

        work_filtered2 = tmp_path / "filtered2"
        work_filtered2.mkdir()
        dm_f2, X_f2 = self._build_dm(filtered, work_filtered2)
        assert np.array_equal(X_f1, X_f2), "filtered DesignMatrix path is not deterministic"

        work_unfiltered = tmp_path / "unfiltered"
        work_unfiltered.mkdir()
        dm_u, X_u = self._build_dm(real_unsrt_subset, work_unfiltered)
        assert dm_u.n_reactions_in_design == 3
        assert X_u.shape[1] == 3
