"""
rmip/tests/test_phase6.py
===========================
Phase-6 test suite — Future.md Part 2, features 1-7 (stdlib + pytest only;
cantera/plotly/PyChemkin-heavy paths are import-skipped when unavailable).

Coverage
--------
- types.py Phase-6 dataclasses (to_dict / classify)
- schema_phase6 + migrations: table creation, idempotency, sim_type CHECK
  widened to JSR/PFR/FLUX, simulation_runs.extra_json, exp_data_points.temp_target_class
- ReverseRateScanner: hand-computed Kc=1 case (k_rev == k_fwd exactly),
  collision-limit flagging, non-bimolecular-reverse skip
- QualityScorer / MechanismDiffEngine / MechanismReport: component scores,
  species/reaction diff counts, HTML export (plotly-optional)
- PlogSurfaceBuilder: grid shape, non-monotonic detection, ill-defined high-P flag
- NTCDetector: onset/end detection on a synthetic S-curve, HTT/LTT split,
  classify_temperature, split_targets_by_ntc
- benchmarking.MechanismBenchmarker / performance_ranker.PerformanceRanker:
  HTT/LTT regime substitution when an ntc_runs row exists, fixed-threshold
  fallback otherwise
- muqsac_bridge.MuqsacBridge: .target line format (parsed back through the
  real combustion_target_class when its heavy deps are available), .add /
  target.opt derivation, mocked subprocess.run, output parsing from fixture
  JSON, DB storage
- Deck generation (src/MUQ-SAC/make_input_file.py): JSR/PFR present for all
  3 solvers x flux on/off; generated cantera/CHEMKIN_PRO scripts are valid
  Python; FlameMaster decks toggle ReactionFluxAnal correctly
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

_MUQSAC_SRC = str(Path(__file__).resolve().parents[3] / "src" / "MUQ-SAC")

# ---------------------------------------------------------------------------
# Helper: in-memory DB fixtures (mirrors test_phase4.py's pattern)
# ---------------------------------------------------------------------------

def _make_db(path: str) -> sqlite3.Connection:
    from rmip.database.schema import init_db
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.database.migrations import run_phase3_migrations, run_phase6_migrations
    conn = init_db(path)
    upgrade_db(conn)
    run_phase3_migrations(conn)
    run_phase6_migrations(conn)
    return conn


def _insert_mech(conn, name="M", fmt="chemkin") -> int:
    from rmip.database.schema import insert_mechanism
    mid = insert_mechanism(conn, name=name, format=fmt)
    conn.commit()
    return mid


def _insert_species(conn, mech_id, name, inchi_key=None, formula="H2",
                     tmin=200.0, tmid=1000.0, tmax=5000.0,
                     coeffs_low=None, coeffs_high=None, h298_kJ_mol=None) -> int:
    coeffs_low = coeffs_low or [3.5, 0, 0, 0, 0, -1000, 3.5]
    coeffs_high = coeffs_high or coeffs_low
    from rmip.database.schema import insert_species
    sid = insert_species(
        conn, mechanism_id=mech_id, name=name, inchi_key=inchi_key,
        formula=formula, thermo_type="nasa7", tmin_K=tmin, tmid_K=tmid, tmax_K=tmax,
        coeffs_low_json=json.dumps(coeffs_low), coeffs_high_json=json.dumps(coeffs_high),
        h298_kJ_mol=h298_kJ_mol,
    )
    conn.commit()
    return sid


def _insert_reaction(conn, mech_id, formula, canonical_key=None,
                      A_si=1e13, n=0.0, Ea_si=100_000.0,
                      kinetics_type="arrhenius", reversible=1,
                      kinetics_data_json=None) -> int:
    from rmip.database.schema import insert_reaction
    rid = insert_reaction(
        conn, mechanism_id=mech_id, formula_string=formula,
        canonical_key=canonical_key or formula,
        A_si=A_si, A_orig=A_si, A_units="SI", n=n, Ea_si=Ea_si, Ea_orig=Ea_si,
        Ea_units="J/mol", kinetics_type=kinetics_type, reversible=reversible,
        kinetics_data_json=kinetics_data_json,
    )
    conn.commit()
    return rid


def _add_stoich(conn, rxn_id, sp_id, coeff, role):
    from rmip.database.schema import insert_stoichiometry
    insert_stoichiometry(conn, rxn_id, sp_id, stoich_coeff=coeff, role=role)
    conn.commit()


# ===========================================================================
# 1. Phase-6 dataclasses
# ===========================================================================

class TestPhase6Dataclasses:
    def test_jsr_result_to_dict(self):
        from rmip.types import JSRResult, SimCondition
        cond = SimCondition(T_K=1200, P_atm=10, phi=1.0, fuel="CH4:1.0")
        r = JSRResult(condition=cond, mechanism_id=1, solver="cantera",
                      residence_time_s=0.1, species_mole_fractions={"CH4": 0.1},
                      converged=True)
        d = r.to_dict()
        assert d["mechanism_id"] == 1
        assert d["solver"] == "cantera"
        assert d["species_mole_fractions"] == {"CH4": 0.1}

    def test_pfr_result_to_dict(self):
        from rmip.types import PFRResult, SimCondition
        cond = SimCondition(T_K=1200, P_atm=10, phi=1.0, fuel="CH4:1.0")
        r = PFRResult(condition=cond, mechanism_id=1, solver="cantera",
                      length_m=0.1, residence_time_s=0.05,
                      species_mole_fractions={"CH4": 0.05}, converged=True)
        d = r.to_dict()
        assert d["length_m"] == 0.1
        assert d["has_axial_profile"] is False

    def test_flux_result_to_dict(self):
        from rmip.types import FluxResult, SimCondition
        cond = SimCondition(T_K=1200, P_atm=1, phi=1.0, fuel="CH4:1.0")
        r = FluxResult(mechanism_id=1, condition=cond, solver="cantera", element="C",
                        rop_by_reaction={"A<=>B": 1.0}, element_flux=[{"source": "A"}],
                        converged=True)
        d = r.to_dict()
        assert d["n_reactions"] == 1
        assert d["n_flux_edges"] == 1

    def test_ntc_report_classify(self):
        from rmip.types import NTCReport
        rep = NTCReport(mechanism_id=1, fuel="NC7H16:1.0", phi=1.0, P_atm=13.5,
                         T_grid_K=[700, 900, 1100], idt_s=[0.01, 0.02, 0.005],
                         has_ntc=True, T_NTC_onset=800.0, T_NTC_end=1000.0,
                         IDT_ratio_NTC=2.0)
        assert rep.classify(700.0) == "LTT"
        assert rep.classify(900.0) == "HTT"

    def test_ntc_report_classify_none_when_no_onset(self):
        from rmip.types import NTCReport
        rep = NTCReport(mechanism_id=1, fuel="X", phi=1.0, P_atm=1.0,
                         T_grid_K=[700], idt_s=[0.1], has_ntc=False,
                         T_NTC_onset=None, T_NTC_end=None, IDT_ratio_NTC=None)
        assert rep.classify(700.0) is None

    def test_reverse_rate_flag_to_dict(self):
        from rmip.types import ReverseRateFlag
        f = ReverseRateFlag(canonical_key="k", formula_string="A<=>B", mechanism_id=1,
                             T_K=1000, k_fwd=1.0, k_rev=1e10, collision_limit=1e8,
                             exceedance_factor=100.0, flagged=True)
        assert f.to_dict()["flagged"] is True


# ===========================================================================
# 2. schema_phase6 + migrations
# ===========================================================================

class TestPhase6SchemaMigrations:
    def test_tables_created(self):
        with tempfile.TemporaryDirectory() as td:
            conn = _make_db(str(Path(td) / "p6.db"))
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            for t in ("ntc_runs", "reverse_rate_flags",
                      "mechanism_quality_scores", "flux_runs"):
                assert t in tables

    def test_idempotent(self):
        from rmip.database.migrations import run_phase6_migrations
        with tempfile.TemporaryDirectory() as td:
            conn = _make_db(str(Path(td) / "p6.db"))
            run_phase6_migrations(conn)
            run_phase6_migrations(conn)  # no error

    def test_sim_type_check_widened(self):
        with tempfile.TemporaryDirectory() as td:
            conn = _make_db(str(Path(td) / "p6.db"))
            mid = _insert_mech(conn)
            for st in ("IDT", "LFS", "JSR", "PFR", "FLUX"):
                conn.execute(
                    "INSERT INTO simulation_runs (mechanism_id, sim_type, T_K, P_atm, "
                    "phi, fuel_name, fuel_composition, oxidizer_composition, converged, "
                    "created_at) VALUES (?,?,0,0,0,'x','x','x',1,'now')",
                    (mid, st),
                )
            conn.commit()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO simulation_runs (mechanism_id, sim_type, T_K, P_atm, "
                    "phi, fuel_name, fuel_composition, oxidizer_composition, converged, "
                    "created_at) VALUES (?,?,0,0,0,'x','x','x',1,'now')",
                    (mid, "NOT_A_TYPE"),
                )

    def test_extra_json_column_and_widen_preserves_rows(self):
        """Regression test: widening sim_type CHECK must not drop existing rows
        or choke on a table that already has extra_json (caught during dev —
        executescript's INSERT SELECT * mismatched column counts)."""
        from rmip.database.migrations import run_phase6_migrations
        with tempfile.TemporaryDirectory() as td:
            conn = _make_db(str(Path(td) / "p6.db"))
            mid = _insert_mech(conn)
            from rmip.database.schema_phase3 import insert_simulation_run
            insert_simulation_run(
                conn, mechanism_id=mid, sim_type="IDT", T_K=1000, P_atm=1, phi=1.0,
                fuel_name="H2", fuel_composition="H2:1", oxidizer_composition="O2:1",
                result=0.01, converged=True, extra_json=json.dumps({"a": 1}),
            )
            # Re-running phase6 migrations must be a no-op that preserves the row.
            run_phase6_migrations(conn)
            run_phase6_migrations(conn)
            rows = conn.execute("SELECT sim_type, extra_json FROM simulation_runs").fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "IDT"
            assert json.loads(rows[0][1]) == {"a": 1}

    def test_temp_target_class_column(self):
        with tempfile.TemporaryDirectory() as td:
            conn = _make_db(str(Path(td) / "p6.db"))
            cols = {r[1] for r in conn.execute("PRAGMA table_info(exp_data_points)")}
            assert "temp_target_class" in cols

    def test_ntc_run_crud(self):
        from rmip.database.schema_phase6 import insert_ntc_run, get_latest_ntc_run
        with tempfile.TemporaryDirectory() as td:
            conn = _make_db(str(Path(td) / "p6.db"))
            mid = _insert_mech(conn)
            insert_ntc_run(conn, mid, "NC7H16:1.0", 1.0, 13.5, [700, 900], [0.05, 0.02],
                            True, 800.0, 900.0, 1.5, 1, 1)
            run = get_latest_ntc_run(conn, mid)
            assert run["T_NTC_onset"] == 800.0
            assert get_latest_ntc_run(conn, 999) is None


# ===========================================================================
# 3. ReverseRateScanner
# ===========================================================================

class TestReverseRateScanner:
    def _setup_kc_one(self, conn, mech_id, A_si=1e10):
        """2A<=>2B with IDENTICAL thermo for A and B => Kc=1 for all T,
        so k_rev must equal k_fwd exactly."""
        coeffs = [3.5, 0.001, 0.0, 0.0, 0.0, -1000.0, 5.0]
        a = _insert_species(conn, mech_id, "A", inchi_key="INCHI_A", coeffs_low=coeffs)
        b = _insert_species(conn, mech_id, "B", inchi_key="INCHI_B", coeffs_low=coeffs)
        rxn = _insert_reaction(conn, mech_id, "2A<=>2B", A_si=A_si, n=0.0, Ea_si=50_000.0)
        _add_stoich(conn, rxn, a, 2.0, "reactant")
        _add_stoich(conn, rxn, b, 2.0, "product")
        return rxn

    def test_k_rev_equals_k_fwd_when_kc_is_one(self):
        from rmip.analysis.reverse_rate_scanner import ReverseRateScanner
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            self._setup_kc_one(conn, mid, A_si=1e10)

            scanner = ReverseRateScanner(db, T_grid_K=[600.0, 1200.0, 2000.0])
            flags = scanner.scan(mid)
            assert len(flags) == 3
            for f in flags:
                assert f.k_rev == pytest.approx(f.k_fwd, rel=1e-9)

    def test_collision_limit_flagging(self):
        from rmip.analysis.reverse_rate_scanner import ReverseRateScanner
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            self._setup_kc_one(conn, mid, A_si=1e25)  # absurdly large -> flagged
            scanner = ReverseRateScanner(db, T_grid_K=[1000.0])
            flags = scanner.scan(mid)
            assert flags[0].flagged is True
            assert flags[0].exceedance_factor > 1000.0

    def test_small_rate_not_flagged(self):
        from rmip.analysis.reverse_rate_scanner import ReverseRateScanner
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            self._setup_kc_one(conn, mid, A_si=1e2)
            scanner = ReverseRateScanner(db, T_grid_K=[1000.0])
            flags = scanner.scan(mid)
            assert flags[0].flagged is False

    def test_non_bimolecular_reverse_skipped(self):
        """A<=>B (unimolecular reverse, product order=1) must not be scanned."""
        from rmip.analysis.reverse_rate_scanner import ReverseRateScanner
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            coeffs = [3.5, 0, 0, 0, 0, -1000.0, 5.0]
            a = _insert_species(conn, mid, "A", inchi_key="IA", coeffs_low=coeffs)
            b = _insert_species(conn, mid, "B", inchi_key="IB", coeffs_low=coeffs)
            rxn = _insert_reaction(conn, mid, "A<=>B", A_si=1e13, Ea_si=50000.0)
            _add_stoich(conn, rxn, a, 1.0, "reactant")
            _add_stoich(conn, rxn, b, 1.0, "product")

            scanner = ReverseRateScanner(db, T_grid_K=[1000.0])
            flags = scanner.scan(mid)
            assert flags == []

    def test_store_and_retrieve(self):
        from rmip.analysis.reverse_rate_scanner import ReverseRateScanner
        from rmip.database.schema_phase6 import get_reverse_rate_flags
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            self._setup_kc_one(conn, mid, A_si=1e25)
            scanner = ReverseRateScanner(db, T_grid_K=[1000.0])
            flags = scanner.scan(mid)
            scanner.store(mid, flags)
            stored = get_reverse_rate_flags(conn, mid, flagged_only=True)
            assert len(stored) == 1


# ===========================================================================
# 4. mechanism_report (QualityScorer + MechanismDiffEngine + MechanismReport)
# ===========================================================================

class TestMechanismReport:
    def _two_mechs_with_overlap(self, conn):
        m1 = _insert_mech(conn, "M1")
        m2 = _insert_mech(conn, "M2")
        coeffs = [3.5, 0, 0, 0, 0, -1000.0, 5.0]
        # Shared species (same inchi_key in both mechs)
        h2_1 = _insert_species(conn, m1, "H2", inchi_key="INCHI_H2", coeffs_low=coeffs)
        h2_2 = _insert_species(conn, m2, "H2", inchi_key="INCHI_H2", coeffs_low=coeffs)
        # Unique species
        _insert_species(conn, m1, "CO", inchi_key="INCHI_CO", coeffs_low=coeffs)
        _insert_species(conn, m2, "CH4", inchi_key="INCHI_CH4", coeffs_low=coeffs)
        # Shared reaction with different A (divergent) -> log-A ratio testable
        r1 = _insert_reaction(conn, m1, "H2+O2<=>H2O2", canonical_key="RXN_SHARED", A_si=1e10)
        r2 = _insert_reaction(conn, m2, "H2+O2<=>H2O2", canonical_key="RXN_SHARED", A_si=1e12)
        # Unique reaction to m1
        _insert_reaction(conn, m1, "CO+OH<=>CO2+H", canonical_key="RXN_UNIQUE_M1")
        return m1, m2

    def test_quality_score_single_mechanism(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            coeffs = [3.5, 0, 0, 0, 0, -1000.0, 5.0]
            _insert_species(conn, mid, "H2", inchi_key="I1", coeffs_low=coeffs)
            _insert_species(conn, mid, "O2", inchi_key=None, coeffs_low=coeffs)  # unresolved

            from rmip.analysis.mechanism_report import QualityScorer
            scorer = QualityScorer(db)
            score = scorer.score(mid)
            assert score.smiles_resolution_pct == pytest.approx(50.0)
            assert 0.0 <= score.composite_score <= 100.0
            assert score.hf_agreement_score is None  # no ATcT ref supplied
            assert score.ntc_coverage_score is None  # no NTC run supplied

    def test_diff_added_removed_shared(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            m1, m2 = self._two_mechs_with_overlap(conn)

            from rmip.analysis.mechanism_report import MechanismDiffEngine
            diff = MechanismDiffEngine(db).diff(m1, m2)
            assert diff.n_species_shared == 1
            assert "INCHI_CH4" in diff.species_added
            assert "INCHI_CO" in diff.species_removed
            assert "RXN_UNIQUE_M1" in diff.reactions_removed
            assert diff.n_reactions_shared == 1
            assert "RXN_SHARED" in diff.shared_reaction_log_a_ratio
            # log10(1e12 / 1e10) = 2
            assert diff.shared_reaction_log_a_ratio["RXN_SHARED"] == pytest.approx(2.0)

    def test_report_single_vs_pair(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            m1, m2 = self._two_mechs_with_overlap(conn)

            from rmip.analysis.mechanism_report import MechanismReport
            rep = MechanismReport(db)
            single = rep.report([m1])
            assert single["diff"] is None
            assert len(single["quality"]) == 1

            paired = rep.report([m1, m2])
            assert paired["diff"] is not None
            assert len(paired["quality"]) == 2

    def test_report_rejects_bad_count(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            m1, m2 = self._two_mechs_with_overlap(conn)
            m3 = _insert_mech(conn, "M3")
            from rmip.analysis.mechanism_report import MechanismReport
            rep = MechanismReport(db)
            with pytest.raises(ValueError):
                rep.report([m1, m2, m3])

    def test_export_html(self):
        pytest.importorskip("plotly")
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            m1, m2 = self._two_mechs_with_overlap(conn)
            from rmip.analysis.mechanism_report import MechanismReport
            rep = MechanismReport(db)
            result = rep.report([m1, m2])
            out = str(Path(td) / "report.html")
            rep.export_html(result, out)
            assert Path(out).exists()
            assert Path(out).stat().st_size > 0


# ===========================================================================
# 5. PlogSurfaceBuilder
# ===========================================================================

class TestPlogSurface:
    def _insert_plog(self, conn, mid):
        kdata = {
            "pressures": [1013.25, 101325.0, 10132500.0],
            "arrhenius": [
                {"A_si": 1e10, "n": 0.5, "Ea_si": 50000.0},
                {"A_si": 5e11, "n": 0.2, "Ea_si": 60000.0},
                {"A_si": 2e12, "n": -0.1, "Ea_si": 70000.0},
            ],
        }
        return _insert_reaction(conn, mid, "CH3+CH3(+M)<=>C2H6(+M)",
                                 kinetics_type="plog", kinetics_data_json=json.dumps(kdata))

    def test_build_grid_shape(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            rid = self._insert_plog(conn, mid)

            from rmip.analysis.plog_surface import PlogSurfaceBuilder
            builder = PlogSurfaceBuilder(db, T_grid_K=[600, 1200, 1800],
                                          P_grid_atm=[0.1, 1.0, 10.0])
            surf = builder.build(mid, rid)
            assert len(surf.log10_k_matrix) == 3
            assert all(len(row) == 3 for row in surf.log10_k_matrix)
            assert surf.high_p_limit_ill_defined is False

    def test_ill_defined_high_p_with_single_level(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            kdata = {"pressures": [101325.0], "arrhenius": [{"A_si": 1e10, "n": 0.0, "Ea_si": 50000.0}]}
            rid = _insert_reaction(conn, mid, "A<=>B", kinetics_type="plog",
                                    kinetics_data_json=json.dumps(kdata))
            from rmip.analysis.plog_surface import PlogSurfaceBuilder
            surf = PlogSurfaceBuilder(db).build(mid, rid)
            assert surf.high_p_limit_ill_defined is True
            assert any("high-P limit" in w for w in surf.warnings)

    def test_rejects_non_plog_reaction(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            rid = _insert_reaction(conn, mid, "A<=>B", kinetics_type="arrhenius")
            from rmip.analysis.plog_surface import PlogSurfaceBuilder
            with pytest.raises(ValueError):
                PlogSurfaceBuilder(db).build(mid, rid)

    def test_export_html(self):
        pytest.importorskip("plotly")
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            rid = self._insert_plog(conn, mid)
            from rmip.analysis.plog_surface import PlogSurfaceBuilder
            builder = PlogSurfaceBuilder(db)
            surf = builder.build(mid, rid)
            out = str(Path(td) / "plog.html")
            builder.export_html(surf, out)
            assert Path(out).stat().st_size > 0


# ===========================================================================
# 6. NTCDetector
# ===========================================================================

class TestNTCDetector:
    def test_find_ntc_on_synthetic_s_curve(self):
        from rmip.analysis.ntc_detector import NTCDetector
        det = NTCDetector(mechanism_id=1, db_path=":memory:")
        T_grid = [650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250]
        idt = [50, 30, 15, 8, 10, 14, 18, 20, 16, 9, 4, 2, 1]
        onset, end, ratio, has_ntc = det._find_ntc(T_grid, idt)
        assert has_ntc is True
        assert onset == 800
        assert end == 1000
        assert ratio == pytest.approx(20 / 8)

    def test_no_ntc_on_monotonic_curve(self):
        from rmip.analysis.ntc_detector import NTCDetector
        det = NTCDetector(mechanism_id=1, db_path=":memory:")
        T_grid = [700, 800, 900, 1000, 1100]
        idt = [0.1, 0.05, 0.02, 0.01, 0.005]
        onset, end, ratio, has_ntc = det._find_ntc(T_grid, idt)
        assert has_ntc is False
        assert onset is None

    def test_noise_does_not_trigger_false_ntc(self):
        """1% rel_tol dead-zone should reject sub-threshold wiggles."""
        from rmip.analysis.ntc_detector import NTCDetector
        det = NTCDetector(mechanism_id=1, db_path=":memory:", rel_tol=0.05)
        T_grid = [700, 800, 900, 1000]
        idt = [0.100, 0.0999, 0.05, 0.02]  # tiny uptick well under 5% tol
        onset, end, ratio, has_ntc = det._find_ntc(T_grid, idt)
        assert has_ntc is False

    def test_classify_temperature_and_split(self):
        from rmip.analysis.ntc_detector import classify_temperature, split_targets_by_ntc
        from rmip.types import NTCReport, SimCondition
        rep = NTCReport(mechanism_id=1, fuel="X", phi=1.0, P_atm=1.0,
                         T_grid_K=[], idt_s=[], has_ntc=True,
                         T_NTC_onset=850.0, T_NTC_end=950.0, IDT_ratio_NTC=1.5)
        assert classify_temperature(700, rep) == "LTT"
        assert classify_temperature(900, rep) == "HTT"

        conds = [SimCondition(T_K=t, P_atm=1, phi=1.0, fuel="X") for t in (700, 800, 900, 1000)]
        htt, ltt = split_targets_by_ntc(conds, rep)
        assert {c.T_K for c in htt} == {900, 1000}
        assert {c.T_K for c in ltt} == {700, 800}

    def test_split_targets_empty_when_no_onset(self):
        from rmip.analysis.ntc_detector import split_targets_by_ntc
        from rmip.types import NTCReport, SimCondition
        rep = NTCReport(mechanism_id=1, fuel="X", phi=1.0, P_atm=1.0,
                         T_grid_K=[], idt_s=[], has_ntc=False,
                         T_NTC_onset=None, T_NTC_end=None, IDT_ratio_NTC=None)
        conds = [SimCondition(T_K=700, P_atm=1, phi=1.0, fuel="X")]
        htt, ltt = split_targets_by_ntc(conds, rep)
        assert htt == [] and ltt == []

    def test_tag_exp_dataset(self):
        from rmip.analysis.ntc_detector import NTCDetector
        from rmip.types import NTCReport
        from rmip.database.schema_phase3 import insert_exp_dataset, insert_exp_data_points

        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            ds_id = insert_exp_dataset(conn, "ds1", "IDT")
            insert_exp_data_points(conn, ds_id, [
                {"T_K": 700, "P_atm": 1, "phi": 1.0, "value": 0.1, "units": "s"},
                {"T_K": 900, "P_atm": 1, "phi": 1.0, "value": 0.02, "units": "s"},
            ])
            rep = NTCReport(mechanism_id=mid, fuel="X", phi=1.0, P_atm=1.0,
                             T_grid_K=[], idt_s=[], has_ntc=True,
                             T_NTC_onset=800.0, T_NTC_end=850.0, IDT_ratio_NTC=1.2)
            det = NTCDetector(mid, db)
            n = det.tag_exp_dataset(ds_id, rep)
            assert n == 2
            rows = conn.execute(
                "SELECT T_K, temp_target_class FROM exp_data_points ORDER BY T_K"
            ).fetchall()
            assert rows[0][1] == "LTT"
            assert rows[1][1] == "HTT"


# ===========================================================================
# 7. benchmarking / performance_ranker: HTT/LTT regime substitution
# ===========================================================================

class TestRegimeSubstitution:
    def test_benchmarking_uses_fixed_thresholds_without_ntc_run(self):
        from rmip.analysis.benchmarking import MechanismBenchmarker
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            b = MechanismBenchmarker(db)
            regimes, regime_set = b._regimes_for_mechanism(
                conn, mid, [{"T_K": 700}, {"T_K": 900}, {"T_K": 1200}], "IDT"
            )
            from rmip.types import Regime
            assert regimes == [Regime.LTC, Regime.NTC, Regime.HTC]

    def test_benchmarking_uses_httltt_with_ntc_run(self):
        from rmip.analysis.benchmarking import MechanismBenchmarker
        from rmip.database.schema_phase6 import insert_ntc_run
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            insert_ntc_run(conn, mid, "X", 1.0, 1.0, [700, 900], [0.1, 0.02],
                            True, 800.0, 850.0, 1.2, 1, 1)
            b = MechanismBenchmarker(db)
            regimes, regime_set = b._regimes_for_mechanism(
                conn, mid, [{"T_K": 700}, {"T_K": 900}], "IDT"
            )
            from rmip.types import Regime
            assert regimes == [Regime.LTT, Regime.HTT]
            assert Regime.LTC not in regime_set

    def test_performance_ranker_uses_httltt_with_ntc_run(self):
        from rmip.simulation.performance_ranker import _regimes_and_bounds_for_mechanism
        from rmip.database.schema_phase6 import insert_ntc_run
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            insert_ntc_run(conn, mid, "X", 1.0, 1.0, [700, 900], [0.1, 0.02],
                            True, 800.0, 850.0, 1.2, 1, 1)
            regimes, bounds = _regimes_and_bounds_for_mechanism(conn, mid, "IDT")
            assert regimes == ["HTT", "LTT", "ALL"]
            assert bounds["HTT"] == (800.0, 1e9)
            assert bounds["LTT"] == (0.0, 800.0)

    def test_performance_ranker_fixed_thresholds_without_ntc_run(self):
        from rmip.simulation.performance_ranker import _regimes_and_bounds_for_mechanism
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            regimes, bounds = _regimes_and_bounds_for_mechanism(conn, mid, "IDT")
            assert regimes == ["LTC", "NTC", "HTC", "ALL"]


# ===========================================================================
# 8. muqsac_bridge
# ===========================================================================

class TestMuqsacBridge:
    def test_build_target_files_structure(self):
        pytest.importorskip("cantera")
        pytest.importorskip("yaml")
        from rmip.simulation.muqsac_bridge import MuqsacBridge
        from rmip.types import SimCondition

        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            _make_db(db)
            base_opt = Path(td) / "target.opt"
            import yaml
            base_opt.write_text(yaml.safe_dump({
                "Locations": {"targets": "x", "addendum": "y"},
                "Counts": {"targets_count": 0},
            }))

            # Use a tiny hand-built Cantera Solution instead of a real mechanism
            # file, to keep this test self-contained (no external .yaml needed).
            import cantera as ct
            gas_yaml = Path(td) / "mini.yaml"
            _write_mini_mechanism(gas_yaml)

            conn = sqlite3.connect(db)
            from rmip.database.schema import insert_mechanism
            mid = insert_mechanism(conn, name="mini", format="cantera_yaml",
                                    filepath=str(gas_yaml))
            conn.commit()

            bridge = MuqsacBridge(db, run_script_path="unused.py")
            cond = SimCondition(T_K=1000.0, P_atm=2.0, phi=1.0, fuel="H2:1.0",
                                oxidizer="O2:1.0, N2:3.76")
            work_dir = str(Path(td) / "work")
            target_opt = bridge.build_target_files(
                base_target_opt=str(base_opt), mechanism_id=mid,
                target_type="JSR", solver="cantera", condition=cond,
                work_dir=work_dir, extra_add={"residenceTime": 0.5, "species": "H2"},
            )
            assert target_opt.exists()
            target_text = (Path(work_dir) / "rmip.target").read_text()
            assert "target -- JSR" in target_text
            assert "T -- 1000.0" in target_text
            add_text = (Path(work_dir) / "rmip.add").read_text()
            assert "solver: cantera" in add_text

    def test_target_line_parses_via_combustion_target_class(self):
        """Full round-trip: bridge-generated .target/.add parses correctly
        through the REAL MUQ-SAC parser (skipped if its heavy deps are absent)."""
        pytest.importorskip("cantera")
        pytest.importorskip("sklearn")
        pytest.importorskip("matplotlib")
        pytest.importorskip("scipy")
        if _MUQSAC_SRC not in sys.path:
            sys.path.insert(0, _MUQSAC_SRC)
        import combustion_target_class as ctc
        import yaml

        from rmip.simulation.muqsac_bridge import MuqsacBridge
        from rmip.types import SimCondition

        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            _make_db(db)
            gas_yaml = Path(td) / "mini.yaml"
            _write_mini_mechanism(gas_yaml)
            conn = sqlite3.connect(db)
            from rmip.database.schema import insert_mechanism
            mid = insert_mechanism(conn, name="mini", format="cantera_yaml",
                                    filepath=str(gas_yaml))
            conn.commit()

            base_opt = Path(td) / "target.opt"
            base_opt.write_text(yaml.safe_dump({
                "Locations": {"targets": "x", "addendum": "y"},
                "Counts": {"targets_count": 0},
            }))

            bridge = MuqsacBridge(db, run_script_path="unused.py")
            cond = SimCondition(T_K=1000.0, P_atm=2.0, phi=1.0, fuel="H2:1.0",
                                oxidizer="O2:1.0, N2:3.76")
            work_dir = str(Path(td) / "work")
            bridge.build_target_files(
                base_target_opt=str(base_opt), mechanism_id=mid,
                target_type="JSR", solver="cantera", condition=cond,
                work_dir=work_dir, extra_add={"residenceTime": 0.5, "species": "H2"},
            )
            target_line = (Path(work_dir) / "rmip.target").read_text().strip()
            addendum = yaml.safe_load((Path(work_dir) / "rmip.add").read_text())
            t = ctc.combustion_target(target_line, addendum, 0)

            assert t.target == "JSR"
            assert t.temperature == pytest.approx(1000.0)
            assert t.pressure == pytest.approx(2.0 * 101325.0)  # atm -> Pa conversion
            assert "H2" in t.species_dict
            assert t.add["solver"] == "cantera"

    def test_parse_jsr_output_from_fixture(self):
        from rmip.simulation.muqsac_bridge import MuqsacBridge
        from rmip.types import SimCondition

        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            _make_db(db)
            out_dir = Path(td) / "output"
            out_dir.mkdir()
            (out_dir / "jsr_full.json").write_text(json.dumps({
                "T_K": 1200.0, "P_atm": 10.0, "residence_time_s": 0.1,
                "converged": True, "species_mole_fractions": {"CH4": 0.05, "N2": 0.7},
            }))
            bridge = MuqsacBridge(db, run_script_path="unused.py")
            cond = SimCondition(T_K=1200, P_atm=10, phi=1.0, fuel="CH4:1.0")
            result = bridge.parse_jsr_output(td, mechanism_id=1, solver="cantera", condition=cond)
            assert result.converged is True
            assert result.species_mole_fractions["CH4"] == 0.05

    def test_parse_jsr_output_missing_file(self):
        from rmip.simulation.muqsac_bridge import MuqsacBridge
        from rmip.types import SimCondition
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            _make_db(db)
            bridge = MuqsacBridge(db, run_script_path="unused.py")
            cond = SimCondition(T_K=1200, P_atm=10, phi=1.0, fuel="CH4:1.0")
            result = bridge.parse_jsr_output(td, mechanism_id=1, solver="cantera", condition=cond)
            assert result.converged is False
            assert "not found" in result.error_msg

    def test_store_reactor_result_with_and_without_tracked_species(self):
        from rmip.simulation.muqsac_bridge import MuqsacBridge
        from rmip.types import JSRResult, SimCondition
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "p6.db")
            conn = _make_db(db)
            mid = _insert_mech(conn)
            bridge = MuqsacBridge(db, run_script_path="unused.py")
            cond = SimCondition(T_K=1200, P_atm=10, phi=1.0, fuel="CH4:1.0")
            result = JSRResult(condition=cond, mechanism_id=mid, solver="cantera",
                                residence_time_s=0.1,
                                species_mole_fractions={"CH4": 0.05, "N2": 0.7},
                                converged=True)
            rid1 = bridge.store_reactor_result(result, tracked_species="CH4")
            row1 = conn.execute("SELECT result FROM simulation_runs WHERE id=?", (rid1,)).fetchone()
            assert row1[0] == pytest.approx(0.05)

            rid2 = bridge.store_reactor_result(result)  # no tracked species
            row2 = conn.execute("SELECT result FROM simulation_runs WHERE id=?", (rid2,)).fetchone()
            assert row2[0] is None

    def test_run_invokes_subprocess_with_expected_command(self, monkeypatch):
        from rmip.simulation.muqsac_bridge import MuqsacBridge
        import subprocess

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        bridge = MuqsacBridge(":memory:", run_script_path="src/MUQ-SAC/run_script.py")
        proc = bridge.run(Path("/tmp/somewhere/target.opt"))
        assert proc.returncode == 0
        assert captured["cmd"] == ["python3", "src/MUQ-SAC/run_script.py",
                                    "/tmp/somewhere/target.opt"]
        assert captured["kwargs"]["cwd"] == "/tmp/somewhere"

    def test_sankey_edges_from_rop(self):
        from rmip.simulation.muqsac_bridge import _sankey_edges_from_rop
        rop = {"H2 + O2 <=> OH + OH": 1e-3, "H + O2 => OH + O": 5e-4}
        edges = _sankey_edges_from_rop(rop, "H")
        # Each reaction has 2 reactants x 2 products -> 4 edges; 2 reactions -> 8.
        assert len(edges) == 8
        assert all("source" in e and "target" in e for e in edges)


def _write_mini_mechanism(path: Path) -> None:
    """A minimal 3-species H2/O2/N2 Cantera YAML mechanism for bridge tests."""
    path.write_text("""
generator: rmip-test-fixture
phases:
- name: gas
  thermo: ideal-gas
  elements: [H, O, N]
  species: [H2, O2, N2]
  kinetics: gas
  reactions: none
  state: {T: 300.0, P: 1e5}
species:
- name: H2
  composition: {H: 2}
  thermo:
    model: NASA7
    temperature-ranges: [200.0, 1000.0, 3500.0]
    data:
    - [3.3, 0.0, 0.0, 0.0, 0.0, -1000.0, -3.2]
    - [3.3, 0.0, 0.0, 0.0, 0.0, -1000.0, -3.2]
- name: O2
  composition: {O: 2}
  thermo:
    model: NASA7
    temperature-ranges: [200.0, 1000.0, 3500.0]
    data:
    - [3.2, 0.0, 0.0, 0.0, 0.0, -1000.0, 5.5]
    - [3.2, 0.0, 0.0, 0.0, 0.0, -1000.0, 5.5]
- name: N2
  composition: {N: 2}
  thermo:
    model: NASA7
    temperature-ranges: [200.0, 1000.0, 3500.0]
    data:
    - [3.5, 0.0, 0.0, 0.0, 0.0, -1000.0, 3.9]
    - [3.5, 0.0, 0.0, 0.0, 0.0, -1000.0, 3.9]
""")


# ===========================================================================
# 9. Deck generation (src/MUQ-SAC/make_input_file.py — Phase-6 additions)
# ===========================================================================

class TestMuqsacDeckGeneration:
    @staticmethod
    def _import_mif():
        if _MUQSAC_SRC not in sys.path:
            sys.path.insert(0, _MUQSAC_SRC)
        import make_input_file as mif
        return mif

    class _FakeTarget:
        def __init__(self, target_type, solver, flux=False):
            self.target = target_type
            self.input_file = None
            self.fuel = "CH4"
            self.pressure = 1013250.0  # Pa (post-unit-conversion, like the real class)
            self.temperature = 1200.0
            self.species_dict = {"CH4": 0.1, "O2": 0.2, "N2": 0.7}
            self.add = {
                "solver": solver, "species": "CH4", "residenceTime": 0.05,
                "MaxSimulationTime": 0.2, "pfr_length": 0.1, "pfr_diameter": 0.02,
                "pfr_mdot": 1e-3, "pfr_n_points": 50,
                "flux_analysis": flux, "flux_element": "C",
            }

    _OPT_DICT = {
        "Locations": {"thermo_file": "x", "trans_file": "y"},
        "StartProfilesData": {}, "Inputs": {"global_reaction": ""},
        "Bin": {"bin": "x", "solver_bin": "y"},
    }

    @pytest.mark.parametrize("target_type", ["JSR", "PFR"])
    @pytest.mark.parametrize("solver", ["cantera", "CHEMKIN_PRO", "FlameMaster"])
    @pytest.mark.parametrize("flux", [False, True])
    def test_deck_generates_and_is_well_formed(self, target_type, solver, flux):
        pytest.importorskip("yaml")
        mif = self._import_mif()
        t = self._FakeTarget(target_type, solver, flux)
        instring, s_convert, s_run, extract = mif.create_input_file(
            0, self._OPT_DICT, t, mech_file="mechanism.yaml", solver=solver
        )
        assert instring

        if solver in ("cantera", "CHEMKIN_PRO"):
            import ast
            ast.parse(instring)  # generated script must be valid Python
        else:
            # "ReactionFluxAnal is TRUE" in instring on its own is a substring
            # match on the commented-out form too ("#ReactionFluxAnal is TRUE"),
            # so distinguish the enabled (uncommented) line explicitly.
            assert ("\nReactionFluxAnal is TRUE" in instring) == flux
            assert ("#ReactionFluxAnal is TRUE" in instring) == (not flux)
            assert "MechanismFile is mechanism.yaml" in instring

    def test_cantera_jsr_pressure_labelled_atm_correctly(self):
        """Regression test for the Pa/atm mixup caught during development:
        target.pressure is in Pa post-conversion; jsr_full.json's P_atm must
        divide by 101325, not re-multiply."""
        pytest.importorskip("yaml")
        mif = self._import_mif()
        t = self._FakeTarget("JSR", "cantera", flux=False)
        instring, *_ = mif.create_input_file(0, self._OPT_DICT, t,
                                              mech_file="mechanism.yaml", solver="cantera")
        assert '"P_atm": reactor_pressure / 101325.0' in instring

    def test_chemkin_pro_pressure_uses_bar_not_pa(self):
        """PyChemkin's writeinputJSR/writeInputPlugFlow divide pressure by
        1.01325 internally (bar -> atm), so the bridge must pass bar, not Pa."""
        pytest.importorskip("yaml")
        mif = self._import_mif()
        for ttype in ("JSR", "PFR"):
            t = self._FakeTarget(ttype, "CHEMKIN_PRO", flux=False)
            instring, *_ = mif.create_input_file(0, self._OPT_DICT, t,
                                                  mech_file="mechanism.yaml", solver="CHEMKIN_PRO")
            # target.pressure = 1013250.0 Pa -> 10.1325 bar
            assert "pressure=10.1325" in instring or "pressure=10.132" in instring
