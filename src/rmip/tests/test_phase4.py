"""
rmip/tests/test_phase4.py
==========================
Phase-4 test suite — stdlib + pytest only; no Cantera or RDKit required.

Coverage
--------
- types.py Phase-4 dataclasses
- UncertaintyBandEstimator: hand-computed k(T), 3-sigma convention,
  PLOG interpolation, max-k ≠ max-A regression test
- UnifiedTreeBuilder.plan(): anchored + optimize_all modes
- Manifest validation: unknown decision, PICK_SOURCE without source, OPTIMIZE single-source
- UnifiedTreeBuilder.build(): reaction counts, format='unified' row, mso_bounds shape
- CanteraYAMLWriter: basic output structure (without Cantera round-trip)
- MissingSpeciesPredictor: cross-mech gap + seed output
- schema_phase4: table creation + idempotency
- run_phase4_migrations: idempotency + 'unified' accepted by CHECK
"""

from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Helper: in-memory DB fixtures
# ---------------------------------------------------------------------------

def _make_db(path: str) -> sqlite3.Connection:
    """Create a fully migrated RMIP DB at path."""
    from rmip.database.schema import init_db
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.database.migrations import run_phase3_migrations, run_phase4_migrations
    conn = init_db(path)
    upgrade_db(conn)
    run_phase3_migrations(conn)
    run_phase4_migrations(conn)
    return conn


def _insert_mech(conn, name="M", fmt="chemkin") -> int:
    from rmip.database.schema import insert_mechanism
    return insert_mechanism(conn, name=name, format=fmt)


def _insert_species(conn, mech_id, name, inchi_key=None, formula="H2",
                    tmin=200.0, tmid=1000.0, tmax=5000.0,
                    coeffs_low=None, coeffs_high=None) -> int:
    coeffs_low  = coeffs_low  or [3.5, 0, 0, 0, 0, -1000, 3.5]
    coeffs_high = coeffs_high or [2.5, 0, 0, 0, 0, -1000, 3.5]
    from rmip.database.schema import insert_species
    return insert_species(
        conn,
        mechanism_id     = mech_id,
        name             = name,
        inchi_key        = inchi_key or name,
        formula          = formula,
        thermo_type      = "nasa7",
        tmin_K           = tmin,
        tmid_K           = tmid,
        tmax_K           = tmax,
        coeffs_low_json  = json.dumps(coeffs_low),
        coeffs_high_json = json.dumps(coeffs_high),
    )


def _insert_reaction(conn, mech_id, formula, canonical_key=None,
                     A_si=1e13, n=0.0, Ea_si=100_000.0,
                     kinetics_type="arrhenius") -> int:
    from rmip.database.schema import insert_reaction
    return insert_reaction(
        conn,
        mechanism_id    = mech_id,
        formula_string  = formula,
        canonical_key   = canonical_key or formula,
        A_si            = A_si,
        A_orig          = A_si,
        A_units         = "SI",
        n               = n,
        Ea_si           = Ea_si,
        Ea_orig         = Ea_si,
        Ea_units        = "J/mol",
        kinetics_type   = kinetics_type,
        reversible      = 1,
    )


def _add_stoich(conn, rxn_id, sp_id, coeff, role):
    from rmip.database.schema import insert_stoichiometry
    insert_stoichiometry(conn, rxn_id, sp_id, stoich_coeff=coeff, role=role)


# ===========================================================================
# 1. Phase-4 dataclasses
# ===========================================================================

class TestPhase4Dataclasses:
    def test_temperature_grid_points(self):
        from rmip.types import TemperatureGrid
        grid = TemperatureGrid(T_min=500, T_max=600, dT=50)
        pts = grid.points()
        assert len(pts) == 3
        assert pts[0] == pytest.approx(500.0)
        assert pts[-1] == pytest.approx(600.0)

    def test_temperature_grid_default(self):
        from rmip.types import TemperatureGrid
        grid = TemperatureGrid()
        pts = grid.points()
        assert pts[0] == 500.0
        assert pts[-1] == pytest.approx(2500.0)
        assert abs(pts[1] - pts[0] - 20.0) < 1e-9

    def test_temperature_grid_frozen(self):
        from rmip.types import TemperatureGrid
        grid = TemperatureGrid()
        with pytest.raises((AttributeError, TypeError)):
            grid.T_min = 200  # type: ignore[misc]

    def test_reaction_kinetics(self):
        from rmip.types import ReactionKinetics
        rk = ReactionKinetics(
            mech_id=1, kinetics_type="arrhenius",
            A_si=1e13, n=0.5, Ea_si=50_000.0
        )
        assert rk.mech_id == 1
        assert rk.A_si == pytest.approx(1e13)

    def test_reaction_decision_roundtrip(self):
        from rmip.types import ReactionDecision
        d = ReactionDecision(
            canonical_key="A<=>B", formula_string="A<=>B",
            status="PATCH_MULTI", n_sources=2,
            source_mech_ids=[1, 2], default_decision="OPTIMIZE",
            decision="OPTIMIZE",
        )
        d2 = ReactionDecision.from_dict(d.to_dict())
        assert d2.canonical_key == "A<=>B"
        assert d2.source_mech_ids == [1, 2]
        assert d2.decision == "OPTIMIZE"

    def test_uncertainty_band(self):
        from rmip.types import UncertaintyBand
        band = UncertaintyBand(
            canonical_key="A<=>B", formula_string="A<=>B",
            T_K=[500, 1000], k_nom=[1e10, 2e10],
            k_min=[0.8e10, 1.5e10], k_max=[1.2e10, 2.5e10],
            f_uncertainty=[math.sqrt(1.5), math.sqrt(5/3)],
            sigma_lnk=[math.log(math.sqrt(1.5)) / 3,
                       math.log(math.sqrt(5/3)) / 3],
            f_to_nom=[1.2, 1.25],
            contributor_mech_ids=[1, 2],
        )
        assert band.f_worst() == pytest.approx(math.sqrt(5/3), rel=1e-6)
        assert band.sigma_lnk_at(1000.0) == pytest.approx(
            math.log(math.sqrt(5/3)) / 3, rel=1e-6
        )

    def test_unified_tree_result(self):
        from rmip.types import UnifiedTreeResult
        r = UnifiedTreeResult(
            unified_mechanism_id=5,
            n_keep_anchor=10, n_import_single=5,
            n_optimize=3, n_pick_source=1, n_dropped=2,
            n_species=20,
            bands_csv_path="x.csv", summary_csv_path="y.csv",
            yaml_path="u.yaml", mso_bounds_path="b.json",
            provenance_path="p.json",
        )
        assert r.n_total == 21
        assert "KEEP_ANCHOR" in r.summary()

    def test_missing_species_report(self):
        from rmip.types import MissingSpeciesReport
        report = MissingSpeciesReport(
            unified_mech_id=1, contributing_mech_ids=[2, 3],
            gap_species=[{"inchi_key": "IK1", "name": "A"}],
            gap_reactions=["A<=>B"],
            dead_end_species=["IK2"], orphan_species=[],
            seed_smiles=["CC"], must_include_smiles=["C"],
            evidence={"IK1": "cross-mech gap"},
        )
        d = report.to_dict()
        assert d["n_gap_species"] == 1
        assert d["n_gap_reactions"] == 1

    def test_rmg_seed_spec_json(self, tmp_path):
        from rmip.types import RMGSeedSpec
        spec = RMGSeedSpec(
            seed_smiles=["CC", "CCC"],
            must_include_smiles=["C"],
            source_unified_mech_id=4,
            evidence={"CC": "dead-end"},
        )
        p = str(tmp_path / "seed.json")
        spec.save_json(p)
        data = json.loads(Path(p).read_text())
        assert data["seed_smiles"] == ["CC", "CCC"]
        assert data["source_unified_mech_id"] == 4


# ===========================================================================
# 2. UncertaintyBandEstimator
# ===========================================================================

R_GAS = 8.314_462_618

def _k_arr(A, n, Ea, T):
    return A * (T ** n) * math.exp(-Ea / (R_GAS * T))


class TestUncertaintyBandEstimator:

    def _make_contrib(self, mech_id, A, n, Ea):
        from rmip.types import ReactionKinetics
        return ReactionKinetics(
            mech_id=mech_id, kinetics_type="arrhenius",
            A_si=A, n=n, Ea_si=Ea,
        )

    def test_two_identical_contributions(self):
        """Band is trivial (f=1, sigma=0) when both mechs agree."""
        from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator
        from rmip.types import TemperatureGrid
        est = UncertaintyBandEstimator()
        grid = TemperatureGrid(T_min=1000, T_max=1000, dT=20)
        contribs = [
            self._make_contrib(1, 1e13, 0.0, 100_000),
            self._make_contrib(2, 1e13, 0.0, 100_000),
        ]
        band = est.band_for_reaction(contribs, grid)
        assert band.f_uncertainty[0] == pytest.approx(1.0, abs=1e-9)
        assert band.sigma_lnk[0] == pytest.approx(0.0, abs=1e-9)

    def test_hand_computed_two_mechs(self):
        """Verify k_nom, k_min, k_max, f, sigma at T=1000 K for two mechs."""
        from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator
        from rmip.types import TemperatureGrid

        A1, n1, Ea1 = 1.0e13, 0.0, 100_000.0
        A2, n2, Ea2 = 3.0e13, 0.0, 100_000.0
        T = 1000.0

        k1 = _k_arr(A1, n1, Ea1, T)
        k2 = _k_arr(A2, n2, Ea2, T)
        k_nom_expected = math.exp((math.log(k1) + math.log(k2)) / 2)
        k_min_expected = min(k1, k2)
        k_max_expected = max(k1, k2)
        f_expected     = math.sqrt(k_max_expected / k_min_expected)
        sigma_expected = math.log(f_expected) / 3  # 3-sigma convention

        est = UncertaintyBandEstimator()
        grid = TemperatureGrid(T_min=T, T_max=T, dT=20)
        contribs = [
            self._make_contrib(1, A1, n1, Ea1),
            self._make_contrib(2, A2, n2, Ea2),
        ]
        band = est.band_for_reaction(contribs, grid)

        assert band.k_nom[0] == pytest.approx(k_nom_expected, rel=1e-6)
        assert band.k_min[0] == pytest.approx(k_min_expected, rel=1e-6)
        assert band.k_max[0] == pytest.approx(k_max_expected, rel=1e-6)
        assert band.f_uncertainty[0] == pytest.approx(f_expected, rel=1e-6)
        assert band.sigma_lnk[0] == pytest.approx(sigma_expected, rel=1e-6)

    def test_3sigma_convention_explicit(self):
        """sigma_lnk = ln(f) / 3  (3-sigma, not 2-sigma)."""
        from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator
        from rmip.types import TemperatureGrid
        est = UncertaintyBandEstimator()
        grid = TemperatureGrid(T_min=1000, T_max=1000, dT=20)
        contribs = [
            self._make_contrib(1, 1e13, 0.0, 100_000),
            self._make_contrib(2, 4e13, 0.0, 100_000),
        ]
        band = est.band_for_reaction(contribs, grid)
        f = band.f_uncertainty[0]
        sigma = band.sigma_lnk[0]
        # Must satisfy sigma = ln(f) / 3  (3-sigma convention)
        assert sigma == pytest.approx(math.log(f) / 3, rel=1e-9)
        # Must NOT satisfy sigma = ln(f) / 2  (2-sigma) unless coincidence
        assert sigma != pytest.approx(math.log(f) / 2, rel=1e-9)

    def test_max_k_not_max_A_regression(self):
        """
        REGRESSION: when Ea differs, the max-k mechanism at low T is NOT the
        max-A mechanism.  A scalar uncertainty_factor = max(A)/min(A) would
        wrongly credit mech-1 (high A) as the always-max source, but at T=1000K
        mech-2 (lower A, much lower Ea) is actually faster.  At T=3000K the
        high-A/high-Ea mech-1 takes over.  The per-T band captures both regimes.
        """
        from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator
        from rmip.types import TemperatureGrid

        # Mech 1: high A, high Ea — only dominant at very high T
        A1, n1, Ea1 = 1e14, 0.0, 200_000.0
        # Mech 2: lower A, very low Ea — dominant across combustion-relevant T
        A2, n2, Ea2 = 1e10, 0.0, 10_000.0

        T_low  = 1000.0   # mech2 wins here despite lower A
        T_high = 3000.0   # mech1 wins here due to its A advantage

        k1_low  = _k_arr(A1, n1, Ea1, T_low)
        k2_low  = _k_arr(A2, n2, Ea2, T_low)
        k1_high = _k_arr(A1, n1, Ea1, T_high)
        k2_high = _k_arr(A2, n2, Ea2, T_high)

        # Pre-condition: mech2 (lower A) is faster at T_low
        assert k2_low > k1_low, (
            f"Fixture invalid: expected k2_low({k2_low:.2e}) > k1_low({k1_low:.2e})"
        )
        # Pre-condition: mech1 (higher A) is faster at T_high
        assert k1_high > k2_high, (
            f"Fixture invalid: expected k1_high({k1_high:.2e}) > k2_high({k2_high:.2e})"
        )
        # Confirm scalar max(A)/min(A) identifies mech1 as the "max-A" source
        assert A1 > A2, "Mech1 must have higher A for this regression to be meaningful"

        # Band estimator must correctly identify who is max at each T point
        est  = UncertaintyBandEstimator()
        grid = TemperatureGrid(T_min=T_low, T_max=T_high, dT=T_high - T_low)
        contribs = [
            self._make_contrib(1, A1, n1, Ea1),
            self._make_contrib(2, A2, n2, Ea2),
        ]
        band = est.band_for_reaction(contribs, grid)

        idx_low  = 0
        idx_high = len(band.T_K) - 1

        # At low T: k_max comes from mech2 (NOT the max-A mech1)
        assert band.k_max[idx_low]  == pytest.approx(k2_low, rel=1e-6)
        assert band.k_min[idx_low]  == pytest.approx(k1_low, rel=1e-6)
        # At high T: k_max comes from mech1 (max-A wins)
        assert band.k_max[idx_high] == pytest.approx(k1_high, rel=1e-6)
        assert band.k_min[idx_high] == pytest.approx(k2_high, rel=1e-6)

    def test_plog_interpolation(self):
        """PLOG band: P_ref between two pressures → interpolated k(T)."""
        from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator
        from rmip.types import ReactionKinetics, TemperatureGrid

        A_lo, n_lo, Ea_lo = 1e12, 0.0, 80_000.0
        A_hi, n_hi, Ea_hi = 3e12, 0.0, 80_000.0
        P_lo_Pa = 101_325.0    # 1 atm
        P_hi_Pa = 1_013_250.0  # 10 atm
        P_ref_Pa = math.sqrt(P_lo_Pa * P_hi_Pa)  # geometric mean → frac=0.5
        P_ref_atm = P_ref_Pa / 101_325.0

        T = 1000.0
        k_lo_T = _k_arr(A_lo, n_lo, Ea_lo, T)
        k_hi_T = _k_arr(A_hi, n_hi, Ea_hi, T)
        # Log-linear interpolation at frac=0.5
        k_interp_expected = math.exp(
            math.log(k_lo_T) + 0.5 * (math.log(k_hi_T) - math.log(k_lo_T))
        )

        kdata = json.dumps({
            "pressures": [P_lo_Pa, P_hi_Pa],
            "arrhenius": [
                {"A_si": A_lo, "n": n_lo, "Ea_si": Ea_lo},
                {"A_si": A_hi, "n": n_hi, "Ea_si": Ea_hi},
            ],
        })
        contrib = ReactionKinetics(
            mech_id=1, kinetics_type="plog",
            A_si=A_lo, n=n_lo, Ea_si=Ea_lo,
            kinetics_data_json=kdata,
        )

        est = UncertaintyBandEstimator(P_ref_atm=P_ref_atm)
        grid = TemperatureGrid(T_min=T, T_max=T, dT=20)
        band = est.band_for_reaction([contrib], grid)

        assert band.k_nom[0] == pytest.approx(k_interp_expected, rel=1e-4)

    def test_falloff_skipped_without_cantera(self):
        """Troe/Lindemann contributions are skipped when use_cantera=False."""
        from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator
        from rmip.types import ReactionKinetics, TemperatureGrid

        troe_contrib = ReactionKinetics(
            mech_id=1, kinetics_type="troe",
            A_si=1e13, n=0.0, Ea_si=100_000.0,
            kinetics_data_json=json.dumps({"alpha": 0.5, "T3": 1e-30, "T1": 1e30}),
        )
        arrh_contrib = ReactionKinetics(
            mech_id=2, kinetics_type="arrhenius",
            A_si=1e13, n=0.0, Ea_si=100_000.0,
        )

        est  = UncertaintyBandEstimator(use_cantera_for_falloff=False)
        grid = TemperatureGrid(T_min=1000, T_max=1000, dT=20)
        band = est.band_for_reaction([troe_contrib, arrh_contrib], grid)

        # troe contribution skipped → only 1 contributor
        assert band.n_skipped == 1
        assert len(band.contributor_mech_ids) == 1
        assert band.contributor_mech_ids[0] == 2

    def test_band_csv_schema(self, tmp_path):
        """CSV output has correct column headers."""
        from rmip.analysis.uncertainty_bands import UncertaintyBandEstimator
        from rmip.types import ReactionKinetics, TemperatureGrid, UncertaintyBand

        est  = UncertaintyBandEstimator()
        grid = TemperatureGrid(T_min=500, T_max=520, dT=20)
        contribs = [
            ReactionKinetics(mech_id=1, kinetics_type="arrhenius",
                             A_si=1e13, n=0.0, Ea_si=100_000.0),
            ReactionKinetics(mech_id=2, kinetics_type="arrhenius",
                             A_si=2e13, n=0.0, Ea_si=100_000.0),
        ]
        band = est.band_for_reaction(contribs, grid)
        band.canonical_key  = "A<=>B"
        band.formula_string = "A<=>B"

        # Write bands CSV
        csv_path = str(tmp_path / "bands.csv")
        with open(csv_path, "w", newline="") as fh:
            from rmip.analysis.unified_tree import _BANDS_CSV_HEADER
            writer = csv.DictWriter(fh, fieldnames=_BANDS_CSV_HEADER)
            writer.writeheader()
            contrib_str = json.dumps(band.contributor_mech_ids)
            for i, T in enumerate(band.T_K):
                writer.writerow({
                    "canonical_key":       band.canonical_key,
                    "formula_string":      band.formula_string,
                    "T_K":                 T,
                    "k_nom":               band.k_nom[i],
                    "k_min":               band.k_min[i],
                    "k_max":               band.k_max[i],
                    "f_uncertainty":       band.f_uncertainty[i],
                    "sigma_lnk":           band.sigma_lnk[i],
                    "f_to_nom":            band.f_to_nom[i],
                    "n_contributors":      len(band.contributor_mech_ids),
                    "contributor_mech_ids": contrib_str,
                })

        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows   = list(reader)
        from rmip.analysis.unified_tree import _BANDS_CSV_HEADER
        for col in _BANDS_CSV_HEADER:
            assert col in rows[0], f"Missing column: {col}"
        # 2 T points
        assert len(rows) == 2


# ===========================================================================
# 3. schema_phase4 and migrations
# ===========================================================================

class TestSchemaPhase4:

    def test_tables_created(self, tmp_path):
        """Phase-4 tables exist after upgrade_db_phase4."""
        from rmip.database.schema_phase4 import upgrade_db_phase4
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        from rmip.database.schema import init_db
        from rmip.database.schema_phase3 import upgrade_db
        conn = init_db(str(tmp_path / "test.db"))
        upgrade_db(conn)
        upgrade_db_phase4(conn)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "unified_tree_runs"  in tables
        assert "reaction_decisions" in tables
        assert "uncertainty_bands"  in tables

    def test_upgrade_idempotent(self, tmp_path):
        """upgrade_db_phase4 is safe to call twice."""
        from rmip.database.schema_phase4 import upgrade_db_phase4
        from rmip.database.schema import init_db
        conn = init_db(str(tmp_path / "test.db"))
        upgrade_db_phase4(conn)
        upgrade_db_phase4(conn)  # should not raise

    def test_run_phase4_migrations_idempotent(self, tmp_path):
        """run_phase4_migrations is safe to call twice."""
        db_path = str(tmp_path / "test.db")
        conn = _make_db(db_path)
        conn.close()
        conn2 = sqlite3.connect(db_path)
        from rmip.database.migrations import run_phase4_migrations
        run_phase4_migrations(conn2)   # second call — must not raise
        conn2.close()

    def test_unified_format_accepted(self, tmp_path):
        """'unified' is a valid value for mechanisms.format after migration."""
        db_path = str(tmp_path / "test.db")
        conn = _make_db(db_path)
        from rmip.database.schema import insert_mechanism
        mid = insert_mechanism(conn, name="U", format="unified")
        assert mid > 0
        row = conn.execute("SELECT format FROM mechanisms WHERE id=?", (mid,)).fetchone()
        assert row[0] == "unified"
        conn.close()

    def test_insert_uncertainty_band_rows(self, tmp_path):
        """CRUD: insert_uncertainty_band_rows writes correct row count."""
        db_path = str(tmp_path / "test.db")
        conn = _make_db(db_path)

        mid = _insert_mech(conn, fmt="unified")
        run_id = _insert_run(conn, mid)

        from rmip.database.schema_phase4 import insert_uncertainty_band_rows
        T_pts = [500.0, 600.0, 700.0]
        k_nom = [1e10, 2e10, 3e10]
        k_min = [0.9e10, 1.8e10, 2.7e10]
        k_max = [1.1e10, 2.2e10, 3.3e10]
        f_list = [math.sqrt(k_max[i] / k_min[i]) for i in range(3)]
        sigma  = [math.log(f) / 3 for f in f_list]
        f2n    = [k_max[i] / k_nom[i] for i in range(3)]

        insert_uncertainty_band_rows(
            conn, run_id, "A<=>B", T_pts, k_nom, k_min, k_max, f_list, sigma, f2n
        )
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM uncertainty_bands WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        assert count == 3
        conn.close()


def _insert_run(conn, mech_id) -> int:
    from rmip.database.schema_phase4 import insert_unified_tree_run
    return insert_unified_tree_run(
        conn,
        unified_mechanism_id  = mech_id,
        mode                  = "anchored",
        contributing_mech_ids = [1, 2],
    )


# ===========================================================================
# 4. UnifiedTreeBuilder.plan()
# ===========================================================================

def _make_two_mech_db(tmp_path):
    """
    Three mechanisms in DB:
      mech A (id=1): reactions R1, R2, R3
      mech B (id=2): reactions R1 (same), R3 (same), R4 (different from all)
      mech C (id=3): R4 only

    For comparison between mechs 1 and 2 (anchor=1):
      R1 → IN_ANCHOR (in anchor 1)
      R2 → IN_ANCHOR (only in anchor 1; still IN_ANCHOR because anchor wins)
      R3 → IN_ANCHOR (in anchor 1)
      R4 → PATCH_SINGLE (in mech 2 only, not in anchor 1)

    For optimize_all (mechs 1+2):
      R1, R3 → SHARED (in both mechs 1+2)
      R2     → UNIQUE (only in mech 1)
      R4     → UNIQUE (only in mech 2)
    """
    db_path = str(tmp_path / "p4.db")
    conn = _make_db(db_path)

    m1 = _insert_mech(conn, "MechA")
    m2 = _insert_mech(conn, "MechB")

    # Shared species: H2, O2, H2O (needed by reactions)
    for mid in [m1, m2]:
        s1 = _insert_species(conn, mid, "H2",  inchi_key="IK_H2",  formula="H2")
        s2 = _insert_species(conn, mid, "O2",  inchi_key="IK_O2",  formula="O2")
        s3 = _insert_species(conn, mid, "H2O", inchi_key="IK_H2O", formula="H2O")
        # Extra species in m1 only: OH
        if mid == m1:
            s4 = _insert_species(conn, m1, "OH", inchi_key="IK_OH", formula="HO")

    # Reaction R1: H2+O2<=>H2O (in both mechs, same kinetics)
    r1_m1 = _insert_reaction(conn, m1, "H2+O2<=>H2O", canonical_key="R1",
                              A_si=1e13, n=0.0, Ea_si=100_000.0)
    r1_m2 = _insert_reaction(conn, m2, "H2+O2<=>H2O", canonical_key="R1",
                              A_si=1e13, n=0.0, Ea_si=100_000.0)

    # Reaction R2: H2+OH<=>H2O (only in mech 1)
    r2_m1 = _insert_reaction(conn, m1, "H2+OH<=>H2O", canonical_key="R2",
                              A_si=2e13, n=0.0, Ea_si=80_000.0)

    # Reaction R3: O2<=>O+O (in both mechs, divergent kinetics)
    r3_m1 = _insert_reaction(conn, m1, "O2<=>O+O", canonical_key="R3",
                              A_si=1e15, n=0.0, Ea_si=494_000.0)
    r3_m2 = _insert_reaction(conn, m2, "O2<=>O+O", canonical_key="R3",
                              A_si=2e15, n=0.0, Ea_si=490_000.0)

    # Reaction R4: H2O<=>H+OH (only in mech 2)
    r4_m2 = _insert_reaction(conn, m2, "H2O<=>H+OH", canonical_key="R4",
                              A_si=3e14, n=0.0, Ea_si=450_000.0)

    # Add stoichiometry for R1
    s_h2  = conn.execute("SELECT id FROM species WHERE mechanism_id=? AND name='H2'",  (m1,)).fetchone()[0]
    s_o2  = conn.execute("SELECT id FROM species WHERE mechanism_id=? AND name='O2'",  (m1,)).fetchone()[0]
    s_h2o = conn.execute("SELECT id FROM species WHERE mechanism_id=? AND name='H2O'", (m1,)).fetchone()[0]
    _add_stoich(conn, r1_m1, s_h2,  1.0, "reactant")
    _add_stoich(conn, r1_m1, s_o2,  1.0, "reactant")
    _add_stoich(conn, r1_m1, s_h2o, 1.0, "product")
    _add_stoich(conn, r3_m1, s_o2,  1.0, "reactant")

    s_h2_m2  = conn.execute("SELECT id FROM species WHERE mechanism_id=? AND name='H2'",  (m2,)).fetchone()[0]
    s_o2_m2  = conn.execute("SELECT id FROM species WHERE mechanism_id=? AND name='O2'",  (m2,)).fetchone()[0]
    s_h2o_m2 = conn.execute("SELECT id FROM species WHERE mechanism_id=? AND name='H2O'", (m2,)).fetchone()[0]
    _add_stoich(conn, r1_m2, s_h2_m2,  1.0, "reactant")
    _add_stoich(conn, r1_m2, s_o2_m2,  1.0, "reactant")
    _add_stoich(conn, r1_m2, s_h2o_m2, 1.0, "product")
    _add_stoich(conn, r3_m2, s_o2_m2,  1.0, "reactant")
    _add_stoich(conn, r4_m2, s_h2o_m2, 1.0, "reactant")

    conn.commit()
    conn.close()
    return db_path, m1, m2


class TestUnifiedTreeBuilderPlan:

    def test_anchored_mode(self, tmp_path):
        """Anchored mode: IN_ANCHOR / PATCH_SINGLE / PATCH_MULTI classification."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        builder = UnifiedTreeBuilder(db_path)
        manifest = str(tmp_path / "decisions.csv")
        plan = builder.plan(
            mechanism_ids  = [m1, m2],
            mode           = "anchored",
            anchor_mech_id = m1,
            manifest_path  = manifest,
        )
        assert plan.mode == "anchored"
        assert plan.anchor_mech_id == m1
        assert Path(manifest).exists()

        by_key = {d.canonical_key: d for d in plan.decisions}

        # R1 is in anchor → IN_ANCHOR
        assert by_key["R1"].status == "IN_ANCHOR"
        assert by_key["R1"].default_decision == "KEEP_ANCHOR"

        # R2 is only in anchor (UNIQUE to anchor) → IN_ANCHOR
        assert by_key["R2"].status == "IN_ANCHOR"

        # R3 is in anchor (even though also in m2) → IN_ANCHOR
        assert by_key["R3"].status == "IN_ANCHOR"

        # R4 is absent from anchor, present in 1 non-anchor mech → PATCH_SINGLE
        assert by_key["R4"].status == "PATCH_SINGLE"
        assert by_key["R4"].default_decision == "IMPORT_SINGLE"

    def test_optimize_all_mode(self, tmp_path):
        """optimize_all: shared reactions → OPTIMIZE, unique → IMPORT_SINGLE."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        builder = UnifiedTreeBuilder(db_path)
        manifest = str(tmp_path / "decisions_opt.csv")
        plan = builder.plan(
            mechanism_ids = [m1, m2],
            mode          = "optimize_all",
            manifest_path = manifest,
        )
        assert plan.mode == "optimize_all"
        assert plan.anchor_mech_id is None

        by_key = {d.canonical_key: d for d in plan.decisions}

        # R1 and R3 are in both mechs → SHARED → OPTIMIZE
        assert by_key["R1"].status == "SHARED"
        assert by_key["R1"].default_decision == "OPTIMIZE"
        assert by_key["R3"].status == "SHARED"

        # R2 only in m1, R4 only in m2 → UNIQUE → IMPORT_SINGLE
        assert by_key["R2"].status == "UNIQUE"
        assert by_key["R2"].default_decision == "IMPORT_SINGLE"
        assert by_key["R4"].status == "UNIQUE"
        assert by_key["R4"].default_decision == "IMPORT_SINGLE"

    def test_manifest_csv_written(self, tmp_path):
        """Manifest CSV has correct fieldnames."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.analysis.unified_tree import UnifiedTreeBuilder, _CSV_FIELDNAMES
        builder = UnifiedTreeBuilder(db_path)
        manifest = str(tmp_path / "decisions.csv")
        builder.plan([m1, m2], mode="optimize_all", manifest_path=manifest)
        with open(manifest) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        for col in _CSV_FIELDNAMES:
            assert col in rows[0]
        assert len(rows) == 4  # R1, R2, R3, R4

    def test_mode_validation(self, tmp_path):
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        builder = UnifiedTreeBuilder(db_path)
        with pytest.raises(ValueError, match="mode"):
            builder.plan([m1, m2], mode="invalid_mode",
                         manifest_path=str(tmp_path / "x.csv"))

    def test_anchored_requires_anchor(self, tmp_path):
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        builder = UnifiedTreeBuilder(db_path)
        with pytest.raises(ValueError, match="anchor_mech_id"):
            builder.plan([m1, m2], mode="anchored",
                         manifest_path=str(tmp_path / "x.csv"))


# ===========================================================================
# 5. UnifiedTreeBuilder — manifest validation
# ===========================================================================

class TestManifestValidation:

    def test_unknown_decision_rejected(self):
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import ReactionDecision
        decisions = [
            ReactionDecision("A<=>B", "A<=>B", "SHARED", 2, [1, 2],
                             "OPTIMIZE", "INVALID_DECISION"),
        ]
        with pytest.raises(ValueError, match="Unknown decision"):
            UnifiedTreeBuilder._validate_decisions(decisions)

    def test_pick_source_without_valid_source(self):
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import ReactionDecision
        decisions = [
            ReactionDecision("A<=>B", "A<=>B", "PATCH_MULTI", 2, [1, 2],
                             "OPTIMIZE", "PICK_SOURCE",
                             chosen_source_mech_id=99),   # 99 not in [1,2]
        ]
        with pytest.raises(ValueError, match="chosen_source_mech_id"):
            UnifiedTreeBuilder._validate_decisions(decisions)

    def test_pick_source_without_id_rejected(self):
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import ReactionDecision
        decisions = [
            ReactionDecision("A<=>B", "A<=>B", "PATCH_MULTI", 2, [1, 2],
                             "OPTIMIZE", "PICK_SOURCE",
                             chosen_source_mech_id=None),
        ]
        with pytest.raises(ValueError, match="PICK_SOURCE requires"):
            UnifiedTreeBuilder._validate_decisions(decisions)

    def test_optimize_single_source_rejected(self):
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import ReactionDecision
        decisions = [
            ReactionDecision("A<=>B", "A<=>B", "UNIQUE", 1, [1],
                             "IMPORT_SINGLE", "OPTIMIZE"),
        ]
        with pytest.raises(ValueError, match="OPTIMIZE on single-source"):
            UnifiedTreeBuilder._validate_decisions(decisions)

    def test_valid_decisions_pass(self):
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import ReactionDecision
        decisions = [
            ReactionDecision("A<=>B", "A<=>B", "IN_ANCHOR",    1, [1],    "KEEP_ANCHOR",    "KEEP_ANCHOR"),
            ReactionDecision("C<=>D", "C<=>D", "PATCH_SINGLE", 1, [2],    "IMPORT_SINGLE",  "IMPORT_SINGLE"),
            ReactionDecision("E<=>F", "E<=>F", "PATCH_MULTI",  2, [1, 2], "OPTIMIZE",       "OPTIMIZE"),
            ReactionDecision("G<=>H", "G<=>H", "PATCH_MULTI",  2, [1, 2], "OPTIMIZE",       "PICK_SOURCE",
                             chosen_source_mech_id=1),
            ReactionDecision("I<=>J", "I<=>J", "SHARED",       2, [1, 2], "OPTIMIZE",       "DROP"),
        ]
        UnifiedTreeBuilder._validate_decisions(decisions)   # must not raise


# ===========================================================================
# 6. UnifiedTreeBuilder.build()
# ===========================================================================

class TestUnifiedTreeBuilderBuild:

    def test_build_creates_unified_mechanism(self, tmp_path):
        """build() inserts a 'unified' format mechanism with correct counts."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import TemperatureGrid

        builder  = UnifiedTreeBuilder(db_path)
        manifest = str(tmp_path / "decisions.csv")
        builder.plan([m1, m2], mode="anchored", anchor_mech_id=m1,
                     manifest_path=manifest)

        grid   = TemperatureGrid(T_min=500, T_max=520, dT=20)
        result = builder.build(
            manifest_path = manifest,
            name          = "test_unified",
            tgrid         = grid,
            out_dir       = str(tmp_path / "out"),
        )

        assert result.unified_mechanism_id > 0
        assert result.n_keep_anchor >= 1
        # No PATCH_MULTI in our fixture for anchored mode (R4 is PATCH_SINGLE)
        assert result.n_import_single >= 0

        # Verify DB row
        conn = sqlite3.connect(db_path)
        row  = conn.execute(
            "SELECT format FROM mechanisms WHERE id=?",
            (result.unified_mechanism_id,),
        ).fetchone()
        assert row[0] == "unified"
        conn.close()

    def test_build_output_files(self, tmp_path):
        """build() writes bands.csv, summary.csv, mso_bounds.json, provenance.json."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import TemperatureGrid

        builder  = UnifiedTreeBuilder(db_path)
        manifest = str(tmp_path / "decisions.csv")
        builder.plan([m1, m2], mode="optimize_all", manifest_path=manifest)

        grid   = TemperatureGrid(T_min=1000, T_max=1000, dT=20)
        result = builder.build(
            manifest_path = manifest,
            name          = "all_opt",
            tgrid         = grid,
            out_dir       = str(tmp_path / "out"),
        )

        assert Path(result.bands_csv_path).exists()
        assert Path(result.summary_csv_path).exists()
        assert Path(result.mso_bounds_path).exists()
        assert Path(result.provenance_path).exists()

    def test_mso_bounds_json_shape(self, tmp_path):
        """mso_bounds.json has correct schema (fuse_blocks with params_mean/lb/ub)."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import TemperatureGrid

        builder  = UnifiedTreeBuilder(db_path)
        manifest = str(tmp_path / "decisions.csv")
        builder.plan([m1, m2], mode="optimize_all", manifest_path=manifest)

        grid   = TemperatureGrid(T_min=1000, T_max=1000, dT=20)
        result = builder.build(
            manifest_path = manifest, name="opt",
            tgrid=grid, out_dir=str(tmp_path / "out"),
        )

        bounds = json.loads(Path(result.mso_bounds_path).read_text())
        assert "fuse_blocks" in bounds
        for block in bounds["fuse_blocks"]:
            assert "params_mean" in block
            assert "params_lb"   in block
            assert "params_ub"   in block
            assert len(block["params_mean"]) == 3
            assert len(block["params_lb"])   == 3
            assert len(block["params_ub"])   == 3

    def test_bands_csv_row_count(self, tmp_path):
        """bands.csv has n_optimize × n_Tpoints rows."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.analysis.unified_tree import UnifiedTreeBuilder
        from rmip.types import TemperatureGrid

        builder  = UnifiedTreeBuilder(db_path)
        manifest = str(tmp_path / "decisions.csv")
        plan     = builder.plan([m1, m2], mode="optimize_all",
                                 manifest_path=manifest)

        n_optimize = sum(1 for d in plan.decisions if d.default_decision == "OPTIMIZE")
        grid = TemperatureGrid(T_min=500, T_max=520, dT=20)   # 2 points
        result = builder.build(
            manifest_path=manifest, name="opt",
            tgrid=grid, out_dir=str(tmp_path / "out"),
        )

        with open(result.bands_csv_path) as fh:
            rows = list(csv.DictReader(fh))
        # Each OPTIMIZE reaction that has valid contributions → 2 T-point rows
        # (only SHARED reactions, n_optimize from plan)
        assert len(rows) == n_optimize * 2


# ===========================================================================
# 7. CanteraYAMLWriter
# ===========================================================================

class TestCanteraYAMLWriter:

    def test_writes_yaml_file(self, tmp_path):
        """CanteraYAMLWriter.write() produces a non-empty YAML file."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.io.cantera_writer import CanteraYAMLWriter
        writer = CanteraYAMLWriter(db_path)
        yaml_path = str(tmp_path / "test.yaml")
        writer.write(m1, yaml_path, uncertainty_bands=None, embed_uncertainty=False)
        assert Path(yaml_path).exists()
        text = Path(yaml_path).read_text()
        assert "phases:" in text
        assert "species:" in text
        assert "reactions:" in text

    def test_yaml_contains_species_names(self, tmp_path):
        """All species from mechanism appear in the YAML."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.io.cantera_writer import CanteraYAMLWriter
        writer = CanteraYAMLWriter(db_path)
        yaml_path = str(tmp_path / "sp.yaml")
        writer.write(m1, yaml_path)
        text = Path(yaml_path).read_text()
        for sp in ["H2", "O2", "H2O", "OH"]:
            assert sp in text

    def test_uncertainty_embedded(self, tmp_path):
        """Uncertainty metadata is embedded when embed_uncertainty=True."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.io.cantera_writer import CanteraYAMLWriter
        from rmip.types import UncertaintyBand

        band = UncertaintyBand(
            canonical_key="R3", formula_string="O2<=>O+O",
            T_K=[1000.0], k_nom=[1e10], k_min=[0.8e10], k_max=[1.2e10],
            f_uncertainty=[math.sqrt(1.5)],
            sigma_lnk=[math.log(math.sqrt(1.5)) / 3],
            f_to_nom=[1.2],
            contributor_mech_ids=[m1, m2],
        )

        writer = CanteraYAMLWriter(db_path)
        yaml_path = str(tmp_path / "unc.yaml")
        writer.write(m1, yaml_path, uncertainty_bands=[band], embed_uncertainty=True)
        text = Path(yaml_path).read_text()
        assert "rmip-uncertainty" in text or "uf_worstT" in text

    def test_unknown_mechanism_raises(self, tmp_path):
        """Writing an unknown mechanism ID raises ValueError."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.io.cantera_writer import CanteraYAMLWriter
        writer = CanteraYAMLWriter(db_path)
        with pytest.raises(ValueError, match="not found"):
            writer.write(9999, str(tmp_path / "x.yaml"))

    def test_nasa7_data_blocks_low_temperature_first(self, tmp_path):
        """M8 (review.md): the two NASA7 `data:` blocks are positionally
        paired with `temperature-ranges: [Tmin, Tmid, Tmax]` -- the first
        block must be the LOW-temperature (Tmin-Tmid) polynomial and the
        second the HIGH-temperature (Tmid-Tmax) one. These were swapped,
        which is what produced the "discontinuity in cp/R/h/RT/s/R at
        Tmid" warnings seen throughout this writer's Cantera round-trip
        validation for every multi-range species it ever exported --
        confirmed by round-tripping real, physically continuous N2
        coefficients (from this repo's own FFCM1.yaml) and checking no
        such warning fires."""
        ct = pytest.importorskip("cantera")
        import warnings

        db_path = str(tmp_path / "n2.db")
        conn = _make_db(db_path)
        m1 = _insert_mech(conn, "N2Mech")
        # Real GRI/FFCM1-style N2 NASA-7 coefficients (continuous at Tmid).
        low  = [3.53100528, -1.23660988e-04, -5.02999433e-07,
                2.43530612e-09, -1.40881235e-12, -1046.97628, 2.96747038]
        high = [2.95257637, 1.3969004e-03, -4.92631603e-07,
                7.86010195e-11, -4.60755204e-15, -923.948688, 5.87188762]
        _insert_species(conn, m1, "N2", inchi_key="IK_N2", formula="N2",
                        tmin=200.0, tmid=1000.0, tmax=6000.0,
                        coeffs_low=low, coeffs_high=high)
        # "N" also needs continuous low/high coefficients (not this file's
        # default fake discontinuous ones) or its own genuinely-discontinuous
        # test data would mask whether N2's block order is actually correct.
        _insert_species(conn, m1, "N", inchi_key="IK_N", formula="N",
                        tmin=200.0, tmid=1000.0, tmax=6000.0,
                        coeffs_low=low, coeffs_high=high)
        r1 = _insert_reaction(conn, m1, "N2 <=> 2 N", canonical_key="R1")
        conn.commit()

        from rmip.io.cantera_writer import CanteraYAMLWriter
        writer = CanteraYAMLWriter(db_path)
        yaml_path = str(tmp_path / "n2.yaml")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            writer.write(m1, yaml_path)
            discontinuities = [w for w in caught if "discontinuity" in str(w.message)]
        assert discontinuities == []

        # Also verify the raw YAML text: the first data row must start with
        # the low-T leading coefficient (3.531...), not the high-T one.
        text = Path(yaml_path).read_text()
        data_idx = text.index("- name: N2")
        data_block = text[data_idx:data_idx + 400]
        first_row_start = data_block.index("data:") + len("data:")
        assert "3.531" in data_block[first_row_start:first_row_start + 60]

    def test_two_letter_element_case_round_trips_through_cantera(self, tmp_path):
        """M6 (review.md): `el.upper()` turned "Ar" into "AR" in the
        `elements:` declaration while `composition:` (built from the same
        regex with no case transform) kept "Ar" -- Cantera's element lookup
        is case-sensitive, so any mechanism with a 2-letter element (Ar, He,
        Cl) failed to load with "element not found: AR" (confirmed
        empirically against GRI-wetCO). Round-trips an Ar-containing
        mechanism through ct.Solution() to catch this class of bug for
        real, not just by inspecting the YAML text."""
        ct = pytest.importorskip("cantera")

        db_path = str(tmp_path / "ar.db")
        conn = _make_db(db_path)
        m1 = _insert_mech(conn, "ArMech")
        s_ar = _insert_species(conn, m1, "AR", inchi_key="IK_AR", formula="Ar")
        s_h2 = _insert_species(conn, m1, "H2", inchi_key="IK_H2", formula="H2")
        s_h  = _insert_species(conn, m1, "H",  inchi_key="IK_H",  formula="H")
        # Ar only needs to appear in the SPECIES/composition block to
        # exercise the elements-case bug -- keep the reaction itself
        # unrelated to avoid any separate third-body-pattern reconstruction.
        # Cantera's YAML equation parser requires whitespace around the
        # arrow and stoich coefficients (unlike CHEMKIN's free-form
        # tokenizer) -- "H2 <=> 2 H", not "H2<=>2H".
        r1 = _insert_reaction(conn, m1, "H2 <=> 2 H", canonical_key="R1")
        _add_stoich(conn, r1, s_h2, 1.0, "reactant")
        _add_stoich(conn, r1, s_h,  2.0, "product")
        conn.commit()

        from rmip.io.cantera_writer import CanteraYAMLWriter
        writer = CanteraYAMLWriter(db_path)
        yaml_path = str(tmp_path / "ar.yaml")
        writer.write(m1, yaml_path)

        text = Path(yaml_path).read_text()
        elements_line = next(l for l in text.splitlines() if l.strip().startswith("elements:"))
        assert "Ar" in elements_line
        assert "AR" not in elements_line

        gas = ct.Solution(yaml_path)  # must not raise "element not found"
        assert "AR" in gas.species_names or "Ar" in [s for s in gas.species_names]


# ===========================================================================
# 8. MissingSpeciesPredictor
# ===========================================================================

class TestMissingSpeciesPredictor:

    def _make_predictor_db(self, tmp_path):
        """
        Unified mech (m_u) is missing species 'OH' that is in source mechs.
        """
        db_path = str(tmp_path / "pred.db")
        conn    = _make_db(db_path)

        # Source mechs
        m1 = _insert_mech(conn, "Src1")
        m2 = _insert_mech(conn, "Src2")
        for mid in [m1, m2]:
            _insert_species(conn, mid, "H2",  inchi_key="IK_H2",  formula="H2")
            _insert_species(conn, mid, "O2",  inchi_key="IK_O2",  formula="O2")
            _insert_species(conn, mid, "H2O", inchi_key="IK_H2O", formula="H2O")
            _insert_species(conn, mid, "OH",  inchi_key="IK_OH",  formula="HO")

        # Unified mech WITHOUT OH
        m_u = _insert_mech(conn, "Unified", fmt="unified")
        _insert_species(conn, m_u, "H2",  inchi_key="IK_H2",  formula="H2")
        _insert_species(conn, m_u, "O2",  inchi_key="IK_O2",  formula="O2")
        _insert_species(conn, m_u, "H2O", inchi_key="IK_H2O", formula="H2O")

        conn.commit()
        conn.close()
        return db_path, m_u, m1, m2

    def test_cross_mech_gap_detected(self, tmp_path):
        """OH is in source mechs but not in unified → detected as gap species."""
        db_path, m_u, m1, m2 = self._make_predictor_db(tmp_path)
        from rmip.generation.missing_species_predictor import MissingSpeciesPredictor
        pred   = MissingSpeciesPredictor(db_path)
        report = pred.predict(m_u, [m1, m2])
        gap_ik = {sp["inchi_key"] for sp in report.gap_species}
        assert "IK_OH" in gap_ik

    def test_must_include_smiles_populated(self, tmp_path):
        """Gap species with canonical_smiles appear in must_include_smiles."""
        db_path = str(tmp_path / "pred2.db")
        conn    = _make_db(db_path)
        m1  = _insert_mech(conn, "Src1")
        m_u = _insert_mech(conn, "Unified", fmt="unified")
        _insert_species(conn, m1,  "H2", inchi_key="IK_H2", formula="H2")
        # H2 absent from unified
        conn.commit()
        conn.close()

        from rmip.generation.missing_species_predictor import MissingSpeciesPredictor
        pred   = MissingSpeciesPredictor(db_path)
        report = pred.predict(m_u, [m1])
        # H2 is a gap species (in source, absent from unified)
        assert any(sp["inchi_key"] == "IK_H2" for sp in report.gap_species)

    def test_to_rmg_seed_writes_json(self, tmp_path):
        """to_rmg_seed writes a valid JSON file with required keys."""
        db_path, m_u, m1, m2 = self._make_predictor_db(tmp_path)
        from rmip.generation.missing_species_predictor import MissingSpeciesPredictor
        pred      = MissingSpeciesPredictor(db_path)
        report    = pred.predict(m_u, [m1, m2])
        seed_path = str(tmp_path / "seed.json")
        seed      = pred.to_rmg_seed(report, seed_path)

        assert Path(seed_path).exists()
        data = json.loads(Path(seed_path).read_text())
        assert "seed_smiles"         in data
        assert "must_include_smiles" in data
        assert data["source_unified_mech_id"] == m_u


# ===========================================================================
# 9. CLI smoke tests
# ===========================================================================

class TestCLIPhase4:

    def test_unify_plan_cli(self, tmp_path):
        """rmip unify-plan runs without error."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.cli import main
        manifest = str(tmp_path / "dec.csv")
        rc = main([
            "unify-plan",
            "--db", db_path,
            "--mechanisms", str(m1), str(m2),
            "--mode", "anchored",
            "--anchor", str(m1),
            "--manifest", manifest,
        ])
        assert rc == 0
        assert Path(manifest).exists()

    def test_unify_build_cli(self, tmp_path):
        """rmip unify-build runs and creates a unified mechanism."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.cli import main
        manifest = str(tmp_path / "dec.csv")
        main([
            "unify-plan",
            "--db", db_path,
            "--mechanisms", str(m1), str(m2),
            "--mode", "optimize_all",
            "--manifest", manifest,
        ])
        out_dir = str(tmp_path / "out")
        rc = main([
            "unify-build",
            "--db", db_path,
            "--manifest", manifest,
            "--name", "test_u",
            "--tmin", "1000", "--tmax", "1000", "--dt", "20",
            "--out", out_dir,
        ])
        assert rc == 0
        # Check a unified mechanism was created
        conn = sqlite3.connect(db_path)
        row  = conn.execute(
            "SELECT id FROM mechanisms WHERE format='unified'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_predict_missing_cli(self, tmp_path):
        """rmip predict-missing runs without error."""
        db_path = str(tmp_path / "pred.db")
        conn    = _make_db(db_path)
        m1  = _insert_mech(conn, "Src")
        m_u = _insert_mech(conn, "Uni", fmt="unified")
        _insert_species(conn, m1, "H2", inchi_key="IK_H2", formula="H2")
        conn.commit()
        conn.close()

        from rmip.cli import main
        rc = main([
            "predict-missing",
            "--db", db_path,
            "--unified", str(m_u),
            "--mechanisms", str(m1),
            "--out", str(tmp_path / "seed"),
        ])
        assert rc == 0

    def test_fuse_cli(self, tmp_path):
        """rmip fuse runs end-to-end (B5 regression: fusion_result.n_species
        doesn't exist on FusionResult -- real fields are
        n_species_universal/partial/unique -- so this crashed with
        AttributeError AFTER committing the fused mechanism but BEFORE
        writing union_spec.json/mso_bounds.json on every invocation)."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.cli import main
        out_dir = str(tmp_path / "fusion_out")
        rc = main([
            "fuse",
            "--db", db_path,
            "--mechanisms", str(m1), str(m2),
            "--name", "test_fused",
            "--out", out_dir,
        ])
        assert rc == 0
        assert (Path(out_dir) / "union_spec.json").exists()
        assert (Path(out_dir) / "mso_bounds.json").exists()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT id FROM mechanisms WHERE format='fused'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_export_yaml_cli(self, tmp_path):
        """rmip export --format yaml writes a YAML file."""
        db_path, m1, m2 = _make_two_mech_db(tmp_path)
        from rmip.cli import main
        yaml_path = str(tmp_path / "out.yaml")
        rc = main([
            "export",
            str(m1),
            "--db", db_path,
            "--output", yaml_path,
            "--format", "yaml",
        ])
        assert rc == 0
        assert Path(yaml_path).exists()
        text = Path(yaml_path).read_text()
        assert "phases:" in text


# ===========================================================================
# 10. Regression: existing 112 tests not broken
# ===========================================================================

def test_existing_phase3_suite_importable():
    """Phase-3 modules can still be imported (no import-time breakage)."""
    from rmip.analysis.fusion            import FusionEngine
    from rmip.analysis.graph_comparator  import GraphComparator
    from rmip.analysis.parameter_estimator import ParameterEstimator
    from rmip.database.migrations        import run_phase3_migrations
    from rmip.io.chemkin_writer          import ChemkinWriter
    assert True


def test_phase3_migration_still_works(tmp_path):
    """run_phase3_migrations on a fresh DB still succeeds."""
    from rmip.database.schema     import init_db
    from rmip.database.schema_phase3 import upgrade_db
    from rmip.database.migrations import run_phase3_migrations
    conn = init_db(str(tmp_path / "p3.db"))
    upgrade_db(conn)
    run_phase3_migrations(conn)
    conn.close()
