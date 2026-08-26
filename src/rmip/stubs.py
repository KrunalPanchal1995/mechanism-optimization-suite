"""
rmip/stubs.py
=============
Full I/O contract stubs for Phase 4 → 6 modules.

Each class has:
  - __init__ signature
  - Fully documented method signatures
  - Body: ``raise NotImplementedError(...)``
  - Docstring describing the data contracts

These stubs are import-safe and can be used by tests to verify the interface.

Phase 4:  M5  PathwayCompleter    (RMG + Steiner tree completion)
          M7  FusionEngine        (build merged CHEMKIN/YAML mechanism)
          M13 UQFusion            (MUQ-SAC covariance-aware fusion)
Phase 5:  M9  DoEFramework        (Design of Experiments + MSO hand-off)
          M10 AIRecommender       (LLM-assisted expert advice)
Phase 6:  M11 ChangeLog           (version-controlled edit log)
          M12 ReportEngine        (PDF/HTML/Excel report generation)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from rmip.types import (
    GraphComparisonResult,
    MSOParameters,
    SimCondition,
    SparsityReport,
    UnionGraphSpec,
)


# ===========================================================================
# Phase 4 — M7: FusionEngine
# ===========================================================================

class FusionEngine:
    """
    Module 7: Merge N mechanisms into a single fused mechanism.

    Input
    -----
    spec : UnionGraphSpec
        Produced by ``GraphComparator.build_union_spec()``.
    db_path : str
        Project SQLite database (for reading raw species/reaction rows).

    Algorithm
    ---------
    1. VERBATIM bucket  → copy all data verbatim from the single-source mechanism.
    2. COPY bucket      → copy from any source (all identical); use mechanism with
                          most complete thermo data.
    3. FUSE bucket      → use ``spec.fuse_params[canonical_key]`` for mean A/n/Ea.
                          If DoEFramework or UQFusion has run, use their optimised
                          values instead (stored in the ``fuse_overrides`` dict).
    4. Write merged CHEMKIN ``.inp`` + ``.dat`` files.
    5. Optionally write Cantera ``.yaml`` via Cantera's built-in writer.

    The fused mechanism is also stored as a new row in the ``mechanisms`` table
    (format = 'chemkin', name = provided by caller).

    Output
    ------
    int
        Mechanism DB id of the newly created fused mechanism.

    Implementation note
    --------------------
    The DB-row-synthesis / VERBATIM-COPY-FUSE logic already exists, fully
    implemented and tested, in ``rmip.analysis.fusion.FusionEngine`` (a
    *different* class, same name, different module — it consumes a flat
    ``GraphComparisonResult`` rather than the bucketed ``UnionGraphSpec`` and
    has no ``fuse_overrides``/file-writing support). This class is a thin
    adapter: it reconstructs an equivalent ``GraphComparisonResult`` from
    ``spec`` (lossless — ``build_union_spec()`` only partitions the same
    ``ReactionClassification``/``SpeciesClassification`` objects into bucket
    dicts, never transforms them), delegates the actual fusion to
    ``analysis.fusion.FusionEngine``, applies any ``fuse_overrides`` as a
    post-insert UPDATE, then adds the two things the core engine doesn't do:
    writing CHEMKIN ``.inp``/``.dat`` files (via ``ChemkinWriter``, already
    used identically by ``zerork_bridge.py``) and, optionally, a Cantera YAML
    (via ``ChemkinWriter.convert_to_cantera_yaml()``'s ``ck2yaml`` wrapper).
    """

    def __init__(
        self,
        spec:    UnionGraphSpec,
        db_path: str,
    ) -> None:
        self._spec    = spec
        self._db_path = db_path

    def fuse(
        self,
        output_name:    str,
        output_dir:     str  = ".",
        fuse_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        write_cantera:  bool = False,
        priority_order: Optional[List[int]] = None,
    ) -> int:
        """
        Build the fused mechanism and write output files.

        Parameters
        ----------
        output_name : str
            Name for the merged mechanism (stored in DB + used for file names).
        output_dir : str
            Directory where .inp / .dat / .yaml files are written.
        fuse_overrides : dict, optional
            ``{canonical_key: {"A_si": float, "n": float, "Ea_si": float}}``
            Optimised values from DoEFramework / UQFusion that override the
            raw geometric mean in ``spec.fuse_params``. Only meaningful for
            FUSE-bucket reactions (VERBATIM/COPY reactions have no estimated
            mean to override) but applied to any ``canonical_key`` present in
            the fused mechanism, regardless of bucket — the caller is
            responsible for only passing FUSE keys. Ignored when
            ``priority_order`` is given (priority-pick reactions have no
            estimated mean to override either).
        write_cantera : bool
            If True, also write a Cantera YAML file via ``ck2yaml``.
        priority_order : list[int], optional
            Feature gap closed 2026-07-17 (bug.md): this GUI-facing adapter
            previously had no way to invoke priority-pick fusion at all,
            even though ``analysis.fusion.FusionEngine.fuse`` (the class
            this method delegates to) has supported it since the 2026-07-14
            fusion-engine stress test -- the only way to use it was the
            separate, CLI-only ``fuse_kld.py``-style driver script. Passed
            straight through to the core engine: for every reaction shared
            across mechanisms, copies A/n/Ea verbatim from the
            highest-priority mechanism ID in this list that's actually
            present, instead of averaging (FUSE bucket). See
            ``analysis.fusion.FusionEngine.fuse``'s own docstring for the
            full algorithm. Mutually exclusive in effect with
            ``fuse_overrides`` (there's nothing to override when nothing
            was averaged).

        Returns
        -------
        int
            DB id of the stored fused mechanism.
        """
        import sqlite3
        from pathlib import Path

        from rmip.analysis.fusion import FusionEngine as _CoreFusionEngine
        from rmip.io.chemkin_writer import ChemkinWriter

        # Reconstruct a GraphComparisonResult from the bucketed spec -- lossless,
        # since build_union_spec() only partitions the same classification
        # objects into verbatim_reactions/copy_reactions/fuse_reactions without
        # transforming them.
        reaction_classifications = (
            list(self._spec.verbatim_reactions.values())
            + list(self._spec.copy_reactions.values())
            + list(self._spec.fuse_reactions.values())
        )
        species_classifications = list(self._spec.species_map.values())
        comparison_result = GraphComparisonResult(
            session_id=self._spec.comparison_session_id,
            mechanism_ids=self._spec.mechanism_ids,
            species_classifications=species_classifications,
            reaction_classifications=reaction_classifications,
        )

        core = _CoreFusionEngine(self._db_path)
        core_result = core.fuse(comparison_result, name=output_name,
                                 priority_order=priority_order)
        fused_mech_id = core_result.fused_mechanism_id

        if fuse_overrides:
            self._apply_fuse_overrides(fused_mech_id, fuse_overrides)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        chemkin_path = str(out_dir / f"{output_name}.inp")
        thermo_path  = str(out_dir / f"{output_name}.dat")

        writer = ChemkinWriter(self._db_path, use_original_units=True)
        writer.write(
            fused_mech_id, chemkin_path,
            thermo_path=thermo_path, include_thermo_in_mech=True,
        )

        if write_cantera:
            # thermo is already embedded in chemkin_path (include_thermo_in_mech=
            # True above) -- passing the separate .dat file too as well would
            # hand ck2yaml two THERMO blocks and it rejects the input.
            writer.convert_to_cantera_yaml(
                chemkin_path, output_yaml=str(out_dir / f"{output_name}.yaml"),
            )

        return fused_mech_id

    def _apply_fuse_overrides(
        self, mech_id: int, fuse_overrides: Dict[str, Dict[str, float]],
    ) -> None:
        """UPDATE the newly-inserted reaction rows with optimiser-supplied
        A/n/Ea, overriding the core engine's geometric/arithmetic-mean
        estimate. ``A_orig``/``Ea_orig``/``A_units``/``Ea_units`` are cleared
        (NULL) rather than mirroring ``A_si``/``Ea_si`` -- an optimiser-fit
        value has no corresponding original-file-units representation, and
        ChemkinWriter's ``use_original_units=True`` export path takes a
        non-NULL ``A_orig`` completely literally (it never actually checks
        ``A_units``). Mirroring SI values into A_orig under a claimed "SI"
        A_units used to make every such export silently write the raw SI
        number as if it were already CGS -- the same 10^6-too-small bug
        found in ``analysis.fusion.FusionEngine`` 2026-07-17 (see bug.md).
        NULL A_orig makes the writer correctly fall back to A_si with a
        proper SI->CGS conversion instead."""
        import sqlite3

        with sqlite3.connect(self._db_path) as conn:
            for canonical_key, params in fuse_overrides.items():
                conn.execute(
                    """
                    UPDATE reactions
                       SET A_si=?, A_orig=NULL, A_units=NULL,
                           n=?, Ea_si=?, Ea_orig=NULL, Ea_units=NULL
                     WHERE mechanism_id=? AND canonical_key=?
                    """,
                    (
                        params["A_si"], params["n"], params["Ea_si"],
                        mech_id, canonical_key,
                    ),
                )
            conn.commit()

    def validate_element_balance(self, mech_id: int) -> Dict[str, Any]:
        """
        Check that every reaction in the fused mechanism is element-balanced.

        Uses ``species.formula`` (a plain chemical formula string, e.g.
        ``"H2O"``) parsed into an element→count dict, weighted by each
        reaction's ``stoichiometry`` rows. Third-body ``M`` never appears as
        a stoichiometry row in this schema (confirmed: third-body reactions
        record their participation via ``has_third_body``/
        ``third_body_efficiencies`` instead), so no special-casing is needed.

        Returns
        -------
        dict
            ``{"balanced": bool, "violations": [formula_string, ...]}``
        """
        import re
        import sqlite3
        from collections import Counter

        def _parse_formula(formula: Optional[str]) -> Counter:
            counts: Counter = Counter()
            for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
                if not elem:
                    continue
                counts[elem] += int(num) if num else 1
            return counts

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            species_formula = {
                row["id"]: _parse_formula(row["formula"])
                for row in conn.execute(
                    "SELECT id, formula FROM species WHERE mechanism_id=?", (mech_id,)
                )
            }
            reactions = conn.execute(
                "SELECT id, formula_string FROM reactions WHERE mechanism_id=?", (mech_id,)
            ).fetchall()

            violations: List[str] = []
            for rxn in reactions:
                reactant_elems: Counter = Counter()
                product_elems:  Counter = Counter()
                for srow in conn.execute(
                    "SELECT species_id, stoich_coeff, role FROM stoichiometry WHERE reaction_id=?",
                    (rxn["id"],),
                ):
                    target = reactant_elems if srow["role"] == "reactant" else product_elems
                    for el, cnt in species_formula.get(srow["species_id"], Counter()).items():
                        target[el] += cnt * srow["stoich_coeff"]

                all_elems = set(reactant_elems) | set(product_elems)
                if any(
                    abs(reactant_elems.get(el, 0.0) - product_elems.get(el, 0.0)) > 1e-6
                    for el in all_elems
                ):
                    violations.append(rxn["formula_string"])

        return {"balanced": len(violations) == 0, "violations": violations}


# ===========================================================================
# Phase 4 — M5: PathwayCompleter
# ===========================================================================

class PathwayCompleter:
    """
    Module 5: Targeted pathway completion via RMG-Py + Steiner tree.

    Workflow
    --------
    1. Load the fused mechanism produced by FusionEngine.
    2. Call RMG-Py programmatically to generate a complete species list for
       the fuel/oxidizer system:
         ``rmgpy.rmg.main.RMG.generate_species_tree(fuel_smiles, ...)``
    3. Identify species present in the RMG tree but absent from the fused mechanism
       (matched by InChI key).
    4. Find minimum-cost subgraph (Steiner tree approximation) connecting the
       missing species to the existing reaction network.
    5. Append new species + reactions to the fused mechanism CHEMKIN file.
    6. Store new species/reactions in the DB as an extension of the fused mechanism.

    Input
    -----
    fused_mech_id : int
        DB id of the mechanism produced by FusionEngine.
    db_path : str
        Project SQLite database.

    Output
    ------
    int
        DB id of the completed mechanism (a new row in mechanisms table).
    """

    def __init__(self, fused_mech_id: int, db_path: str) -> None:
        self._mech_id = fused_mech_id
        self._db_path = db_path

    def identify_missing_species(
        self,
        fuel_smiles: str,
        oxidizer_smiles: str = "O=O",
        max_carbon_number: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Return species in the RMG tree but absent from the fused mechanism.

        Returns
        -------
        list[dict]
            Each entry: ``{"smiles": str, "inchi_key": str, "formula": str,
                           "rmg_label": str}``
        """
        raise NotImplementedError("M5 Phase 4 pending — requires RMG-Py install.")

    def complete(
        self,
        fuel_smiles:       str,
        output_name:       str = "",
        oxidizer_smiles:   str = "O=O",
        max_carbon_number: int = 10,
        steiner_weight:    str = "sensitivity",  # 'sensitivity'|'uniform'
    ) -> int:
        """
        Run the full completion workflow.

        Returns
        -------
        int
            DB id of the completed mechanism.
        """
        raise NotImplementedError("M5 Phase 4 pending.")


# ===========================================================================
# Phase 4 — M13: UQFusion
# ===========================================================================

class UQFusion:
    """
    Module 13: Uncertainty-Aware Mechanism Fusion using MUQ-SAC covariances.

    This module consumes the ``uq_data`` dict produced by KP's MUQ-SAC
    adapter and replaces the simple geometric-mean rate parameters in the
    FUSE bucket with covariance-weighted estimates.

    The merged covariance matrix for each shared reaction is computed as
    the precision-weighted mean:
        Σ_fused⁻¹ = Σᵢ Σᵢ⁻¹
        μ_fused   = Σ_fused · Σᵢ Σᵢ⁻¹ μᵢ

    Input
    -----
    spec    : UnionGraphSpec
    uq_data : dict
        Keyed by reaction id (int); each value is a dict carrying, per
        contributing mechanism, the Arrhenius mean vector ``μᵢ`` (A_si,
        n, Ea_si) and its covariance ``Σᵢ`` from that mechanism's MUQ-SAC
        run — i.e. the per-reaction inputs to the precision-weighted-mean
        formula above. This adapter (turning a raw MUQ-SAC run into that
        per-reaction dict) does not exist yet; that is what "Phase 4
        pending" below refers to.
    db_path : str

    Output
    ------
    dict
        ``fuse_overrides`` dict ready to pass to ``FusionEngine.fuse()``.
    """

    def __init__(
        self,
        spec:    UnionGraphSpec,
        uq_data: Dict[int, Dict[str, Any]],
        db_path: str,
    ) -> None:
        self._spec    = spec
        self._uq_data = uq_data
        self._db_path = db_path

    def compute_fuse_overrides(self) -> Dict[str, Dict[str, float]]:
        """
        Compute precision-weighted mean Arrhenius parameters for every FUSE reaction.

        Returns
        -------
        dict
            ``{canonical_key: {"A_si": float, "n": float, "Ea_si": float,
                                "cov_fused": [[float]], "confidence": float}}``
        """
        raise NotImplementedError(
            "M13 UQFusion.compute_fuse_overrides() — Phase 4 pending. "
            "Requires uq_data from KP's MUQ-SAC covariance adapter (see "
            "the UQFusion class docstring for the expected uq_data shape); "
            "that adapter does not exist yet."
        )

    def export_covariance_report(self, path: str) -> None:
        """Export per-reaction covariance comparison to JSON."""
        raise NotImplementedError("Phase 4 pending.")


# ===========================================================================
# Phase 5 — M9: DoEFramework
# ===========================================================================

class DoEFramework:
    """
    Module 9: Design of Experiments → MSO parameter preparation.

    Consumes the ``fuse_params`` block from ``UnionGraphSpec`` (or overrides
    from ``UQFusion``) and packages them into ``MSOParameters`` ready for
    KP's ``OptimizationTool_v4.py``.

    Optionally runs a Latin-Hypercube or Sobol design to identify the most
    influential parameters before handing off to MSO.

    Input
    -----
    spec           : UnionGraphSpec
    exp_dataset_id : int   (DB id from exp-load / ExpDataReader)
    db_path        : str

    Output
    ------
    MSOParameters  — serialisable to JSON via .save_json()
    """

    def __init__(
        self,
        spec:           UnionGraphSpec,
        exp_dataset_id: int,
        db_path:        str,
    ) -> None:
        self._spec     = spec
        self._ds_id    = exp_dataset_id
        self._db_path  = db_path

    def prepare_mso_input(
        self,
        fuel_name:          str,
        session_id:         int,
        n_screening_samples: int = 256,
        screening_method:   str = "sobol",  # 'sobol'|'lhs'|'none'
        regime_filter:      Optional[str] = None,  # 'LTC'|'NTC'|'HTC'|None
        condition:          Optional[SimCondition] = None,
        zerork_install_dir: str = "/home/krithika/zero-rk/build2/inst_dir",
    ) -> MSOParameters:
        """
        Build the MSOParameters object for KP's MSO framework.

        Screening (optional)
        --------------------
        If ``n_screening_samples > 0`` and ``screening_method != "none"``,
        runs a real Sobol/Morris screening via :meth:`run_screening`
        (Zero-RK GSA, ZERORK_GSA_PLAN.md Phase 3 — NOT "lhs"; see that
        parameter's note below) and screens out FUSE parameters with
        ``|S1| < 0.01`` (Sobol) or ``|mu_star| < 0.01`` (Morris) from the
        MSO search space.

        Parameter bounds
        ----------------
        From ``spec.fuse_params[canonical_key]``:
            lb = (min_A_si, min_n, min_Ea_si)
            ub = (max_A_si, max_n, max_Ea_si)
            x0 = (mean_A_si, mean_n, mean_Ea_si)

        Parameters
        ----------
        condition : SimCondition, optional
            Required if ``screening_method != "none"`` — ``exp_data_points``
            only stores ``T_K``/``P_atm``/``phi`` per point, never a
            Zero-RK-consumable fuel/oxidizer composition string, so it
            cannot be derived automatically from ``exp_dataset_id`` alone.
        regime_filter : str, optional
            Accepted for interface compatibility but NOT applied — no
            regime classification is stored on ``exp_data_points`` rows in
            this schema (no ``regime``/``ntc``-style column exists to filter
            on). Passed through into ``metadata`` so a caller can see it was
            requested but ignored, rather than silently dropped.
        screening_method : str
            ``"sobol"`` or ``"morris"`` (delegates to :meth:`run_screening`,
            i.e. Zero-RK GSA) or ``"none"`` to skip screening entirely and
            use every FUSE reaction. ``"lhs"`` from the original docstring
            is not implemented — :meth:`run_screening`/``ZerorkGSARunner``
            only support Sobol/Morris (SALib's samplers, per
            ``ZERORK_GSA_PLAN.md``'s own design commitment) — passing
            ``"lhs"`` raises ``ValueError``.

        Returns
        -------
        MSOParameters
        """
        import sqlite3

        if screening_method not in ("sobol", "morris", "none"):
            raise ValueError(
                f"screening_method={screening_method!r} not supported — "
                'use "sobol", "morris", or "none" (see this method\'s '
                "docstring: \"lhs\" was never implemented).",
            )

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT T_K, P_atm, phi FROM exp_data_points "
                "WHERE dataset_id=? AND T_K IS NOT NULL",
                (self._ds_id,),
            ).fetchall()
        target_conditions = [
            {"T_K": t, "P_atm": p, "phi": phi} for t, p, phi in rows
        ]

        screening_indices: Dict[str, Any] = {}
        if n_screening_samples > 0 and screening_method != "none":
            if condition is None:
                raise ValueError(
                    "prepare_mso_input(): screening_method != 'none' requires "
                    "an explicit `condition` (SimCondition) — see this "
                    "parameter's docstring note for why it can't be derived "
                    "from exp_dataset_id automatically.",
                )
            screening_indices = self.run_screening(
                n_samples=n_screening_samples, method=screening_method,
                condition=condition, zerork_install_dir=zerork_install_dir,
            )

        param_blocks: List[Dict[str, Any]] = []
        # M9: fuse_params/fuse_reactions are keyed on (canonical_key,
        # occurrence_index) tuples now (so CHEMKIN DUPLICATE reactions
        # don't collide) -- unpack to get the real string canonical_key for
        # screening_indices lookups (screening_indices is keyed by plain
        # canonical_key, from run_screening()'s per-reaction Sobol/Morris
        # indices) and for the output param block's own "canonical_key"
        # field, which downstream MSO tooling expects to be a string.
        for (canonical_key, occ), params in self._spec.fuse_params.items():
            if screening_indices:
                idx = screening_indices.get(canonical_key, {})
                coeff = idx.get("S1", idx.get("mu_star"))
                if coeff is not None and abs(coeff) < 0.01:
                    continue  # screened out — not MSO-worthy
            rc = self._spec.fuse_reactions.get((canonical_key, occ))
            formula_string = rc.formula_string if rc else canonical_key
            param_blocks.append({
                "canonical_key":  canonical_key,
                "formula_string": formula_string,
                "params_mean": [params["mean_A_si"], params["mean_n"], params["mean_Ea_si"]],
                "params_lb":   [params["unc_A_si"][0], params["unc_n"][0], params["unc_Ea_si"][0]],
                "params_ub":   [params["unc_A_si"][1], params["unc_n"][1], params["unc_Ea_si"][1]],
                "param_names": ["A_si", "n", "Ea_si"],
                "units":       ["m^3/mol/s", "1", "J/mol"],
            })

        return MSOParameters(
            session_id=session_id,
            fuel_name=fuel_name,
            param_blocks=param_blocks,
            target_conditions=target_conditions,
            metadata={
                "regime_filter": regime_filter,  # requested but not applied, see docstring
                "screening_method": screening_method,
                "n_screening_samples": n_screening_samples,
                "n_fuse_reactions_total": len(self._spec.fuse_reactions),
                "n_fuse_reactions_after_screening": len(param_blocks),
            },
        )

    def run_screening(
        self,
        n_samples: int = 256,
        method:    str = "sobol",
        condition: Optional[SimCondition] = None,
        zerork_install_dir: str = "/home/krithika/zero-rk/build2/inst_dir",
    ) -> Dict[str, Any]:
        """
        Run screening design and return sensitivity indices per parameter.

        Delegates to ``rmip.simulation.zerork_gsa.ZerorkGSARunner``
        (ZERORK_GSA_PLAN.md Phase 3) over a mechanism built from ``self._spec``
        via :class:`FusionEngine` (M7) — only FUSE-bucket reactions are
        screened (VERBATIM/COPY reactions have a single fixed value, nothing
        to screen).

        Parameters
        ----------
        condition : SimCondition, required
            Zero-RK needs a real ``T_K``/``P_atm``/``phi``/``fuel``/
            ``oxidizer`` to run — see :meth:`prepare_mso_input`'s note on
            why this can't be derived from ``exp_dataset_id`` alone.

        Returns
        -------
        dict
            ``{canonical_key: {"S1": float, "S1_conf": float, ...}}``
            (SALib Morris/Sobol indices)
        """
        import sqlite3
        import tempfile

        if condition is None:
            raise ValueError(
                "DoEFramework.run_screening() needs a SimCondition (T_K/P_atm/"
                "phi/fuel/oxidizer) to actually run Zero-RK — it can't be "
                "derived from exp_dataset_id alone (exp_data_points stores "
                "T_K/P_atm/phi but never a fuel/oxidizer composition "
                "string). Pass one explicitly.",
            )
        if not self._spec.fuse_reactions:
            return {}

        from rmip.simulation.zerork_gsa import ZerorkGSARunner
        from rmip.types import GSARunConfig

        work_dir = tempfile.mkdtemp(prefix=f"doe_screening_{self._ds_id}_")
        fusion_engine = FusionEngine(self._spec, self._db_path)
        fused_mech_id = fusion_engine.fuse(
            f"doe_screening_ds{self._ds_id}", output_dir=work_dir,
        )

        # M9: fuse_reactions is keyed on (canonical_key, occurrence_index)
        # tuples now (so CHEMKIN DUPLICATE reactions don't collide) -- this
        # SQL IN-clause only needs the canonical_key strings, deduplicated.
        fuse_keys = sorted({ck for ck, _occ in self._spec.fuse_reactions.keys()})
        with sqlite3.connect(self._db_path) as conn:
            placeholders = ",".join("?" * len(fuse_keys))
            rows = conn.execute(
                f"SELECT id, canonical_key FROM reactions "
                f"WHERE mechanism_id=? AND canonical_key IN ({placeholders})",
                [fused_mech_id] + fuse_keys,
            ).fetchall()
        # Preserve DB row order -- ZerorkGSARunner.run_screening() returns
        # SobolResult.parameter_names in the same order as the reaction_ids
        # list it's given, so positional zip below is safe.
        reaction_ids = [rid for rid, _ in rows]
        canonical_key_by_id = dict(rows)

        runner = ZerorkGSARunner(self._db_path, zerork_install_dir)
        config = GSARunConfig(
            mechanism_id=fused_mech_id, method=method, n_samples=n_samples,
            reaction_ids=reaction_ids,
            T_K=condition.T_K, P_atm=condition.P_atm, phi=condition.phi,
            fuel=condition.fuel, oxidizer=condition.oxidizer,
        )
        result = runner.run_screening(config, work_dir=work_dir)
        screening_dict = result.to_screening_dict()  # keyed by "label:formula"

        return {
            canonical_key_by_id[rid]: screening_dict[name]
            for rid, name in zip(reaction_ids, result.parameter_names)
        }


# ===========================================================================
# Phase 5 — M10: AIRecommender
# ===========================================================================

class AIRecommender:
    """
    Module 10: AI-assisted expert recommendation.

    Uses an LLM (via the Anthropic API) to:
    1. Summarise which mechanisms performed best in which regimes.
    2. Recommend targeted pathway additions based on sparsity analysis.
    3. Flag thermodynamically suspicious rate constants.
    4. Suggest literature searches for divergent rate parameters.

    Input
    -----
    benchmark_scores : pd.DataFrame  (from MechanismBenchmarker.run_and_score())
    graph_comparison : GraphComparisonResult
    sparsity_reports : list[SparsityReport]

    Output
    ------
    dict  with keys: ``recommendations``, ``regime_winner``, ``risk_flags``
    """

    def __init__(
        self,
        benchmark_scores,  # pd.DataFrame
        graph_comparison:  GraphComparisonResult,
        sparsity_reports:  List[SparsityReport],
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        raise NotImplementedError(
            "M10 AIRecommender.__init__() — Phase 5 pending. "
            "Requires Anthropic API key and benchmark data."
        )

    def recommend(self, fuel_name: str, target_conditions: List[Dict]) -> Dict[str, Any]:
        """
        Generate targeted recommendations.

        Returns
        -------
        dict
            ``{"recommendations": [str], "regime_winner": {regime: mech_id},
               "risk_flags": [{reaction, risk_type, explanation}]}``
        """
        raise NotImplementedError("Phase 5 pending.")


# ===========================================================================
# Phase 6 — M11: ChangeLog
# ===========================================================================

class ChangeLog:
    """
    Module 11: Automated version-controlled edit log.

    Records every modification to a mechanism (parameter changes from fusion,
    pathway additions from PathwayCompleter, optimisation updates from MSO)
    as a JSON diff stored in the DB and optionally committed to a Git repo.

    Input
    -----
    db_path  : str
    git_repo : str | None   path to Git repository root

    Output
    ------
    str   — Markdown-formatted change summary.
    """

    def __init__(self, db_path: str, git_repo: Optional[str] = None) -> None:
        raise NotImplementedError("M11 ChangeLog — Phase 6 pending.")

    def record_fusion(
        self,
        source_mech_ids: List[int],
        fused_mech_id:   int,
        spec:            UnionGraphSpec,
    ) -> int:
        """Record a fusion event; return change log entry id."""
        raise NotImplementedError("Phase 6 pending.")

    def record_optimisation(
        self,
        mech_id:     int,
        mso_params:  MSOParameters,
        old_values:  Dict[str, Any],
        new_values:  Dict[str, Any],
    ) -> int:
        """Record an MSO optimisation event."""
        raise NotImplementedError("Phase 6 pending.")

    def to_markdown(self, mech_id: int) -> str:
        """Return a Markdown change summary for a mechanism."""
        raise NotImplementedError("Phase 6 pending.")


# ===========================================================================
# Phase 6 — M12: ReportEngine
# ===========================================================================

class ReportEngine:
    """
    Module 12: Automated reporting.

    Generates a full PDF/HTML/Excel report combining:
    - Mechanism summary tables
    - Benchmark performance plots (IDT, LFS)
    - Sparsity analysis figures
    - Graph comparison summary
    - Kinetic rate ratio plots for divergent reactions
    - DoE/MSO convergence plots
    - Change log summary

    Input
    -----
    db_path : str

    Dependencies
    ------------
    ``jinja2``, ``weasyprint`` (for PDF), ``openpyxl`` (for Excel),
    ``matplotlib`` (for figures).
    """

    def __init__(self, db_path: str) -> None:
        raise NotImplementedError("M12 ReportEngine — Phase 6 pending.")

    def generate_pdf(
        self,
        mechanism_ids: List[int],
        output_path:   str,
        title:         str = "RMIP Mechanism Analysis Report",
    ) -> None:
        """Generate a PDF report."""
        raise NotImplementedError("Phase 6 pending.")

    def generate_excel(
        self,
        mechanism_ids: List[int],
        output_path:   str,
    ) -> None:
        """Generate an Excel workbook with one sheet per analysis section."""
        raise NotImplementedError("Phase 6 pending.")

    def generate_html(
        self,
        mechanism_ids: List[int],
        output_path:   str,
    ) -> None:
        """Generate a self-contained HTML report."""
        raise NotImplementedError("Phase 6 pending.")
