"""
rmip/core/parser.py
===================
Module 1: Mechanism Parsing + Canonicalization orchestrator.

Parses a CHEMKIN or Cantera YAML mechanism file, resolves species SMILES
from the glossary, canonicalises all identities via RDKit, and stores
everything in the RMIP SQLite database.

Usage::

    from rmip.config    import RMIPConfig
    from rmip.core.parser import MechanismParser

    cfg     = RMIPConfig(db_path="project.db")
    parser  = MechanismParser(cfg)
    mech_id = parser.parse(
        mech_path     = "n_heptane.inp",
        glossary_path = "species_smiles.csv",
        thermo_path   = "thermo.dat",
        name          = "Curran n-heptane 2000",
    )
    summary = parser.get_mechanism_summary(mech_id)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rmip.config            import RMIPConfig
from rmip.core.canonicalize import Canonicalizer, smiles_to_formula, smiles_to_mw
from rmip.database.schema   import (
    init_db, insert_mechanism, update_mechanism_counts,
    insert_species, insert_reaction, insert_stoichiometry,
    insert_third_body_efficiency, get_mechanism_summary,
    BatchInserter,
)
from rmip.database.schema_phase3 import upgrade_db
from rmip.io.glossary_reader import load_glossary, lookup_smiles, load_variants

log = logging.getLogger("rmip.core.parser")


class MechanismParser:
    """
    Parses and stores a combustion kinetic mechanism.

    Parameters
    ----------
    cfg : RMIPConfig
        Project configuration.
    """

    def __init__(self, cfg: RMIPConfig) -> None:
        self.cfg    = cfg
        self._conn  = init_db(cfg.db_path)
        upgrade_db(self._conn)
        self._canon = Canonicalizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self,
        mech_path:      str,
        glossary_path:  Optional[str] = None,
        thermo_path:    Optional[str] = None,
        name:           str           = "",
        description:    Optional[str] = None,
        format:         Optional[str] = None,
        dict_path:      Optional[str] = None,
        check_duplicates: bool         = False,
        variants_path:  Optional[str] = None,
    ) -> int:
        """
        Parse and store a mechanism.

        Parameters
        ----------
        mech_path : str
            CHEMKIN .inp or Cantera .yaml / .cti file.
        glossary_path : str, optional
            CSV/YAML file mapping species names to SMILES.
        thermo_path : str, optional
            Separate thermo data file (CHEMKIN only).
        name : str
            Human-readable name stored in the DB.
        format : str, optional
            'chemkin' | 'cantera_yaml' | 'cti'.  Auto-detected if None.
        dict_path : str, optional
            RMG species dictionary .txt file (CHEMKIN only, for resonance structures).
        check_duplicates : bool
            Flag duplicate reactions during CHEMKIN loading.
        variants_path : str, optional
            Phase-5D species-variant override CSV.  Maps variant species names
            (e.g. OHEX) to their canonical equivalents (e.g. OH) so they share
            the same InChIKey in the union graph.

        Returns
        -------
        int
            The new mechanism's DB id.
        """
        fmt = format or self._detect_format(mech_path)
        log.info("Parsing mechanism: %s (format=%s)", mech_path, fmt)

        # Load glossary
        glossary: Dict[str, str] = {}
        if glossary_path:
            glossary = load_glossary(glossary_path)

        # Load variant overrides (Phase 5D)
        variants: Dict[str, Dict[str, str]] = {}
        if variants_path:
            from rmip.database.migrations import run_phase5d_migrations
            run_phase5d_migrations(self._conn)
            variants = load_variants(variants_path)

        # Read mechanism
        if fmt == "chemkin":
            from rmip.io.chemkin_reader import load_chemkin
            species_list, rxn_list = load_chemkin(
                mech_path, thermo_path,
                dict_path=dict_path,
                check_duplicates=check_duplicates,
            )
        elif fmt in ("cantera_yaml", "cti"):
            from rmip.io.cantera_reader import load_cantera_yaml
            species_list, rxn_list = load_cantera_yaml(mech_path)
        else:
            raise ValueError(f"Unsupported mechanism format: '{fmt}'")

        # Insert mechanism row
        metadata: Dict[str, Any] = {}
        if thermo_path:
            metadata["thermo_filepath"] = thermo_path
        mech_id = insert_mechanism(
            self._conn, name=name or Path(mech_path).stem,
            format=fmt, filepath=mech_path,
            description=description, metadata=metadata,
        )

        # Insert species (with optional Phase-5D variant overrides)
        sp_name_to_id = self._insert_species(mech_id, species_list, glossary, variants)

        # Insert reactions
        self._insert_reactions(mech_id, rxn_list, sp_name_to_id)

        # Assign sequential R-labels (R1, R2, …) in file order
        self._assign_reaction_labels(mech_id)

        # Update counts
        n_sp = self._conn.execute(
            "SELECT COUNT(*) FROM species WHERE mechanism_id=?", (mech_id,)
        ).fetchone()[0]
        n_rx = self._conn.execute(
            "SELECT COUNT(*) FROM reactions WHERE mechanism_id=?", (mech_id,)
        ).fetchone()[0]
        update_mechanism_counts(self._conn, mech_id, n_sp, n_rx)

        log.info("Stored mechanism id=%d: %d species, %d reactions", mech_id, n_sp, n_rx)
        return mech_id

    def get_mechanism_summary(self, mech_id: int) -> Dict[str, Any]:
        return get_mechanism_summary(self._conn, mech_id)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assign_reaction_labels(self, mech_id: int) -> None:
        """Assign R1, R2, … labels in insertion (file) order."""
        rows = self._conn.execute(
            "SELECT id FROM reactions WHERE mechanism_id=? ORDER BY id",
            (mech_id,),
        ).fetchall()
        self._conn.executemany(
            "UPDATE reactions SET reaction_label=? WHERE id=?",
            [(f"R{i}", row[0]) for i, row in enumerate(rows, start=1)],
        )
        self._conn.commit()

    @staticmethod
    def _detect_format(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in (".yaml", ".yml"):
            return "cantera_yaml"
        if suffix == ".cti":
            return "cti"
        return "chemkin"  # default for .inp, .dat, .txt, etc.

    def _insert_species(
        self,
        mech_id:      int,
        species_list: list,
        glossary:     Dict[str, str],
        variants:     Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, int]:
        """Insert all species; return {name: db_id} mapping."""
        # Build a name→(can_smiles, inchi_key) cache so variant lookups can
        # reference canonical species that may appear later in the list.
        # Two-pass: first resolve SMILES for all non-variant species.
        resolved: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}
        for sp in species_list:
            name   = sp["name"]
            smiles = lookup_smiles(glossary, name) if glossary else None
            can_smiles: Optional[str] = None
            inchi:      Optional[str] = None
            inchi_key:  Optional[str] = None
            if smiles:
                can_smiles       = self._canon.canonicalize(smiles)
                inchi, inchi_key = self._canon.to_inchi(can_smiles or smiles)
            resolved[name] = (can_smiles, inchi, inchi_key)

        name_to_id: Dict[str, int] = {}
        for sp in species_list:
            name       = sp["name"]
            can_smiles, inchi, inchi_key = resolved[name]
            smiles     = lookup_smiles(glossary, name) if glossary else None

            # Phase 5D: apply variant override if present
            variant_tag:   Optional[str] = None
            canonical_name: Optional[str] = None
            if variants:
                entry = variants.get(name.upper())
                if entry:
                    cname = entry["canonical_name"]
                    canon_resolved = resolved.get(cname)
                    if canon_resolved and canon_resolved[2]:  # has inchi_key
                        can_smiles, inchi, inchi_key = canon_resolved
                        smiles = lookup_smiles(glossary, cname) if glossary else smiles
                        variant_tag    = entry.get("reason", "")
                        canonical_name = cname
                        log.info(
                            "Species %s → variant of %s (%s), sharing InChIKey %s",
                            name, cname, variant_tag, inchi_key,
                        )
                    else:
                        log.warning(
                            "Variant override for %s → %s: canonical species has no "
                            "InChIKey (not in glossary?). Override skipped.",
                            name, cname,
                        )

            # Fallback: compute formula and MW from SMILES if reader didn't provide them
            formula  = sp.get("formula")
            mw       = sp.get("mw_kg_mol")
            if smiles and not formula:
                formula = smiles_to_formula(smiles)
            if smiles and not mw:
                mw = smiles_to_mw(smiles)

            extra: Dict[str, Any] = {}
            if variant_tag is not None:
                extra["variant_tag"]    = variant_tag
                extra["canonical_name"] = canonical_name

            sp_id = insert_species(
                self._conn,
                mechanism_id     = mech_id,
                name             = name,
                canonical_smiles = can_smiles,
                source_smiles    = smiles,
                inchi            = inchi,
                inchi_key        = inchi_key,
                formula          = formula,
                mw_kg_mol        = mw,
                glossary_matched = 1 if smiles else 0,
                thermo_type      = sp.get("thermo_type"),
                tmin_K           = sp.get("tmin_K"),
                tmid_K           = sp.get("tmid_K"),
                tmax_K           = sp.get("tmax_K"),
                coeffs_low_json  = json.dumps(sp["coeffs_low"])  if sp.get("coeffs_low")  else None,
                coeffs_high_json = json.dumps(sp["coeffs_high"]) if sp.get("coeffs_high") else None,
                h298_kJ_mol      = sp.get("h298_kJ_mol"),
                s298_J_molK      = sp.get("s298_J_molK"),
                thermo_comment   = sp.get("thermo_comment"),
                **extra,
            )
            name_to_id[name] = sp_id
        self._conn.commit()
        log.debug("Inserted %d species for mech_id=%d", len(name_to_id), mech_id)
        return name_to_id

    def _insert_reactions(
        self,
        mech_id:      int,
        rxn_list:     list,
        sp_name_to_id: Dict[str, int],
    ) -> None:
        """Insert all reactions plus stoichiometry and third-body efficiencies."""
        for rx in rxn_list:
            # Build canonical key
            can_key = self._build_canonical_key(rx, sp_name_to_id)

            kdata_json = json.dumps(rx.get("kinetics_data") or {})
            rx_id = insert_reaction(
                self._conn,
                mechanism_id      = mech_id,
                formula_string    = rx["formula_string"],
                canonical_key     = can_key,
                reversible        = int(rx.get("reversible", True)),
                family            = rx.get("family"),
                duplicate         = int(rx.get("duplicate", False)),
                kinetics_type     = rx.get("kinetics_type"),
                pressure_dependent = int(rx.get("pressure_dependent", False)),
                has_third_body    = int(rx.get("has_third_body", False)),
                A_si              = rx.get("A_si"),
                A_orig            = rx.get("A_orig"),
                A_units           = rx.get("A_units"),
                n                 = rx.get("n"),
                Ea_si             = rx.get("Ea_si"),
                Ea_orig           = rx.get("Ea_orig"),
                Ea_units          = rx.get("Ea_units"),
                kinetics_data_json = kdata_json,
                rmg_comment       = rx.get("rmg_comment"),
            )

            # Stoichiometry
            for sp_name, coeff in rx.get("reactants", []):
                sp_id = sp_name_to_id.get(sp_name)
                if sp_id:
                    insert_stoichiometry(self._conn, rx_id, sp_id, coeff, "reactant")
            for sp_name, coeff in rx.get("products", []):
                sp_id = sp_name_to_id.get(sp_name)
                if sp_id:
                    insert_stoichiometry(self._conn, rx_id, sp_id, coeff, "product")

            # Third-body efficiencies
            for sp_name, eff in rx.get("third_body_efficiencies", []):
                sp_id = sp_name_to_id.get(sp_name)
                insert_third_body_efficiency(
                    self._conn, rx_id, eff,
                    species_id=sp_id, species_name=sp_name,
                )

        self._conn.commit()
        log.debug("Inserted %d reactions for mech_id=%d", len(rxn_list), mech_id)

    def _build_canonical_key(
        self,
        rx:            dict,
        sp_name_to_id: Dict[str, int],
    ) -> Optional[str]:
        """
        Resolve species SMILES from DB and build the reaction canonical key.
        Returns None if any species lacks a SMILES.
        """
        def get_smiles(name: str) -> Optional[str]:
            sp_id = sp_name_to_id.get(name)
            if sp_id is None:
                return None
            row = self._conn.execute(
                "SELECT canonical_smiles FROM species WHERE id=?", (sp_id,)
            ).fetchone()
            return row["canonical_smiles"] if row else None

        r_smiles = [get_smiles(n) for n, _ in rx.get("reactants", [])]
        p_smiles = [get_smiles(n) for n, _ in rx.get("products", [])]

        if None in r_smiles or None in p_smiles:
            return None

        return self._canon.reaction_key(
            r_smiles, p_smiles, reversible=rx.get("reversible", True)
        )
