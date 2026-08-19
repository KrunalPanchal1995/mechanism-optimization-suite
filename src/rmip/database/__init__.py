"""
rmip/database/__init__.py
=========================
Re-exports all CRUD helpers from schema.py and schema_phase3.py so callers
can write:

    from rmip.database import init_db, insert_mechanism, upgrade_db
"""
from rmip.database.schema import (
    init_db,
    insert_mechanism,
    get_mechanism,
    get_mechanisms,
    update_mechanism_counts,
    delete_mechanism,
    insert_species,
    get_species_for_mechanism,
    get_species_by_inchi_key,
    insert_reaction,
    get_reactions_for_mechanism,
    get_reaction_by_canonical_key,
    insert_stoichiometry,
    get_stoichiometry_for_reaction,
    insert_third_body_efficiency,
    get_mechanism_summary,
    BatchInserter,
)
from rmip.database.schema_phase3 import (
    upgrade_db,
    insert_simulation_run,
    get_simulation_runs,
    insert_exp_dataset,
    insert_exp_data_points,
    get_exp_data_points,
    list_exp_datasets,
    upsert_mechanism_score,
    get_mechanism_scores,
    insert_comparison_session,
    insert_reaction_classification,
    insert_species_classification,
    get_comparison_session,
    get_reaction_classifications_for_session,
)

__all__ = [
    # Phase 1 schema
    "init_db", "insert_mechanism", "get_mechanism", "get_mechanisms",
    "update_mechanism_counts", "delete_mechanism",
    "insert_species", "get_species_for_mechanism", "get_species_by_inchi_key",
    "insert_reaction", "get_reactions_for_mechanism", "get_reaction_by_canonical_key",
    "insert_stoichiometry", "get_stoichiometry_for_reaction",
    "insert_third_body_efficiency",
    "get_mechanism_summary", "BatchInserter",
    # Phase 3 schema
    "upgrade_db",
    "insert_simulation_run", "get_simulation_runs",
    "insert_exp_dataset", "insert_exp_data_points", "get_exp_data_points",
    "list_exp_datasets",
    "upsert_mechanism_score", "get_mechanism_scores",
    "insert_comparison_session",
    "insert_reaction_classification", "insert_species_classification",
    "get_comparison_session", "get_reaction_classifications_for_session",
]
