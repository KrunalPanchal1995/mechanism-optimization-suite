"""
mso_project.report.rules
=========================
``ReportRules``: what decides whether a tracked artifact lands in the MAIN
report, the SUPPLEMENTARY document, or is excluded entirely.

Routing precedence (see ``builder.py``):
  1. An artifact's own ``report_include`` column in ``artifacts.db``, if
     not ``"auto"`` -- an explicit per-artifact override always wins.
  2. ``rules.role_routing[role]``, if that role has an explicit entry.
  3. ``rules.default_routing_for_kind(kind)`` -- a coded, kind-based
     fallback, so a role nobody has thought to route explicitly yet still
     gets a sane answer instead of silently vanishing from the report.

Defaults here are CODE (importable, testable, the ground truth); the
wizard (``wizard.py``) is what ASKS a user to review/override them, per
this project's explicit requirement that report rules be both coded and
user-configurable. The wizard writes a ``report_rules.yaml`` that
round-trips through :meth:`ReportRules.to_yaml`/:meth:`from_yaml` --
editable by hand, diffable in git, and NOT re-asked every build (contrast
with a "prompt every time" design, which can't reproduce a report later
without re-answering identically).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

RULES_SCHEMA_VERSION = 1

# Headline artifacts -- explicit "main" routing regardless of kind.
_MAIN_ROLES = {
    "t_p_phi_plot", "fit_quality_summary", "optimized_mechanism",
    "cost_summary", "posterior_plot",
    "forward_uq_summary", "sobol_indices",
    "rmip_comparison_report",
    "fused_mechanism_yaml", "fusion_summary",
    "merged_comparison",
}

# Operational/cache artifacts -- never report content, explicit "exclude".
_EXCLUDE_ROLES = {
    "progress_marker", "uncertainty_cache", "sensitivity_cache",
    "partial_prs_cache", "partial_prs_yaml", "perturbed_mechanism",
    "outlier_perturbed_mechanism", "design_matrix", "archived_study",
    "study_marker", "mechanism_database", "rmip_database",
    "graph_sa_database", "forward_uq_config", "sa_config", "sa_plot_style",
    "unclassified",
}

# Kind-based fallback for any role not listed above (new footprint roles,
# generic-classified files, etc.) -- errs toward supplementary rather than
# silently dropping content a role-routing table hasn't caught up with yet.
_DEFAULT_ROUTING_BY_KIND = {
    "doc": "main",         # user notes/readmes are short and user-curated
    "csv": "supplementary",
    "png": "supplementary",
    "pdf": "supplementary",
    "json": "supplementary",
    "yaml": "supplementary",
    "notebook": "supplementary",
    "log": "exclude",
    "pkl": "exclude",
    "txt": "supplementary",
    "other": "exclude",
}


@dataclass
class ReportRules:
    rules_version: int = RULES_SCHEMA_VERSION
    title: str = "Project Report"
    author: str = ""
    sections_main: List[str] = field(default_factory=lambda: [
        "overview", "methodology", "cost_summary", "fit_quality",
        "key_figures", "outlier_retraining_history", "conclusions",
    ])
    sections_supplementary: List[str] = field(default_factory=lambda: [
        "full_run_table", "per_case_diagnostics", "raw_data_index", "provenance",
    ])
    role_routing: Dict[str, str] = field(default_factory=lambda: {
        **{r: "main" for r in _MAIN_ROLES},
        **{r: "exclude" for r in _EXCLUDE_ROLES},
    })
    max_figures_main: int = 12
    holdout_error_flag_pct: float = 2.0  # flag cases whose hold-out error exceeds this
    include_notes: bool = True
    bib_path: Optional[str] = None

    def routing_for(self, role: str, kind: str) -> str:
        """auto|main|supplementary|exclude for one artifact's (role, kind),
        BEFORE any per-artifact database override is applied (see this
        module's docstring for the full precedence chain)."""
        if role in self.role_routing:
            return self.role_routing[role]
        return _DEFAULT_ROUTING_BY_KIND.get(kind, "supplementary")

    # ------------------------------------------------------------------
    # YAML round-trip
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "rules_version": self.rules_version,
            "title": self.title,
            "author": self.author,
            "sections_main": list(self.sections_main),
            "sections_supplementary": list(self.sections_supplementary),
            "role_routing": dict(self.role_routing),
            "max_figures_main": self.max_figures_main,
            "holdout_error_flag_pct": self.holdout_error_flag_pct,
            "include_notes": self.include_notes,
            "bib_path": self.bib_path,
        }

    def to_yaml(self, path: str) -> None:
        """Write this rule set to ``path``, keeping a ``.bak`` of whatever
        was there before (the wizard calls this on every save, so a user
        can always recover the previous rule set after an edit)."""
        p = Path(path)
        if p.exists():
            shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
        p.write_text(yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False))

    @classmethod
    def from_dict(cls, data: dict) -> "ReportRules":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> "ReportRules":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    @classmethod
    def load_or_default(cls, path: str) -> "ReportRules":
        """The usual entry point: use an existing report_rules.yaml if the
        project has one, else the coded defaults (unsaved -- call
        to_yaml() explicitly, or run the wizard, to persist them). Always
        returns a FRESH instance, never the shared ``DEFAULT_RULES``
        singleton, so a caller mutating the result (e.g. the wizard) can't
        corrupt the module-level default for everyone else in-process."""
        if Path(path).exists():
            return cls.from_yaml(path)
        return cls()


DEFAULT_RULES = ReportRules()
