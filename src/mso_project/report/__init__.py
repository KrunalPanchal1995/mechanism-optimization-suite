"""mso_project.report -- convert a project's tracked artifacts into a LaTeX
report + supplementary document, built from coded defaults (``rules.py``)
that a user can review/edit through an interactive wizard (``wizard.py``)
or by hand-editing the resulting ``report_rules.yaml``."""
from .builder import ReportResult, build_report
from .rules import DEFAULT_RULES, ReportRules

__all__ = ["ReportRules", "DEFAULT_RULES", "build_report", "ReportResult"]
