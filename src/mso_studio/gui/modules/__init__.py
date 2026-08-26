"""
mso_studio.gui.modules
========================
One module per capability page. :data:`PAGE_FACTORIES` is what
``mso_studio.py`` hands to :class:`~mso_studio.gui.main_window.MainWindow` --
a ``(module_id, nav_label, factory)`` list, built lazily so importing this
package never imports Qt widgets that aren't needed yet.
"""
from __future__ import annotations


def _lazy(module_path: str, class_name: str):
    def factory():
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)()
    return factory


PAGE_FACTORIES: list[tuple[str, str, callable]] = [
    ("mechanism_converter", "Mechanism Converter", _lazy("mso_studio.gui.modules.converter", "ConverterPage")),
    ("csv2xml", "CSV → ReSpecTh XML", _lazy("mso_studio.gui.modules.csv2xml", "Csv2XmlPage")),
    ("mechanism_check", "Mechanism Health-Check", _lazy("mso_studio.gui.modules.mechanism_check", "MechanismCheckPage")),
    ("mechanism_fusion", "Mechanism Fusion", _lazy("mso_studio.gui.modules.fusion", "FusionPage")),
    ("uncertainty", "Uncertainty Generator", _lazy("mso_studio.gui.modules.uncertainty", "UncertaintyPage")),
    ("forward_uq", "Forward UQ", _lazy("mso_studio.gui.modules.forward_uq", "ForwardUQPage")),
    ("nominal_sim", "Nominal Simulation", _lazy("mso_studio.gui.modules.nominal", "NominalSimPage")),
    ("mso_compare", "Mechanism Comparison", _lazy("mso_studio.gui.modules.compare", "ComparePage")),
    ("sensitivity", "Sensitivity Analysis", _lazy("mso_studio.gui.modules.sensitivity", "SensitivityPage")),
    ("optimization", "Optimization", _lazy("mso_studio.gui.modules.optimization", "OptimizationPage")),
    ("target_workbench", "Target Workbench", _lazy("mso_studio.gui.modules.target_workbench", "TargetWorkbenchPage")),
    ("automation", "Mechanism-Comparison Automation", _lazy("mso_studio.gui.modules.automation", "AutomationPage")),
    ("rmip_suite", "RMIP Suite", _lazy("mso_studio.gui.modules.rmip_suite", "RMIPSuitePage")),
    ("tools", "Tools", _lazy("mso_studio.gui.modules.tools", "ToolsPage")),
    ("project_manager", "Project Manager", _lazy("mso_studio.gui.modules.project_manager", "ProjectManagerPage")),
]
