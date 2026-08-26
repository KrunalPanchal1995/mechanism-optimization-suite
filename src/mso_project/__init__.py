"""
mso_project
===========
Cross-pipeline project persistence: remembers every artifact location a run
produces -- no matter which pipeline or working directory produced it -- and
converts a project's data into a LaTeX report + supplementary document.

See ``project.py``'s module docstring for the full design rationale and the
canonical ``src/MUQ-SAC/run_script.py`` capture/resume recipe.

Public API
----------
    from mso_project import Project
    proj = Project.create("/data/PROJECTS/AMMONIA_2026", name="AMMONIA_2026")
    # ... or, on a later run:
    proj = Project.open("/data/PROJECTS/AMMONIA_2026")
"""
from .project import Project

__all__ = ["Project"]
__version__ = "0.1.0"
