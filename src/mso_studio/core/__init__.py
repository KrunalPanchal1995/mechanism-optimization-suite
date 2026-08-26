"""
mso_studio.core -- UI-agnostic science adapters.

Zero Qt imports anywhere in this package (enforced by
``tests/test_no_gui_imports.py``: ``grep -r "PySide6" core/`` must return
nothing). Every public function here takes a plain dataclass and returns a
:class:`~mso_studio.core.runrecord.RunRecord`.
"""
