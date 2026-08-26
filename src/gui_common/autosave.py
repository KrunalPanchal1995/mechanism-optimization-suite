"""
gui_common/autosave.py
========================
Dirty-flag autosave timer, extracted verbatim from generate_target_opt.py's
AutoSaveManager class. Generic -- has no knowledge of what's being saved.

Each split-out app should use its OWN checkpoint file (e.g.
``.gui_settings/<app_name>_checkpoint.json``), not a single shared path --
see gui_common/settings.py.
"""
from __future__ import annotations


class AutoSaveManager:
    """Automatically save GUI state every N ms (default 5000) if changed."""

    def __init__(self, root, save_callback, interval_ms=5000):
        self.root = root
        self.save_callback = save_callback
        self.interval_ms = interval_ms
        self.timer_id = None
        self.dirty = False

    def mark_dirty(self):
        """Mark state as dirty and schedule save."""
        self.dirty = True
        self._schedule_save()

    def _schedule_save(self):
        """Cancel pending timer and schedule a new one."""
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        self.timer_id = self.root.after(self.interval_ms, self._do_save)

    def _do_save(self):
        """Perform the actual save."""
        if self.dirty:
            try:
                self.save_callback()
                self.dirty = False
            except Exception as e:
                print(f"Auto-save error: {e}")

    def on_close(self):
        """Save final state before exit."""
        if self.dirty:
            self._do_save()
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
