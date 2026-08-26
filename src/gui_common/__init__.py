"""
gui_common
==========
Shared infrastructure for the split-out per-task GUI apps (see
MarkDownFiles/GUI_SEPARATION_AND_IMPROVEMENT_PLAN_2026-07-12_142230.md).

Extracted from generate_target_opt.py's monolithic App class -- these pieces
were already self-contained (no dependency on App internals) and are reused
as-is by each new standalone app rather than copy-pasted into every one.
"""
