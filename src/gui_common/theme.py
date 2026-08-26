"""
gui_common/theme.py
====================
Dark theme (palette + fonts + ttk style configuration) for the legacy
Tkinter apps -- there is no light-mode toggle here, this is a single fixed
dark theme applied unconditionally (the Tk stack is frozen, not redesigned;
see the visual-system plan's Session A4).

PALETTE is now a thin adapter over ``mso_studio.gui.tokens.DARK`` -- the
same 13 hex values used to live independently in three places
(``generate_target_opt.py``'s own PALETTE dict, this file, and
``mso_studio/gui/theme.py``'s DARK dict); this is now the single import
both this module and ``generate_target_opt.py`` use, so a future palette
fix lands once. ``tokens.Tokens`` has no dedicated tooltip pair (Qt has a
native ``ToolTipBase``/``ToolTipText`` palette role; Tk has no built-in
tooltip widget, so these apps roll their own popup) -- ``tip_bg``/``tip_fg``
stay as Tk-only literals, not a duplicate of anything in the shared tokens.
Falls back to the original literal values if the import ever fails (e.g.
run standalone without ``src/`` on ``sys.path``), so no Tk app can be fully
broken by this indirection.
"""
from __future__ import annotations

try:
    import tkinter as tk
    from tkinter import ttk
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

try:
    from mso_studio.gui.tokens import DARK as _DARK
    PALETTE = {
        "bg":       _DARK.bg,
        "surface":  _DARK.surface,
        "surface2": _DARK.surface_raised,
        "accent":   _DARK.accent,
        "accent2":  _DARK.accent2,
        "success":  _DARK.success,
        "warning":  _DARK.warning,
        "error":    _DARK.error,
        "text":     _DARK.text,
        "text_dim": _DARK.text_dim,
        "border":   _DARK.border,
        "entry_bg": _DARK.entry_bg,
        "tip_bg":   "#1e40af",
        "tip_fg":   "#dbeafe",
    }
except ImportError:
    PALETTE = {
        "bg":       "#1e2433",
        "surface":  "#252b3b",
        "surface2": "#2e3650",
        "accent":   "#4f8ef7",
        "accent2":  "#38bdf8",
        "success":  "#22c55e",
        "warning":  "#f59e0b",
        "error":    "#ef4444",
        "text":     "#e2e8f0",
        "text_dim": "#94a3b8",
        "border":   "#374151",
        "entry_bg": "#111827",
        "tip_bg":   "#1e40af",
        "tip_fg":   "#dbeafe",
    }

# Font definitions — system-agnostic families with fallbacks.
_FONT_FAMILY = "DejaVu Sans"      # available on most Linux; Tk falls back gracefully
_MONO_FAMILY = "DejaVu Sans Mono"

FONTS = {
    "body":       (_FONT_FAMILY, 10),
    "body_bold":  (_FONT_FAMILY, 10, "bold"),
    "header":     (_FONT_FAMILY, 12, "bold"),
    "subheader":  (_FONT_FAMILY, 11, "bold"),
    "section":    (_FONT_FAMILY, 10, "bold"),
    "small":      (_FONT_FAMILY,  9),
    "mono":       (_MONO_FAMILY, 10),
    "mono_small": (_MONO_FAMILY,  9),
    "status":     (_FONT_FAMILY,  9),
    "tab":        (_FONT_FAMILY, 10, "bold"),
    "tip":        (_FONT_FAMILY,  9),
}


def apply_theme(root: "tk.Tk") -> None:
    """Apply the shared dark theme to the root window and all ttk widgets."""
    if not _HAS_TK:
        return
    root.configure(bg=PALETTE["bg"])
    style = ttk.Style(root)
    style.theme_use("clam")

    # ── Global widget defaults ─────────────────────────────────────────────
    style.configure(".",
        background=PALETTE["surface"],
        foreground=PALETTE["text"],
        fieldbackground=PALETTE["entry_bg"],
        bordercolor=PALETTE["border"],
        darkcolor=PALETTE["bg"],
        lightcolor=PALETTE["surface2"],
        troughcolor=PALETTE["bg"],
        selectbackground=PALETTE["accent"],
        selectforeground=PALETTE["text"],
        font=FONTS["body"],
        relief="flat",
        borderwidth=1,
    )

    # ── Frames ─────────────────────────────────────────────────────────────
    style.configure("TFrame",      background=PALETTE["surface"])
    style.configure("TLabelframe", background=PALETTE["surface"],
                    foreground=PALETTE["accent"], font=FONTS["body_bold"])
    style.configure("TLabelframe.Label", background=PALETTE["surface"],
                    foreground=PALETTE["accent"])

    # ── Labels ─────────────────────────────────────────────────────────────
    style.configure("TLabel",      background=PALETTE["surface"],  foreground=PALETTE["text"])
    style.configure("Dim.TLabel",  background=PALETTE["surface"],  foreground=PALETTE["text_dim"])
    style.configure("Header.TLabel", background=PALETTE["surface"],
                    foreground=PALETTE["accent"], font=FONTS["header"])
    style.configure("SubHeader.TLabel", background=PALETTE["surface"],
                    foreground=PALETTE["accent2"], font=FONTS["subheader"])
    style.configure("Section.TLabel", background=PALETTE["surface"],
                    foreground=PALETTE["text"], font=FONTS["section"])
    style.configure("Status.TLabel", background=PALETTE["bg"],
                    foreground=PALETTE["text_dim"], font=FONTS["status"])

    # ── Buttons ─────────────────────────────────────────────────────────────
    style.configure("TButton",
        background=PALETTE["accent"],
        foreground=PALETTE["text"],
        font=FONTS["body_bold"],
        padding=(8, 4),
        relief="flat",
        borderwidth=0,
    )
    style.map("TButton",
        background=[("active", PALETTE["accent2"]), ("pressed", PALETTE["bg"])],
        foreground=[("active", PALETTE["text"])],
    )
    style.configure("Small.TButton",
        background=PALETTE["surface2"],
        foreground=PALETTE["text_dim"],
        font=FONTS["small"],
        padding=(4, 2),
    )
    style.map("Small.TButton",
        background=[("active", PALETTE["accent"])],
        foreground=[("active", PALETTE["text"])],
    )
    style.configure("Danger.TButton",
        background=PALETTE["error"],
        foreground=PALETTE["text"],
        font=FONTS["small"],
        padding=(3, 1),
    )
    style.map("Danger.TButton",
        background=[("active", "#b91c1c")],
    )

    # ── Entries ─────────────────────────────────────────────────────────────
    style.configure("TEntry",
        fieldbackground=PALETTE["entry_bg"],
        foreground=PALETTE["text"],
        insertcolor=PALETTE["text"],
        bordercolor=PALETTE["border"],
        lightcolor=PALETTE["entry_bg"],
        darkcolor=PALETTE["entry_bg"],
    )
    style.map("TEntry",
        fieldbackground=[("focus", "#1f2937")],
        bordercolor=[("focus", PALETTE["accent"])],
    )

    # ── Combobox ────────────────────────────────────────────────────────────
    style.configure("TCombobox",
        fieldbackground=PALETTE["entry_bg"],
        foreground=PALETTE["text"],
        background=PALETTE["surface"],
        arrowcolor=PALETTE["accent"],
        bordercolor=PALETTE["border"],
    )
    style.map("TCombobox",
        fieldbackground=[("readonly", PALETTE["entry_bg"]), ("focus", "#1f2937")],
        foreground=[("readonly", PALETTE["text"])],
        bordercolor=[("focus", PALETTE["accent"])],
    )
    root.option_add("*TCombobox*Listbox.background", PALETTE["entry_bg"])
    root.option_add("*TCombobox*Listbox.foreground", PALETTE["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", PALETTE["accent"])

    # ── Notebook ────────────────────────────────────────────────────────────
    style.configure("TNotebook",
        background=PALETTE["bg"],
        tabmargins=[2, 4, 0, 0],
        borderwidth=0,
    )
    style.configure("TNotebook.Tab",
        background=PALETTE["surface2"],
        foreground=PALETTE["text_dim"],
        font=FONTS["tab"],
        padding=[12, 6],
        borderwidth=0,
        relief="flat",
    )
    style.map("TNotebook.Tab",
        background=[("selected", PALETTE["surface"]), ("active", PALETTE["surface2"])],
        foreground=[("selected", PALETTE["accent"]), ("active", PALETTE["text"])],
        padding=[("selected", [14, 7])],
    )

    # ── Scrollbar ────────────────────────────────────────────────────────────
    style.configure("TScrollbar",
        background=PALETTE["surface2"],
        troughcolor=PALETTE["bg"],
        arrowcolor=PALETTE["text_dim"],
        borderwidth=0,
        relief="flat",
    )
    style.map("TScrollbar",
        background=[("active", PALETTE["accent"])],
    )

    # ── Separator ────────────────────────────────────────────────────────────
    style.configure("TSeparator", background=PALETTE["border"])

    # ── Canvas / scrolled frames ─────────────────────────────────────────────
    root.option_add("*Canvas.background", PALETTE["surface"])
    root.option_add("*Text.background",   PALETTE["entry_bg"])
    root.option_add("*Text.foreground",   PALETTE["text"])
    root.option_add("*Text.font",         "{%s} %d" % (FONTS['mono'][0], FONTS['mono'][1]))


def style_text_widget(widget) -> None:
    """Explicitly theme a raw ``tk.Text``/``scrolledtext.ScrolledText``
    widget. ``apply_theme()``'s ``option_add("*Text...")`` calls above only
    reach a Text widget that is (a) created after ``apply_theme`` ran on
    that same interpreter and (b) not given an explicit competing color at
    construction -- several widgets across the Tk apps are built with no
    color args at all yet still render with Tk's plain white/black system
    default, so this applies the theme directly rather than depending on
    the option-database's implicit propagation. Safe to call even when Tk
    isn't available (no-op)."""
    if not _HAS_TK:
        return
    widget.configure(
        bg=PALETTE["entry_bg"],
        fg=PALETTE["text"],
        insertbackground=PALETTE["text"],
        selectbackground=PALETTE["accent"],
        selectforeground=PALETTE["text"],
        relief="flat",
        borderwidth=1,
        highlightthickness=1,
        highlightbackground=PALETTE["border"],
        highlightcolor=PALETTE["accent"],
    )
