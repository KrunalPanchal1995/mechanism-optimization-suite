"""
pdf_panel.py
------------
Tkinter panel: load a PDF → extract text (pdfplumber) → Gemini summary + Q&A chat.

Usage (standalone)::

    python pdf_panel.py           # opens a bare Tk window with the panel

Usage (embedded in the main GUI)::

    from notebook_gemini.pdf_panel import PDFNotebookPanel
    panel = PDFNotebookPanel(parent_frame, client=GeminiClient())
    panel.pack(fill=tk.BOTH, expand=True)
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional


class PDFNotebookPanel(ttk.Frame):
    """
    PDF Q&A panel.

    Parameters
    ----------
    parent : tk widget
    client : GeminiClient
        A ``GeminiClient`` instance (may have ``available=False``).
    palette : dict, optional
        Colour palette from the main GUI; falls back to plain defaults.
    """

    def __init__(self, parent, client, palette: Optional[dict] = None):
        super().__init__(parent)
        self._client = client
        self._pal = palette or {
            "bg": "#1e1e2e", "surface": "#2a2a3e",
            "accent": "#7aa2f7", "text": "#cdd6f4",
        }
        self._pdf_bytes: Optional[bytes] = None
        self._pdf_text: str = ""
        self._pdf_path: str = ""
        self._history: list[tuple[str, str]] = []   # (role, text)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.configure(padding=8)

        # ── top bar ──────────────────────────────────────────────────
        top = ttk.Frame(self)
        top.pack(fill=tk.X, pady=(0, 6))

        self._path_var = tk.StringVar(value="No PDF loaded")
        ttk.Label(top, textvariable=self._path_var, width=60, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Button(top, text="Open PDF…",   command=self._open_pdf).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Summarise ↵", command=self._do_summary).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Clear chat",  command=self._clear_chat).pack(side=tk.LEFT, padx=2)

        self._gemini_status = tk.StringVar(
            value="Gemini: ✓ ready" if self._client.available else "Gemini: ✗ no API key"
        )
        ttk.Label(top, textvariable=self._gemini_status, foreground="#a6e3a1").pack(side=tk.RIGHT)

        # ── chat history ─────────────────────────────────────────────
        chat_frame = ttk.Frame(self)
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self._chat_scroll = tk.Scrollbar(chat_frame)
        self._chat_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._chat_text = tk.Text(
            chat_frame,
            yscrollcommand=self._chat_scroll.set,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("DejaVu Sans Mono", 9),
            bg=self._pal["surface"],
            fg=self._pal["text"],
            relief=tk.FLAT,
            padx=8, pady=6,
        )
        self._chat_text.pack(fill=tk.BOTH, expand=True)
        self._chat_scroll.config(command=self._chat_text.yview)

        # Tags for colour-coding
        self._chat_text.tag_config("user",   foreground="#7aa2f7", font=("DejaVu Sans", 9, "bold"))
        self._chat_text.tag_config("gemini", foreground="#a6e3a1", font=("DejaVu Sans", 9, "bold"))
        self._chat_text.tag_config("body",   foreground=self._pal["text"])
        self._chat_text.tag_config("error",  foreground="#f38ba8")

        # ── input row ────────────────────────────────────────────────
        inp_frame = ttk.Frame(self)
        inp_frame.pack(fill=tk.X, pady=(6, 0))

        self._question_var = tk.StringVar()
        entry = ttk.Entry(inp_frame, textvariable=self._question_var, font=("DejaVu Sans", 10))
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.bind("<Return>", lambda _e: self._ask())
        ttk.Button(inp_frame, text="Ask ↵", command=self._ask).pack(side=tk.LEFT, padx=6)

        self._spinner_var = tk.StringVar(value="")
        ttk.Label(inp_frame, textvariable=self._spinner_var, width=10).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _open_pdf(self):
        path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf"), ("All files", "*")],
            title="Select PDF",
        )
        if not path:
            return
        try:
            import pdfplumber
            with open(path, "rb") as fh:
                self._pdf_bytes = fh.read()
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            self._pdf_text = "\n".join(pages)
            self._pdf_path = path
            self._path_var.set(os.path.basename(path))
            self._append_chat("system", f"Loaded: {os.path.basename(path)} "
                              f"({len(pdf.pages)} pages, {len(self._pdf_text)} chars extracted)")
        except Exception as exc:
            messagebox.showerror("PDF load error", str(exc))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _do_summary(self):
        if not self._pdf_bytes:
            messagebox.showinfo("No PDF", "Load a PDF first.")
            return
        self._append_chat("user", "[Summarise this paper]")
        self._run_in_thread(
            lambda: self._client.summarise_pdf_bytes(self._pdf_bytes),
            label="Summary",
        )

    def _ask(self):
        q = self._question_var.get().strip()
        if not q:
            return
        self._question_var.set("")
        self._append_chat("user", q)

        if self._pdf_bytes and self._client.available:
            fn = lambda: self._client.chat_with_pdf_bytes(self._pdf_bytes, q)
        else:
            # Fallback: pass extracted text as context
            ctx = self._pdf_text[:30_000] if self._pdf_text else ""
            fn = lambda: self._client.chat(q, context=ctx)

        self._run_in_thread(fn, label="Answer")

    def _clear_chat(self):
        self._chat_text.config(state=tk.NORMAL)
        self._chat_text.delete("1.0", tk.END)
        self._chat_text.config(state=tk.DISABLED)
        self._history.clear()

    # ------------------------------------------------------------------
    # Threading / chat helpers
    # ------------------------------------------------------------------

    def _run_in_thread(self, fn, label: str):
        self._spinner_var.set("⏳ …")

        def _worker():
            reply = fn()
            self.after(0, lambda: self._on_reply(reply, label))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_reply(self, text: str, label: str):
        self._spinner_var.set("")
        tag = "error" if text.startswith("[Gemini error") or text.startswith("[Gemini not") else "gemini"
        self._append_chat("gemini", text, body_tag=tag if tag == "error" else "body")

    def _append_chat(self, role: str, text: str, body_tag: str = "body"):
        self._chat_text.config(state=tk.NORMAL)
        if role == "user":
            self._chat_text.insert(tk.END, "\nYou: ", "user")
        elif role == "gemini":
            self._chat_text.insert(tk.END, "\nGemini: ", "gemini")
        else:
            self._chat_text.insert(tk.END, "\n[" + role + "] ", "gemini")
        self._chat_text.insert(tk.END, text + "\n", body_tag)
        self._chat_text.see(tk.END)
        self._chat_text.config(state=tk.DISABLED)
        self._history.append((role, text))


# ---------------------------------------------------------------------------
# Standalone launcher
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from notebook_gemini.gemini_client import GeminiClient

    root = tk.Tk()
    root.title("Gemini PDF Notebook")
    root.geometry("900x640")
    client = GeminiClient()
    panel = PDFNotebookPanel(root, client=client)
    panel.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
