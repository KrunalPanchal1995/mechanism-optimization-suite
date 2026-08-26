"""
gemini_client.py
----------------
Thin wrapper around `google.genai` (Gemini 2.0 Flash).

Reads API key from ``GOOGLE_API_KEY`` environment variable.
If the key is absent or the package is not installed, all methods return
a polite "not available" string — the GUI never crashes.

Usage::

    from notebook_gemini.gemini_client import GeminiClient
    client = GeminiClient()            # reads GOOGLE_API_KEY from env
    print(client.available)            # True / False

    reply = client.chat("Summarise this text: ...", context="...")
    reply = client.chat_with_pdf_bytes(pdf_bytes, "What are the ignition delay conditions?")
"""

from __future__ import annotations

import os
from typing import Optional


_UNAVAILABLE_MSG = (
    "[Gemini not available — set GOOGLE_API_KEY environment variable "
    "and ensure google-genai is installed]"
)

_DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiClient:
    """
    Parameters
    ----------
    api_key : str, optional
        Gemini API key.  Defaults to ``os.environ['GOOGLE_API_KEY']``.
    model : str
        Gemini model name (default ``gemini-2.0-flash``).
    """

    def __init__(self, api_key: Optional[str] = None, model: str = _DEFAULT_MODEL):
        self.model = model
        self._client = None
        self.available = False

        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            return

        try:
            from google import genai as _genai
            self._client = _genai.Client(api_key=key)
            self.available = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, prompt: str, context: str = "") -> str:
        """Send a plain-text prompt (with optional context) → reply string."""
        if not self.available:
            return _UNAVAILABLE_MSG
        full = f"{context}\n\n{prompt}".strip() if context else prompt
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=full
            )
            return resp.text
        except Exception as exc:
            return f"[Gemini error: {exc}]"

    def chat_with_pdf_bytes(self, pdf_bytes: bytes, question: str) -> str:
        """Upload PDF bytes to Gemini and ask a question about the document."""
        if not self.available:
            return _UNAVAILABLE_MSG
        try:
            from google.genai import types as _types
            part = _types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            resp = self._client.models.generate_content(
                model=self.model,
                contents=[part, question],
            )
            return resp.text
        except Exception as exc:
            return f"[Gemini error: {exc}]"

    def chat_with_image_bytes(self, image_bytes: bytes, mime: str, question: str) -> str:
        """Upload image bytes (PNG/JPG) to Gemini and ask about it."""
        if not self.available:
            return _UNAVAILABLE_MSG
        try:
            from google.genai import types as _types
            part = _types.Part.from_bytes(data=image_bytes, mime_type=mime)
            resp = self._client.models.generate_content(
                model=self.model,
                contents=[part, question],
            )
            return resp.text
        except Exception as exc:
            return f"[Gemini error: {exc}]"

    def summarise_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """One-shot PDF summary focused on combustion-relevant content."""
        prompt = (
            "You are assisting a combustion researcher. "
            "Summarise this paper concisely:\n"
            "• Fuel / oxidizer system and conditions studied\n"
            "• Key experimental observables (IDT, burning velocity, species)\n"
            "• Mechanism(s) tested and their performance\n"
            "• Main conclusions and any published uncertainty estimates"
        )
        return self.chat_with_pdf_bytes(pdf_bytes, prompt)
