"""Loader for plain .txt and .md files: no page structure, so a single PageText."""

from __future__ import annotations

from pathlib import Path

from rag.types import PageText


class TextLoader:
    def load(self, path: Path) -> list[PageText]:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return []
        return [PageText(text=text, page_number=None)]
