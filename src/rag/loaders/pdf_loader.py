"""Loader for .pdf files, one PageText per page. Scanned/image-only pages have no
extractable text and are skipped with a warning rather than silently ingested as
empty chunks (OCR is out of scope for v1)."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from rag.types import PageText
from rag.utils.logging_setup import get_logger

logger = get_logger(__name__)


class PDFLoader:
    def load(self, path: Path) -> list[PageText]:
        reader = PdfReader(str(path))
        pages: list[PageText] = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                logger.warning(
                    "%s: page %d has no extractable text (scanned image? OCR not supported) — skipping",
                    path.name,
                    i,
                )
                continue
            pages.append(PageText(text=text, page_number=i))
        return pages
