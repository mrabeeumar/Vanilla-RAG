"""Shared lightweight data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageText:
    """One page (or the whole document, for formats without pages) of extracted text."""

    text: str
    page_number: int | None = None


@dataclass
class Chunk:
    """A chunk ready for embedding, tagged with its position in the source document."""

    text: str
    chunk_index: int
    source: str
    page_number: int | None = None


@dataclass
class RetrievedChunk:
    """A chunk returned from the vector store along with its similarity score."""

    text: str
    source: str
    chunk_index: int
    page_number: int | None
    score: float
