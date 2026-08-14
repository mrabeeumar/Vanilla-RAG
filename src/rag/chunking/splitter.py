"""Recursive character-based text splitter with overlap.

Character-based (not token-based) for v1 to avoid a tokenizer mismatch between the
splitter and whichever embedding model is configured (e.g. tiktoken doesn't match
the BGE tokenizer). Token-aware chunking reusing the HF tokenizer already loaded in
Embedder is a documented future upgrade.
"""

from __future__ import annotations

from rag.types import Chunk, PageText

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_on_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split `text` into pieces no larger than chunk_size, trying separators in
    order and falling back to a hard char-level split so nothing is ever too big."""
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # Hard fallback: slice by character.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_separators = separators[0], separators[1:]
    parts = _split_on_separator(text, sep)

    # Re-attach the separator (except for the empty-string char-split case) so
    # rejoining reproduces the original text faithfully.
    if sep != "":
        rejoined = [p + sep for p in parts[:-1]] + parts[-1:]
    else:
        rejoined = parts

    pieces: list[str] = []
    for part in rejoined:
        if not part:
            continue
        if len(part) <= chunk_size:
            pieces.append(part)
        else:
            pieces.extend(_recursive_split(part, chunk_size, rest_separators))
    return pieces


def _pack_pieces(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack small pieces (each already <= chunk_size, guaranteed by
    _recursive_split) into windows up to chunk_size, carrying the trailing
    `chunk_overlap` chars of each window into the next one. The chunk_size cap is
    a hard invariant: if adding the overlap prefix back onto a large piece would
    exceed it, the overlap is truncated from the front rather than growing the
    chunk past chunk_size."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
            continue

        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue

        chunks.append(current)
        overlap_tail = current[-chunk_overlap:] if chunk_overlap > 0 else ""
        candidate = overlap_tail + piece
        current = candidate[-chunk_size:] if len(candidate) > chunk_size else candidate

    if current:
        chunks.append(current)

    return chunks


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text or not text.strip():
        return []

    separators = separators if separators is not None else DEFAULT_SEPARATORS
    pieces = _recursive_split(text, chunk_size, separators)
    return _pack_pieces(pieces, chunk_size, chunk_overlap)


def chunk_document(
    pages: list[PageText],
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Chunk each page independently (chunks never span a page boundary) and assign
    a running chunk_index across the whole document."""
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        for piece in split_text(page.text, chunk_size, chunk_overlap, separators):
            stripped = piece.strip()
            if not stripped:
                continue
            chunks.append(
                Chunk(
                    text=stripped,
                    chunk_index=index,
                    source=source,
                    page_number=page.page_number,
                )
            )
            index += 1
    return chunks
