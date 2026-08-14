"""Deterministic ID generation so re-ingesting an unchanged file is a no-op upsert."""

from __future__ import annotations

import hashlib


def chunk_id(source_path: str, chunk_index: int) -> str:
    """Stable, deterministic ID for a chunk based on its source file and position."""
    key = f"{source_path}::{chunk_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
