"""Loader protocol: every loader turns a file path into a list of PageText."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag.types import PageText


class Loader(Protocol):
    def load(self, path: Path) -> list[PageText]: ...


class UnsupportedFileTypeError(ValueError):
    """Raised when a file's extension has no registered loader."""
