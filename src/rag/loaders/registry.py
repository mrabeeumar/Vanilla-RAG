"""Maps file extensions to loader instances."""

from __future__ import annotations

from pathlib import Path

from rag.loaders.base import Loader, UnsupportedFileTypeError
from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.text_loader import TextLoader
from rag.types import PageText

_LOADERS: dict[str, Loader] = {
    ".txt": TextLoader(),
    ".md": TextLoader(),
    ".pdf": PDFLoader(),
}

SUPPORTED_EXTENSIONS = tuple(_LOADERS.keys())


def get_loader(path: Path) -> Loader:
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFileTypeError(
            f"No loader registered for extension '{path.suffix}' ({path.name}). "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return loader


def load_file(path: Path) -> list[PageText]:
    return get_loader(path).load(path)
