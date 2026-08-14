"""Orchestrates: discover files -> load -> chunk -> embed -> upsert."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking.splitter import chunk_document
from rag.config import Settings
from rag.embeddings.embedder import Embedder
from rag.loaders.base import UnsupportedFileTypeError
from rag.loaders.registry import SUPPORTED_EXTENSIONS, load_file
from rag.utils.logging_setup import get_logger
from rag.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)


@dataclass
class IngestReport:
    files_ingested: int = 0
    chunks_added: int = 0
    unsupported_files: list[str] = field(default_factory=list)
    empty_files: list[str] = field(default_factory=list)
    total_chunks_in_collection: int = 0


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file())
    raise FileNotFoundError(f"Path does not exist: {path}")


def ingest_path(
    path: Path,
    settings: Settings,
    embedder: Embedder,
    store: ChromaStore,
    overwrite: bool = False,
) -> IngestReport:
    files = discover_files(path)
    report = IngestReport()

    if not files:
        logger.warning("No files found under %s", path)
        report.total_chunks_in_collection = store.count()
        return report

    for file_path in files:
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            report.unsupported_files.append(str(file_path))
            continue

        try:
            pages = load_file(file_path)
        except UnsupportedFileTypeError:
            report.unsupported_files.append(str(file_path))
            continue

        if not pages:
            report.empty_files.append(str(file_path))
            continue

        source = str(file_path)
        chunks = chunk_document(
            pages,
            source=source,
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        )

        if not chunks:
            report.empty_files.append(source)
            continue

        if overwrite:
            store.delete_by_source(source)

        embeddings = embedder.embed_documents([c.text for c in chunks])
        store.upsert_chunks(chunks, embeddings)

        report.files_ingested += 1
        report.chunks_added += len(chunks)
        logger.info("Ingested %s (%d chunks)", source, len(chunks))

    report.total_chunks_in_collection = store.count()
    return report
