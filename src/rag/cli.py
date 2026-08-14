"""Click-based CLI entry point. Run with `python -m rag.cli` or the `rag` console script."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from rag.config import MissingAPIKeyError, get_settings
from rag.embeddings.embedder import Embedder
from rag.llm.factory import get_llm_client
from rag.pipeline.ingest import ingest_path
from rag.pipeline.query import format_sources, run_query
from rag.vectorstore.chroma_store import ChromaStore


def _build_store(collection: str | None) -> ChromaStore:
    settings = get_settings()
    return ChromaStore(
        persist_directory=settings.vectorstore.persist_directory,
        collection_name=collection or settings.vectorstore.collection_name,
    )


def _build_embedder() -> Embedder:
    settings = get_settings()
    e = settings.embedding
    return Embedder(
        model_name=e.model_name,
        device=e.device,
        pooling=e.pooling,
        query_prefix=e.query_prefix,
        passage_prefix=e.passage_prefix,
        max_seq_length=e.max_seq_length,
        batch_size=e.batch_size,
        normalize=e.normalize,
    )


@click.group()
def cli():
    """Vanilla RAG — a from-scratch retrieval-augmented generation CLI."""


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--collection", default=None, help="Chroma collection name.")
@click.option("--chunk-size", default=None, type=int, help="Override chunk size (chars).")
@click.option("--chunk-overlap", default=None, type=int, help="Override chunk overlap (chars).")
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Delete existing chunks for each source file before re-ingesting it.",
)
def ingest(path: Path, collection: str | None, chunk_size: int | None, chunk_overlap: int | None, overwrite: bool):
    """Ingest a file or directory of documents into the vector store."""
    settings = get_settings()
    if chunk_size is not None:
        settings.chunking.chunk_size = chunk_size
    if chunk_overlap is not None:
        settings.chunking.chunk_overlap = chunk_overlap

    embedder = _build_embedder()
    store = _build_store(collection)

    report = ingest_path(path, settings, embedder, store, overwrite=overwrite)

    if report.unsupported_files:
        click.echo(f"Skipped {len(report.unsupported_files)} unsupported file(s):")
        for f in report.unsupported_files:
            click.echo(f"  - {f}")

    if report.empty_files:
        click.echo(f"Skipped {len(report.empty_files)} empty/unextractable file(s):")
        for f in report.empty_files:
            click.echo(f"  - {f}")

    if report.files_ingested == 0 and not report.unsupported_files and not report.empty_files:
        click.echo(f"No files found under {path}", err=True)
        sys.exit(1)

    click.echo(
        f"Ingested {report.files_ingested} file(s), {report.chunks_added} chunk(s) added. "
        f"Collection now has {report.total_chunks_in_collection} chunk(s)."
    )


@cli.command()
@click.argument("question")
@click.option("--top-k", default=None, type=int, help="Number of chunks to retrieve.")
@click.option("--score-threshold", default=None, type=float, help="Minimum similarity score.")
@click.option("--collection", default=None, help="Chroma collection name.")
def query(question: str, top_k: int | None, score_threshold: float | None, collection: str | None):
    """Ask a question against the ingested documents."""
    settings = get_settings()

    try:
        llm_client = get_llm_client(settings)
    except MissingAPIKeyError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    embedder = _build_embedder()
    store = _build_store(collection)

    result = run_query(
        question,
        embedder,
        store,
        llm_client,
        top_k=top_k if top_k is not None else settings.retrieval.top_k,
        score_threshold=(
            score_threshold if score_threshold is not None else settings.retrieval.score_threshold
        ),
    )

    if result.status == "empty_collection":
        click.echo("The collection is empty. Run 'rag ingest <path>' first.", err=True)
        sys.exit(1)

    if result.status == "no_matches":
        click.echo(
            "No chunks passed the similarity score threshold for this question. "
            "Try lowering --score-threshold or rephrasing the question."
        )
        return

    click.echo(result.answer)
    click.echo("\nSources:")
    click.echo(format_sources(result.sources))


@cli.command(name="list-collections")
def list_collections():
    """List all Chroma collections in the persistent store."""
    settings = get_settings()
    import chromadb

    client = chromadb.PersistentClient(path=settings.vectorstore.persist_directory)
    names = [c.name for c in client.list_collections()]
    if not names:
        click.echo("No collections found.")
        return
    for name in names:
        click.echo(name)


@cli.command()
@click.option("--collection", default=None, help="Chroma collection name.")
def info(collection: str | None):
    """Show chunk count and persist path for a collection."""
    settings = get_settings()
    store = _build_store(collection)
    click.echo(f"Persist directory: {settings.vectorstore.persist_directory}")
    click.echo(f"Collection: {store.collection_name}")
    click.echo(f"Chunk count: {store.count()}")


@cli.command()
@click.option("--collection", default=None, help="Chroma collection name.")
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def reset(collection: str | None, yes: bool):
    """Delete all chunks in a collection."""
    store = _build_store(collection)
    if not yes:
        click.confirm(
            f"This will delete all {store.count()} chunk(s) in collection "
            f"'{store.collection_name}'. Continue?",
            abort=True,
        )
    store.reset()
    click.echo(f"Collection '{store.collection_name}' reset.")


if __name__ == "__main__":
    cli()
