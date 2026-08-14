"""Interactive chatbot entry point. Run with `python main.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from rag.config import MissingAPIKeyError, get_settings
from rag.embeddings.embedder import Embedder
from rag.llm.factory import get_llm_client
from rag.pipeline.ingest import ingest_path
from rag.pipeline.query import run_query
from rag.vectorstore.chroma_store import ChromaStore

_BANNER_FONT = {
    "V": [
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".#.#.",
        ".#.#.",
        "..#..",
    ],
    "A": [
        "..#..",
        ".#.#.",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
    ],
    "N": [
        "#...#",
        "##..#",
        "#.#.#",
        "#..##",
        "#...#",
        "#...#",
        "#...#",
    ],
    "I": [
        ".###.",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        ".###.",
    ],
    "L": [
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        "#####",
    ],
    "R": [
        "####.",
        "#...#",
        "#...#",
        "####.",
        "#.#..",
        "#..#.",
        "#...#",
    ],
    "G": [
        ".###.",
        "#...#",
        "#....",
        "#.##.",
        "#...#",
        "#...#",
        ".###.",
    ],
    " ": ["..."] * 7,
}

_BANNER_TEXT_COLOR = "bright_white"
_BANNER_BORDER_COLOR = "yellow"


def _make_banner(text: str, padding: int = 2) -> str:
    letters = [c.upper() for c in text]
    glyph_rows = []
    for row_idx in range(7):
        parts = [_BANNER_FONT[letter][row_idx].replace(".", " ") for letter in letters]
        glyph_rows.append(" ".join(parts))

    content_width = max(len(row) for row in glyph_rows)
    inner_width = content_width + padding * 2

    border = click.style("+" + "-" * inner_width + "+", fg=_BANNER_BORDER_COLOR, bold=True)
    blank_line = (
        click.style("|", fg=_BANNER_BORDER_COLOR, bold=True)
        + " " * inner_width
        + click.style("|", fg=_BANNER_BORDER_COLOR, bold=True)
    )

    lines = [border, blank_line]
    for row in glyph_rows:
        padded_row = row.ljust(content_width)
        lines.append(
            click.style("|", fg=_BANNER_BORDER_COLOR, bold=True)
            + " " * padding
            + click.style(padded_row, fg=_BANNER_TEXT_COLOR, bold=True)
            + " " * padding
            + click.style("|", fg=_BANNER_BORDER_COLOR, bold=True)
        )
    lines.append(blank_line)
    lines.append(border)
    return "\n".join(lines)


BANNER = _make_banner("VANILLA RAG")

HELP_TEXT = """\
Commands:
  /ingest <path>   Ingest a file or directory into the vector store
  /reset           Wipe all chunks in the current collection
  /info            Show persist directory, collection, and chunk count
  /help            Show this message
  /exit, /quit     Leave the chat
Anything else is treated as a question."""


def _echo_bot(text: str) -> None:
    click.echo(click.style("Bot", fg="green", bold=True) + click.style(" > ", fg="green") + text)


def _echo_system(text: str) -> None:
    click.echo(click.style(text, fg="yellow"))


def _echo_error(text: str) -> None:
    click.echo(click.style(text, fg="red"))


def main() -> None:
    click.echo(BANNER)
    click.echo(click.style("A from-scratch RAG chatbot. Type /help for commands.\n", fg="bright_black"))

    settings = get_settings()

    try:
        llm_client = get_llm_client(settings)
    except MissingAPIKeyError as e:
        _echo_error(str(e))
        sys.exit(1)

    _echo_system("Loading embedding model (first run downloads it, ~130MB)...")
    embedder = Embedder(
        model_name=settings.embedding.model_name,
        device=settings.embedding.device,
        pooling=settings.embedding.pooling,
        query_prefix=settings.embedding.query_prefix,
        passage_prefix=settings.embedding.passage_prefix,
        max_seq_length=settings.embedding.max_seq_length,
        batch_size=settings.embedding.batch_size,
        normalize=settings.embedding.normalize,
    )
    store = ChromaStore(
        persist_directory=settings.vectorstore.persist_directory,
        collection_name=settings.vectorstore.collection_name,
    )

    if settings.ingest.auto_ingest:
        store.reset()
        data_dir = Path(settings.ingest.data_dir)
        if data_dir.exists():
            _echo_system(f"Auto-ingesting {data_dir}...")
            report = ingest_path(data_dir, settings, embedder, store)
            _echo_system(
                f"Ingested {report.files_ingested} file(s), {report.chunks_added} chunk(s)."
            )
        else:
            _echo_system(
                f"Auto-ingest enabled but {data_dir} does not exist; collection left empty."
            )

    click.echo(
        click.style(
            f"Ready. Provider: {settings.llm.provider} | "
            f"Collection: {store.collection_name} | Chunks: {store.count()}\n",
            fg="bright_black",
        )
    )

    while True:
        try:
            question = click.prompt(click.style("You >", fg="blue", bold=True), prompt_suffix=" ")
        except (EOFError, click.exceptions.Abort):
            click.echo()
            _echo_system("Goodbye!")
            break

        question = question.strip()
        if not question:
            continue

        if question in ("/exit", "/quit"):
            _echo_system("Goodbye!")
            break

        if question == "/help":
            click.echo(HELP_TEXT)
            continue

        if question == "/info":
            click.echo(f"Persist directory: {settings.vectorstore.persist_directory}")
            click.echo(f"Collection: {store.collection_name}")
            click.echo(f"Chunk count: {store.count()}")
            continue

        if question == "/reset":
            if click.confirm(
                f"This will delete all {store.count()} chunk(s) in collection "
                f"'{store.collection_name}'. Continue?",
                default=False,
            ):
                store.reset()
                _echo_system(f"Collection '{store.collection_name}' reset.")
            continue

        if question.startswith("/ingest"):
            parts = question.split(maxsplit=1)
            if len(parts) != 2:
                _echo_error("Usage: /ingest <path>")
                continue
            path = Path(parts[1].strip().strip('"'))
            if not path.exists():
                _echo_error(f"Path does not exist: {path}")
                continue
            try:
                report = ingest_path(path, settings, embedder, store)
            except Exception as e:
                _echo_error(f"Ingest failed: {e}")
                continue
            if report.unsupported_files:
                click.echo(f"Skipped {len(report.unsupported_files)} unsupported file(s).")
            if report.empty_files:
                click.echo(f"Skipped {len(report.empty_files)} empty/unextractable file(s).")
            _echo_system(
                f"Ingested {report.files_ingested} file(s), {report.chunks_added} chunk(s) added. "
                f"Collection now has {report.total_chunks_in_collection} chunk(s)."
            )
            continue

        if question.startswith("/"):
            _echo_error(f"Unknown command: {question}. Type /help for a list of commands.")
            continue

        try:
            with click.progressbar(length=1, label="Thinking", show_eta=False, show_percent=False) as bar:
                result = run_query(
                    question,
                    embedder,
                    store,
                    llm_client,
                    top_k=settings.retrieval.top_k,
                    score_threshold=settings.retrieval.score_threshold,
                )
                bar.update(1)
        except Exception as e:
            _echo_error(f"Something went wrong: {e}")
            continue

        if result.status == "empty_collection":
            _echo_error("The collection is empty. Use /ingest <path> to add documents first.")
            continue

        if result.status == "no_matches":
            _echo_bot("I couldn't find anything relevant. Try rephrasing, or lower the score threshold in config/default.yaml.")
            continue

        _echo_bot(result.answer or "")
        click.echo()


if __name__ == "__main__":
    main()
