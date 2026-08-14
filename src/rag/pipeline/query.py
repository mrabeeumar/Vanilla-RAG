"""Orchestrates: embed query -> retrieve -> build prompt -> generate -> format answer."""

from __future__ import annotations

from dataclasses import dataclass

from rag.embeddings.embedder import Embedder
from rag.llm.base import LLMClient
from rag.prompting.template import build_prompt
from rag.types import RetrievedChunk
from rag.utils.logging_setup import get_logger
from rag.vectorstore.chroma_store import ChromaStore

logger = get_logger(__name__)


@dataclass
class QueryResult:
    answer: str | None
    sources: list[RetrievedChunk]
    status: str  # "ok" | "empty_collection" | "no_matches"


def run_query(
    question: str,
    embedder: Embedder,
    store: ChromaStore,
    llm_client: LLMClient,
    top_k: int = 5,
    score_threshold: float | None = 0.3,
) -> QueryResult:
    if store.count() == 0:
        return QueryResult(answer=None, sources=[], status="empty_collection")

    query_embedding = embedder.embed_query(question)
    retrieved = store.query(
        query_embedding, top_k=top_k, score_threshold=score_threshold
    )

    if not retrieved:
        return QueryResult(answer=None, sources=[], status="no_matches")

    system_prompt, user_prompt = build_prompt(question, retrieved)
    answer = llm_client.generate(user_prompt, system_prompt=system_prompt)

    return QueryResult(answer=answer, sources=retrieved, status="ok")


def format_sources(sources: list[RetrievedChunk]) -> str:
    """Deduplicated Sources footer, independent of whether the model's [n] markers
    in the answer text are reliable."""
    seen: dict[tuple[str, int | None], float] = {}
    for chunk in sources:
        key = (chunk.source, chunk.page_number)
        if key not in seen or chunk.score > seen[key]:
            seen[key] = chunk.score

    lines = []
    for (source, page), score in seen.items():
        loc = f"{source}"
        if page is not None:
            loc += f" (page {page})"
        lines.append(f"  - {loc} [score: {score:.2f}]")
    return "\n".join(lines)
