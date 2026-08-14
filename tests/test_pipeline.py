"""End-to-end pipeline test using a fake embedder and fake LLM client so it runs
fast, offline, and without downloading any model."""

import numpy as np
import pytest

pytest.importorskip("chromadb")

from rag.config import get_settings
from rag.llm.base import LLMClient
from rag.pipeline.ingest import ingest_path
from rag.pipeline.query import run_query
from rag.vectorstore.chroma_store import ChromaStore


class FakeEmbedder:
    """Deterministic bag-of-words-ish embedding so semantically similar fake text
    maps to similar vectors, without needing a real model."""

    dim = 16

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text.lower())) % (2**32))
        v = rng.normal(size=self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.empty((0, self.dim), dtype=np.float32)

    def embed_query(self, text):
        return self._vec(text)


class FakeLLMClient(LLMClient):
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return "FAKE ANSWER"


@pytest.fixture
def settings(tmp_path):
    s = get_settings.__wrapped__()  # bypass lru_cache to get a fresh instance
    s.chunking.chunk_size = 200
    s.chunking.chunk_overlap = 20
    s.vectorstore.persist_directory = str(tmp_path / "chroma")
    return s


def test_ingest_then_query_roundtrip(tmp_path, settings):
    doc = tmp_path / "notes.txt"
    doc.write_text(
        "The mitochondria is the powerhouse of the cell. " * 10,
        encoding="utf-8",
    )

    embedder = FakeEmbedder()
    store = ChromaStore(
        persist_directory=settings.vectorstore.persist_directory,
        collection_name="test",
    )

    report = ingest_path(doc, settings, embedder, store)
    assert report.files_ingested == 1
    assert report.chunks_added > 0
    assert store.count() == report.chunks_added

    result = run_query(
        "What is the mitochondria?",
        embedder,
        store,
        FakeLLMClient(),
        top_k=3,
        score_threshold=None,
    )
    assert result.status == "ok"
    assert result.answer == "FAKE ANSWER"
    assert len(result.sources) > 0


def test_query_on_empty_collection(tmp_path, settings):
    embedder = FakeEmbedder()
    store = ChromaStore(
        persist_directory=settings.vectorstore.persist_directory,
        collection_name="empty",
    )
    result = run_query("anything", embedder, store, FakeLLMClient())
    assert result.status == "empty_collection"


def test_reingest_unchanged_file_is_idempotent(tmp_path, settings):
    doc = tmp_path / "notes.txt"
    doc.write_text("Stable content that does not change between runs. " * 5, encoding="utf-8")

    embedder = FakeEmbedder()
    store = ChromaStore(
        persist_directory=settings.vectorstore.persist_directory,
        collection_name="idempotent",
    )

    ingest_path(doc, settings, embedder, store)
    count_after_first = store.count()
    ingest_path(doc, settings, embedder, store)
    count_after_second = store.count()

    assert count_after_first == count_after_second
