import numpy as np
import pytest

pytest.importorskip("chromadb")

from rag.types import Chunk
from rag.vectorstore.chroma_store import ChromaStore


def _unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_directory=str(tmp_path / "chroma"), collection_name="test")


def test_upsert_and_count(store):
    chunks = [
        Chunk(text="the cat sat on the mat", chunk_index=0, source="a.txt", page_number=None),
        Chunk(text="dogs are loyal animals", chunk_index=1, source="a.txt", page_number=None),
    ]
    embeddings = np.stack([_unit_vector(0), _unit_vector(1)])
    store.upsert_chunks(chunks, embeddings)
    assert store.count() == 2


def test_query_returns_closest_first(store):
    chunks = [
        Chunk(text="near", chunk_index=0, source="a.txt", page_number=1),
        Chunk(text="far", chunk_index=1, source="a.txt", page_number=2),
    ]
    v_near = _unit_vector(42)
    # a vector very close to v_near
    v_near_dup = v_near + np.random.default_rng(1).normal(scale=0.01, size=v_near.shape).astype(np.float32)
    v_near_dup = v_near_dup / np.linalg.norm(v_near_dup)
    v_far = -v_near  # opposite direction -> low cosine similarity

    store.upsert_chunks(chunks, np.stack([v_near_dup, v_far]))
    results = store.query(v_near, top_k=2, score_threshold=None)

    assert len(results) == 2
    assert results[0].text == "near"
    assert results[0].score > results[1].score


def test_query_applies_score_threshold(store):
    chunks = [Chunk(text="only chunk", chunk_index=0, source="a.txt", page_number=None)]
    v = _unit_vector(7)
    store.upsert_chunks(chunks, np.stack([v]))

    opposite = -v
    results = store.query(opposite, top_k=5, score_threshold=0.9)
    assert results == []


def test_upsert_is_idempotent_for_unchanged_chunks(store):
    chunks = [Chunk(text="stable content", chunk_index=0, source="a.txt", page_number=None)]
    v = _unit_vector(3)
    store.upsert_chunks(chunks, np.stack([v]))
    store.upsert_chunks(chunks, np.stack([v]))  # re-ingest same file
    assert store.count() == 1


def test_reset_clears_collection(store):
    chunks = [Chunk(text="temp", chunk_index=0, source="a.txt", page_number=None)]
    store.upsert_chunks(chunks, np.stack([_unit_vector(5)]))
    assert store.count() == 1
    store.reset()
    assert store.count() == 0


def test_delete_by_source(store):
    chunks = [
        Chunk(text="keep me", chunk_index=0, source="keep.txt", page_number=None),
        Chunk(text="delete me", chunk_index=0, source="drop.txt", page_number=None),
    ]
    store.upsert_chunks(chunks, np.stack([_unit_vector(10), _unit_vector(11)]))
    store.delete_by_source("drop.txt")
    assert store.count() == 1
