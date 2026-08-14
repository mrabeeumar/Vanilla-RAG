"""These tests download a real (small) HF model on first run, so they're marked
slow and skipped if the model can't be fetched (e.g. no network in CI)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from rag.embeddings.embedder import Embedder

MODEL_NAME = "hf-internal-testing/tiny-random-BertModel"


@pytest.fixture(scope="module")
def embedder():
    try:
        return Embedder(model_name=MODEL_NAME, pooling="cls", batch_size=2)
    except Exception as e:
        pytest.skip(f"Could not load test embedding model (offline?): {e}")


def test_embed_documents_shape(embedder):
    texts = ["hello world", "another sentence", "a third one here"]
    out = embedder.embed_documents(texts)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, embedder.dim)
    assert out.dtype == np.float32


def test_embed_query_shape(embedder):
    out = embedder.embed_query("what is this about?")
    assert out.shape == (embedder.dim,)


def test_embeddings_are_normalized(embedder):
    out = embedder.embed_documents(["some text", "more text"])
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_mean_pooling_ignores_padding():
    """A short and a long text batched together get right-padded; mean pooling
    must mask out the pad positions or the short text's embedding gets polluted."""
    try:
        e = Embedder(model_name=MODEL_NAME, pooling="mean", batch_size=2)
    except Exception as exc:
        pytest.skip(f"Could not load test embedding model (offline?): {exc}")

    short_alone = e.embed_documents(["hi"])[0]
    short_batched = e.embed_documents(["hi", "this is a much longer sentence than the other one"])[0]
    np.testing.assert_allclose(short_alone, short_batched, atol=1e-4)


def test_embed_documents_empty_list(embedder):
    out = embedder.embed_documents([])
    assert out.shape == (0, embedder.dim)
