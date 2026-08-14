"""Thin wrapper around a persistent local ChromaDB collection."""

from __future__ import annotations

import numpy as np
import chromadb

from rag.types import Chunk, RetrievedChunk
from rag.utils.hashing import chunk_id
from rag.utils.logging_setup import get_logger

logger = get_logger(__name__)


class ChromaStore:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Chroma defaults to L2 distance; embeddings are L2-normalized so cosine
        # similarity is what we actually want, and it must be set explicitly here.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        ids = [chunk_id(c.source, c.chunk_index) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number if c.page_number is not None else -1,
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        n_results = min(top_k, self.count()) or top_k
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=max(n_results, 1),
        )

        ids = results.get("ids", [[]])[0]
        if not ids:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved: list[RetrievedChunk] = []
        for doc, meta, distance in zip(documents, metadatas, distances):
            # Cosine space: Chroma returns distance = 1 - cosine_similarity.
            score = 1.0 - distance
            if score_threshold is not None and score < score_threshold:
                continue
            page_number = meta.get("page_number")
            retrieved.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    chunk_index=meta.get("chunk_index", -1),
                    page_number=None if page_number == -1 else page_number,
                    score=score,
                )
            )
        return retrieved

    def delete_by_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()
