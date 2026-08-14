"""Manual embedding module built directly on transformers' AutoTokenizer/AutoModel
(no sentence-transformers wrapper) so tokenization, pooling, normalization, and
batching are all explicit and visible.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from rag.utils.logging_setup import get_logger

logger = get_logger(__name__)


class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        pooling: str = "cls",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        passage_prefix: str = "",
        max_seq_length: int = 512,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got '{pooling}'")

        self.model_name = model_name
        self.pooling = pooling
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.normalize = normalize

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading embedding model '%s' on %s...", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Embedding model loaded.")

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a list of passages, returning a (N, dim) float32 array."""
        prefixed = [self.passage_prefix + t for t in texts]
        return self._embed_all(prefixed)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query, returning a (dim,) float32 array."""
        result = self._embed_all([self.query_prefix + text])
        return result[0]

    def _embed_all(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        all_embeddings: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_embeddings.append(self._embed_batch(batch))

        stacked = torch.cat(all_embeddings, dim=0)
        return stacked.cpu().numpy().astype(np.float32)

    def _embed_batch(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)

        pooled = self._pool(output.last_hidden_state, encoded["attention_mask"])

        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        return pooled

    def _pool(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]

        # Mean pooling: naive .mean(dim=1) would pollute the average with padding
        # vectors, so padding positions must be masked out before averaging.
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
