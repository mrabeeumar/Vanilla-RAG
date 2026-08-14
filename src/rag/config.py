"""Central Settings object: env vars override config/default.yaml which overrides
hardcoded defaults. Everything else in the package imports `get_settings()`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class MissingAPIKeyError(RuntimeError):
    """Raised when the selected LLM provider has no API key configured."""


@dataclass
class EmbeddingSettings:
    model_name: str = "BAAI/bge-small-en-v1.5"
    pooling: str = "cls"
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""
    max_seq_length: int = 512
    batch_size: int = 32
    normalize: bool = True
    device: str | None = None


@dataclass
class ChunkingSettings:
    chunk_size: int = 800
    chunk_overlap: int = 120


@dataclass
class VectorStoreSettings:
    persist_directory: str = "./chroma_db"
    collection_name: str = "documents"


@dataclass
class RetrievalSettings:
    top_k: int = 5
    score_threshold: float | None = 0.3


@dataclass
class IngestSettings:
    data_dir: str = "./data"
    auto_ingest: bool = True


@dataclass
class LLMSettings:
    provider: str = "groq"
    groq_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-1.5-flash"
    temperature: float = 0.2
    max_tokens: int = 1024
    groq_api_key: str | None = None
    gemini_api_key: str | None = None


@dataclass
class Settings:
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    vectorstore: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)

    def require_llm_api_key(self) -> str:
        """Return the API key for the active provider, or raise MissingAPIKeyError."""
        if self.llm.provider == "groq":
            if not self.llm.groq_api_key:
                raise MissingAPIKeyError(
                    "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                    "Groq API key (get one free at https://console.groq.com)."
                )
            return self.llm.groq_api_key
        if self.llm.provider == "gemini":
            if not self.llm.gemini_api_key:
                raise MissingAPIKeyError(
                    "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
                    "Gemini API key (get one free at https://aistudio.google.com)."
                )
            return self.llm.gemini_api_key
        raise MissingAPIKeyError(
            f"Unknown LLM_PROVIDER '{self.llm.provider}'. Expected 'groq' or 'gemini'."
        )


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> Settings:
    """Load settings once and cache them. env vars > yaml > dataclass defaults."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    yaml_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw = _load_yaml(yaml_path)

    emb_raw = raw.get("embedding", {})
    chunk_raw = raw.get("chunking", {})
    vs_raw = raw.get("vectorstore", {})
    ret_raw = raw.get("retrieval", {})
    ingest_raw = raw.get("ingest", {})
    llm_raw = raw.get("llm", {})

    embedding = EmbeddingSettings(
        model_name=emb_raw.get("model_name", EmbeddingSettings.model_name),
        pooling=emb_raw.get("pooling", EmbeddingSettings.pooling),
        query_prefix=emb_raw.get("query_prefix", EmbeddingSettings.query_prefix),
        passage_prefix=emb_raw.get("passage_prefix", EmbeddingSettings.passage_prefix),
        max_seq_length=emb_raw.get("max_seq_length", EmbeddingSettings.max_seq_length),
        batch_size=emb_raw.get("batch_size", EmbeddingSettings.batch_size),
        normalize=emb_raw.get("normalize", EmbeddingSettings.normalize),
        device=emb_raw.get("device"),
    )

    chunking = ChunkingSettings(
        chunk_size=chunk_raw.get("chunk_size", ChunkingSettings.chunk_size),
        chunk_overlap=chunk_raw.get("chunk_overlap", ChunkingSettings.chunk_overlap),
    )

    vectorstore = VectorStoreSettings(
        persist_directory=vs_raw.get(
            "persist_directory", VectorStoreSettings.persist_directory
        ),
        collection_name=vs_raw.get(
            "collection_name", VectorStoreSettings.collection_name
        ),
    )

    retrieval = RetrievalSettings(
        top_k=ret_raw.get("top_k", RetrievalSettings.top_k),
        score_threshold=ret_raw.get(
            "score_threshold", RetrievalSettings.score_threshold
        ),
    )

    ingest = IngestSettings(
        data_dir=ingest_raw.get("data_dir", IngestSettings.data_dir),
        auto_ingest=ingest_raw.get("auto_ingest", IngestSettings.auto_ingest),
    )

    llm = LLMSettings(
        provider=os.environ.get("LLM_PROVIDER", llm_raw.get("provider", LLMSettings.provider)),
        groq_model=llm_raw.get("groq_model", LLMSettings.groq_model),
        gemini_model=llm_raw.get("gemini_model", LLMSettings.gemini_model),
        temperature=llm_raw.get("temperature", LLMSettings.temperature),
        max_tokens=llm_raw.get("max_tokens", LLMSettings.max_tokens),
        groq_api_key=os.environ.get("GROQ_API_KEY") or None,
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
    )

    return Settings(
        embedding=embedding,
        chunking=chunking,
        vectorstore=vectorstore,
        retrieval=retrieval,
        ingest=ingest,
        llm=llm,
    )
