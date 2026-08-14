# Vanilla RAG

A Retrieval-Augmented Generation pipeline built from scratch — no LangChain, no
LlamaIndex. Chunking, embeddings, vector search, prompt assembly, and generation
are all implemented directly so the internals stay visible.

## Stack

- **Embeddings**: raw `transformers` (`AutoTokenizer`/`AutoModel`), manual
  tokenization, pooling, and normalization — no `sentence-transformers` wrapper.
- **Vector store**: [ChromaDB](https://www.trychroma.com/), persistent local client.
- **LLM**: [Groq](https://console.groq.com) (free tier, default) or
  [Gemini](https://aistudio.google.com) (alternative), behind a swappable interface.
- **Interface**: CLI (`click`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

copy .env.example .env
# edit .env and set GROQ_API_KEY=... (or GEMINI_API_KEY=... + LLM_PROVIDER=gemini)
```

The first embedding call downloads `BAAI/bge-small-en-v1.5` (~130MB) to the local
Hugging Face cache (`~/.cache/huggingface`, override with `HF_HOME`). This can take
a minute; subsequent runs are instant.

## Usage

### Interactive chat

```bash
python main.py
```

Starts a REPL-style chatbot. On launch it loads the embedding model and, if
`ingest.auto_ingest` is enabled in `config/default.yaml` (default: on), wipes
the collection and re-ingests everything under `ingest.data_dir` (default:
`./data`). Then it drops into a chat loop:

```
Commands:
  /ingest <path>   Ingest a file or directory into the vector store
  /reset           Wipe all chunks in the current collection
  /info            Show persist directory, collection, and chunk count
  /help            Show this message
  /exit, /quit     Leave the chat
Anything else is treated as a question.
```

### CLI

```bash
# Ingest a file or a whole directory (.txt, .md, .pdf supported)
python -m rag.cli ingest ./data/raw/

# Ask a question
python -m rag.cli query "What does the document say about X?"

# Inspect the store
python -m rag.cli info
python -m rag.cli list-collections

# Wipe a collection
python -m rag.cli reset --yes
```

Re-running `ingest` on an unchanged file is a no-op (chunk IDs are deterministic,
derived from `source_path::chunk_index`). If a file's content shrinks (fewer
chunks than before), use `--overwrite` to delete-then-reinsert and avoid stale
trailing chunks.

## Configuration

- Secrets (`GROQ_API_KEY`, `GEMINI_API_KEY`, `LLM_PROVIDER`) live in `.env`.
- Tunables (chunk size/overlap, top_k, score threshold, model names, Chroma
  paths, `ingest.auto_ingest`/`ingest.data_dir`) live in `config/default.yaml`.

Alternative embedding models (swap `embedding.model_name` in the config):

- `BAAI/bge-small-en-v1.5` (default) — asymmetric, CLS pooling, query prefix required.
- `intfloat/e5-small-v2` — asymmetric, mean pooling, `query:` / `passage:` prefixes.
- `sentence-transformers/all-MiniLM-L6-v2` — symmetric, no prefixes needed, simplest.

## Architecture

```
src/rag/
├── cli.py                  # click subcommands: ingest, query, info, list-collections, reset
├── config.py                # Settings: env > yaml > defaults
├── types.py                  # PageText, Chunk, RetrievedChunk
├── loaders/                  # txt/markdown/pdf -> list[PageText]
├── chunking/splitter.py      # recursive char-based splitter with overlap
├── embeddings/embedder.py    # manual PyTorch embedding module
├── vectorstore/chroma_store.py # persistent Chroma wrapper (cosine space)
├── llm/                      # LLMClient ABC + Groq/Gemini implementations + factory
├── prompting/template.py     # context-injection prompt builder
└── pipeline/                 # ingest.py, query.py orchestration
```

## Testing

```bash
pytest
```

Chunking, Chroma, prompt template, and pipeline tests run fully offline. Embedder
tests download a tiny HF test model on first run and are skipped automatically if
there's no network access.

## Known limitations (v1)

- Chunking is character-based, not token-aware (avoids a tokenizer mismatch with
  whichever embedding model is configured).
- Scanned/image-only PDF pages have no extractable text and are skipped with a
  warning — OCR is out of scope.
- No streaming LLM output, no hybrid/BM25 + rerank, no web API layer.
