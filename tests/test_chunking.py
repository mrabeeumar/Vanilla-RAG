from rag.chunking.splitter import chunk_document, split_text
from rag.types import PageText


def test_split_text_respects_chunk_size():
    text = "word " * 500  # 2500 chars
    chunks = split_text(text, chunk_size=200, chunk_overlap=20)
    assert all(len(c) <= 200 for c in chunks)


def test_split_text_empty_returns_empty():
    assert split_text("", chunk_size=100, chunk_overlap=10) == []
    assert split_text("   \n  ", chunk_size=100, chunk_overlap=10) == []


def test_split_text_overlap_carries_context():
    words = ["A" * 10, "B" * 10, "C" * 10, "D" * 10, "E" * 10]
    text = " ".join(words)
    chunks = split_text(text, chunk_size=25, chunk_overlap=8, separators=[" "])
    assert len(chunks) >= 2
    assert all(len(c) <= 25 for c in chunks)
    # end of chunk[0] should reappear at start of chunk[1] (the overlap)
    assert chunks[0][-8:] in chunks[1]


def test_split_text_hard_fallback_no_separators():
    text = "x" * 1000  # single token, no whitespace/newlines at all
    chunks = split_text(text, chunk_size=100, chunk_overlap=10, separators=["\n\n", "\n", " ", ""])
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) > 1


def test_split_text_rejects_overlap_gte_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        split_text("hello world", chunk_size=10, chunk_overlap=10)


def test_chunk_document_assigns_running_index_across_pages():
    pages = [
        PageText(text="Para one. " * 20, page_number=1),
        PageText(text="Para two. " * 20, page_number=2),
    ]
    chunks = chunk_document(pages, source="doc.pdf", chunk_size=50, chunk_overlap=10)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
    assert all(c.source == "doc.pdf" for c in chunks)
    # page numbers preserved per-chunk
    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]
    assert page1_chunks and page2_chunks


def test_chunk_document_skips_blank_pages():
    pages = [PageText(text="   ", page_number=1), PageText(text="Real content here.", page_number=2)]
    chunks = chunk_document(pages, source="doc.txt", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].page_number == 2
