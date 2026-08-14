from pathlib import Path

import pytest

from rag.loaders.base import UnsupportedFileTypeError
from rag.loaders.registry import get_loader, load_file
from rag.loaders.text_loader import TextLoader


def test_text_loader_reads_content(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("hello world", encoding="utf-8")
    pages = TextLoader().load(p)
    assert len(pages) == 1
    assert pages[0].text == "hello world"
    assert pages[0].page_number is None


def test_text_loader_empty_file_returns_no_pages(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n  ", encoding="utf-8")
    assert TextLoader().load(p) == []


def test_registry_dispatches_by_extension(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# heading", encoding="utf-8")
    assert isinstance(get_loader(p), TextLoader)
    pages = load_file(p)
    assert pages[0].text == "# heading"


def test_registry_raises_on_unsupported_extension(tmp_path):
    p = tmp_path / "doc.xyz"
    p.write_text("data", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError):
        get_loader(p)
