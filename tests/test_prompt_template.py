from rag.prompting.template import build_prompt
from rag.types import RetrievedChunk


def test_build_prompt_numbers_and_includes_all_chunks():
    chunks = [
        RetrievedChunk(text="Paris is the capital of France.", source="geo.txt", chunk_index=0, page_number=None, score=0.91),
        RetrievedChunk(text="The Eiffel Tower is in Paris.", source="geo.txt", chunk_index=1, page_number=2, score=0.77),
    ]
    system_prompt, user_prompt = build_prompt("What is the capital of France?", chunks)

    assert "context" in system_prompt.lower()
    assert "[1]" in user_prompt
    assert "[2]" in user_prompt
    assert "Paris is the capital of France." in user_prompt
    assert "The Eiffel Tower is in Paris." in user_prompt
    assert "geo.txt" in user_prompt
    assert "page 2" in user_prompt
    assert "What is the capital of France?" in user_prompt


def test_build_prompt_empty_chunks_still_returns_valid_prompt():
    system_prompt, user_prompt = build_prompt("anything?", [])
    assert "anything?" in user_prompt
    assert system_prompt
