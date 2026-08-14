"""Dispatches to a concrete LLMClient based on settings.llm.provider."""

from __future__ import annotations

from rag.config import Settings
from rag.llm.base import LLMClient


def get_llm_client(settings: Settings) -> LLMClient:
    api_key = settings.require_llm_api_key()  # raises MissingAPIKeyError if unset

    if settings.llm.provider == "groq":
        from rag.llm.groq_client import GroqClient

        return GroqClient(
            api_key=api_key,
            model=settings.llm.groq_model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

    if settings.llm.provider == "gemini":
        from rag.llm.gemini_client import GeminiClient

        return GeminiClient(
            api_key=api_key,
            model=settings.llm.gemini_model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )

    raise ValueError(f"Unknown LLM provider: {settings.llm.provider!r}")
