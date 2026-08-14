"""Abstract interface every LLM backend implements. pipeline/query.py depends only
on this, never on a concrete client, so adding a provider is a one-file change."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str: ...
