from __future__ import annotations

from groq import Groq

from rag.llm.base import LLMClient


class GroqClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""
