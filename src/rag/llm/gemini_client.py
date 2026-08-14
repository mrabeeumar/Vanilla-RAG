from __future__ import annotations

import google.generativeai as genai

from rag.llm.base import LLMClient


class GeminiClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        genai.configure(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text or ""
