"""Async Gemini API client wrapper.

Single responsibility: send a prompt string, return response text.
No domain logic, no schema enforcement, no story state.
"""
from __future__ import annotations

import asyncio

import google.genai as genai
from google.genai import types as genai_types


class GeminiClientError(Exception):
    """Raised when the Gemini API call fails unrecoverably.

    Attributes:
        reason: "network" | "safety_block" | "malformed_response" | "timeout"
        details: Raw SDK exception message or finish_reason string.
    """

    def __init__(self, reason: str, details: str) -> None:
        self.reason = reason
        self.details = details
        super().__init__(f"GeminiClient error [{reason}]: {details}")


class GeminiClient:
    """Async Gemini wrapper. Sends a prompt, returns accumulated text."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._base_delay = base_delay

    async def generate(self, prompt: str, *, temperature: float = 0.9) -> str:
        """Send prompt to Gemini, return accumulated response text.

        Raises:
            GeminiClientError: On network failure, safety block, or empty response.
        """
        config = genai_types.GenerateContentConfig(temperature=temperature)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await self._stream(prompt, config)
            except GeminiClientError:
                raise  # non-retryable
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._base_delay * (2**attempt))

        raise GeminiClientError(
            reason="network",
            details=str(last_exc) if last_exc else "unknown error after retries",
        )

    async def _stream(self, prompt: str, config: genai_types.GenerateContentConfig) -> str:
        chunks: list[str] = []
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=config,
        )
        async for chunk in stream:
            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.finish_reason == genai_types.FinishReason.SAFETY:
                    raise GeminiClientError(
                        reason="safety_block",
                        details=f"finish_reason=SAFETY on model={self._model}",
                    )
            if chunk.text:
                chunks.append(chunk.text)

        result = "".join(chunks)
        if not result.strip():
            raise GeminiClientError(
                reason="malformed_response",
                details="Empty response after streaming",
            )
        return result
