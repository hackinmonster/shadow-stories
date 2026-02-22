"""Tests for shadow_stories.narrate — all mocked, no real API calls."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from shadow_stories.narrate import _build_prompt, narrate


def test_build_prompt_includes_inputs() -> None:
    prompt = _build_prompt("the dragon roars", "wings spread wide")
    assert "the dragon roars" in prompt
    assert "wings spread wide" in prompt
    assert "{v}" in prompt
    assert "{s}" in prompt


def test_build_prompt_includes_system_instruction() -> None:
    prompt = _build_prompt("hi", "wave")
    assert "shadow puppet" in prompt.lower()
    assert "2 sentences maximum" in prompt


@patch("shadow_stories.narrate.GeminiClient")
def test_narrate_returns_text(MockClient: object) -> None:
    mock_instance = MockClient.return_value  # type: ignore[attr-defined]
    mock_instance.generate = AsyncMock(return_value="The dragon soars high above the trees.")

    result = narrate(voice_input="fly", shadow_input="puppet moves up")

    assert result == "The dragon soars high above the trees."
    mock_instance.generate.assert_awaited_once()


@patch("shadow_stories.narrate.GeminiClient")
def test_narrate_passes_correct_prompt(MockClient: object) -> None:
    mock_instance = MockClient.return_value  # type: ignore[attr-defined]
    mock_instance.generate = AsyncMock(return_value="Narration text.")

    narrate(voice_input="roar", shadow_input="claws out")

    call_args = mock_instance.generate.call_args
    prompt_sent = call_args[0][0]
    assert "roar" in prompt_sent
    assert "claws out" in prompt_sent


@patch("shadow_stories.narrate.GeminiClient")
@patch.dict("os.environ", {"GEMINI_MODEL": "gemini-2.0-flash"})
def test_narrate_uses_env_model(MockClient: object) -> None:
    mock_instance = MockClient.return_value  # type: ignore[attr-defined]
    mock_instance.generate = AsyncMock(return_value="Story continues.")

    narrate(voice_input="jump", shadow_input="leaps forward")

    MockClient.assert_called_once()  # type: ignore[attr-defined]
    _, kwargs = MockClient.call_args  # type: ignore[attr-defined]
    assert kwargs.get("model") == "gemini-2.0-flash"
