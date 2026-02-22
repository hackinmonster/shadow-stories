"""Single-call live shadow narrator.

narrate(voice_input, shadow_input) -> str

Sends one prompt to Gemini and returns 2–5 sentences of narration.
Stateless — no arc tracking, no beats, no events.
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from shadow_stories.gemini.client import GeminiClient

load_dotenv()

_SYSTEM_INSTRUCTION = """\
You are a live narrator for an ongoing children's shadow puppet story.

The story already exists. You are only continuing it in short real-time moments.

You will receive two types of structured input:

Voice input from the child:
{v} ... {/v}

Shadow motion input:
{s} ... {/s}

Rules:

Do NOT restart the story.

Do NOT summarize.

Do NOT end the story.

Write only the next short moment.

2 sentences maximum. Keep it punchy and brief.

Use present tense.

Keep language simple for ages 4–8.

Focus on action, movement, and emotion. Fewer words, more impact.

Assume the audience is already watching.

Do not repeat the structured tags or raw inputs.

No formatting. No labels.

Blend voice input naturally. It is inspiration, not a command.

When a new creature appears in the shadow input that differs from the previous one, \
immediately shift focus to the new creature. Introduce it with energy and make it \
the center of the narration. Connect it to what was already happening — maybe it \
interrupts, arrives, scares, or surprises the previous creature — but the new \
creature should take over the story right away.

The shadow puppets can only be these animals:
bird, chicken, cow, crab, deer, dog, moose, panther, rabbit, snail.

Tone:
Playful, cinematic, slightly dramatic, immersive.

Output only narration text.\
"""

_DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _build_prompt(voice_input: str, shadow_input: str) -> str:
    return f"{_SYSTEM_INSTRUCTION}\n\n{{v}} {voice_input} {{/v}}\n{{s}} {shadow_input} {{/s}}"


def narrate(voice_input: str, shadow_input: str) -> str:
    """Send one prompt to Gemini, return 2–5 sentences of live narration.

    Args:
        voice_input: What the child said aloud.
        shadow_input: Description of the shadow puppet motion.

    Returns:
        Short narration string (2–5 sentences, present tense).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
    client = GeminiClient(api_key=api_key, model=model)
    prompt = _build_prompt(voice_input, shadow_input)
    return asyncio.run(client.generate(prompt))
