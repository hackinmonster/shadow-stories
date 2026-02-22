"""Live shadow narrator — wired to root main.py classifier.

Uses the real camera + ConvNext classifier from main.py to drive Gemini narration
with rolling context memory.

Usage:
    # Run forever, pulling from real camera classifier
    uv run python -m shadow_stories.live_stream_test

    # Interactive voice mode — capture mic speech each turn
    uv run python -m shadow_stories.live_stream_test --interactive-voice

    # Debug mode: shows live camera window with bbox + label overlay
    uv run python -m shadow_stories.live_stream_test --debug

    uv run python -m shadow_stories.live_stream_test --inference-interval 1.5 --min-obs 5 --count 5
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
import wave
from collections import deque
from datetime import datetime

import google.genai as genai
import numpy as np
import sounddevice as sd

# Ensure the project root is on sys.path so `import main` resolves to root main.py
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
import main as shadow_classifier  # noqa: E402

from dotenv import load_dotenv

from shadow_stories.gemini.client import GeminiClient, GeminiClientError
from shadow_stories.narrate import _DEFAULT_MODEL, _SYSTEM_INSTRUCTION
from shadow_stories.tts import init_tts, enqueue, is_speaking, cleanup_tts
from shadow_stories.tts.sfx import is_playing_sfx, play_sfx, start_ambient, stop_ambient

load_dotenv()

_CONTEXT_MAXLEN = 4
# min seconds between Gemini inferences
_INFERENCE_INTERVAL = 10.0
# min observations to collect before allowing an inference
_MIN_OBSERVATIONS = 5
# how often to poll the classifier (seconds)
_POLL_INTERVAL = 0.4
_VOICE_SAMPLE_RATE = 16000
_VOICE_MAX_SECONDS = 4.0
_VOICE_MIN_RMS = 0.01
_VOICE_MIN_PEAK = 0.04
_VOICE_OUTPUT_IDLE_TIMEOUT = 6.0
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_log_file = None


def _open_log():
    global _log_file
    os.makedirs(_LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_LOG_DIR, f"session_{stamp}.txt")
    _log_file = open(path, "w", encoding="utf-8")
    print(f"Logging to {path}")
    return _log_file


def _log(iteration, classifier, voice, prompt, narration):
    if _log_file is None:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log_file.write(f"{'='*64}\n")
    _log_file.write(f"[{ts}] #{iteration}\n")
    _log_file.write(f"classifier: {classifier}\n")
    _log_file.write(f"voice: {voice or '(silent)'}\n")
    _log_file.write(f"\n--- prompt ---\n{prompt}\n")
    _log_file.write(f"\n--- narration ---\n{narration}\n\n")
    _log_file.flush()


def _summarize_observations(observations: list[str]) -> str:
    """Return the single most common observation from the batch."""
    from collections import Counter
    if not observations:
        return ""
    return Counter(observations).most_common(1)[0][0]


def _build_prompt(
    voice: str,
    shadow_input: str,
    context: deque[str],
    persistent_facts: list[str],
) -> str:
    """Build prompt with optional facts/context and explicit voice-priority policy."""
    facts_block = ""
    if persistent_facts:
        lines = "\n".join(f"- {f}" for f in persistent_facts)
        facts_block = f"Story facts to maintain:\n\n{lines}\n\n"

    ctx_block = ""
    if context:
        ctx_block = "Current scene context:\n\n" + "\n".join(context) + "\n\n"

    voice_priority_block = ""
    if voice.strip():
        voice_priority_block = (
            "Voice + shadow integration rules:\n"
            "1) BOTH inputs are important: use both voice and shadow every turn.\n"
            "2) Prioritize voice when deciding emphasis and direction.\n"
            "3) The first sentence should reflect the latest voice input.\n"
            "4) Also include at least one concrete shadow detail (animal and/or motion).\n"
            "5) If voice and shadow conflict, resolve toward voice while still acknowledging shadow.\n\n"
        )

    return (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        f"{voice_priority_block}"
        f"{facts_block}"
        f"{ctx_block}"
        f"{{v}}\n{voice}\n{{/v}}\n\n"
        f"{{s}}\n{shadow_input}\n{{/s}}"
    )


def _print_banner(
    iteration: int,
    shadow_input: str,
    voice: str,
    ctx_len: int,
    facts_len: int,
) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    voice_display = repr(voice) if voice else "(silent)"
    print(
        f"\n[{ts}] #{iteration}"
        f"  facts={facts_len}"
        f"  ctx={ctx_len}/{_CONTEXT_MAXLEN}"
        f"  classifier={shadow_input!r}"
        f"  voice={voice_display}"
    )
    print("  Narrating...", end="", flush=True)


def _record_voice_clip() -> np.ndarray | None:
    """Record a short mono mic clip and return float32 samples in [-1, 1]."""
    frames = int(_VOICE_SAMPLE_RATE * _VOICE_MAX_SECONDS)
    print(f"\n  Listening on mic ({_VOICE_MAX_SECONDS:.1f}s)...", end="", flush=True)
    try:
        audio = sd.rec(frames, samplerate=_VOICE_SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:
        print(f"\n  [voice] microphone error: {exc}")
        return None

    print(" done.")
    clip = audio.reshape(-1)
    if clip.size == 0:
        return None

    rms = float(np.sqrt(np.mean(np.square(clip))))
    peak = float(np.max(np.abs(clip)))
    if rms < _VOICE_MIN_RMS and peak < _VOICE_MIN_PEAK:
        return None
    return np.clip(clip, -1.0, 1.0)


def _to_wav_bytes(samples: np.ndarray) -> bytes:
    pcm16 = (samples * 32767.0).astype(np.int16)
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(_VOICE_SAMPLE_RATE)
            wav.writeframes(pcm16.tobytes())
        return buf.getvalue()


async def _transcribe_mic_audio(
    stt_client: genai.Client,
    stt_model: str,
    wav_bytes: bytes,
) -> str:
    from google.genai import types as genai_types

    prompt = (
        "Transcribe this child speech. Return only the transcript text. "
        "If no speech is present, return an empty string."
    )
    response = await stt_client.aio.models.generate_content(
        model=stt_model,
        contents=[
            prompt,
            genai_types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
        ],
        config=genai_types.GenerateContentConfig(temperature=0.0),
    )
    text = (response.text or "").strip()
    low = text.lower()
    if low in {"", "silence", "(silence)", "[silence]", "no speech"}:
        return ""
    return text


async def _wait_for_output_idle(timeout: float = _VOICE_OUTPUT_IDLE_TIMEOUT) -> bool:
    """Wait until TTS and one-shot SFX are quiet before recording from mic."""
    import time

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if not is_speaking() and not is_playing_sfx():
            return True
        await asyncio.sleep(0.05)
    return False


async def _get_voice(
    interactive: bool,
    stt_client: genai.Client | None,
    stt_model: str,
) -> str:
    """Return voice input from microphone in interactive mode, else silent."""
    if interactive:
        if stt_client is None:
            print("\n  [voice] GEMINI_API_KEY not set; skipping microphone transcription.")
            return ""
        try:
            # Avoid recording our own speaker output.
            await stop_ambient()
            quiet = await _wait_for_output_idle()
            if not quiet:
                print("\n  [voice] speaker output still active; skipping mic capture this turn.")
                return ""

            clip = await asyncio.to_thread(_record_voice_clip)
            if clip is None:
                return ""
            wav_bytes = await asyncio.to_thread(_to_wav_bytes, clip)

            try:
                text = await _transcribe_mic_audio(stt_client, stt_model, wav_bytes)
                if text:
                    print(f"  [voice] {text}")
                return text
            except Exception as exc:
                print(f"  [voice] transcription error: {exc}")
                return ""
        finally:
            await start_ambient()
    return ""


async def _run(
    count: int | None,
    interactive: bool,
    debug: bool,
    inference_interval: float = _INFERENCE_INTERVAL,
    min_obs: int = _MIN_OBSERVATIONS,
) -> None:
    import time

    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)
    stt_model = os.environ.get("VOICE_STT_MODEL", "gemini-2.5-flash")
    client = GeminiClient(api_key=api_key, model=model)
    stt_client = genai.Client(api_key=api_key) if (interactive and api_key) else None

    context: deque[str] = deque(maxlen=_CONTEXT_MAXLEN)
    persistent_facts: list[str] = []
    observations: list[str] = []
    last_inference = 0.0
    prev_animal: str | None = None
    _sfx_task: asyncio.Task | None = None

    _open_log()

    await init_tts()
    await start_ambient()

    print("=== LIVE SHADOW STREAM MODE ===")
    print(f"Classifier: root main.py predict()")
    print(f"Gemini model: {model}")
    print(f"Context window: {_CONTEXT_MAXLEN}")
    print(f"Inference interval: {inference_interval}s  min obs: {min_obs}")
    print(f"Interactive voice: {interactive} ({'microphone' if interactive else 'off'})")
    print(f"Debug window: {debug}")
    print("-" * 64)

    iteration = 1
    try:
        while count is None or iteration <= count:
            t_cls = time.monotonic()
            try:
                shadow_label, shadow_motion = shadow_classifier.predict(debug=debug)
            except Exception:
                shadow_label, shadow_motion = "unknown shadow", "still"
            cls_ms = (time.monotonic() - t_cls) * 1000

            if shadow_label:
                desc = f"{shadow_label} {shadow_motion}" if shadow_motion and shadow_motion != "still" else shadow_label
                observations.append(desc)
                print(f"  [classify] {desc} ({cls_ms:.0f}ms)")

            now = time.monotonic()
            elapsed = now - last_inference
            ready = (
                len(observations) >= min_obs
                and elapsed >= inference_interval
            )

            if not ready:
                await asyncio.sleep(_POLL_INTERVAL)
                continue

            # batch the collected observations into a single shadow description
            shadow_input = _summarize_observations(observations)
            observations.clear()
            last_inference = now

            # extract bare animal name (first word before any motion descriptor)
            current_animal = shadow_input.split()[0] if shadow_input else None

            voice = await _get_voice(interactive, stt_client, stt_model)

            # fire SFX on animal transition (runs in background, won't block narration)
            if current_animal and current_animal != prev_animal:
                if _sfx_task is not None and not _sfx_task.done():
                    _sfx_task.cancel()
                _sfx_task = asyncio.create_task(play_sfx(current_animal))
                prev_animal = current_animal

            _print_banner(iteration, shadow_input, voice, len(context), len(persistent_facts))

            try:
                prompt = _build_prompt(voice, shadow_input, context, persistent_facts)
                t_gen = time.monotonic()
                text = await client.generate(prompt)
                gen_ms = (time.monotonic() - t_gen) * 1000
                text = text.strip()
                print(f"\n  [gemini] {gen_ms:.0f}ms")
                print(f"  {text}")
                await enqueue(text)
                context.append(text)
                _log(iteration, shadow_input, voice, prompt, text)
            except GeminiClientError as exc:
                print(f"\n  [ERROR] {exc}", file=sys.stderr)
                _log(iteration, shadow_input, voice, prompt, f"[ERROR] {exc}")

            if voice:
                persistent_facts.append(voice)

            iteration += 1
    finally:
        await stop_ambient()
        await cleanup_tts()

    print("\n" + "-" * 64)
    print(f"[live-stream-test] Done — {iteration - 1} narration(s) generated.")
    if persistent_facts:
        print(f"Story facts accumulated ({len(persistent_facts)}):")
        for f in persistent_facts:
            print(f"  - {f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live shadow narration using real camera classifier."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N narrations (default: run forever).",
    )
    parser.add_argument(
        "--interactive-voice",
        action="store_true",
        help="Capture kid voice from microphone and transcribe each turn.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show a live camera window with detected bbox and classifier label overlay.",
    )
    parser.add_argument(
        "--inference-interval",
        type=float,
        default=_INFERENCE_INTERVAL,
        metavar="SEC",
        help=f"Min seconds between Gemini inferences (default: {_INFERENCE_INTERVAL}).",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=_MIN_OBSERVATIONS,
        metavar="N",
        help=f"Min shadow observations to collect before inferring (default: {_MIN_OBSERVATIONS}).",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            _run(
                args.count,
                args.interactive_voice,
                args.debug,
                inference_interval=args.inference_interval,
                min_obs=args.min_obs,
            )
        )
    except KeyboardInterrupt:
        print("\n[live-stream-test] Interrupted.")
        asyncio.run(cleanup_tts())
    finally:
        shadow_classifier.cleanup()
        if _log_file is not None:
            _log_file.close()


if __name__ == "__main__":
    main()
