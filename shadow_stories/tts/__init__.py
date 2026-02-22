"""ElevenLabs streaming TTS — async speak() for live narration."""
from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from shadow_stories.tts.audio_engine import AudioEngine
from shadow_stories.tts.elevenlabs_ws import ElevenLabsWSManager

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_VOICE_ID = "0lp4RIz96WD1RUtvEu3Q"

VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.25,
    "style": 0,
    "use_speaker_boost": False,
    "speed": 1.15,
}

SAMPLE_RATE = 22050
OUTPUT_FORMAT = "pcm_22050"
MODEL_ID = "eleven_turbo_v2_5"
CHUNK_LENGTH_SCHEDULE = [50, 80, 120, 150]

# module-level singleton state
_audio: AudioEngine | None = None
_ws_mgr: ElevenLabsWSManager | None = None
_initialized = False
_tts_queue: asyncio.Queue[str | None] | None = None
_tts_worker_task: asyncio.Task | None = None
_tts_send_time: float = 0.0
_tts_first_chunk_logged: bool = True


def _get_api_key() -> str:
    return os.environ.get("ELEVENLABS_API_KEY", "")


def _on_audio_chunk(b64: str) -> None:
    import time
    global _tts_first_chunk_logged
    if not _tts_first_chunk_logged and _tts_send_time > 0:
        dt = (time.monotonic() - _tts_send_time) * 1000
        print(f"  [tts] first audio in {dt:.0f}ms")
        _tts_first_chunk_logged = True
    _audio.push_base64(b64)


async def _tts_worker() -> None:
    """Background worker that pulls text from the queue and sends to ElevenLabs."""
    import time
    global _tts_send_time, _tts_first_chunk_logged
    while True:
        text = await _tts_queue.get()
        if text is None:
            _tts_queue.task_done()
            break
        try:
            _tts_first_chunk_logged = False
            _tts_send_time = time.monotonic()
            await _ws_mgr.send_phrase(text, flush=True)
        except Exception as e:
            logger.warning("TTS send error: %s", e)
        finally:
            _tts_queue.task_done()


async def init_tts() -> None:
    """Start audio engine and background TTS worker."""
    global _audio, _ws_mgr, _initialized, _tts_queue, _tts_worker_task
    if _initialized:
        return

    api_key = _get_api_key()
    if not api_key:
        logger.warning("ELEVENLABS_API_KEY not set — TTS will be silent")
        return

    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)

    _audio = AudioEngine(sample_rate=SAMPLE_RATE)
    _audio.start()

    _ws_mgr = ElevenLabsWSManager(
        on_audio_chunk=_on_audio_chunk,
        api_key=api_key,
        voice_settings=VOICE_SETTINGS,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
        chunk_length_schedule=CHUNK_LENGTH_SCHEDULE,
    )
    _ws_mgr.set_current_voice(voice_id)

    _tts_queue = asyncio.Queue()
    _tts_worker_task = asyncio.create_task(_tts_worker())

    _initialized = True
    logger.info("ElevenLabs TTS initialized (voice=%s)", voice_id[:8])


async def interrupt() -> None:
    """Stop current speech immediately — clear queue and audio buffer."""
    global _tts_worker_task
    if not _initialized:
        return

    # drain the text queue
    if _tts_queue is not None:
        while not _tts_queue.empty():
            try:
                _tts_queue.get_nowait()
                _tts_queue.task_done()
            except asyncio.QueueEmpty:
                break

    # flush buffered audio so speakers go silent right away
    if _audio is not None:
        _audio.flush()

    # close current websocket so any in-flight stream stops
    if _ws_mgr is not None:
        await _ws_mgr.close_all()


async def speak(text: str) -> None:
    """Interrupt any current speech and start speaking new text immediately."""
    if not _initialized or _tts_queue is None:
        logger.warning("TTS not initialized — skipping speech")
        return

    text = text.strip()
    if not text:
        return

    await interrupt()
    _tts_queue.put_nowait(text)


async def enqueue(text: str) -> None:
    """Queue text to play after current speech finishes (no interrupt)."""
    if not _initialized or _tts_queue is None:
        logger.warning("TTS not initialized — skipping speech")
        return

    text = text.strip()
    if not text:
        return

    _tts_queue.put_nowait(text)


def is_speaking() -> bool:
    """True if TTS is still sending or audio is still playing."""
    if _tts_queue is not None and not _tts_queue.empty():
        return True
    if _audio is not None and (_audio.queue_size > 0 or _audio.has_leftover):
        return True
    return False


async def cleanup_tts() -> None:
    """Tear down TTS worker, websocket connections, and audio engine."""
    global _audio, _ws_mgr, _initialized, _tts_queue, _tts_worker_task
    if _tts_queue is not None and _tts_worker_task is not None:
        _tts_queue.put_nowait(None)
        try:
            await asyncio.wait_for(_tts_worker_task, timeout=5.0)
        except asyncio.TimeoutError:
            _tts_worker_task.cancel()
        _tts_worker_task = None
        _tts_queue = None
    if _ws_mgr is not None:
        await _ws_mgr.close_all()
        _ws_mgr = None
    if _audio is not None:
        await asyncio.to_thread(_audio.wait_for_drain, 30.0)
        _audio.stop()
        _audio = None
    _initialized = False
