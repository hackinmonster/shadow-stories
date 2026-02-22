"""Persistent WebSocket connection to ElevenLabs stream-input TTS API."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional
from urllib.parse import urlencode

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

WS_BASE = "wss://api.elevenlabs.io/v1/text-to-speech"


def _build_url(voice_id: str, model_id: str, output_format: str) -> str:
    params = urlencode({
        "model_id": model_id,
        "output_format": output_format,
        "optimize_streaming_latency": 4,
    })
    return f"{WS_BASE}/{voice_id}/stream-input?{params}"


def _ws_is_open(ws) -> bool:
    """Check if a websocket is still open, compatible with websockets v12-v15+."""
    if ws is None:
        return False
    try:
        # websockets v13+ new-style connections
        from websockets.asyncio.client import ClientConnection
        if isinstance(ws, ClientConnection):
            return ws.protocol.state.name == "OPEN"
    except (ImportError, AttributeError):
        pass
    try:
        # websockets v12 legacy or fallback
        return ws.open
    except AttributeError:
        pass
    try:
        return ws.protocol.state.name == "OPEN"
    except Exception:
        return False


async def _receive_loop(ws, on_audio_chunk: Callable[[str], None]) -> None:
    try:
        async for raw in ws:
            msg = json.loads(raw if isinstance(raw, str) else raw.decode())
            if msg.get("audio"):
                on_audio_chunk(msg["audio"])
            if msg.get("isFinal"):
                logger.debug("received isFinal from ElevenLabs")
    except ConnectionClosed:
        pass
    except Exception as e:
        logger.exception("receive loop error: %s", e)


class ElevenLabsVoiceConnection:
    """Single persistent WebSocket for one voice_id."""

    def __init__(
        self,
        voice_id: str,
        on_audio_chunk: Callable[[str], None],
        api_key: str,
        voice_settings: dict[str, Any],
        model_id: str,
        output_format: str,
        chunk_length_schedule: list[int],
    ):
        self.voice_id = voice_id
        self.on_audio_chunk = on_audio_chunk
        self.api_key = api_key
        self._voice_settings = voice_settings
        self._model_id = model_id
        self._output_format = output_format
        self._chunk_length_schedule = chunk_length_schedule
        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        return _ws_is_open(self._ws)

    async def connect(self) -> None:
        # clean up any previous connection
        if self._ws is not None:
            await self._force_close()

        self._ws = await websockets.connect(
            _build_url(self.voice_id, self._model_id, self._output_format),
            additional_headers={"xi-api-key": self.api_key},
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        )
        init_msg = {
            "text": " ",
            "voice_settings": self._voice_settings,
            "generation_config": {"chunk_length_schedule": self._chunk_length_schedule},
        }
        await self._ws.send(json.dumps(init_msg))
        self._recv_task = asyncio.create_task(
            _receive_loop(self._ws, self.on_audio_chunk)
        )
        logger.info("connected to ElevenLabs for voice %s", self.voice_id[:8])

    async def _ensure_connected(self) -> None:
        if not self.is_connected:
            logger.info("reconnecting to ElevenLabs for voice %s", self.voice_id[:8])
            await self.connect()

    async def send_phrase(self, text: str, *, flush: bool = False) -> None:
        await self._ensure_connected()
        t = text.strip()
        if t:
            if not t.endswith(" "):
                t += " "
            await self._ws.send(json.dumps({"text": t, "flush": flush}))

    async def send_text_streamed(self, text: str) -> None:
        """Send text word by word — matches the stream-input API's expected usage."""
        await self._ensure_connected()
        for word in text.split():
            await self._ws.send(json.dumps({"text": word + " "}))
            await asyncio.sleep(0.01)

    async def flush(self) -> None:
        await self.send_phrase(" ", flush=True)

    async def _force_close(self) -> None:
        """Close websocket and cancel receive task without regard for _closed flag."""
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def close(self) -> None:
        """Send EOS and wait for remaining audio before closing."""
        if self._ws is not None and _ws_is_open(self._ws):
            try:
                await self._ws.send(json.dumps({"text": ""}))
            except (ConnectionClosed, Exception):
                pass
            if self._recv_task and not self._recv_task.done():
                try:
                    await asyncio.wait_for(self._recv_task, timeout=30.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        await self._force_close()
        logger.debug("closed connection for voice %s", self.voice_id[:8])


class ElevenLabsWSManager:
    """Manages one WebSocket connection per voice_id for TTS streaming."""

    def __init__(
        self,
        on_audio_chunk: Callable[[str], None],
        api_key: str,
        voice_settings: dict[str, Any],
        model_id: str,
        output_format: str,
        chunk_length_schedule: list[int],
    ):
        self.on_audio_chunk = on_audio_chunk
        self.api_key = api_key
        self._voice_settings = voice_settings
        self._model_id = model_id
        self._output_format = output_format
        self._chunk_length_schedule = chunk_length_schedule
        self._connections: dict[str, ElevenLabsVoiceConnection] = {}
        self._current_voice_id: Optional[str] = None
        self._lock = asyncio.Lock()

    def _make_connection(self, voice_id: str) -> ElevenLabsVoiceConnection:
        return ElevenLabsVoiceConnection(
            voice_id,
            self.on_audio_chunk,
            self.api_key,
            self._voice_settings,
            self._model_id,
            self._output_format,
            self._chunk_length_schedule,
        )

    async def get_connection(self, voice_id: str) -> ElevenLabsVoiceConnection:
        async with self._lock:
            conn = self._connections.get(voice_id)
            if conn is None:
                conn = self._make_connection(voice_id)
                await conn.connect()
                self._connections[voice_id] = conn
            # auto-reconnect handled inside send methods via _ensure_connected
            return conn

    def set_current_voice(self, voice_id: str) -> None:
        self._current_voice_id = voice_id

    async def send_phrase(self, text: str, *, flush: bool = False) -> None:
        if not self._current_voice_id:
            logger.warning("no current voice set; skipping phrase")
            return
        conn = await self.get_connection(self._current_voice_id)
        await conn.send_phrase(text, flush=flush)

    async def send_text_streamed(self, text: str) -> None:
        if not self._current_voice_id:
            logger.warning("no current voice set; skipping text")
            return
        conn = await self.get_connection(self._current_voice_id)
        await conn.send_text_streamed(text)

    async def flush_current(self) -> None:
        if self._current_voice_id:
            conn = self._connections.get(self._current_voice_id)
            if conn:
                await conn.flush()

    async def close_all(self) -> None:
        async with self._lock:
            for conn in self._connections.values():
                await conn.close()
            self._connections.clear()
            self._current_voice_id = None
