"""PCM stream playback: push-pull buffer with sounddevice callback."""
from __future__ import annotations

import base64
import logging
import queue
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

DTYPE = np.int16
BYTES_PER_SAMPLE = 2


class AudioEngine:
    """Push-pull PCM buffer between websocket producer and sounddevice consumer."""

    def __init__(self, sample_rate: int = 22050, block_duration_ms: float = 50.0):
        self.sample_rate = sample_rate
        self.block_duration_ms = block_duration_ms
        self._queue: queue.Queue = queue.Queue(maxsize=2048)
        self._leftover: bytes = b""
        self._closed = False
        self._stream: sd.OutputStream | None = None

    def push_base64(self, b64_audio: str) -> None:
        if self._closed:
            return
        try:
            self._queue.put(base64.b64decode(b64_audio))
        except Exception as e:
            logger.warning("failed to decode/push audio: %s", e)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def has_leftover(self) -> bool:
        return bool(self._leftover)

    def wait_for_drain(self, timeout: float = 10.0) -> None:
        """Block until the audio queue and leftover buffer have been empty for a while.

        Waits for a sustained period of silence to ensure all audio from the
        websocket has arrived and finished playing, not just a momentary gap.
        """
        idle_streak = 0
        # require ~1.5s of sustained empty queue to be confident playback is done
        required_streak = 30
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.qsize() == 0 and not self._leftover:
                idle_streak += 1
                if idle_streak >= required_streak:
                    return
            else:
                idle_streak = 0
            time.sleep(0.05)
        logger.warning("wait_for_drain timed out (queue_size=%d)", self._queue.qsize())

    def _fill_buffer(self, outdata, frames, time_info, status):
        """sounddevice callback — runs on real-time thread, must never block."""
        need = frames * BYTES_PER_SAMPLE
        chunks: list[bytes] = []
        got = 0
        while got < need:
            try:
                chunk = self._queue.get_nowait()
                chunks.append(chunk)
                got += len(chunk)
            except queue.Empty:
                break

        if chunks or self._leftover:
            data = self._leftover + b"".join(chunks)
            self._leftover = b""
            if len(data) > need:
                self._leftover = data[need:]
                data = data[:need]
            if len(data) % BYTES_PER_SAMPLE:
                data = data[: len(data) - len(data) % BYTES_PER_SAMPLE]
            if data:
                arr = np.frombuffer(data, dtype=DTYPE)
                outdata[: len(arr)] = arr.reshape(-1, 1)
                if len(arr) < frames:
                    outdata[len(arr):] = 0
            else:
                outdata.fill(0)
        else:
            outdata.fill(0)

    def start(self) -> None:
        if self._stream is not None:
            return
        self._closed = False
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=DTYPE,
            blocksize=int(self.sample_rate * self.block_duration_ms / 1000),
            callback=self._fill_buffer,
        )
        self._stream.start()
        logger.info("audio engine started (sample_rate=%d)", self.sample_rate)

    def flush(self) -> None:
        """discard all queued audio immediately"""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._leftover = b""

    def stop(self) -> None:
        self._closed = True
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
