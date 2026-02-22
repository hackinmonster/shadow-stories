"""Local sound effects and ambient playback from the project's audio folder."""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(__file__).resolve().parents[2]
_AUDIO_DIR = _ROOT_DIR / "audio"
_AMBIENT_VOLUME = 0.05
_ANIMAL_FILES: dict[str, str] = {
    "bird": "bird.mp3",
    "chicken": "chicken.mp3",
    "crab": "crab.mp3",
    "dog": "dog.mp3",
    "moose": "moose.mp3",
    "panther": "panther.mp3",
    "snail": "snail.mp3",
}
_ANIMAL_VOLUMES: dict[str, float] = {
    "panther": 0.5,
}
_playing = False
_ambient_proc: subprocess.Popen | None = None


def _resolve_animal_file(animal: str) -> Path | None:
    name = _ANIMAL_FILES.get(animal)
    if not name:
        return None
    path = _AUDIO_DIR / name
    return path if path.exists() else None


def _resolve_background_file() -> Path | None:
    if not _AUDIO_DIR.exists():
        return None
    # Prefer filenames that clearly look like the ambient bed.
    candidates = sorted(p for p in _AUDIO_DIR.glob("*.mp3") if "background" in p.name.lower())
    if candidates:
        return candidates[0]
    return None


def _run_ffplay(path: Path, *, loop: bool = False, volume: float = 1.0) -> int:
    cmd = [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "error",
        "-af",
        f"volume={volume}",
    ]
    if loop:
        cmd.extend(["-loop", "-1"])
    cmd.append(str(path))
    try:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    except Exception as e:
        logger.warning("ffplay playback failed for %s: %s", path.name, e)
        return 1


def _play_file_blocking(path: Path, volume: float = 1.0) -> None:
    global _playing
    _playing = True
    try:
        _run_ffplay(path, loop=False, volume=volume)
    finally:
        _playing = False


async def play_sfx(animal: str) -> None:
    """Play one-shot local SFX for the given animal if available."""
    path = _resolve_animal_file(animal)
    if path is None:
        return
    volume = _ANIMAL_VOLUMES.get(animal, 1.0)
    await asyncio.to_thread(_play_file_blocking, path, volume)


def is_playing_sfx() -> bool:
    return _playing


async def start_ambient() -> None:
    """Start looping ambient background from audio folder."""
    global _ambient_proc
    if _ambient_proc is not None and _ambient_proc.poll() is None:
        return

    ambient_path = _resolve_background_file()
    if ambient_path is None:
        logger.warning("no background audio found in %s", _AUDIO_DIR)
        return

    try:
        _ambient_proc = subprocess.Popen(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                "-af",
                f"volume={_AMBIENT_VOLUME}",
                "-loop",
                "-1",
                str(ambient_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  [sfx] ambient loop started: {ambient_path.name}")
    except Exception as e:
        logger.warning("failed to start ambient loop: %s", e)
        _ambient_proc = None


async def stop_ambient() -> None:
    """Stop looping ambient background."""
    global _ambient_proc
    if _ambient_proc is None:
        return
    try:
        if _ambient_proc.poll() is None:
            _ambient_proc.terminate()
            await asyncio.to_thread(_ambient_proc.wait, 2.0)
    except Exception:
        try:
            _ambient_proc.kill()
        except Exception:
            pass
    finally:
        _ambient_proc = None
