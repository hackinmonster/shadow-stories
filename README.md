# Shadow Stories

Shadow Stories is a real-time shadow-puppet storyteller.

It watches a hand-shadow animal on camera, classifies the animal + motion, generates a short story continuation with Gemini, then speaks it with ElevenLabs. Local ambient audio and animal SFX are included.

## What It Does

- Detects shadow silhouettes from webcam frames.
- Classifies supported animals (for example: `snail`, `panther`, `moose`).
- Detects simple motion states (`still`, `walking ...`, `jumping`).
- Feeds that context into a constrained narration prompt.
- Streams spoken narration with low-latency TTS.

## Prerequisites

- Python `3.11+`
- Webcam
- `ffplay` on your `PATH` (from ffmpeg)
- Model weights file: `HSPR_ConvNextLarge_Aug_CB.pt`
  - Source model repo: `https://github.com/Starscream-11813/HaSPeR`
- API keys:
  - Gemini (`GEMINI_API_KEY`)
  - ElevenLabs (`ELEVENLABS_API_KEY`)

## Setup

1. Clone and enter the repo.
2. Create env file:

```bash
cp .env.example .env
```

3. Fill required keys in `.env`:

```env
GEMINI_API_KEY=...
ELEVENLABS_API_KEY=...
```

4. Make model weights available (choose one):
- Place file at `./models/HSPR_ConvNextLarge_Aug_CB.pt`
- or set `SHADOW_MODEL_PATH=/absolute/path/to/HSPR_ConvNextLarge_Aug_CB.pt`
- Model source: `https://github.com/Starscream-11813/HaSPeR`

5. Install dependencies.

With `uv`:

```bash
uv sync
```

With `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Main live loop (camera -> narration -> speech):

```bash
uv run python -m shadow_stories.live_stream_test
```

Add microphone input each turn:

```bash
uv run python -m shadow_stories.live_stream_test --interactive-voice
```

Show debug camera overlay window:

```bash
uv run python -m shadow_stories.live_stream_test --debug
```

Tune response cadence:

```bash
uv run python -m shadow_stories.live_stream_test --inference-interval 1.5 --min-obs 5
```

Stop:
- Terminal loop: `Ctrl+C`
- Debug camera window: `q` (also works via loop shutdown)

## One-Shot CLI

For quick prompt testing without live camera loop:

```bash
uv run shadow-narrate --voice "the dragon roars" --shadow "wings spread wide"
```

## Configuration

Environment variables used by the app:

- `GEMINI_API_KEY` (required)
- `GEMINI_MODEL` (default: `gemini-2.5-flash-lite`)
- `VOICE_STT_MODEL` (default: `gemini-2.5-flash`)
- `ELEVENLABS_API_KEY` (required for speech)
- `ELEVENLABS_VOICE_ID` (default in code: `0lp4RIz96WD1RUtvEu3Q`)
- `SHADOW_MODEL_PATH` (optional model path override)
- `SHADOW_CAMERA_INDEX` (default: `1`)

Runtime knobs (in `shadow_stories/live_stream_test.py`):
- `--inference-interval`
- `--min-obs`

## Troubleshooting

- Camera not found:
  - Set `SHADOW_CAMERA_INDEX` in `.env` (try `0`, then `1`).
- No speech output:
  - Check `ELEVENLABS_API_KEY`.
  - Confirm `ffplay` is installed and on `PATH`.
- Model file error:
  - Verify `SHADOW_MODEL_PATH` or place weights under `./models/`.
- Very noisy character switching:
  - Increase `--min-obs` and/or `--inference-interval`.

## Tests

```bash
uv run pytest -q
```

## Notes

- Logs are written to `logs/`.
- Secrets and local artifacts are gitignored (`.env`, model files, caches, logs).
- Do not commit API keys.
