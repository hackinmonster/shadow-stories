# Shadow Stories

Live shadow-puppet narrator with:
- webcam puppet detection/classification,
- Gemini-generated narration,
- ElevenLabs TTS output,
- local animal SFX + looping ambient background.

## Requirements

- Python 3.11+
- `ffplay` available on PATH (part of ffmpeg)
- Webcam
- Model weights file: `HSPR_ConvNextLarge_Aug_CB.pt`

## Quick Start

1. Clone and enter repo.
2. Create env file:

```bash
cp .env.example .env
```

3. Fill API keys in `.env`:
- `GEMINI_API_KEY`
- `ELEVENLABS_API_KEY`

4. Point to your model file (choose one):
- Place model at `./models/HSPR_ConvNextLarge_Aug_CB.pt`, or
- set `SHADOW_MODEL_PATH=/absolute/path/to/HSPR_ConvNextLarge_Aug_CB.pt`

5. Install dependencies.

Using `uv`:
```bash
uv sync
```

Using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Live stream (camera + narration):
```bash
uv run python -m shadow_stories.live_stream_test
```

Live stream with microphone voice input:
```bash
uv run python -m shadow_stories.live_stream_test --interactive-voice
```

Debug camera overlay:
```bash
uv run python -m shadow_stories.live_stream_test --debug
```

One-shot CLI narrator:
```bash
uv run shadow-narrate --voice "the dragon roars" --shadow "wings spread wide"
```

## Useful Environment Variables

- `GEMINI_API_KEY`
- `GEMINI_MODEL` (default: `gemini-2.5-flash-lite`)
- `VOICE_STT_MODEL` (default: `gemini-2.5-flash`)
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID` (default in code: `0lp4RIz96WD1RUtvEu3Q`)
- `SHADOW_MODEL_PATH` (optional model path override)
- `SHADOW_CAMERA_INDEX` (default: `1`)

## Tests

```bash
uv run pytest -q
```

## Notes for GitHub Sharing

- `.env`, `logs/`, caches, local models, and local assistant artifacts are gitignored.
- Do not commit API keys.
