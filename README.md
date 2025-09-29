# VideoMaker CLI

CLI helpers for turning a narrated audio track (Nepali, Hindi, English, …) into a finished video with subtitles. The toolkit keeps the workflow local: transcribe with faster‑whisper, clean the narration, loop footage or images to match the runtime, and burn (or embed) subtitles with a bundled Devanagari-friendly font.

## Features
- **Multilingual transcription** via `faster-whisper`, with auto language detection or explicit hints, producing cleaned SRT subtitle files.
- **Automatic audio denoising** using FFmpeg’s RNNoise (`arnndn`) filter – enabled by default with a one-time model download.
- **Video builder** that loops clips from a directory or creates slideshows from still images (alphabetical or random order) to cover the narration.
- **Subtitle handling** with a single `--burn` switch: `force` hard-burns text using a Mukta font (downloaded automatically), while `soft` embeds a toggleable subtitle track with custom language tags.
- **Smart path resolution** – relative audio paths are resolved against `audio/` (and `audios/` for legacy folders), while clip/image directories fall back to `media/` (or `clips/`) if not found beside the CLI.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Install FFmpeg (with libass) if you have not already:

```bash
brew install ffmpeg  # macOS (Homebrew)
# or use your Linux package manager / Windows build
```

For development linting tools:

```bash
pip install -r requirements-dev.txt
ruff check src
```

## Directory Layout Tips

```
project/
├── audio/              # narration sources (e.g. Saas.m4a)
├── media/
│   ├── clips/          # mp4/mov clips for looping
│   └── slides/         # jpg/png stills for slideshows
├── subs/               # optional subtitle exports
└── videomaker.toml   # optional defaults
```

The CLI automatically searches `audio/` (and `audios/`) for audio files and `media/` (plus `media/*` folders, then `clips/`) for media directories when you pass a relative path.

## Usage

### Transcribe

```bash
videomaker transcribe \
  --audio Saas.m4a \
  --language ne \
  --model Systran/faster-whisper-large-v3
```

This writes `Saas.srt` next to the audio (override with `--out`). Audio is denoised with RNNoise by default—disable with `--no-clean-audio`. Useful switches:

- `--vad/--no-vad` – enable voice activity detection (default on).
- `--condition-on-previous-text` – reuse the previously decoded text to keep context across segments.
- `--max-line-words`, `--min-line-words`, `--max-line-duration`, `--max-gap` – control subtitle splitting.

### Build a Video

Loop a directory of clips:

```bash
videomaker build-video \
  --audio Saas.m4a \
  --subs Saas.srt \
  --clips-dir clips/Saas \
  --burn force
```

Or drive a slideshow from still images:

```bash
videomaker build-video \
  --audio Saas.m4a \
  --subs Saas.srt \
  --images-dir slides/Saas \
  --image-order random \
  --image-duration 4 \
  --burn soft
```

Key flags:

- `--burn {force|soft}` – choose hard-burned subtitles or an embedded track (`soft`).
- `--font-size 40` – adjust the rendered subtitle size (Mukta font downloads automatically when first needed).
- `--clean-audio/--no-clean-audio` – keep RNNoise on by default or skip it.
- `--subtitle-language` – set the ISO 639-2 code used for embedded subtitle tracks (e.g., `nep`, `hin`).
- `--image-order` (`alphabetical`/`random`) & `--image-seed` – control slideshow sequencing.

## Configuration (`videomaker.toml`)

You can store defaults in `videomaker.toml` at the project root. Example:

```toml
[transcribe]
model = "Systran/faster-whisper-large-v3"
language = "auto"
clean_audio = true
vad_filter = true
beam_size = 8
word_timestamps = true
max_line_words = 12
min_line_words = 4
max_line_duration = 5.0
max_gap = 0.8
cpu_threads = 8
num_workers = 4
chunk_length = 30

[build_video]
burn = "force"
subtitle_language = "und"
font_size = 40
font_file = ""
```

Leave entries out to stick with CLI defaults. The `burn` value maps directly to the `--burn` option (`force` or `soft`).

## Advanced Notes

- **`--condition-on-previous-text`** feeds the last decoded text back into Whisper for the next chunk. Enable it if your narration has long pauses but consistent context.
- **Noise model caching:** the RNNoise `.rnnn` model downloads into your cache folder on first use; pass `--rnnoise-model` to supply a custom model path.
- **Font overrides:** use `--font-file /path/to/font.ttf` to burn subtitles with a specific typeface.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a summary of recent updates.

## License

MIT
