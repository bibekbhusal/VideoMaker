# VideoMaker CLI

CLI helpers for turning a narrated audio track (Nepali, Hindi, English, …) into a finished video with subtitles. The toolkit keeps the workflow local: transcribe with faster‑whisper, clean the narration, loop footage or images to match the runtime, and burn (or embed) subtitles with a bundled Devanagari-friendly font.

## Features
- **Multilingual transcription** via `faster-whisper`, with auto language detection or explicit hints, producing cleaned SRT subtitle files.
- **Automatic audio denoising** using FFmpeg’s RNNoise (`arnndn`) filter – enabled by default with a one-time model download.
- **Video builder** that loops clips from a directory or creates slideshows from still images (alphabetical or random order) to cover the narration.
- **Subtitle handling** with a single `--burn` switch: `force` hard-burns text using a Mukta font (downloaded automatically), while `soft` embeds a toggleable subtitle track with custom language tags.
- **Smart path resolution** – relative audio paths are resolved against `audio/` (and `audios/` for legacy folders), while clip/image directories fall back to `media/` (or `clips/`) if not found beside the CLI.
- **Structured logging** – unified `--log-level` control plus FFmpeg log propagation for easier debugging.
- **Prompt templates** – editable text files (see `prompts/initial_prompt.txt`) to keep Whisper grounded in your desired style.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# editable install for the CLI entry point
pip install -e .
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
│   ├── slides/         # jpg/png stills for slideshows
│   └── photos/         # alternate folder auto-discovered for images
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
  --model Systran/faster-whisper-large-v3 \
  --initial-prompt-file prompts/initial_prompt.txt
```

This writes `Saas.srt` next to the audio (override with `--out`). Audio is denoised with RNNoise by default—disable with `--no-clean-audio`. The bundled prompt lives at `prompts/initial_prompt.txt`; edit it or pass `--initial-prompt` for ad-hoc overrides. Useful switches:

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
- `--initial-prompt` / `--initial-prompt-file` – steer decoding with a reusable text template.
- `--temperature` – tweak decoder creativity (defaults to 0.2).
- `--image-order` (`alphabetical`/`random`) & `--image-seed` – control slideshow sequencing.
- `--image-config durations.json` – map individual images to custom durations (seconds).
- `--log-level` – change runtime logging (DEBUG, INFO, WARN, …).

If you omit `--image-duration`, VideoMaker divides the narration length evenly across every image discovered in the directory, so each still is shown for the same number of seconds (minimum 0.5 s).

Per-image timing configuration example (`durations.json`):

```json
{
  "img01.jpg": 4.5,
  "img02.jpg": 6.0,
  "closing.png": 8
}
```

Any images not listed fall back to `--image-duration` (if provided) or the automatic even split.

### Voice Activity Detection tips

- Leave `--vad` enabled for most workflows; it helps drop long silences.
- If quieter speech is being skipped, lower `--vad-threshold` (e.g., `0.3`) and/or reduce the silence requirement with `--vad-min-silence-ms 200`.
- You can disable VAD altogether (`--no-vad`) if your audio is clean and already trimmed.

### Temperature tuning

- Temperature controls how adventurous Whisper’s decoder is: `0.0` is fully greedy, higher values sample alternative wordings.
- For Nepali (and most Indic languages) a small non-zero value such as `0.2` helps the model reconsider uncertain segments without straying far.
- Increase towards `0.4`–`0.5` only if the decoder gets stuck or repeats words; lower it again if transcripts become inconsistent.

## Configuration (`videomaker.toml`)

You can store defaults in `videomaker.toml` at the project root. Example:

```toml
[global]
log_level = "WARN"

[transcribe]
backend = "whisper"
model = "Systran/faster-whisper-large-v3"
language = "auto"
temperature = 0.2
clean_audio = true
vad_filter = true
vad_threshold = 0.35
vad_min_silence_ms = 250
beam_size = 8
word_timestamps = true
max_line_words = 12
min_line_words = 4
max_line_duration = 5.0
max_gap = 0.8
cpu_threads = 8
num_workers = 4
chunk_length = 30
initial_prompt_file = "prompts/initial_prompt.txt"

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
- **Alternate ASR models:** If `Systran/faster-whisper-large-v3` struggles with Nepali speech, try [sarvamai/indic-whisper](https://huggingface.co/sarvamai/indic-whisper) (CT2 export available) or the Hindi-focussed [GauravSharma/indic-whisper-medium-hi](https://huggingface.co/GauravSharma/indic-whisper-medium-hi). Swap the `--model` argument; both models cover Nepali/Hindi phonetics and can outperform general Whisper on low-resource accents.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a summary of recent updates.

## License

MIT
