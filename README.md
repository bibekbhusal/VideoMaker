# Nepali Narration → Village Video (with Subtitles)

Create a video from:
- **Narration audio** (Nepali language)
- **Story text** (optional, saved alongside for your records)
- **Stock, open-license Nepal village clip** you supply (from Pexels/Pixabay, etc.)

The tool:
1. **Transcribes** the Nepali narration to **SRT** (time-coded subtitles).
2. **Builds a video**: loops the supplied village clip to match audio duration, muxes the audio, and either **burns** or **embeds** the subtitles.

> 💡 _Subtitle source choice_: For easiest automation, we **extract text from audio** (ASR) and produce SRT. If you need the subtitles to exactly match your story text, you can later correct the SRT manually, or use a **forced aligner** (advanced) to align the story text to the audio.

---

## Quick start (macOS Apple Silicon)

1) **Install FFmpeg** (with libass for subtitle burn):
```bash
brew install ffmpeg
ffmpeg -filters | grep subtitles   # should list 'subtitles' filter
```

2) **Create venv & install**:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3) **Run transcription (Nepali)**:
```bash
nepali-video transcribe \
  --audio /path/to/narration.wav \  --out subs.srt \
  --model Systran/faster-whisper-large-v3 \
  --device auto \
  --compute-type int8 \
  --language ne
```

4) **Build the video** (burned subtitles):
```bash
nepali-video build-video \
  --video /path/to/nepal_village.mp4 \  --audio /path/to/narration.wav \
  --subs  subs.srt \
  --out   output.mp4 \
  --burn  --font-size 30
```

Or **embed soft subtitles** (no burn; toggle on/off in players):
```bash
nepali-video build-video \
  --video /path/to/nepal_village.mp4 \  --audio /path/to/narration.wav \
  --subs  subs.srt \
  --out   output.mp4 \
  --soft
```

5) **All-in-one** (save story text & produce video):
```bash
nepali-video all-in-one \  --audio /path/to/narration.wav \
  --story /path/to/story.txt \
  --video /path/to/nepal_village.mp4 \
  --out   output.mp4 \
  --burn
```

---

## Models & recommendations (Nepali ASR)

- **Default (local, accurate, practical):** `Systran/faster-whisper-large-v3` (CTranslate2 format of OpenAI Whisper Large-v3). Good multilingual accuracy including Nepali; runs well on Apple Silicon **CPU**; GPU optional on Linux/Windows with CUDA.  \
  Sources: OpenAI Whisper repo & Large‑v3 model card; faster‑whisper project.  
  See: openai/whisper【turn0search9】, whisper‑large‑v3 card【turn0search1】, faster‑whisper PyPI notes【turn0search15】, SYSTRAN ctranslate2 model【turn0search21】.

- **Best word‑level timing (GPU recommended):** **WhisperX** (Whisper + alignment) with MMS aligner for **word‑level timestamps** and diarization; supports Nepali in Seamless/MMS.    See: WhisperX repo【turn0search10】, MMS ASR (supports Nepali “npi”)【turn0search11】【turn0search20】.

- **Research alternatives:** Meta **SeamlessM4T v2** (ASR+ST) supports Nepali input; more complex integration for SRT-only use.    See: SeamlessM4T v2 card (Nepali listed)【turn0search5】【turn0search20】.

> **Apple Silicon**: OpenAI Whisper via PyTorch can use **MPS/Metal**; `faster-whisper` uses CTranslate2 which is highly optimized on CPU on macOS (and GPU on Linux/Windows). For max Mac performance you can also explore MLX/whisper.cpp variants, but this project ships with `faster-whisper` for simplicity.  > See: Whisper MPS notes【turn1search0】【turn1search5】 and Mac benchmarks/community projects【turn1search8】【turn1search10】【turn1search11】.

---

## Stock footage (open license)

Grab Nepal village clips from:
- **Pexels** videos (free; no attribution required; some restrictions):【turn0search6】【turn4search4】【turn4search6】
- **Pixabay** videos (free; no attribution required; some restrictions):【turn0search7】【turn4search7】

> Always read the license pages and keep the download URL for records (esp. if uploading to YouTube).  > Pexels license summary【turn4search4】; Pixabay license summary【turn4search7】. More ToS: Pexels【turn4search0】, Pixabay【turn4search1】.

---

## CLI

```text
nepali-video transcribe --audio <AUDIO> --out <SRT> [--model ...] [--device auto|cpu|cuda] [--compute-type int8|int8_float16|float16|float32] [--language ne]
nepali-video build-video --video <MP4> --audio <AUDIO> --subs <SRT> --out <MP4> [--burn|--soft] [--font-size 28] [--font-file "/path/font.ttf"]
nepali-video all-in-one --audio <AUDIO> --story <TXT> --video <MP4> --out <MP4> [--burn|--soft]
```
- `--burn`: hard-burn subtitles into pixels using FFmpeg `subtitles` (needs libass).
- `--soft`: embed as a subtitle track (`mov_text`) you can toggle.
- If your clip is shorter than the narration, it will be **looped** and trimmed to match the audio duration.

### Tips
- Prefer **WAV/FLAC** inputs for best results (we auto-convert via ffmpeg if needed).
- Set `--language ne` to force Nepali; omit to let the model auto-detect.
- If memory is tight on Mac, try `--compute-type int8` or use a smaller model (e.g. `medium`, `small`).

---

## GPU notes (Linux/Windows, optional)
- For CUDA 12 + cuDNN 9, latest `ctranslate2` is fine. If you’re on older CUDA, pin ctranslate2 as noted on faster‑whisper PyPI【turn0search15】.
- WhisperX is fastest/most accurate with a recent NVIDIA GPU (e.g., RTX 4090).

---

## Roadmap / advanced (optional)
- **Forced alignment of your story text** to audio (for exact words & timings): try **WhisperX alignment** or classical aligners (e.g., aeneas/MFA).  
- **Diarization** (multiple speakers) with WhisperX.
- **B‑roll playlists**: pass a directory of multiple clips to randomize shot changes.

---

## License
MIT
