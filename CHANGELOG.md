# Changelog

## 2025-09-29
- Renamed the project to **VideoMaker**, updated CLI entry point, and adopted `videomaker.toml` for configuration.
- Added RNNoise cleaning to the transcription workflow with `--clean-audio/--no-clean-audio` toggles and custom model support.
- Switched to auto language detection by default and exposed ISO language options for both transcription and subtitle tracks.
- Introduced `--subtitle-language` so embedded subtitles carry accurate metadata across languages.
- Added centralized logging configuration (`--log-level`) and FFmpeg log-level integration for consistent checkpoint output.
- Externalised the Whisper initial prompt to `prompts/initial_prompt.txt`, added `--initial-prompt-file`, and updated default temperature guidance (0.2).
- Added an MMS ASR backend (`--backend mms`) using Meta's `facebook/mms-1b-nep` checkpoint for higher-quality Nepali/Hindi transcription.

## 2025-09-28
- Added automatic slideshow builder for image directories with alphabetical or random ordering.
- Integrated RNNoise-based noise suppression (FFmpeg `arnndn`) and made audio cleaning the default, with an opt-out flag.
- Bundled a Mukta (Devanagari) font downloader and raised the burned subtitle size to 40pt by default.
- Simplified CLI options: directories now drive clip/image discovery (with automatic lookups in `media/` and `clips/`), and the subtitle mode is configured via `--burn {force|soft}`.
- Defaulted output filenames to the source audio name and relaxed reliance on maximum character limits during transcription.
