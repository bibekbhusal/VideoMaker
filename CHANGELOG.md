# Changelog

## 2025-09-28
- Added automatic slideshow builder for image directories with alphabetical or random ordering.
- Integrated RNNoise-based noise suppression (FFmpeg `arnndn`) and made audio cleaning the default, with an opt-out flag.
- Bundled a Nepali Mukta font downloader and raised the burned subtitle size to 40pt by default.
- Simplified CLI options: directories now drive clip/image discovery (with automatic lookups in `media/` and `clips/`), and the subtitle mode is configured via `--burn {force|soft}`.
- Defaulted output filenames to the source audio name and relaxed reliance on maximum character limits during transcription.
