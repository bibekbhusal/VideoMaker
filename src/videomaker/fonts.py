from __future__ import annotations

from pathlib import Path

from .utils import cache_dir, download_file

DEFAULT_FONT_URL = "https://github.com/google/fonts/raw/main/ofl/mukta/Mukta-Bold.ttf"
DEFAULT_FONT_FILENAME = "Mukta-Bold.ttf"


def ensure_default_font() -> Path:
    """Download and cache a Devanagari-friendly font for subtitles."""

    fonts_dir = cache_dir() / "fonts"
    target = fonts_dir / DEFAULT_FONT_FILENAME
    if not target.is_file():
        download_file(DEFAULT_FONT_URL, target)
    return target
