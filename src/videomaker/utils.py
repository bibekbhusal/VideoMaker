import json
import logging
import os
import shlex
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_FFMPEG_LOG_LEVEL = "warning"


def set_ffmpeg_log_level(level: str) -> None:
    """Set the log level used for FFmpeg/ffprobe commands."""

    global _FFMPEG_LOG_LEVEL
    _FFMPEG_LOG_LEVEL = level


def ffmpeg_log_level() -> str:
    """Return the configured FFmpeg log level string."""

    return _FFMPEG_LOG_LEVEL


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def require_ffmpeg():
    if not which("ffmpeg") or not which("ffprobe"):
        raise RuntimeError("FFmpeg/ffprobe not found. Install with Homebrew: brew install ffmpeg")


def ffprobe_duration(path: str) -> float:
    require_ffmpeg()
    cmd = [
        "ffprobe",
        "-loglevel",
        ffmpeg_log_level(),
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    logger.debug("Running ffprobe: %s", " ".join(cmd))
    out = subprocess.check_output(cmd)
    data = json.loads(out.decode("utf-8"))
    return float(data["format"]["duration"])


def safe(path: str) -> str:
    # Quote for ffmpeg command
    return shlex.quote(os.fspath(path))


def cache_dir() -> Path:
    """Return (and create) a cache directory for downloadable assets."""

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    target = base / "videomaker"
    target.mkdir(parents=True, exist_ok=True)
    return target


def download_file(url: str, destination: Path) -> Path:
    """Download ``url`` into ``destination`` atomically."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return destination
