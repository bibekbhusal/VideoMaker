
import json
import os
import shlex
import shutil
import subprocess
from typing import Optional

def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)

def require_ffmpeg():
    if not which("ffmpeg") or not which("ffprobe"):
        raise RuntimeError("FFmpeg/ffprobe not found. Install with Homebrew: brew install ffmpeg")

def ffprobe_duration(path: str) -> float:
    require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", path
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out.decode("utf-8"))
    return float(data["format"]["duration"])

def safe(path: str) -> str:
    # Quote for ffmpeg command
    return shlex.quote(os.fspath(path))
