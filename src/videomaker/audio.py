from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError

from .utils import cache_dir, download_file, require_ffmpeg, safe

RNNOISE_MODEL_SOURCES = (
    # Attempt a Hugging Face mirror first (may be unavailable in the future).
    (
        "Hugging Face",
        "https://huggingface.co/datasets/ashudeep/rnnoise/resolve/main/conjoined-burgers-2018-08-28/cb.rnnn",
    ),
    (
        "GitHub",
        "https://raw.githubusercontent.com/GregorR/rnnoise-models/master/conjoined-burgers-2018-08-28/cb.rnnn",
    ),
)


def ensure_rnnoise_model(model_path: Optional[str] = None) -> str:
    """Ensure an RNNoise model exists locally and return its path."""

    if model_path:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RNNoise model not found at {path}")
        return str(path)

    cache = cache_dir()
    target = cache / "rnnoise-model-2018-08-28.rnnn"
    if target.is_file():
        return str(target)

    last_error: Optional[Exception] = None
    for _, url in RNNOISE_MODEL_SOURCES:
        try:
            download_file(url, target)
            return str(target)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
        except Exception as exc:  # pragma: no cover - network failures, disk errors
            last_error = exc

    if last_error:
        raise FileNotFoundError(
            "Unable to download an RNNoise model automatically. Provide --rnnoise-model manually."
        ) from last_error
    raise FileNotFoundError("Unable to obtain RNNoise model.")


def denoise_audio(
    audio_path: str,
    *,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Run FFmpeg's arnndn (RNNoise) filter to suppress background noise."""

    require_ffmpeg()
    model = ensure_rnnoise_model(model_path)

    if output_path is None:
        suffix = Path(audio_path).suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="cleaned_")
        tmp.close()
        output_path = tmp.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-af",
        f"arnndn=m={safe(model)}",
        "-c:a",
        "pcm_s16le" if Path(output_path).suffix.lower() in {".wav", ".wave"} else "aac",
        output_path,
    ]
    subprocess.check_call(cmd)
    return output_path
