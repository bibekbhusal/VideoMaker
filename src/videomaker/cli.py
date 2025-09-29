from __future__ import annotations

import json
import logging
import pathlib
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import typer
from rich import box, print
from rich.table import Table

from .audio import denoise_audio
from .config import get_default
from .fonts import ensure_default_font
from .image_config import (
    build_anchor_schedule,
    parse_image_config_payload,
    serialise_schedule,
)
from .image_timings import calculate_image_durations, sort_image_paths
from .stt import transcribe_to_srt
from .utils import ffprobe_duration, set_ffmpeg_log_level
from .video import (
    build_slideshow_from_images,
    build_video_playlist,
    loop_and_build_video,
)

app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)
logger = logging.getLogger(__name__)


_PY_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_FFMPEG_LEVELS = {
    "CRITICAL": "fatal",
    "ERROR": "error",
    "WARN": "warning",
    "WARNING": "warning",
    "INFO": "info",
    "DEBUG": "verbose",
}


_AUDIO_SEARCH_DIRS: Sequence[pathlib.Path] = (
    pathlib.Path("."),
    pathlib.Path("audio"),
    pathlib.Path("audios"),
)

_ASSET_SEARCH_DIRS: Sequence[pathlib.Path] = (
    pathlib.Path("."),
    pathlib.Path("media"),
    pathlib.Path("media/videos"),
    pathlib.Path("media/images"),
    pathlib.Path("media/photos"),
    pathlib.Path("media/clips"),
    pathlib.Path("clips"),
)

_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".m4v", ".webm")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

_NUM_RE = re.compile(r"(\d+)")


def _natural_key(value: str) -> List[tuple[int, object]]:
    """Return a key suitable for natural sorting (numbers ordered numerically)."""

    parts = _NUM_RE.split(value)
    key: List[tuple[int, object]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def _configure_logging(level: str) -> None:
    normalized = level.upper()
    py_level = _PY_LOG_LEVELS.get(normalized)
    if py_level is None:
        raise ValueError(
            f"Unsupported log level '{level}'. Choose from {', '.join(sorted(_PY_LOG_LEVELS))}."
        )

    logging.basicConfig(
        level=py_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        force=True,
    )
    set_ffmpeg_log_level(_FFMPEG_LEVELS.get(normalized, "warning"))
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@app.callback()
def main(
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Logging level (DEBUG, INFO, WARN, ERROR, CRITICAL). Defaults to config/global.",
    )
) -> None:
    configured = log_level or str(get_default("global", "log_level", "WARN"))
    try:
        _configure_logging(configured)
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise typer.BadParameter(str(exc)) from exc
    logger.debug("Logging configured at %s", configured.upper())


def _dedupe(paths: Sequence[pathlib.Path]) -> List[pathlib.Path]:
    seen = set()
    ordered: List[pathlib.Path] = []
    for path in paths:
        key = path.expanduser()
        s = str(key)
        if s in seen:
            continue
        seen.add(s)
        ordered.append(key)
    return ordered


def _resolve_existing(
    path: pathlib.Path,
    *,
    expect_dir: bool,
    search_roots: Sequence[pathlib.Path],
    description: str,
) -> pathlib.Path:
    """Resolve a possibly-relative path by searching known roots.

    The helper walks through candidate directories such as ``audio/`` or
    ``media/`` until it finds a matching file/directory. A descriptive
    ``typer.BadParameter`` is raised if the path cannot be located.
    """

    candidates: List[pathlib.Path] = [path.expanduser()]
    if not candidates[0].is_absolute():
        for root in search_roots:
            base = root.expanduser()
            candidate = (base / path).expanduser()
            candidates.append(candidate)

    for candidate in _dedupe(candidates):
        if expect_dir and candidate.is_dir():
            return candidate.resolve()
        if not expect_dir and candidate.is_file():
            return candidate.resolve()

    checked = ", ".join(str(c) for c in _dedupe(candidates))
    raise typer.BadParameter(f"{description} '{path}' not found. Checked: {checked}")


def _collect_media_files(directory: pathlib.Path, extensions: Sequence[str]) -> List[str]:
    files: List[str] = []
    for child in sorted(directory.iterdir(), key=lambda p: _natural_key(p.name)):
        if child.is_file() and child.suffix.lower() in extensions:
            files.append(str(child))
    return files


@app.command()
def transcribe(
    audio: pathlib.Path = typer.Option(
        ...,
        "--audio",
        help="Path to narration audio (WAV/MP3/FLAC/etc.)",
    ),
    out: Optional[pathlib.Path] = typer.Option(
        None,
        "--out",
        help="Output SRT path",
    ),
    model: str = typer.Option(
        get_default("transcribe", "model", "Systran/faster-whisper-large-v3"),
        "--model",
        help="Whisper model name (CTranslate2)",
    ),
    device: str = typer.Option(
        get_default("transcribe", "device", "auto"),
        "--device",
        help="auto|cpu|cuda",
    ),
    compute_type: str = typer.Option(
        get_default("transcribe", "compute_type", "int8"),
        "--compute-type",
        help="int8|int8_float16|float16|float32",
    ),
    language: str = typer.Option(
        get_default("transcribe", "language", "auto"),
        "--language",
        help="Whisper language hint (use 'auto' for automatic detection)",
    ),
    vad_filter: bool = typer.Option(
        get_default("transcribe", "vad_filter", True),
        "--vad/--no-vad",
        help="Enable VAD filtering",
    ),
    vad_threshold: Optional[float] = typer.Option(
        get_default("transcribe", "vad_threshold", None),
        "--vad-threshold",
        help="Adjust VAD detection threshold (lower keeps quieter speech)",
    ),
    vad_min_silence_ms: Optional[int] = typer.Option(
        get_default("transcribe", "vad_min_silence_ms", None),
        "--vad-min-silence-ms",
        help="Minimum silence duration (ms) before closing a segment",
    ),
    beam_size: Optional[int] = typer.Option(
        get_default("transcribe", "beam_size", 5),
        "--beam-size",
        help="Beam width (higher = better accuracy, slower)",
    ),
    best_of: Optional[int] = typer.Option(
        get_default("transcribe", "best_of", None),
        "--best-of",
        help="Sampling candidates when beam-size=1",
    ),
    temperature: float = typer.Option(
        get_default("transcribe", "temperature", 0.2),
        "--temperature",
        help="Sampling temperature (0 = greedy)",
    ),
    initial_prompt: Optional[str] = typer.Option(
        get_default("transcribe", "initial_prompt", None),
        "--initial-prompt",
        help="Optional initial prompt to steer transcription",
    ),
    initial_prompt_file: Optional[pathlib.Path] = typer.Option(
        (
            pathlib.Path(get_default("transcribe", "initial_prompt_file", ""))
            if get_default("transcribe", "initial_prompt_file", "")
            else None
        ),
        "--initial-prompt-file",
        help="Path to a text file containing the initial prompt",
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
    ),
    condition_on_previous_text: bool = typer.Option(
        get_default("transcribe", "condition_on_previous_text", False),
        "--condition-on-previous-text/--no-condition-on-previous-text",
        help="Reuse previous text to condition decoding",
    ),
    word_timestamps: bool = typer.Option(
        get_default("transcribe", "word_timestamps", True),
        "--word-timestamps/--segment-timestamps",
        help="Generate word-level timestamps for better subtitle splits",
    ),
    max_line_words: int = typer.Option(
        get_default("transcribe", "max_line_words", 12),
        "--max-line-words",
        help="Maximum words per subtitle line when word timestamps are enabled",
    ),
    min_line_words: int = typer.Option(
        get_default("transcribe", "min_line_words", 2),
        "--min-line-words",
        help="Minimum words per subtitle line when word timestamps are enabled",
    ),
    max_line_duration: float = typer.Option(
        get_default("transcribe", "max_line_duration", 5.0),
        "--max-line-duration",
        help="Maximum seconds per subtitle line",
    ),
    max_gap: float = typer.Option(
        get_default("transcribe", "max_gap", 0.8),
        "--max-gap",
        help="Split when silence gap exceeds this many seconds",
    ),
    cpu_threads: int = typer.Option(
        get_default("transcribe", "cpu_threads", 0),
        "--cpu-threads",
        help="CPU threads for faster-whisper (0 = auto)",
    ),
    num_workers: int = typer.Option(
        get_default("transcribe", "num_workers", 2),
        "--num-workers",
        help="Workers used for decoding (1-4 recommended)",
    ),
    chunk_length: Optional[int] = typer.Option(
        get_default("transcribe", "chunk_length", None),
        "--chunk-length",
        help="Chunk length in seconds (e.g., 30)",
    ),
    compression_ratio_threshold: Optional[float] = typer.Option(
        get_default("transcribe", "compression_ratio_threshold", None),
        "--compression-threshold",
        help="Compress ratio guard (defaults to model's)",
    ),
    log_prob_threshold: Optional[float] = typer.Option(
        get_default("transcribe", "log_prob_threshold", None),
        "--logprob-threshold",
        help="Discard if avg log prob < threshold",
    ),
    no_speech_threshold: Optional[float] = typer.Option(
        get_default("transcribe", "no_speech_threshold", None),
        "--no-speech-threshold",
        help="Silence probability threshold",
    ),
    clean_audio: bool = typer.Option(
        get_default("transcribe", "clean_audio", True),
        "--clean-audio/--no-clean-audio",
        help="Run RNNoise noise suppression before transcription",
    ),
    rnnoise_model: Optional[pathlib.Path] = typer.Option(
        None,
        "--rnnoise-model",
        help="Optional custom RNNoise model used for noise suppression",
        exists=True,
        readable=True,
    ),
):
    """Transcribe narration audio to SRT subtitles."""

    audio_path = _resolve_existing(
        audio,
        expect_dir=False,
        search_roots=_AUDIO_SEARCH_DIRS,
        description="Audio file",
    )
    logger.info("Starting transcription for %s", audio_path)

    configured_out = get_default("transcribe", "out", "")
    if out is not None:
        out_path = out
    elif configured_out:
        out_path = pathlib.Path(configured_out)
    else:
        out_path = audio_path.with_suffix(".srt")
    logger.debug("Output subtitle path set to %s", out_path)

    narration_path = audio_path
    cleanup_audio = False
    if clean_audio:
        model_path = str(rnnoise_model) if rnnoise_model else None
        logger.info(
            "Running RNNoise denoising%s",
            " with custom model" if model_path else "",
        )
        try:
            cleaned_path = denoise_audio(str(audio_path), model_path=model_path)
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
        narration_path = pathlib.Path(cleaned_path)
        cleanup_audio = True

    language_arg = None
    if language:
        lang_value = language.strip()
        if lang_value and lang_value.lower() != "auto":
            language_arg = lang_value
            logger.info("Language hint set to '%s'", language_arg)
        else:
            logger.info("Language auto-detection enabled")
    else:
        logger.info("Language auto-detection enabled")

    prompt_text = initial_prompt.strip() if initial_prompt else None
    if prompt_text:
        logger.info("Using inline initial prompt (%d chars)", len(prompt_text))
    elif initial_prompt_file:
        try:
            prompt_text = initial_prompt_file.read_text(encoding="utf-8").strip()
        except OSError as exc:  # pragma: no cover - file issues
            raise typer.BadParameter(f"Failed to read prompt file: {exc}") from exc
        logger.info(
            "Loaded initial prompt from %s (%d chars)",
            initial_prompt_file,
            len(prompt_text),
        )
    if vad_filter:
        if vad_threshold is not None:
            logger.info("VAD threshold set to %.3f", vad_threshold)
        if vad_min_silence_ms is not None:
            logger.info("VAD minimum silence %d ms", vad_min_silence_ms)

    try:
        logger.info(
            "Loading Whisper model %s (device=%s, compute_type=%s)",
            model,
            device,
            compute_type,
        )
        srt_path, rtf = transcribe_to_srt(
            audio_path=str(narration_path),
            out_srt=str(out_path),
            model_name=model,
            device=device,
            compute_type=compute_type,
            language=language_arg,
            vad_filter=vad_filter,
            vad_threshold=vad_threshold,
            vad_min_silence_ms=vad_min_silence_ms,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            initial_prompt=prompt_text,
            condition_on_previous_text=condition_on_previous_text,
            word_timestamps=word_timestamps,
            max_line_words=max_line_words,
            max_line_duration=max_line_duration,
            max_gap=max_gap,
            min_line_words=min_line_words,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            chunk_length=chunk_length,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
        )
    finally:
        if cleanup_audio:
            narration_path.unlink(missing_ok=True)
            logger.debug("Removed temporary denoised file %s", narration_path)

    logger.info("Transcription complete -> %s", srt_path)
    table = Table(title="Transcription complete", show_header=False, box=box.SIMPLE)
    table.add_row("SRT file", str(srt_path))
    if rtf and rtf > 0:
        table.add_row("Approx. realtime factor (duration/speed)", f"{rtf:.2f}")
    print(table)


@app.command("build-video")
def build_video(
    clips_dir: Optional[pathlib.Path] = typer.Option(
        None,
        "--clips-dir",
        help="Directory with video clips (relative paths also search assets/ and clips/)",
    ),
    images_dir: Optional[pathlib.Path] = typer.Option(
        None,
        "--images-dir",
        help="Directory with still images (relative paths also search assets/)",
    ),
    image_order: str = typer.Option(
        str(get_default("build_video", "image_order", "alphabetical")),
        "--image-order",
        help="Order to cycle still images: alphabetical or random",
        show_default=True,
    ),
    image_duration: Optional[float] = typer.Option(
        None,
        "--image-duration",
        min=0.2,
        help="Seconds each image is shown before advancing (auto if omitted)",
    ),
    image_seed: Optional[int] = typer.Option(
        None,
        "--image-seed",
        help="Optional seed used when --image-order=random",
    ),
    image_config: Optional[pathlib.Path] = typer.Option(
        (
            pathlib.Path(cfg) if (cfg := get_default("build_video", "image_config", "")) else None
        ),
        "--image-config",
        help="JSON file mapping image filename to duration (seconds)",
        exists=True,
        readable=True,
        dir_okay=False,
        file_okay=True,
    ),
    audio: pathlib.Path = typer.Option(
        ...,
        "--audio",
    ),
    subs: pathlib.Path = typer.Option(
        ...,
        "--subs",
        exists=True,
        readable=True,
    ),
    out: Optional[pathlib.Path] = typer.Option(
        None,
        "--out",
        help="Output MP4 path (defaults to audio name)",
    ),
    burn: str = typer.Option(
        str(get_default("build_video", "burn", "force")),
        "--burn",
        help="Subtitle mode: force (burn-in) or soft (embedded track)",
        show_default=True,
    ),
    subtitle_language: str = typer.Option(
        str(get_default("build_video", "subtitle_language", "und")),
        "--subtitle-language",
        help="ISO 639-2 language code stored on subtitle tracks",
        show_default=True,
    ),
    font_size: int = typer.Option(
        get_default("build_video", "font_size", 40),
        "--font-size",
    ),
    font_file: Optional[pathlib.Path] = typer.Option(
        None,
        "--font-file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Override font file used when burning subtitles",
    ),
    clean_audio: bool = typer.Option(
        True,
        "--clean-audio/--no-clean-audio",
        help="Run RNNoise noise suppression on the narration before muxing",
    ),
    rnnoise_model: Optional[pathlib.Path] = typer.Option(
        None,
        "--rnnoise-model",
        help="Optional custom RNNoise model for noise suppression",
    ),
):
    """Loop clips or images to cover the narration, mux audio, and add subtitles."""

    logger.info("Building video for audio %s", audio)
    clips: List[str] = []
    if clips_dir:
        clip_directory = _resolve_existing(
            clips_dir,
            expect_dir=True,
            search_roots=_ASSET_SEARCH_DIRS,
            description="Video directory",
        )
        logger.info("Using clip directory %s", clip_directory)
        clips = _collect_media_files(clip_directory, _VIDEO_EXTS)
        if not clips:
            raise typer.BadParameter(f"No video files found in {clip_directory}")
        logger.info("Discovered %d clip(s)", len(clips))

    image_inputs: List[str] = []
    image_config_data: Optional[Dict[str, Tuple[float, float]]] = None
    if images_dir:
        image_directory = _resolve_existing(
            images_dir,
            expect_dir=True,
            search_roots=_ASSET_SEARCH_DIRS,
            description="Image directory",
        )
        logger.info("Using image directory %s", image_directory)
        raw_images = _collect_media_files(image_directory, _IMAGE_EXTS)
        if not raw_images:
            raise typer.BadParameter(f"No image files found in {image_directory}")
        try:
            image_inputs = sort_image_paths(raw_images, image_order.lower(), image_seed)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        logger.info("Discovered %d image(s)", len(image_inputs))
        if image_config:
            try:
                payload = json.loads(image_config.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - file errors
                raise typer.BadParameter(f"Failed to read image config: {exc}") from exc
            try:
                config_entries = parse_image_config_payload(payload)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc

            image_config_data = {
                name: (start, end) for name, start, end in config_entries if end > start
            }

            if image_inputs:
                by_name: Dict[str, List[str]] = {}
                for path in image_inputs:
                    name = pathlib.Path(path).name
                    by_name.setdefault(name, []).append(path)

                duplicates = [name for name, paths in by_name.items() if len(paths) > 1]
                if duplicates:
                    joined = ", ".join(sorted(duplicates))
                    raise typer.BadParameter(
                        f"Duplicate filenames detected when applying image config: {joined}"
                    )

                ordered_inputs: List[str] = []
                missing: List[str] = []
                used = set()
                for name, *_ in config_entries:
                    if name not in by_name:
                        missing.append(name)
                        continue
                    ordered_inputs.append(by_name[name][0])
                    used.add(name)

                if missing:
                    raise typer.BadParameter(
                        "Image config references files not found in images directory: "
                        + ", ".join(missing)
                    )

                remaining = [path for path in image_inputs if pathlib.Path(path).name not in used]
                image_inputs = ordered_inputs + remaining
                logger.info(
                    "Ordering %d image(s) according to image config", len(ordered_inputs)
                )

            logger.info("Loaded image timing config for %d file(s)", len(image_config_data))

    if clips and image_inputs:
        raise typer.BadParameter(
            "Choose either video clips or still images for the background, not both."
        )

    if not clips and not image_inputs:
        raise typer.BadParameter(
            "Provide at least one --clips-dir or --images-dir with media files."
        )

    audio_path = _resolve_existing(
        audio,
        expect_dir=False,
        search_roots=_AUDIO_SEARCH_DIRS,
        description="Audio file",
    )
    logger.debug("Resolved audio path %s", audio_path)
    audio_seconds = ffprobe_duration(str(audio_path))

    if image_inputs:
        try:
            image_inputs, image_duration_list, log_events = calculate_image_durations(
                image_inputs,
                audio_seconds,
                image_config_data,
                image_duration,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        for level, message in log_events:
            getattr(logger, level)(message)
    else:
        image_duration_list = None

    mode = (burn or "force").strip().lower()
    if mode not in {"force", "soft"}:
        raise typer.BadParameter("--burn must be 'force' or 'soft'")
    burn_subtitles = mode == "force"
    soft_subtitles = mode == "soft"

    subtitle_language = subtitle_language.strip().lower() or "und"
    logger.info(
        "Subtitle mode: %s (language tag %s)",
        "burn" if burn_subtitles else "soft",
        subtitle_language,
    )

    configured_out = get_default("build_video", "out", "")
    if out is not None:
        out_path = out
    elif configured_out:
        out_path = pathlib.Path(configured_out)
    else:
        out_path = audio_path.with_suffix(".mp4")

    configured_font = get_default("build_video", "font_file", "")
    if font_file:
        font_path: Optional[pathlib.Path] = font_file.resolve()
    elif configured_font:
        candidate = pathlib.Path(configured_font).expanduser()
        font_path = candidate.resolve() if candidate.is_file() else None
        if configured_font and font_path is None:
            typer.secho(
                f"Configured font file {configured_font!r} not found; falling back to bundled font",
                err=True,
                fg=typer.colors.YELLOW,
            )
    else:
        font_path = None

    if burn_subtitles and font_path is None:
        try:
            font_path = ensure_default_font()
        except Exception as exc:  # pragma: no cover - network failures
            typer.secho(
                f"Failed to download default subtitle font: {exc}",
                err=True,
                fg=typer.colors.RED,
            )
            font_path = None

    narration_path = str(audio_path)
    cleanup_audio = False
    if clean_audio:
        model_path = str(rnnoise_model) if rnnoise_model else None
        try:
            narrated = denoise_audio(narration_path, model_path=model_path)
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from exc
        narration_path = narrated
        cleanup_audio = True

    try:
        if image_inputs:
            try:
                sample_duration = (
                    image_duration
                    if image_duration is not None
                    else (image_duration_list[0] if image_duration_list else 0.0)
                )
                slideshow_order = "manual" if image_config_data else image_order.lower()
                logger.info(
                    "Rendering slideshow (%d images, order=%s, ~%.2fs each)",
                    len(image_inputs),
                    slideshow_order,
                    sample_duration,
                )
                final = build_slideshow_from_images(
                    image_inputs,
                    narration_path,
                    str(subs),
                    str(out_path),
                    order=slideshow_order,
                    seed=image_seed,
                    image_duration=(
                        image_duration if image_duration is not None else sample_duration
                    ),
                    image_durations=image_duration_list,
                    burn=burn_subtitles,
                    soft=soft_subtitles,
                    font_size=font_size,
                    font_file=str(font_path) if font_path else "",
                    subtitle_language=subtitle_language,
                )
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
        else:
            font_arg = str(font_path) if font_path else ""
            try:
                if len(clips) == 1:
                    logger.info("Rendering looped single clip")
                    final = loop_and_build_video(
                        clips[0],
                        narration_path,
                        str(subs),
                        str(out_path),
                        burn=burn_subtitles,
                        soft=soft_subtitles,
                        font_size=font_size,
                        font_file=font_arg,
                        subtitle_language=subtitle_language,
                    )
                else:
                    logger.info("Rendering playlist with %d clips", len(clips))
                    final = build_video_playlist(
                        clips,
                        narration_path,
                        str(subs),
                        str(out_path),
                        burn=burn_subtitles,
                        soft=soft_subtitles,
                        font_size=font_size,
                        font_file=font_arg,
                        subtitle_language=subtitle_language,
                    )
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
    finally:
        if cleanup_audio:
            pathlib.Path(narration_path).unlink(missing_ok=True)

    logger.info("Video render complete -> %s", final)
    print(f"[bold green]Done:[/bold green] {final}")


@app.command("init-image-config")
def init_image_config(
    audio: pathlib.Path = typer.Argument(..., exists=True, readable=True),
    images_dir: pathlib.Path = typer.Argument(..., exists=True, readable=True),
    out: Optional[pathlib.Path] = typer.Option(
        None,
        "--out",
        help="Output JSON path (defaults to <images_dir>/image_config.json)",
    ),
    order: str = typer.Option(
        "alphabetical",
        "--image-order",
        help="Ordering used when assigning windows",
        show_default=True,
    ),
) -> None:
    """Generate an image_config JSON with equal-duration timestamps."""

    audio_path = _resolve_existing(
        audio,
        expect_dir=False,
        search_roots=_AUDIO_SEARCH_DIRS,
        description="Audio file",
    )
    directory = images_dir.expanduser().resolve()
    if not directory.is_dir():
        raise typer.BadParameter(f"{directory} is not a directory")

    images = _collect_media_files(directory, _IMAGE_EXTS)
    if not images:
        raise typer.BadParameter(f"No images found in {directory}")

    duration = ffprobe_duration(str(audio_path))
    base_duration = duration / len(images) if images else 0.0

    order_key = order.lower()
    if order_key not in {"alphabetical", "random"}:
        raise typer.BadParameter("--image-order must be 'alphabetical' or 'random'")
    if order_key == "alphabetical":
        ordered = sorted(images, key=lambda p: _natural_key(pathlib.Path(p).name))
    else:
        ordered = images[:]
        random.shuffle(ordered)

    entries: List[Tuple[str, float, float]] = []
    current = 0.0
    for path in ordered:
        name = pathlib.Path(path).name
        if current >= duration:
            entries.append((name, -1.0, -1.0))
            continue
        end = min(duration, current + base_duration)
        entries.append((name, current, end))
        current = end

    if out is None:
        out_path = directory / "image_config.json"
    else:
        out_path = out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps(serialise_schedule(entries), indent=2, ensure_ascii=False))
    typer.echo(f"Wrote {len(entries)} entries to {out_path}")


@app.command("set-image-start")
def set_image_start(
    audio: pathlib.Path = typer.Option(
        ...,
        "--audio",
        "-a",
        help="Path to narration audio (searched in default audio directories if relative)",
    ),
    images_dir: pathlib.Path = typer.Option(
        ...,
        "--images-dir",
        "-i",
        help="Directory containing the images",
    ),
    start_image: str = typer.Option(
        ...,
        "--start-image",
        "-s",
        help="Filename of the image to anchor",
    ),
    start_timestamp: float = typer.Option(
        ...,
        "--start-timestamp",
        "-t",
        help="When the start image should appear (seconds)",
    ),
    out: Optional[pathlib.Path] = typer.Option(
        None,
        "--out",
        help="Output JSON path (defaults to <images_dir>/image_config.json)",
    ),
    order: str = typer.Option(
        "alphabetical",
        "--image-order",
        help="Ordering used when assigning windows",
        show_default=True,
    ),
) -> None:
    """Rewrite image timings so a chosen image starts at a specific timestamp."""

    if start_timestamp < 0:
        raise typer.BadParameter("Start timestamp must be non-negative")

    audio_path = _resolve_existing(
        audio,
        expect_dir=False,
        search_roots=_AUDIO_SEARCH_DIRS,
        description="Audio file",
    )

    directory = images_dir.expanduser().resolve()
    if not directory.is_dir():
        raise typer.BadParameter(f"{directory} is not a directory")

    raw_images = _collect_media_files(directory, _IMAGE_EXTS)
    if not raw_images:
        raise typer.BadParameter(f"No images found in {directory}")

    order_key = order.lower()
    if order_key != "alphabetical":
        raise typer.BadParameter(
            "set-image-start currently supports only alphabetical ordering"
        )

    ordered_images = sort_image_paths(raw_images, "alphabetical")

    image_names = [pathlib.Path(p).name for p in ordered_images]
    target_name = pathlib.Path(start_image).name

    if target_name not in image_names:
        raise typer.BadParameter(
            f"Image {target_name!r} not found in {directory}. Available: {', '.join(image_names)}"
        )

    out_path = (directory / "image_config.json") if out is None else out.expanduser().resolve()
    if out is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_entries: List[Tuple[str, float, float]] = []
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            existing_entries = parse_image_config_payload(payload)
        except FileNotFoundError:
            existing_entries = []
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid existing image config: {exc}") from exc
        except Exception as exc:  # pragma: no cover - file errors
            raise typer.BadParameter(f"Failed to read existing image config: {exc}") from exc

    audio_seconds = ffprobe_duration(str(audio_path))
    if start_timestamp >= audio_seconds:
        raise typer.BadParameter(
            f"Start timestamp {start_timestamp:.3f}s must be less than audio duration {audio_seconds:.3f}s"
        )

    existing_lookup = {name: (start, end) for name, start, end in existing_entries}

    anchor_index = image_names.index(target_name)
    prefix_names = image_names[:anchor_index]

    final_schedule: List[Tuple[str, float, float]] = []
    current = 0.0
    for idx, name in enumerate(prefix_names):
        if name in existing_lookup:
            start, end = existing_lookup[name]
            final_schedule.append((name, start, end))
            current = max(current, end)
            continue

        remaining_slots = len(prefix_names) - idx
        remaining_time = max(start_timestamp - current, 0.0)
        segment = remaining_time / remaining_slots if remaining_slots else 0.0
        end = start_timestamp if idx == len(prefix_names) - 1 else current + segment
        final_schedule.append((name, current, end))
        current = end

    try:
        tail_schedule = build_anchor_schedule(
            image_names,
            audio_seconds,
            target_name,
            start_timestamp,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    final_schedule.extend(tail_schedule)

    serialised = serialise_schedule(final_schedule)

    out_path.write_text(json.dumps(serialised, indent=2, ensure_ascii=False))
    typer.echo(
        f"Updated {out_path} with {len(serialised)} entries. Anchor {target_name!r} starts at {start_timestamp:.3f}s"
    )


if __name__ == "__main__":
    app()
