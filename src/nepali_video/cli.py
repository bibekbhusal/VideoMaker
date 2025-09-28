from __future__ import annotations

import pathlib
from typing import List, Optional

import typer
from rich import box, print
from rich.table import Table

from .config import get_default
from .stt import transcribe_to_srt
from .video import build_video_playlist, loop_and_build_video


app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def transcribe(
    audio: pathlib.Path = typer.Option(
        ...,
        "--audio",
        exists=True,
        readable=True,
        help="Path to narration audio (WAV/MP3/FLAC/etc.)",
    ),
    out: pathlib.Path = typer.Option(
        pathlib.Path(get_default("transcribe", "out", "subs.srt")),
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
        get_default("transcribe", "language", "ne"),
        "--language",
        help="Force language code (e.g., ne for Nepali)",
    ),
    vad_filter: bool = typer.Option(
        get_default("transcribe", "vad_filter", True),
        "--vad/--no-vad",
        help="Enable VAD filtering",
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
        get_default("transcribe", "temperature", 0.0),
        "--temperature",
        help="Sampling temperature (0 = greedy)",
    ),
    initial_prompt: Optional[str] = typer.Option(
        get_default("transcribe", "initial_prompt", None),
        "--initial-prompt",
        help="Optional initial prompt to steer transcription",
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
    max_line_chars: int = typer.Option(
        get_default("transcribe", "max_line_chars", 48),
        "--max-line-chars",
        help="Maximum characters per subtitle line when word timestamps are enabled",
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
):
    """Transcribe Nepali narration to SRT subtitles."""

    srt_path, rtf = transcribe_to_srt(
        audio_path=str(audio),
        out_srt=str(out),
        model_name=model,
        device=device,
        compute_type=compute_type,
        language=language,
        vad_filter=vad_filter,
        beam_size=beam_size,
        best_of=best_of,
        temperature=temperature,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        word_timestamps=word_timestamps,
        max_line_chars=max_line_chars,
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

    table = Table(title="Transcription complete", show_header=False, box=box.SIMPLE)
    table.add_row("SRT file", str(srt_path))
    if rtf and rtf > 0:
        table.add_row("Approx. realtime factor (duration/speed)", f"{rtf:.2f}")
    print(table)


@app.command("build-video")
def build_video(
    video: List[pathlib.Path] = typer.Option(
        None,
        "--video",
        help="Repeatable --video path(s) for a playlist",
        exists=True,
        readable=True,
    ),
    clips_dir: Optional[pathlib.Path] = typer.Option(
        None,
        "--clips-dir",
        help="Directory with clips (mp4/mov)",
    ),
    audio: pathlib.Path = typer.Option(
        ...,
        "--audio",
        exists=True,
        readable=True,
    ),
    subs: pathlib.Path = typer.Option(
        ...,
        "--subs",
        exists=True,
        readable=True,
    ),
    out: pathlib.Path = typer.Option(
        pathlib.Path(get_default("build_video", "out", "output.mp4")),
        "--out",
    ),
    burn: bool = typer.Option(
        get_default("build_video", "burn", False),
        "--burn",
    ),
    soft: bool = typer.Option(
        get_default("build_video", "soft", False),
        "--soft",
    ),
    font_size: int = typer.Option(
        get_default("build_video", "font_size", 28),
        "--font-size",
    ),
    font_file: str = typer.Option(
        get_default("build_video", "font_file", ""),
        "--font-file",
    ),
):
    """Loop clips to cover the narration, mux audio, and add subtitles."""

    clips: List[str] = []
    if video:
        clips.extend([str(v) for v in video])
    if clips_dir:
        for ext in ("*.mp4", "*.mov", "*.mkv", "*.m4v", "*.webm"):
            clips.extend([str(p) for p in clips_dir.glob(ext)])

    if not clips:
        raise typer.BadParameter("Provide at least one --video or a --clips-dir with video files.")

    if len(clips) == 1:
        final = loop_and_build_video(
            clips[0],
            str(audio),
            str(subs),
            str(out),
            burn=burn,
            soft=soft,
            font_size=font_size,
            font_file=font_file,
        )
    else:
        final = build_video_playlist(
            clips,
            str(audio),
            str(subs),
            str(out),
            burn=burn,
            soft=soft,
            font_size=font_size,
            font_file=font_file,
        )

    print(f"[bold green]Done:[/bold green] {final}")


if __name__ == "__main__":
    app()
