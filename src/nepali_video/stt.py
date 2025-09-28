from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import datetime as dt
import srt
from faster_whisper import WhisperModel


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


def _format_segments(segments: Iterable, keep_words: bool) -> List[Segment]:
    out: List[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        words: List[Word] = []
        if keep_words and getattr(seg, "words", None):
            for w in seg.words:
                wtext = (w.word or "").strip()
                if not wtext:
                    continue
                start = float(w.start) if w.start is not None else float(seg.start)
                end = float(w.end) if w.end is not None else float(seg.end)
                words.append(Word(start=start, end=end, text=wtext))

        out.append(
            Segment(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                words=words,
            )
        )
    return out


def _split_segments_words(
    segments: List[Segment],
    max_chars: int,
    max_words: int,
    max_duration: float,
    max_gap: float,
    min_words: int,
) -> List[Segment]:
    refined: List[Segment] = []

    min_words = max(1, min_words)

    def flush(chunk: List[Word]):
        if not chunk:
            return
        text = " ".join(word.text for word in chunk).strip()
        if refined and len(chunk) < min_words:
            prev = refined[-1]
            prev.text = f"{prev.text} {text}".strip()
            prev.end = chunk[-1].end
        else:
            refined.append(
                Segment(
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    text=text,
                )
            )

    for seg in segments:
        if not seg.words:
            refined.append(seg)
            continue

        chunk: List[Word] = []
        prev_end: Optional[float] = None

        for word in seg.words:
            start = word.start
            end = word.end
            gap = 0.0
            if prev_end is not None:
                gap = max(0.0, start - prev_end)

            projected = chunk + [word]
            projected_text = " ".join(w.text for w in projected).strip()
            projected_duration = 0.0
            if projected and projected[0].start is not None and projected[-1].end is not None:
                projected_duration = projected[-1].end - projected[0].start

            should_split = False
            if chunk:
                if max_chars and len(projected_text) > max_chars:
                    should_split = True
                if max_words and len(projected) > max_words:
                    should_split = True
                if max_duration and projected_duration > max_duration:
                    should_split = True
                if max_gap and gap > max_gap:
                    should_split = True

            if should_split and len(chunk) >= min_words:
                flush(chunk)
                chunk = [word]
            else:
                chunk.append(word)

            prev_end = end

        flush(chunk)

    return refined if refined else segments


def to_srt(segments: List[Segment]) -> str:
    items = []
    for idx, seg in enumerate(segments, start=1):
        start = dt.timedelta(seconds=seg.start)
        end = dt.timedelta(seconds=seg.end)
        items.append(srt.Subtitle(index=idx, start=start, end=end, content=seg.text))
    return srt.compose(items)


def transcribe_to_srt(
    audio_path: str,
    out_srt: str,
    model_name: str = "Systran/faster-whisper-large-v3",
    device: str = "auto",
    compute_type: str = "int8",
    language: Optional[str] = "ne",
    vad_filter: bool = True,
    beam_size: Optional[int] = None,
    best_of: Optional[int] = None,
    temperature: Optional[float] = None,
    initial_prompt: Optional[str] = None,
    condition_on_previous_text: bool = False,
    word_timestamps: bool = False,
    max_line_chars: int = 48,
    max_line_words: int = 12,
    max_line_duration: float = 5,
    max_gap: float = 0.8,
    min_line_words: int = 1,
    cpu_threads: int = 0,
    num_workers: int = 1,
    chunk_length: Optional[int] = None,
    compression_ratio_threshold: Optional[float] = None,
    log_prob_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
) -> Tuple[str, float]:
    """Transcribe audio to SRT using faster-whisper and optionally word-level splitting."""

    if cpu_threads < 0:
        cpu_threads = 0
    if num_workers < 1:
        num_workers = 1

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )

    kwargs = dict(
        language=language,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
        word_timestamps=word_timestamps,
    )

    if beam_size:
        kwargs["beam_size"] = beam_size
    if best_of:
        kwargs["best_of"] = best_of
    if temperature is not None:
        kwargs["temperature"] = temperature
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    if chunk_length:
        kwargs["chunk_length"] = chunk_length
    if compression_ratio_threshold is not None:
        kwargs["compression_ratio_threshold"] = compression_ratio_threshold
    if log_prob_threshold is not None:
        kwargs["log_prob_threshold"] = log_prob_threshold
    if no_speech_threshold is not None:
        kwargs["no_speech_threshold"] = no_speech_threshold

    segments, info = model.transcribe(audio_path, **kwargs)

    segs = _format_segments(segments, keep_words=word_timestamps)
    if word_timestamps and segs:
        segs = _split_segments_words(
            segs,
            max_chars=max_line_chars,
            max_words=max_line_words,
            max_duration=max_line_duration,
            max_gap=max_gap,
            min_words=min_line_words,
        )

    srt_text = to_srt(segs)
    with open(out_srt, "w", encoding="utf-8") as fh:
        fh.write(srt_text)

    dur = getattr(info, "duration", None)
    speed = getattr(info, "transcription_speed", None)
    rtf = None
    if dur and speed and speed > 0:
        rtf = dur / speed

    return out_srt, (rtf or -1.0)
