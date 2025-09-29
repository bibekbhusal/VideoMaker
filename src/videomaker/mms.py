from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
from transformers import AutoProcessor, pipeline

from .stt import Segment, Word, _split_segments_words, to_srt

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_pipeline(
    model_id: str,
    device: str = "auto",
    chunk_length_s: float = 20.0,
    stride_length_s: float = 5.0,
) -> Tuple[object, int]:
    """Load and cache an MMS ASR pipeline instance."""

    logger.info(
        "Loading MMS pipeline %s (chunk=%.1fs, stride=%.1fs)",
        model_id,
        chunk_length_s,
        stride_length_s,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    normalized_device = device.lower()
    if normalized_device == "auto":
        normalized_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = 0 if normalized_device.startswith("cuda") else -1
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
        return_timestamps="word",
        device=torch_device,
    )
    sampling_rate = processor.feature_extractor.sampling_rate
    return asr, sampling_rate


def _words_to_segments(
    words: List[Word],
    max_line_words: int,
    min_line_words: int,
    max_line_duration: float,
    max_gap: float,
) -> List[Segment]:
    if not words:
        return []

    full_text = " ".join(w.text for w in words)
    segment = Segment(
        start=words[0].start,
        end=words[-1].end,
        text=full_text,
        words=words,
    )
    return _split_segments_words(
        [segment],
        max_chars=0,
        max_words=max_line_words,
        max_duration=max_line_duration,
        max_gap=max_gap,
        min_words=min_line_words,
    )


def transcribe_mms_to_srt(
    audio_path: str,
    out_srt: str,
    *,
    model_id: str = "facebook/mms-1b-nep",
    device: str = "auto",
    chunk_length_s: float = 20.0,
    stride_length_s: float = 5.0,
    max_line_words: int = 12,
    min_line_words: int = 2,
    max_line_duration: float = 5.0,
    max_gap: float = 0.8,
) -> Tuple[str, Optional[float]]:
    """Run Meta MMS ASR and emit SRT subtitles."""

    asr, sampling_rate = _load_pipeline(model_id, device, chunk_length_s, stride_length_s)
    logger.info(
        "Transcribing with MMS (%s) [sampling rate %d]",
        model_id,
        sampling_rate,
    )
    result = asr(audio_path)
    chunks = result.get("chunks") or []
    words: List[Word] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        start, end = chunk.get("timestamp") or (None, None)
        if not text or start is None or end is None:
            continue
        words.append(Word(start=float(start), end=float(end), text=text))

    if not words:
        raise RuntimeError("MMS transcription returned no word timestamps.")

    segments = _words_to_segments(
        words,
        max_line_words=max_line_words,
        min_line_words=min_line_words,
        max_line_duration=max_line_duration,
        max_gap=max_gap,
    )

    logger.info("Formatting %d MMS segments", len(segments))
    srt_text = to_srt(segments)
    with open(out_srt, "w", encoding="utf-8") as fh:
        fh.write(srt_text)
    logger.debug("MMS subtitle file written to %s", out_srt)
    return out_srt, None
