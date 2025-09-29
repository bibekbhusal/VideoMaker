import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from .image_timings import natural_key
from .utils import ffmpeg_log_level, ffprobe_duration, require_ffmpeg, safe

logger = logging.getLogger(__name__)

def _subtitles_filter(subs_path: str, font_size: int, font_file: str = "") -> str:
    """Build a subtitles filter string with optional font directory/name overrides."""

    style_parts = [f"Fontsize={font_size}"]
    filter_parts = [f"subtitles={safe(subs_path)}"]

    font_dir = ""
    font_name = ""
    if font_file:
        fpath = Path(font_file).expanduser()
        if fpath.is_file():
            font_dir = str(fpath.parent)
            font_name = fpath.stem.split("-")[0].replace("_", " ").strip()
            if font_name:
                style_parts.append(f"Fontname={font_name}")
    if font_dir:
        filter_parts.append(f"fontsdir={safe(font_dir)}")

    filter_parts.append(f"force_style='{','.join(style_parts)}'")
    return ":".join(filter_parts)


def _validate_ordered_paths(paths: Iterable[str], order: str, seed: int | None = None) -> List[str]:
    items = [os.fspath(p) for p in paths]
    if not items:
        raise ValueError("Provide at least one media path.")

    order_key = order.lower()
    if order_key not in {"alphabetical", "random", "manual"}:
        raise ValueError("order must be either 'alphabetical' or 'random'")

    if order_key == "manual":
        return list(paths)
    if order_key == "alphabetical":
        items.sort(key=lambda p: natural_key(Path(p).name))
    else:
        rnd = random.Random(seed)
        rnd.shuffle(items)
    return items


def build_video_playlist(
    video_paths: list,
    audio_path: str,
    subs_path: str,
    out_path: str,
    burn: bool = True,
    soft: bool = False,
    font_size: int = 28,
    font_file: str = "",
    subtitle_language: str = "und",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
):
    """Concatenate clips, align duration with audio, and mux subtitles/audio."""

    if burn and soft:
        raise ValueError("Choose either burn OR soft subtitles, not both.")

    require_ffmpeg()
    adur = ffprobe_duration(audio_path)
    logger.info("Preparing playlist with %d clips to cover %.2fs narration", len(video_paths), adur)

    durs = [ffprobe_duration(p) for p in video_paths]
    seq = []
    total = 0.0
    i = 0
    while total < adur + 1.0:
        idx = i % len(video_paths)
        seq.append((video_paths[idx], durs[idx]))
        total += durs[idx]
        i += 1

    cmd = ["ffmpeg", "-loglevel", ffmpeg_log_level(), "-y"]
    for v, _ in seq:
        cmd += ["-i", v]
    cmd += ["-i", audio_path]
    audio_idx = len(seq)

    subs_idx = None
    if soft:
        cmd += ["-i", subs_path]
        subs_idx = audio_idx + 1

    filter_parts = []
    labels = []
    for idx in range(len(seq)):
        filter_parts.append(
            f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps},setpts=PTS-STARTPTS[v{idx}]"
        )
        labels.append(f"[v{idx}]")
    filter_parts.append(f"{''.join(labels)}concat=n={len(seq)}:v=1:a=0[vcat]")

    vmap = "[vcat]"
    if burn:
        subf = _subtitles_filter(subs_path, font_size, font_file)
        filter_parts.append(f"{vmap}{subf}[vout]")
        vmap = "[vout]"

    output_opts = [
        "-filter_complex",
        ";".join(filter_parts),
        "-t",
        f"{adur:.3f}",
        "-map",
        vmap,
        "-map",
        f"{audio_idx}:a:0",
    ]

    if soft and subs_idx is not None:
        output_opts += ["-map", f"{subs_idx}:0"]

    output_opts += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]

    if soft:
        output_opts += [
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            f"language={subtitle_language}",
        ]

    output_opts += ["-shortest", out_path]

    cmd += output_opts
    logger.info("Running ffmpeg playlist render -> %s", out_path)
    logger.debug("ffmpeg command: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    return out_path


def loop_and_build_video(
    video_path: str,
    audio_path: str,
    subs_path: str,
    out_path: str,
    burn: bool = True,
    soft: bool = False,
    font_size: int = 28,
    font_file: str = "",
    subtitle_language: str = "und",
):
    """Loop a single clip to cover the audio duration and mux subtitles/audio."""

    if burn and soft:
        raise ValueError("Choose either burn OR soft subtitles, not both.")

    require_ffmpeg()
    adur = ffprobe_duration(audio_path)
    logger.info("Looping clip %s to cover %.2fs narration", video_path, adur)

    cmd = [
        "ffmpeg",
        "-loglevel",
        ffmpeg_log_level(),
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        video_path,
        "-i",
        audio_path,
    ]

    audio_idx = 1
    subs_idx = None
    if soft:
        cmd += ["-i", subs_path]
        subs_idx = 2

    cmd += ["-t", f"{adur:.3f}"]

    vf_filters = []
    if burn:
        sub_filter = _subtitles_filter(subs_path, font_size, font_file)
        vf_filters.append(sub_filter)

    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]

    map_opts = ["-map", "0:v:0", "-map", f"{audio_idx}:a:0"]
    if soft and subs_idx is not None:
        map_opts += ["-map", f"{subs_idx}:0"]

    cmd += map_opts + [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]

    if soft:
        cmd += [
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            f"language={subtitle_language}",
        ]

    cmd += ["-shortest", out_path]
    logger.info("Running ffmpeg loop render -> %s", out_path)
    logger.debug("ffmpeg command: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    return out_path


def build_slideshow_from_images(
    image_paths: list,
    audio_path: str,
    subs_path: str,
    out_path: str,
    *,
    order: str = "alphabetical",
    seed: int | None = None,
    image_duration: float = 5.0,
    image_durations: Optional[List[float]] = None,
    burn: bool = True,
    soft: bool = False,
    font_size: int = 28,
    font_file: str = "",
    subtitle_language: str = "und",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> str:
    """Create a looping slideshow from still images to match the narration length."""

    if burn and soft:
        raise ValueError("Choose either burn OR soft subtitles, not both.")

    require_ffmpeg()

    manual = image_durations is not None
    if manual:
        if len(image_durations) != len(image_paths):
            raise ValueError("image_durations must match the number of images")
        for dur in image_durations:
            if dur <= 0:
                raise ValueError("Image durations must be positive")
        ordered = list(image_paths)
        base_durations = image_durations
    else:
        if image_duration <= 0:
            raise ValueError("image_duration must be positive")
        ordered = _validate_ordered_paths(image_paths, order, seed=seed)
        base_durations = [image_duration] * len(ordered)

    adur = ffprobe_duration(audio_path)
    logger.info(
        "Building slideshow with %d images (order=%s)",
        len(ordered),
        order,
    )

    if manual:
        seq = list(zip(ordered, base_durations))
        total = sum(base_durations)
    else:
        total = 0.0
        seq: list[tuple[str, float]] = []
        idx = 0
        safety = adur + 1.0
        while total < safety and ordered:
            base_idx = idx % len(ordered)
            dur = base_durations[base_idx]
            seq.append((ordered[base_idx], dur))
            total += dur
            idx += 1

    cmd = ["ffmpeg", "-loglevel", ffmpeg_log_level(), "-y"]
    for path, dur in seq:
        cmd += ["-loop", "1", "-t", f"{dur:.3f}", "-i", path]
    cmd += ["-i", audio_path]

    audio_idx = len(seq)
    subs_idx = None
    if soft:
        cmd += ["-i", subs_path]
        subs_idx = audio_idx + 1

    filter_parts = []
    labels = []
    for idx in range(len(seq)):
        filter_parts.append(
            f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps},setpts=PTS-STARTPTS[v{idx}]"
        )
        labels.append(f"[v{idx}]")
    filter_parts.append(f"{''.join(labels)}concat=n={len(seq)}:v=1:a=0[vcat]")

    vmap = "[vcat]"
    if burn:
        subf = _subtitles_filter(subs_path, font_size, font_file)
        filter_parts.append(f"{vmap}{subf}[vout]")
        vmap = "[vout]"

    output_opts = [
        "-filter_complex",
        ";".join(filter_parts),
        "-t",
        f"{adur:.3f}",
        "-map",
        vmap,
        "-map",
        f"{audio_idx}:a:0",
    ]

    if soft and subs_idx is not None:
        output_opts += ["-map", f"{subs_idx}:0"]

    output_opts += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]

    if soft:
        output_opts += [
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            f"language={subtitle_language}",
        ]

    output_opts += ["-shortest", out_path]

    cmd += output_opts
    logger.info("Running ffmpeg slideshow render -> %s", out_path)
    logger.debug("ffmpeg command: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    return out_path
