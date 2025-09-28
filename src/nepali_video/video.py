
import subprocess, shlex
from .utils import require_ffmpeg, ffprobe_duration, safe

def build_video_playlist(
    video_paths: list,
    audio_path: str,
    subs_path: str,
    out_path: str,
    burn: bool = True,
    soft: bool = False,
    font_size: int = 28,
    font_file: str = "",
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
):
    """
    Concatenate multiple clips (hard cuts), repeat the sequence until it covers
    the narration duration, trim to length, then add audio + subtitles.
    """
    if burn and soft:
        raise ValueError("Choose either burn OR soft subtitles, not both.")

    require_ffmpeg()
    adur = ffprobe_duration(audio_path)

    # Measure durations and build a repeated list long enough
    durs = [ffprobe_duration(p) for p in video_paths]
    seq = []
    total = 0.0
    i = 0
    while total < adur + 1.0:  # small safety margin
        idx = i % len(video_paths)
        seq.append((video_paths[idx], durs[idx]))
        total += durs[idx]
        i += 1

    # Inputs: all videos first, then audio (and maybe subs)
    cmd = ["ffmpeg", "-y"]
    for v, _ in seq:
        cmd += ["-i", v]
    cmd += ["-i", audio_path]
    audio_idx = len(seq)

    subs_idx = None
    if soft:
        cmd += ["-i", subs_path]
        subs_idx = audio_idx + 1

    # Build filter_complex:
    # - Normalize each video to same size (scale+pad), fps, and reset PTS
    # - Concat videos
    filter_parts = []
    labels = []
    for idx in range(len(seq)):
        filter_parts.append(
            f"[{idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps},setpts=PTS-STARTPTS[v{idx}]"
        )
        labels.append(f"[v{idx}]")
    filter_parts.append(f"{''.join(labels)}concat=n={len(seq)}:v=1:a=0[vcat]")

    # Burn subtitles or keep them soft
    vmap = "[vcat]"
    if burn:
        subf = f"subtitles={safe(subs_path)}:force_style='Fontsize={font_size}'"
        filter_parts.append(f"{vmap}{subf}[vout]")
        vmap = "[vout]"

    output_opts = [
        "-filter_complex", ";".join(filter_parts),
        "-t", f"{adur:.3f}",
        "-map", vmap,
        "-map", f"{audio_idx}:a:0",
    ]

    if soft and subs_idx is not None:
        output_opts += ["-map", f"{subs_idx}:0"]

    output_opts += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
    ]

    if soft:
        output_opts += ["-c:s", "mov_text", "-metadata:s:s:0", "language=nep"]

    output_opts += ["-shortest", out_path]

    cmd += output_opts
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
    font_file: str = ""
):
    """
    Loops the village clip to match audio length, muxes audio, and either burns or embeds subtitles.
    """
    if burn and soft:
        raise ValueError("Choose either burn OR soft subtitles, not both.")

    require_ffmpeg()

    # Get audio duration to know how long to loop the video
    adur = ffprobe_duration(audio_path)

    # Build the base command:
    #  - loop the video indefinitely (-stream_loop -1), then trim to audio duration (-t <adur>)
    #  - re-encode video to H.264, ensure yuv420p for compatibility
    #  - map audio from file, normalize sample rate, and -shortest to cut to the shorter stream
    base = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", video_path,
        "-i", audio_path,
    ]

    audio_idx = 1
    subs_idx = None
    if soft:
        base += ["-i", subs_path]
        subs_idx = 2

    base += ["-t", f"{adur:.3f}"]

    vf_filters = []
    if burn:
        # Subtitles burn-in (requires libass; ffmpeg from Homebrew includes it)
        style_parts = [f"Fontsize={font_size}"]
        if font_file:
            # Use font file via FONTCONFIG path or drawtext; simplest: rely on libass fontconfig discovery
            # Users can install fonts to ~/Library/Fonts and refer by family name if needed.
            pass
        sub_filter = f"subtitles={safe(subs_path)}:force_style='{','.join(style_parts)}'"
        vf_filters.append(sub_filter)

    if vf_filters:
        base += ["-vf", ",".join(vf_filters)]

    map_opts = ["-map", "0:v:0", "-map", f"{audio_idx}:a:0"]
    if soft and subs_idx is not None:
        map_opts += ["-map", f"{subs_idx}:0"]

    base += map_opts + [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
    ]

    if soft:
        base += ["-c:s", "mov_text", "-metadata:s:s:0", "language=nep"]

    base += [
        "-shortest",
        out_path
    ]

    # Run
    subprocess.check_call(base)
    return out_path
