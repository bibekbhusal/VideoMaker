from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


ImageWindow = Tuple[str, float, float]


def build_anchor_schedule(
    image_names: Sequence[str],
    audio_seconds: float,
    anchor_name: str,
    start_timestamp: float,
) -> List[ImageWindow]:
    """Return ordered (name, start, end) windows with ``anchor_name`` anchored."""

    if not image_names:
        raise ValueError("At least one image is required")
    if start_timestamp < 0:
        raise ValueError("Start timestamp must be non-negative")
    if audio_seconds <= 0:
        raise ValueError("Audio duration must be positive")
    if start_timestamp >= audio_seconds:
        raise ValueError("Start timestamp must be within the audio duration")

    order = list(image_names)
    try:
        start_index = order.index(anchor_name)
    except ValueError as exc:  # pragma: no cover - caller validates names
        raise ValueError(f"Anchor image {anchor_name!r} not found") from exc

    after_names = order[start_index:]
    after_count = len(after_names)
    if after_count <= 0:
        raise ValueError("Anchor image list computation failed")

    if start_index > 0 and start_timestamp <= 0:
        raise ValueError(
            "Start timestamp must be greater than zero when the anchor image is not first"
        )

    remaining_audio = audio_seconds - start_timestamp
    if remaining_audio <= 0:
        raise ValueError(
            f"Audio duration {audio_seconds:.3f}s leaves no room after {start_timestamp:.3f}s"
        )

    per_after = remaining_audio / after_count

    schedule: List[ImageWindow] = []
    current = start_timestamp

    for offset, name in enumerate(after_names):
        start = current
        end = start + per_after
        if end <= start:
            raise ValueError(
                f"Computed non-positive duration for {name!r}. Check timestamp and audio length"
            )
        schedule.append((name, start, end))
        current = end

    if schedule:
        last_name, last_start, last_end = schedule[-1]
        leftover = audio_seconds - last_end
        if abs(leftover) > 1e-6:
            adjusted_end = last_end + leftover
            if adjusted_end <= last_start:
                raise ValueError(
                    f"Computed non-positive duration for {last_name!r}. Check timestamp and audio length"
                )
            schedule[-1] = (last_name, last_start, adjusted_end)

    return schedule


def serialise_schedule(schedule: Iterable[ImageWindow]) -> List[Dict[str, float | str]]:
    """Round schedule to milliseconds for JSON output in a stable order."""

    serialised: List[Dict[str, float | str]] = []
    for name, start, end in schedule:
        serialised.append(
            {
                "file": name,
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )
    return serialised


def parse_image_config_payload(payload: Any) -> List[ImageWindow]:
    """Parse JSON payload into ordered image windows."""

    entries: List[ImageWindow] = []

    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            if isinstance(item, dict):
                try:
                    name = str(item["file"])
                    start = float(item["start"])
                    end = float(item["end"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid image config entry at index {idx}: expected keys file/start/end"
                    ) from exc
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                name = str(item[0])
                try:
                    start = float(item[1])
                    end = float(item[2])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid timestamps at index {idx}: {item}") from exc
            else:
                raise ValueError(
                    "Image config list entries must be objects with file/start/end or [file, start, end]"
                )

            if end <= start and not (start < 0 and end < 0):
                raise ValueError(
                    f"Image config entry for {name!r} must have end greater than start"
                )
            entries.append((name, start, end))
    elif isinstance(payload, dict):
        for name, window in payload.items():
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise ValueError(
                    f"Image config entry for {name!r} must be a [start, end] list"
                )
            try:
                start = float(window[0])
                end = float(window[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid timestamps for {name!r}: {window}") from exc
            if end <= start and not (start < 0 and end < 0):
                raise ValueError(
                    f"Image config entry for {name!r} must have end greater than start"
                )
            entries.append((str(name), start, end))
    else:
        raise ValueError("Image config must be a JSON array or object")

    seen: Dict[str, None] = {}
    for name, *_ in entries:
        if name in seen:
            raise ValueError(f"Duplicate image entry {name!r} in image config")
        seen[name] = None

    return entries
