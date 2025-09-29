from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_NUM_RE = re.compile(r"(\d+)")


def natural_key(name: str) -> List[tuple[int, object]]:
    parts = _NUM_RE.split(name)
    key: List[tuple[int, object]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return key


def sort_image_paths(
    paths: Iterable[str],
    order: str,
    seed: Optional[int] = None,
) -> List[str]:
    order_key = order.lower()
    items = list(paths)
    if order_key not in {"alphabetical", "random"}:
        raise ValueError("order must be either 'alphabetical' or 'random'")
    if order_key == "alphabetical":
        items.sort(key=lambda p: natural_key(Path(p).name))
    else:
        rng = random.Random(seed)
        rng.shuffle(items)
    return items


def calculate_image_durations(
    image_paths: List[str],
    audio_seconds: float,
    config_map: Optional[Dict[str, Tuple[float, float]]],
    default_duration: Optional[float],
    min_auto_duration: float = 0.5,
) -> Tuple[List[str], List[float], List[Tuple[str, str]]]:
    """Return filtered image list, durations, and log events."""

    tol = 1e-3
    filtered: List[str] = []
    durations: List[float] = []
    events: List[Tuple[str, str]] = []

    total_images = len(image_paths)
    current_time = 0.0

    for idx, path in enumerate(image_paths):
        name = Path(path).name
        duration_value: Optional[float] = None

        if config_map and name in config_map:
            start, end = config_map[name]
            if start >= audio_seconds - tol:
                events.append(("warning", f"Skipping {name}; start {start:.2f}s beyond audio"))
                continue
            if start > current_time + tol:
                raise ValueError(
                    f"Image config for {name!r} leaves a gap before {start:.3f}s"
                )
            if start < current_time - tol:
                events.append((
                    "warning",
                    f"Adjusting start of {name} from {start:.3f}s to {current_time:.3f}s",
                ))
                start = current_time
            end = min(end, audio_seconds)
            if end <= start + tol:
                events.append((
                    "warning",
                    f"Skipping {name}; configured window too small ({end - start:.3f}s)",
                ))
                continue
            duration_value = end - start
            current_time = end
            events.append(("info", f"Using configured window {start:.2f}–{end:.2f}s for {name}"))
        elif default_duration is not None:
            if current_time >= audio_seconds - tol:
                events.append(("warning", f"Skipping {name}; audio already covered"))
                continue
            remaining = audio_seconds - current_time
            duration_value = min(default_duration, remaining)
            current_time += duration_value
            events.append(("info", f"Using fixed duration {duration_value:.2f}s for {name}"))
        else:
            if current_time >= audio_seconds - tol:
                events.append(("warning", f"Skipping {name}; audio already covered"))
                continue
            remaining = audio_seconds - current_time
            remaining_slots = total_images - len(filtered)
            if remaining_slots <= 0 or remaining <= tol:
                events.append(("warning", f"Skipping {name}; no remaining audio coverage"))
                continue
            auto_duration = max(remaining / remaining_slots, min_auto_duration)
            auto_duration = min(auto_duration, remaining)
            duration_value = auto_duration
            current_time += duration_value
            events.append((
                "info",
                f"Auto duration {duration_value:.2f}s based on {remaining_slots} slot(s)",
            ))

        if duration_value is None or duration_value <= tol:
            events.append(("warning", f"Skipping {name}; negligible duration"))
            continue

        filtered.append(path)
        durations.append(duration_value)

    if not filtered:
        raise ValueError("No valid images remain after applying durations/config")

    leftover = audio_seconds - current_time
    if leftover > tol:
        durations[-1] += leftover
        events.append(("info", f"Extending last image by {leftover:.2f}s to match audio"))

    return filtered, durations, events
