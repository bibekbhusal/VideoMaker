from __future__ import annotations

import os
import pathlib
from functools import lru_cache
from typing import Any, Dict

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore


_DEFAULT_FILENAMES = (
    "videomaker.toml",
    ".videomaker.toml",
)


def _candidate_paths() -> "list[pathlib.Path]":
    paths: list[pathlib.Path] = []
    env_path = os.environ.get("VIDEOMAKER_CONFIG")
    if env_path:
        paths.append(pathlib.Path(env_path).expanduser())

    cwd = pathlib.Path.cwd()
    for name in _DEFAULT_FILENAMES:
        paths.append((cwd / name).resolve())

    package_root = pathlib.Path(__file__).resolve().parent.parent
    for name in _DEFAULT_FILENAMES:
        paths.append(package_root / name)

    return paths


def _load_config() -> Dict[str, Dict[str, Any]]:
    for path in _candidate_paths():
        if path.is_file():
            try:
                with path.open("rb") as fh:
                    data = tomllib.load(fh)
            except Exception:
                continue
            return data if isinstance(data, dict) else {}
    return {}


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Dict[str, Any]]:
    """Load configuration once and cache the parsed dictionary."""

    return _load_config()


def command_defaults(command: str) -> Dict[str, Any]:
    """Fetch defaults for a given command from the config file."""

    return load_config().get(command, {})


def get_default(command: str, option: str, fallback: Any) -> Any:
    """Return config value if present; otherwise use provided fallback."""

    return command_defaults(command).get(option, fallback)
