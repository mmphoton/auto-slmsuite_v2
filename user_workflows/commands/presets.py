"""Preset/profile loading helpers for workflow commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_PRESET_FILE = Path("user_workflows/presets.json")


def load_preset(profile: str, preset_file: str | Path = DEFAULT_PRESET_FILE) -> dict[str, Any]:
    path = Path(preset_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Preset file '{path}' was not found. Create it or rerun without --preset."
        )

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    profiles = data.get("profiles", data)
    if profile not in profiles:
        available = ", ".join(sorted(profiles.keys())) if isinstance(profiles, dict) else "<none>"
        raise KeyError(f"Preset '{profile}' not found in {path}. Available presets: {available}")

    selected = profiles[profile]
    if not isinstance(selected, dict):
        raise ValueError(f"Preset '{profile}' must map to an object/dict, got {type(selected).__name__}")
    return selected


def apply_preset(args, parser):
    if not getattr(args, "preset", ""):
        return args

    overrides = load_preset(args.preset, getattr(args, "preset_file", DEFAULT_PRESET_FILE))
    for key, value in overrides.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            parser.error(f"Preset key '{key}' is not a valid argument for this command.")
    return args
