"""YAML configuration loading, merging, and validation helpers."""

from __future__ import annotations

import ast
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from user_workflows.config.schema import (
    CalibrationConfig,
    FeedbackConfig,
    HardwareConfig,
    OutputConfig,
    PatternConfig,
    WorkflowConfig,
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_nested(root: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cursor = root
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _from_dataclass(cls, data: dict[str, Any]):
    valid_names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in valid_names}
    return cls(**kwargs)


def _defaults() -> dict[str, Any]:
    return WorkflowConfig().to_dict()


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("Top-level config must be a mapping")
    return loaded


def _parse_override_value(value_text: str) -> Any:
    if yaml is not None:
        return yaml.safe_load(value_text)
    try:
        return ast.literal_eval(value_text)
    except (ValueError, SyntaxError):
        return value_text


def load_workflow_config(config_path: str | Path | None = None, overrides: list[str] | None = None) -> WorkflowConfig:
    raw = _defaults()

    if config_path:
        raw = _deep_merge(raw, _load_mapping(Path(config_path)))

    for entry in overrides or []:
        if "=" not in entry:
            raise ValueError(f"Invalid override '{entry}'. Expected key=value")
        key, value_text = entry.split("=", maxsplit=1)
        _set_nested(raw, key.strip(), _parse_override_value(value_text))

    config = WorkflowConfig(
        hardware=_from_dataclass(HardwareConfig, raw.get("hardware", {})),
        pattern=PatternConfig(**raw.get("pattern", {})),
        feedback=_from_dataclass(FeedbackConfig, raw.get("feedback", {})),
        output=_from_dataclass(OutputConfig, raw.get("output", {})),
        calibration=_from_dataclass(CalibrationConfig, raw.get("calibration", {})),
        lut_file=raw.get("lut_file", WorkflowConfig().lut_file),
        lut_key=raw.get("lut_key", WorkflowConfig().lut_key),
    )
    config.validate()
    return config


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(data, f, sort_keys=False)
        else:
            f.write(json.dumps(data, indent=2))
            f.write("\n")
