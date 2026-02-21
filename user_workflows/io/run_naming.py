"""Run directory naming helpers for user workflows."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_TEMPLATE = "{date}_{run_name}_{pattern}_{camera}_{iter}"


@dataclass(frozen=True)
class RunNamingConfig:
    run_name: str
    output_root: Path
    name_template: str = DEFAULT_TEMPLATE
    overwrite: bool = False
    resume: bool = False


def add_naming_args(parser: argparse.ArgumentParser) -> None:
    """Attach common run naming/output flags to a parser."""
    parser.add_argument("--run-name", default="run", help="User-facing run identifier")
    parser.add_argument("--output-root", default="user_workflows/output", help="Root directory for all run outputs")
    parser.add_argument(
        "--name-template",
        default=DEFAULT_TEMPLATE,
        help=(
            "Directory template using fields: {date}, {run_name}, {pattern}, {camera}, {iter}. "
            "Default: {date}_{run_name}_{pattern}_{camera}_{iter}"
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--overwrite", action="store_true", help="Overwrite files in a pre-existing run directory")
    mode_group.add_argument("--resume", action="store_true", help="Reuse a pre-existing run directory")


def config_from_args(args: argparse.Namespace) -> RunNamingConfig:
    return RunNamingConfig(
        run_name=args.run_name,
        output_root=Path(args.output_root),
        name_template=args.name_template,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )


def _slugify(value: str) -> str:
    value = str(value).strip().replace(" ", "-")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    return value.strip("-_") or "na"


def render_name(template: str, *, run_name: str, pattern: str, camera: str, iter_value: int, now: datetime | None = None) -> str:
    now = now or datetime.now()
    payload = {
        "date": now.strftime("%Y%m%d"),
        "run_name": _slugify(run_name),
        "pattern": _slugify(pattern),
        "camera": _slugify(camera),
        "iter": f"{int(iter_value):03d}",
    }
    try:
        rendered = template.format(**payload)
    except KeyError as exc:
        allowed = ", ".join(sorted(payload))
        raise ValueError(f"Unknown name-template field {exc!s}. Allowed fields: {allowed}") from exc
    return _slugify(rendered)


def choose_run_directory(config: RunNamingConfig, *, pattern: str, camera: str, now: datetime | None = None) -> Path:
    """Choose/create a run directory following collision policies.

    Default policy auto-increments suffixes `_001`, `_002`, ... when a collision is found.
    """
    root = config.output_root
    root.mkdir(parents=True, exist_ok=True)

    base_name = render_name(
        config.name_template,
        run_name=config.run_name,
        pattern=pattern,
        camera=camera,
        iter_value=0,
        now=now,
    )
    candidate = root / base_name
    if not candidate.exists():
        return candidate

    if config.overwrite or config.resume:
        return candidate

    idx = 1
    while True:
        suffixed = root / f"{base_name}_{idx:03d}"
        if not suffixed.exists():
            return suffixed
        idx += 1
