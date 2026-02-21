"""Backward-compatible wrapper around the new workflow CLI commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from user_workflows.commands.acquire import add_acquire_args, run_acquire
from user_workflows.commands.feedback import add_feedback_args, run_feedback
from user_workflows.commands.pattern import add_pattern_args, run_pattern


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_pattern_args(parser)
    add_acquire_args(parser)
    add_feedback_args(parser)

    parser.add_argument("--use-camera", action="store_true", help="Enable Andor full-frame acquisition")
    parser.add_argument("--feedback", action="store_true", help="Run experimental feedback optimization")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and file paths without touching hardware")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.feedback and not args.use_camera:
        parser.error("--feedback requires --use-camera")

    if not args.use_camera:
        run_pattern(args)
    elif args.feedback:
        run_feedback(args)
    else:
        run_acquire(args)


if __name__ == "__main__":
    main()
