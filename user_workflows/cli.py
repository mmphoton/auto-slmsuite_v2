"""Unified CLI for user workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from user_workflows.commands.acquire import add_acquire_args, run_acquire
from user_workflows.commands.calibrate import add_calibration_args, run_calibration
from user_workflows.commands.doctor import add_doctor_args, run_doctor
from user_workflows.commands.feedback import add_feedback_args, run_feedback
from user_workflows.commands.pattern import add_pattern_args, run_pattern
from user_workflows.commands.presets import apply_preset
from user_workflows.bootstrap import bootstrap_runtime


def add_common_profile_args(parser: argparse.ArgumentParser):
    parser.add_argument("--preset", default="", help="Preset profile name to load")
    parser.add_argument("--preset-file", default="user_workflows/presets.json", help="JSON file that defines profiles")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and paths without touching hardware")
    parser.add_argument("--repo-root", default=None, help="Optional repo root for runtime bootstrap")
    parser.add_argument("--sdk-root", default=None, help="Optional Holoeye SDK root for runtime bootstrap")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root_subparsers = parser.add_subparsers(dest="root_command", required=True)

    workflow = root_subparsers.add_parser("workflow", help="Workflow entrypoint")
    wf_subparsers = workflow.add_subparsers(dest="workflow_command", required=True)

    calibrate = wf_subparsers.add_parser("calibrate", help="Run SLM calibration workflow")
    add_common_profile_args(calibrate)
    add_calibration_args(calibrate)
    calibrate.set_defaults(handler=run_calibration)

    pattern = wf_subparsers.add_parser("pattern", help="Display a selected SLM pattern")
    add_common_profile_args(pattern)
    add_pattern_args(pattern)
    pattern.set_defaults(handler=run_pattern)

    acquire = wf_subparsers.add_parser("acquire", help="Display pattern and acquire camera frames")
    add_common_profile_args(acquire)
    add_pattern_args(acquire)
    add_acquire_args(acquire)
    acquire.set_defaults(handler=run_acquire)

    feedback = wf_subparsers.add_parser("feedback", help="Run camera-driven feedback optimization")
    add_common_profile_args(feedback)
    add_pattern_args(feedback)
    add_acquire_args(feedback)
    add_feedback_args(feedback)
    feedback.set_defaults(handler=run_feedback)

    doctor = wf_subparsers.add_parser("doctor", help="Validate SDK, hardware discoverability, and files")
    add_common_profile_args(doctor)
    add_doctor_args(doctor)
    doctor.set_defaults(handler=run_doctor)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args = apply_preset(args, parser)
    bootstrap_runtime(repo_root=getattr(args, "repo_root", None), sdk_root=getattr(args, "sdk_root", None))
    args.handler(args)


if __name__ == "__main__":
    main()
