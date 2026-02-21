"""Backward-compatible wrapper for `workflow calibrate`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from user_workflows.commands.calibrate import add_calibration_args, run_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_calibration_args(parser)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and paths without touching hardware")
    args = parser.parse_args(argv)
    run_calibration(args)


if __name__ == "__main__":
    main()
