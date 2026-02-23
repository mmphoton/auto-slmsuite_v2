"""System diagnostics for workflow prerequisites."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from user_workflows.andor_camera import verify_camera_discoverable
from user_workflows.calibration_io import calibration_paths
from user_workflows.commands.calibrate import load_phase_lut


def add_doctor_args(parser: argparse.ArgumentParser):
    parser.add_argument("--lut-file", default="deep_1024.mat")
    parser.add_argument("--lut-key", default="deep")
    parser.add_argument("--calibration-root", default="user_workflows/calibrations")
    parser.add_argument("--output-dir", default="user_workflows/output")


def _check_pylablib_import():
    from pylablib.devices import Andor as _Andor  # noqa: F401


def _check_camera_discovery():
    verify_camera_discoverable("")


def _check_lut(path: Path, key: str):
    lut = load_phase_lut(path, key)
    if lut.size < 2:
        raise ValueError("LUT has too few entries.")


def _check_calibrations(root: Path):
    paths = calibration_paths(root)
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing calibration artifacts: " + ", ".join(str(p) for p in missing))

    amp = np.asarray(np.load(paths["source_amplitude"]))
    if amp.ndim != 2:
        raise ValueError(f"source-amplitude-corrected.npy must be 2D, got shape {amp.shape}")


def _check_writable(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".doctor_write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def _run_check(name: str, fn, fix: str):
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        print(f"        Fix: {fix}")
        return False


def run_doctor(args):
    checks = [
        (
            "pylablib Andor import availability",
            _check_pylablib_import,
            "Install pylablib bindings: `pip install pylablib`.",
        ),
        (
            "LUT file existence/shape",
            lambda: _check_lut(Path(args.lut_file), args.lut_key),
            "Generate or point to a valid LUT and rerun `python user_workflows/cli.py workflow doctor --lut-file <path>`."
        ),
        (
            "calibration artifact presence/compatibility",
            lambda: _check_calibrations(Path(args.calibration_root)),
            "Run `python user_workflows/cli.py workflow calibrate --factory <module:function> --phase-lut <lut>`."
        ),
        (
            "writable output directories",
            lambda: _check_writable(Path(args.output_dir)),
            "Change permissions or choose another folder using `--output-dir`.",
        ),
    ]

    if not args.dry_run:
        checks.insert(1, (
            "camera discoverability",
            _check_camera_discovery,
            "Connect/power the camera, then rerun `python user_workflows/cli.py workflow doctor`.",
        ))
    else:
        print("[SKIP] camera discoverability check skipped in --dry-run mode")

    ok = True
    for name, fn, fix in checks:
        ok = _run_check(name, fn, fix) and ok

    if not ok:
        raise SystemExit(2)

    print("Doctor checks passed.")
