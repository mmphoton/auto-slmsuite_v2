"""Acquire camera frames while showing an SLM pattern."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from user_workflows.calibration_io import assert_required_calibration_files
from user_workflows.andor_camera import AndorConnectionConfig, PylablibAndorCamera
from user_workflows.commands.pattern import build_pattern, hold_until_interrupt, load_phase_lut


def add_acquire_args(parser: argparse.ArgumentParser):
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--exposure-s", type=float, default=0.03)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--calibration-root", default="user_workflows/calibrations")
    parser.add_argument("--save-frames", default="", help="Optional .npy output path for acquired frames")


def create_fourier_slm(args):
    from slmsuite.hardware.cameraslms import FourierSLM
    from slmsuite.hardware.slms.holoeye import Holoeye

    slm = Holoeye(preselect="index:0")
    cam = PylablibAndorCamera(
        AndorConnectionConfig(camera_serial=args.camera_serial, exposure_s=args.exposure_s, shutter_mode="auto"),
        verbose=True,
    )

    fs = FourierSLM(cam, slm)
    calibration_paths = assert_required_calibration_files(args.calibration_root)
    fs.load_calibration("fourier", str(calibration_paths["fourier"]))
    fs.load_calibration("wavefront_superpixel", str(calibration_paths["wavefront_superpixel"]))
    fs.slm.source["amplitude"] = np.load(calibration_paths["source_amplitude"])
    return fs


def run_acquire(args):
    lut_path = Path(args.lut_file)
    if not lut_path.exists():
        raise FileNotFoundError(
            f"LUT file '{lut_path}' does not exist. Fix: provide --lut-file or run workflow doctor first."
        )
    deep = load_phase_lut(lut_path, args.lut_key)
    calibration_paths = assert_required_calibration_files(args.calibration_root)

    if args.dry_run:
        print("[dry-run] acquisition inputs validated:")
        print(f"  LUT: {lut_path.resolve()}")
        for key, path in calibration_paths.items():
            print(f"  calibration[{key}]: {path.resolve()}")
        if args.save_frames:
            Path(args.save_frames).parent.mkdir(parents=True, exist_ok=True)
        return

    fs = create_fourier_slm(args)
    try:
        pattern = build_pattern(args, fs.slm, deep)
        fs.slm.set_phase(pattern, settle=True)

        frames = [fs.cam.get_image() for _ in range(max(1, args.frames))]
        frames = np.asarray(frames)
        print(f"Acquired {frames.shape[0]} Andor full-frame image(s): shape={frames.shape[1:]}")

        if args.save_frames:
            out = Path(args.save_frames)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, frames)
            print(f"Saved frames to {out.resolve()}")

        hold_until_interrupt(fs.slm)
    finally:
        fs.cam.close()
        fs.slm.close()
