"""Acquire camera frames while showing an SLM pattern."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
    return fs


def run_acquire(args):
    deep = None
    if args.use_phase_depth_correction:
        lut_path = Path(args.lut_file)
        if not lut_path.exists():
            raise FileNotFoundError(
                f"LUT file '{lut_path}' does not exist. Fix: provide --lut-file or use --no-phase-depth-correction."
            )
        deep = load_phase_lut(lut_path, args.lut_key)

    if args.dry_run:
        print("[dry-run] acquisition inputs validated:")
        if args.use_phase_depth_correction:
            print(f"  LUT: {lut_path.resolve()}")
        else:
            print("  LUT: skipped (--no-phase-depth-correction)")
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
