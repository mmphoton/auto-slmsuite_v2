"""Acquire camera frames while showing an SLM pattern."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from user_workflows.andor_camera import AndorConnectionConfig, PylablibAndorCamera
from user_workflows.commands.pattern import build_pattern, hold_until_interrupt, load_phase_lut


def add_acquire_args(parser: argparse.ArgumentParser):
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--exposure-s", type=float, default=0.03)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--calibration-root", default="user_workflows/calibrations")
    parser.add_argument("--save-frames", default="", help="Optional .npy output path for acquired frames")
    parser.add_argument(
        "--experimental-wgs-iters",
        type=int,
        default=0,
        help="Run this many iterations of experimental WGS before acquiring frames (0 disables).",
    )
    parser.add_argument(
        "--show-camera-phase-plot",
        action="store_true",
        default=False,
        help="Display side-by-side camera intensity and SLM phase after acquisition.",
    )
    parser.add_argument(
        "--save-camera-phase-plot",
        default="",
        help="Optional path to save side-by-side camera intensity + phase plot (.png recommended).",
    )


def create_fourier_slm(args):
    from slmsuite.hardware.cameraslms import FourierSLM
    from slmsuite.hardware.slms.holoeye import Holoeye

    slm = Holoeye(preselect="index:0")
    cam = PylablibAndorCamera(
        AndorConnectionConfig(camera_serial=args.camera_serial, exposure_s=args.exposure_s, shutter_mode="auto"),
        verbose=True,
    )

    fs = FourierSLM(cam, slm)

    calibration_root = Path(getattr(args, "calibration_root", "user_workflows/calibrations"))
    fourier_path = calibration_root / "fourier-calibration.h5"
    wavefront_path = calibration_root / "wavefront-superpixel-calibration.h5"
    source_amp_path = calibration_root / "source-amplitude-corrected.npy"

    if fourier_path.exists():
        fs.load_calibration("fourier", str(fourier_path))
    else:
        warnings.warn(
            f"Fourier calibration not found at '{fourier_path}'. Experimental feedback will be unavailable until calibration is provided."
        )

    if wavefront_path.exists():
        fs.load_calibration("wavefront_superpixel", str(wavefront_path))

    if source_amp_path.exists():
        fs.slm.source["amplitude"] = np.load(source_amp_path)

    return fs


def _run_experimental_wgs(fs, iterations: int) -> np.ndarray:
    from slmsuite.holography.algorithms import FeedbackHologram

    img = fs.cam.get_image().astype(float)
    peak = np.nanmax(img)
    target_ij = img / peak if peak > 0 else img

    holo = FeedbackHologram(shape=fs.slm.shape, target_ij=target_ij, cameraslm=fs)
    holo.optimize(
        method="WGS-Kim",
        feedback="experimental",
        maxiter=int(iterations),
        stat_groups=["experimental", "computational"],
    )
    return np.mod(holo.get_phase(include_propagation=True), 2 * np.pi)


def _plot_camera_phase(camera_frame: np.ndarray, phase: np.ndarray, save_path: str = "", show_plot: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    im0 = axes[0].imshow(camera_frame, cmap="inferno")
    axes[0].set_title("Andor intensity")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.mod(phase, 2 * np.pi), cmap="twilight", vmin=0.0, vmax=2 * np.pi)
    axes[1].set_title("SLM phase")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved camera/phase plot to {out.resolve()}")

    if show_plot:
        plt.show(block=False)
    else:
        plt.close(fig)


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
        if args.save_camera_phase_plot:
            Path(args.save_camera_phase_plot).parent.mkdir(parents=True, exist_ok=True)
        return

    fs = create_fourier_slm(args)
    try:
        phase_for_display = build_pattern(args, fs.slm, deep)
        fs.slm.set_phase(phase_for_display, settle=True)

        if args.experimental_wgs_iters > 0:
            optimized_phase = _run_experimental_wgs(fs, iterations=args.experimental_wgs_iters)
            phase_for_display = optimized_phase
            fs.slm.set_phase(phase_for_display, settle=True)
            print(f"Experimental WGS completed ({args.experimental_wgs_iters} iterations).")

        frames = [fs.cam.get_image() for _ in range(max(1, args.frames))]
        frames = np.asarray(frames)
        print(f"Acquired {frames.shape[0]} Andor full-frame image(s): shape={frames.shape[1:]}")

        if args.save_frames:
            out = Path(args.save_frames)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, frames)
            print(f"Saved frames to {out.resolve()}")

        if args.show_camera_phase_plot or args.save_camera_phase_plot:
            _plot_camera_phase(
                camera_frame=frames[-1],
                phase=phase_for_display,
                save_path=args.save_camera_phase_plot,
                show_plot=args.show_camera_phase_plot,
            )

        hold_until_interrupt(fs.slm)
    finally:
        fs.cam.close()
        fs.slm.close()
