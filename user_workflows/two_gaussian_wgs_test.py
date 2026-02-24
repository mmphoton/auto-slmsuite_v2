"""Setup validation script: two Gaussian spots + optional experimental WGS.

This script intentionally allows pattern generation without loading lab calibration
artifacts. Only the deep phase LUT is used when depth correction is enabled.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from user_workflows.andor_camera import AndorConnectionConfig, PylablibAndorCamera
from user_workflows.bootstrap import bootstrap_runtime
from user_workflows.commands.pattern import depth_correct, load_phase_lut


def _hold_until_interrupt(slm, cam=None):
    print("Holding SLM phase. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.close()
        slm.set_phase(None, settle=True)
        slm.close()


def _compute_spot_knm(shape, args):
    """Return 2xN knm spot array using center offsets relative to FFT center."""
    half_sep = float(args.separation_knm) / 2.0
    center_x = (float(shape[1]) / 2.0) + float(args.center_knm_x)
    center_y = (float(shape[0]) / 2.0) + float(args.center_knm_y)
    return np.array(
        [
            [center_x - half_sep, center_x + half_sep],
            [center_y, center_y],
        ]
    )


def _build_two_spot_phase(args, slm):
    """Build two-spot phase using SpotHologram in SLM-only coordinates.

    We intentionally use the "knm" basis here so generation works with an SLM
    object directly (no camera/Fourier calibration wrapper required).
    """
    from slmsuite.holography.algorithms import SpotHologram

    shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)
    spot_knm = _compute_spot_knm(shape, args)
    hologram = SpotHologram(shape, spot_vectors=spot_knm, basis="knm", cameraslm=None)
    hologram.optimize(method=args.holo_method, maxiter=args.holo_maxiter, feedback="computational")
    return np.mod(hologram.get_phase(), 2 * np.pi)


def run(args):
    bootstrap_runtime(repo_root=args.repo_root, sdk_root=args.sdk_root)

    from slmsuite.hardware.cameraslms import FourierSLM
    from slmsuite.hardware.slms.holoeye import Holoeye
    from slmsuite.holography.algorithms import FeedbackHologram
    from slmsuite.holography.toolbox.phase import blaze

    deep = None
    if args.use_phase_depth_correction:
        deep = load_phase_lut(Path(args.lut_file), key=args.lut_key)

    slm = Holoeye(preselect="index:0")

    if args.separation_kxy is not None:
        args.separation_knm = float(args.separation_kxy)

    phase_wrapped = _build_two_spot_phase(args, slm)
    phase_wrapped = np.mod(phase_wrapped + blaze(grid=slm, vector=(args.blaze_k, args.blaze_k)), 2 * np.pi)
    phase_to_show = depth_correct(phase_wrapped, deep) if args.use_phase_depth_correction else phase_wrapped
    slm.set_phase(phase_to_show, settle=True)
    print(
        "Displayed two-spot Gaussian pattern "
        f"(radius_hint={args.radius_hint_px}px, separation_knm={args.separation_knm}, blaze={args.blaze_k})."
    )

    if not args.run_experimental_wgs:
        _hold_until_interrupt(slm)
        return

    cam = PylablibAndorCamera(
        AndorConnectionConfig(camera_serial=args.camera_serial, exposure_s=args.exposure_s, shutter_mode="auto"),
        verbose=True,
    )
    fs = FourierSLM(cam, slm)

    try:
        img = fs.cam.get_image().astype(float)
        peak = np.nanmax(img)
        target_ij = img / peak if peak > 0 else img

        feedback = FeedbackHologram(shape=fs.slm.shape, target_ij=target_ij, cameraslm=fs)
        feedback.optimize(
            method="WGS-Kim",
            feedback="experimental",
            maxiter=int(args.feedback_iters),
            stat_groups=["experimental_ij", "computational"],
        )

        corrected = np.mod(feedback.get_phase(include_propagation=True), 2 * np.pi)
        corrected = depth_correct(corrected, deep) if args.use_phase_depth_correction else corrected
        fs.slm.set_phase(corrected, settle=True)
        print(f"Experimental WGS correction finished ({args.feedback_iters} iterations).")
        _hold_until_interrupt(fs.slm, cam=fs.cam)
    finally:
        fs.cam.close()
        fs.slm.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--sdk-root", default=None)
    parser.add_argument("--lut-file", default="deep_1024.mat")
    parser.add_argument("--lut-key", default="deep")
    parser.add_argument("--use-phase-depth-correction", action="store_true", default=True)
    parser.add_argument("--no-phase-depth-correction", dest="use_phase_depth_correction", action="store_false")

    parser.add_argument("--radius-hint-px", type=float, default=25.0, help="Operator note for desired spot radius.")
    parser.add_argument("--center-knm-x", type=float, default=0.0, help="X offset from FFT center in knm basis.")
    parser.add_argument("--center-knm-y", type=float, default=0.0, help="Y offset from FFT center in knm basis.")
    parser.add_argument("--separation-knm", type=float, default=30.0, help="Spot separation in knm basis.")

    parser.add_argument(
        "--separation-kxy",
        type=float,
        default=None,
        help="Deprecated alias for --separation-knm (kept for compatibility).",
    )
    parser.add_argument("--blaze-k", type=float, default=0.003)
    parser.add_argument("--holo-method", default="WGS-Kim")
    parser.add_argument("--holo-maxiter", type=int, default=40)

    parser.add_argument("--run-experimental-wgs", action="store_true", default=False)
    parser.add_argument("--feedback-iters", type=int, default=10)
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--exposure-s", type=float, default=0.03)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
