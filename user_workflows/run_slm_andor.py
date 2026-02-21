"""Generate configurable SLM analytical patterns with optional Andor iDus acquisition/feedback."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io

from slmsuite.hardware.cameras.andor_idus import AndorIDus
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.holoeye import Holoeye
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram
from slmsuite.holography.toolbox.phase import blaze
from slmsuite.holography.toolbox import phase

from user_workflows.calibration_io import assert_required_calibration_files
from user_workflows.patterns.schemas import (
    PatternValidationError,
    pattern_field_descriptions,
    pattern_params_from_flat_dict,
)


def load_phase_lut(path: Path, key: str = "deep") -> np.ndarray:
    mat = scipy.io.loadmat(path)
    if key not in mat:
        raise ValueError(f"LUT file must contain variable '{key}'")
    deep = np.asarray(mat[key]).squeeze()
    if deep.ndim != 1:
        raise ValueError(f"Expected 1D LUT, got {deep.shape}")
    return deep


def _depth_correct(phi, deep):
    idx = np.clip(np.rint((phi / (2 * np.pi)) * (deep.size - 1)), 0, deep.size - 1).astype(int)
    F = deep[idx]
    corrected = (phi - np.pi) * F + np.pi
    return np.mod(corrected, 2 * np.pi)


def build_pattern(args, slm, deep):
    """Build one of several user-selectable analytical pattern families."""
    params = pattern_params_from_flat_dict(args.pattern, vars(args))

    if args.pattern == "laguerre-gaussian":
        lg_phase = phase.laguerre_gaussian(slm, l=params.l, p=params.p)
        phi = np.mod(lg_phase + blaze(grid=slm, vector=(args.blaze_kx, args.blaze_ky)), 2 * np.pi)
        return _depth_correct(phi, deep)

    # Spot-based patterns use hologram optimization so users get Gaussian-like focused spots.
    shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)

    if args.pattern == "single-gaussian":
        spot_kxy = np.array([[params.kx], [params.ky]])
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=slm)
    elif args.pattern == "double-gaussian":
        dx = float(params.sep_kxy) / 2.0
        spot_kxy = np.array(
            [
                [params.center_kx - dx, params.center_kx + dx],
                [params.center_ky, params.center_ky],
            ]
        )
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=slm)
    elif args.pattern == "gaussian-lattice":
        hologram = SpotHologram.make_rectangular_array(
            shape,
            array_shape=(params.nx, params.ny),
            array_pitch=(params.pitch_x, params.pitch_y),
            array_center=(params.center_kx, params.center_ky),
            basis="kxy",
            cameraslm=slm,
        )
    else:
        raise ValueError(f"Unknown pattern '{args.pattern}'")

    hologram.optimize(
        method=args.holo_method,
        maxiter=args.holo_maxiter,
        feedback="computational",
        stat_groups=["computational"],
    )
    phi = np.mod(hologram.get_phase(), 2 * np.pi)
    return _depth_correct(phi, deep)


def hold_until_interrupt(slm, cam=None):
    print("Holding SLM phase. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        slm.set_phase(None, settle=True)
        slm.close()
        if cam is not None:
            cam.close()


def run_feedback(fs: FourierSLM, iterations: int):
    img = fs.cam.get_image()
    target_ij = img.astype(float)
    peak = np.nanmax(target_ij)
    target_ij /= peak if peak > 0 else 1.0

    holo = FeedbackHologram(shape=fs.slm.shape, target_ij=target_ij, cameraslm=fs)
    holo.optimize(
        method="WGS-Kim",
        feedback="experimental",
        maxiter=int(iterations),
        stat_groups=["experimental_ij", "computational"],
    )
    fs.slm.set_phase(holo.get_phase(include_propagation=True), settle=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Pattern selection + easy parameter knobs.
    parser.add_argument(
        "--pattern",
        default="laguerre-gaussian",
        choices=["single-gaussian", "double-gaussian", "gaussian-lattice", "laguerre-gaussian"],
        help="Pattern family to generate on SLM",
    )

    parser.add_argument("--lut-file", default="deep_1024.mat")
    parser.add_argument("--lut-key", default="deep")
    parser.add_argument("--blaze-kx", type=float, default=0.0)
    parser.add_argument("--blaze-ky", type=float, default=0.0045)

    # LG params
    lg_help = pattern_field_descriptions("laguerre-gaussian")
    parser.add_argument("--lg-l", type=int, default=3, help=lg_help["l"])
    parser.add_argument("--lg-p", type=int, default=0, help=lg_help["p"])

    # Single gaussian spot params.
    single_help = pattern_field_descriptions("single-gaussian")
    parser.add_argument("--single-kx", type=float, default=0.0, help=single_help["kx"])
    parser.add_argument("--single-ky", type=float, default=0.0, help=single_help["ky"])

    # Two gaussian spot params.
    double_help = pattern_field_descriptions("double-gaussian")
    parser.add_argument("--double-center-kx", type=float, default=0.0, help=double_help["center_kx"])
    parser.add_argument("--double-center-ky", type=float, default=0.0, help=double_help["center_ky"])
    parser.add_argument("--double-sep-kxy", type=float, default=0.02, help=double_help["sep_kxy"])

    # Lattice params.
    lattice_help = pattern_field_descriptions("gaussian-lattice")
    parser.add_argument("--lattice-nx", type=int, default=5, help=lattice_help["nx"])
    parser.add_argument("--lattice-ny", type=int, default=5, help=lattice_help["ny"])
    parser.add_argument("--lattice-pitch-x", type=float, default=0.01, help=lattice_help["pitch_x"])
    parser.add_argument("--lattice-pitch-y", type=float, default=0.01, help=lattice_help["pitch_y"])
    parser.add_argument("--lattice-center-kx", type=float, default=0.0, help=lattice_help["center_kx"])
    parser.add_argument("--lattice-center-ky", type=float, default=0.0, help=lattice_help["center_ky"])

    # Hologram optimization knobs.
    parser.add_argument("--holo-method", default="WGS-Kim")
    parser.add_argument("--holo-maxiter", type=int, default=30)

    # Camera/feedback knobs.
    parser.add_argument("--use-camera", action="store_true", help="Enable Andor full-frame acquisition")
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--exposure-s", type=float, default=0.03)
    parser.add_argument("--frames", type=int, default=1, help="Number of full frames to acquire")
    parser.add_argument("--feedback", action="store_true", help="Run experimental feedback optimization")
    parser.add_argument("--feedback-iters", type=int, default=10)
    parser.add_argument("--calibration-root", default="user_workflows/calibrations")
    parser.add_argument("--save-frames", default="", help="Optional .npy output path for acquired frames")

    args = parser.parse_args()

    try:
        pattern_params_from_flat_dict(args.pattern, vars(args))
    except PatternValidationError as exc:
        parser.error(str(exc))

    slm = Holoeye(preselect="index:0")
    deep = load_phase_lut(Path(args.lut_file), args.lut_key)
    pattern = build_pattern(args, slm, deep)
    slm.set_phase(pattern, settle=True)
    print(f"Pattern '{args.pattern}' displayed on SLM")

    if not args.use_camera:
        hold_until_interrupt(slm)
        return

    calibration_paths = assert_required_calibration_files(args.calibration_root)

    cam = AndorIDus(
        serial=args.camera_serial,
        target_temperature_c=-65,
        shutter_mode="auto",
        verbose=True,
    )
    cam.set_exposure(args.exposure_s)

    fs = FourierSLM(cam, slm)
    fs.load_calibration("fourier", str(calibration_paths["fourier"]))
    fs.load_calibration("wavefront_superpixel", str(calibration_paths["wavefront_superpixel"]))
    fs.slm.source["amplitude"] = np.load(calibration_paths["source_amplitude"])

    if args.feedback:
        run_feedback(fs, iterations=args.feedback_iters)

    frames = [cam.get_image() for _ in range(max(1, args.frames))]
    frames = np.asarray(frames)
    print(f"Acquired {frames.shape[0]} Andor full-frame image(s): shape={frames.shape[1:]}")

    if args.save_frames:
        out = Path(args.save_frames)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, frames)
        print(f"Saved frames to {out.resolve()}")

    hold_until_interrupt(slm, cam=cam)


if __name__ == "__main__":
    main()
