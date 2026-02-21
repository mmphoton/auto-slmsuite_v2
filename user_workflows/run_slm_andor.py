"""Generate SLM patterns with optional Andor iDus full-frame acquisition and feedback optimization."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io

from slmsuite.hardware.cameras.andor_idus import AndorIDus
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.holoeye import Holoeye
from slmsuite.holography.algorithms import FeedbackHologram
from slmsuite.holography.toolbox.phase import blaze
from slmsuite.holography.toolbox import phase

from user_workflows.calibration_io import assert_required_calibration_files


def load_phase_lut(path: Path, key: str = "deep") -> np.ndarray:
    mat = scipy.io.loadmat(path)
    if key not in mat:
        raise ValueError(f"LUT file must contain variable '{key}'")
    deep = np.asarray(mat[key]).squeeze()
    if deep.ndim != 1:
        raise ValueError(f"Expected 1D LUT, got {deep.shape}")
    return deep


def build_pattern(slm, deep, sigma, blaze_vector):
    ny, nx = slm.shape
    x = np.linspace(-nx / 2, nx / 2, nx)
    y = np.linspace(-ny / 2, ny / 2, ny)
    X, Y = np.meshgrid(x, y)

    target_amp = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    indices = np.clip(np.rint(target_amp * (deep.size - 1)), 0, deep.size - 1).astype(int)
    F = deep[indices]

    lg30_phase = phase.laguerre_gaussian(slm, l=3, p=0)
    blaze_phase = blaze(grid=slm, vector=blaze_vector)

    phi = np.mod(lg30_phase + blaze_phase, 2 * np.pi)
    corrected = (phi - np.pi) * F + np.pi
    return np.mod(corrected, 2 * np.pi)


def hold_until_interrupt(slm):
    print("Holding SLM phase. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        slm.set_phase(None, settle=True)
        slm.close()


def run_feedback(fs: FourierSLM, iterations: int):
    img = fs.cam.get_image()
    target_ij = img.astype(float)
    target_ij /= np.nanmax(target_ij) if np.nanmax(target_ij) > 0 else 1.0

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
    parser.add_argument("--lut-file", default="deep_1024.mat")
    parser.add_argument("--lut-key", default="deep")
    parser.add_argument("--sigma", type=float, default=1.0e8)
    parser.add_argument("--blaze-kx", type=float, default=0.0)
    parser.add_argument("--blaze-ky", type=float, default=0.0045)
    parser.add_argument("--use-camera", action="store_true", help="Enable Andor full-frame acquisition")
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--exposure-s", type=float, default=0.03)
    parser.add_argument("--frames", type=int, default=1, help="Number of full frames to acquire")
    parser.add_argument("--feedback", action="store_true", help="Run experimental feedback optimization")
    parser.add_argument("--feedback-iters", type=int, default=10)
    parser.add_argument("--calibration-root", default="user_workflows/calibrations")
    parser.add_argument("--save-frames", default="", help="Optional .npy output path for acquired frames")
    args = parser.parse_args()

    slm = Holoeye(preselect="index:0")
    deep = load_phase_lut(Path(args.lut_file), args.lut_key)
    pattern = build_pattern(slm, deep, args.sigma, (args.blaze_kx, args.blaze_ky))
    slm.set_phase(pattern, settle=True)
    print("Pattern displayed on SLM")

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

    frames = []
    for _ in range(max(1, args.frames)):
        frames.append(cam.get_image())
    frames = np.asarray(frames)
    print(f"Acquired {frames.shape[0]} Andor full-frame image(s): shape={frames.shape[1:]}")

    if args.save_frames:
        out = Path(args.save_frames)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, frames)
        print(f"Saved frames to {out.resolve()}")

    hold_until_interrupt(slm)


if __name__ == "__main__":
    main()
