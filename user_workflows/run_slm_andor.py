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
from slmsuite.holography.toolbox import phase
from slmsuite.holography.toolbox.phase import blaze

from user_workflows.calibration_io import assert_required_calibration_files
from user_workflows.io.output_manager import OutputManager
from user_workflows.io.run_naming import add_naming_args, config_from_args


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
    if args.pattern == "laguerre-gaussian":
        lg_phase = phase.laguerre_gaussian(slm, l=args.lg_l, p=args.lg_p)
        phi = np.mod(lg_phase + blaze(grid=slm, vector=(args.blaze_kx, args.blaze_ky)), 2 * np.pi)
        return _depth_correct(phi, deep)

    shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)

    if args.pattern == "single-gaussian":
        spot_kxy = np.array([[args.single_kx], [args.single_ky]])
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=slm)
    elif args.pattern == "double-gaussian":
        dx = float(args.double_sep_kxy) / 2.0
        spot_kxy = np.array(
            [
                [args.double_center_kx - dx, args.double_center_kx + dx],
                [args.double_center_ky, args.double_center_ky],
            ]
        )
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=slm)
    elif args.pattern == "gaussian-lattice":
        hologram = SpotHologram.make_rectangular_array(
            shape,
            array_shape=(args.lattice_nx, args.lattice_ny),
            array_pitch=(args.lattice_pitch_x, args.lattice_pitch_y),
            array_center=(args.lattice_center_kx, args.lattice_center_ky),
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


def run_feedback(fs: FourierSLM, iterations: int) -> dict[str, float]:
    img = fs.cam.get_image()
    target_ij = img.astype(float)
    peak = float(np.nanmax(target_ij))
    target_ij /= peak if peak > 0 else 1.0

    holo = FeedbackHologram(shape=fs.slm.shape, target_ij=target_ij, cameraslm=fs)
    holo.optimize(
        method="WGS-Kim",
        feedback="experimental",
        maxiter=int(iterations),
        stat_groups=["experimental_ij", "computational"],
    )
    fs.slm.set_phase(holo.get_phase(include_propagation=True), settle=True)
    return {"feedback_iters": int(iterations), "target_peak": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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

    parser.add_argument("--lg-l", type=int, default=3)
    parser.add_argument("--lg-p", type=int, default=0)
    parser.add_argument("--single-kx", type=float, default=0.0)
    parser.add_argument("--single-ky", type=float, default=0.0)
    parser.add_argument("--double-center-kx", type=float, default=0.0)
    parser.add_argument("--double-center-ky", type=float, default=0.0)
    parser.add_argument("--double-sep-kxy", type=float, default=0.02, help="Separation in kxy units")
    parser.add_argument("--lattice-nx", type=int, default=5)
    parser.add_argument("--lattice-ny", type=int, default=5)
    parser.add_argument("--lattice-pitch-x", type=float, default=0.01)
    parser.add_argument("--lattice-pitch-y", type=float, default=0.01)
    parser.add_argument("--lattice-center-kx", type=float, default=0.0)
    parser.add_argument("--lattice-center-ky", type=float, default=0.0)
    parser.add_argument("--holo-method", default="WGS-Kim")
    parser.add_argument("--holo-maxiter", type=int, default=30)

    parser.add_argument("--use-camera", action="store_true", help="Enable Andor full-frame acquisition")
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--exposure-s", type=float, default=0.03)
    parser.add_argument("--frames", type=int, default=1, help="Number of full frames to acquire")
    parser.add_argument("--feedback", action="store_true", help="Run experimental feedback optimization")
    parser.add_argument("--feedback-iters", type=int, default=10)
    parser.add_argument("--calibration-root", default="user_workflows/calibrations")
    parser.add_argument("--save-frames", default="", help="Optional legacy .npy output path for acquired frames")
    add_naming_args(parser)

    args = parser.parse_args()

    camera_mode = "andor" if args.use_camera else "none"
    output = OutputManager(
        config_from_args(args),
        pattern=args.pattern,
        camera=camera_mode,
        metadata={"workflow": "run_slm_andor"},
    )

    slm = Holoeye(preselect="index:0")
    deep = load_phase_lut(Path(args.lut_file), args.lut_key)
    pattern = build_pattern(args, slm, deep)
    output.save_phase(pattern)
    slm.set_phase(pattern, settle=True)
    print(f"Pattern '{args.pattern}' displayed on SLM")

    if not args.use_camera:
        output.save_metrics({"mode": "slm_only", "pattern": args.pattern})
        output.save_manifest()
        print(f"Run directory: {output.run_dir.resolve()}")
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

    feedback_metrics = {}
    if args.feedback:
        feedback_metrics = run_feedback(fs, iterations=args.feedback_iters)

    frames = [cam.get_image() for _ in range(max(1, args.frames))]
    frames = np.asarray(frames)
    print(f"Acquired {frames.shape[0]} Andor full-frame image(s): shape={frames.shape[1:]}")

    for i, frame in enumerate(frames):
        output.save_frame(frame, index=i)
    output.save_plot(frames[0], filename="first_frame.png")

    metrics = {
        "mode": "camera",
        "frames": int(frames.shape[0]),
        "frame_shape": list(frames.shape[1:]),
        "exposure_s": float(args.exposure_s),
        "feedback": bool(args.feedback),
        **feedback_metrics,
    }
    output.save_metrics(metrics)

    if args.save_frames:
        out = Path(args.save_frames)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, frames)
        output.register_file(out, "legacy_frame_stack")
        print(f"Saved frames to {out.resolve()}")

    output.save_manifest({"calibration_root": str(Path(args.calibration_root).resolve())})
    print(f"Run directory: {output.run_dir.resolve()}")

    hold_until_interrupt(slm, cam=cam)


if __name__ == "__main__":
    main()
