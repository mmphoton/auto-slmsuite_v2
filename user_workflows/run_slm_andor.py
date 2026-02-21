"""Generate configurable SLM analytical patterns with optional Andor iDus acquisition/feedback."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from slmsuite.hardware.cameras.andor_idus import AndorIDus
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.holoeye import Holoeye
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram
from slmsuite.holography.toolbox.phase import blaze
from slmsuite.holography.toolbox import phase

from user_workflows.calibration_io import assert_required_calibration_files


@dataclass
class PatternResult:
    """Container for generated SLM pattern data and debug artifacts."""

    phase: np.ndarray
    expected_farfield: np.ndarray | None
    target_descriptor: dict[str, Any]
    artifacts: dict[str, np.ndarray]


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


def build_pattern(args, slm, deep) -> PatternResult:
    """Build one of several user-selectable analytical pattern families."""
    ny, nx = slm.shape
    descriptor: dict[str, Any]
    artifacts: dict[str, np.ndarray] = {}

    if args.pattern == "laguerre-gaussian":
        lg_phase = phase.laguerre_gaussian(slm, l=args.lg_l, p=args.lg_p)
        blaze_phase = blaze(grid=slm, vector=(args.blaze_kx, args.blaze_ky))
        phi = np.mod(lg_phase + blaze_phase, 2 * np.pi)
        corrected_phase = _depth_correct(phi, deep)

        descriptor = {
            "type": "laguerre-gaussian",
            "centers": [],
            "radii": [],
            "spots": [],
            "lg_l": int(args.lg_l),
            "lg_p": int(args.lg_p),
            "blaze_kxy": [float(args.blaze_kx), float(args.blaze_ky)],
        }
        artifacts.update(
            {
                "laguerre_phase": np.asarray(lg_phase),
                "blaze_phase": np.asarray(blaze_phase),
                "wrapped_phase": np.asarray(phi),
            }
        )
        return PatternResult(
            phase=corrected_phase,
            expected_farfield=None,
            target_descriptor=descriptor,
            artifacts=artifacts,
        )

    # Spot-based patterns use hologram optimization so users get Gaussian-like focused spots.
    shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)

    if args.pattern == "single-gaussian":
        spot_kxy = np.array([[args.single_kx], [args.single_ky]])
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=slm)
        descriptor = {
            "type": "single-gaussian",
            "spots": spot_kxy.T.tolist(),
            "centers": spot_kxy.T.tolist(),
            "radii": [],
        }
    elif args.pattern == "double-gaussian":
        dx = float(args.double_sep_kxy) / 2.0
        spot_kxy = np.array(
            [
                [args.double_center_kx - dx, args.double_center_kx + dx],
                [args.double_center_ky, args.double_center_ky],
            ]
        )
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=slm)
        descriptor = {
            "type": "double-gaussian",
            "spots": spot_kxy.T.tolist(),
            "centers": spot_kxy.T.tolist(),
            "radii": [],
        }
    elif args.pattern == "gaussian-lattice":
        hologram = SpotHologram.make_rectangular_array(
            shape,
            array_shape=(args.lattice_nx, args.lattice_ny),
            array_pitch=(args.lattice_pitch_x, args.lattice_pitch_y),
            array_center=(args.lattice_center_kx, args.lattice_center_ky),
            basis="kxy",
            cameraslm=slm,
        )
        descriptor = {
            "type": "gaussian-lattice",
            "spots": np.asarray(hologram.spot_kxy).T.tolist(),
            "centers": [
                [float(args.lattice_center_kx), float(args.lattice_center_ky)],
            ],
            "radii": [
                [float(args.lattice_pitch_x), float(args.lattice_pitch_y)],
            ],
        }
    else:
        raise ValueError(f"Unknown pattern '{args.pattern}'")

    hologram.optimize(
        method=args.holo_method,
        maxiter=args.holo_maxiter,
        feedback="computational",
        stat_groups=["computational"],
    )
    phi = np.mod(hologram.get_phase(), 2 * np.pi)
    corrected_phase = _depth_correct(phi, deep)

    expected_farfield = None
    if hasattr(hologram, "extract_farfield"):
        expected_farfield = np.asarray(hologram.extract_farfield())

    artifacts.update(
        {
            "spot_kxy": np.asarray(hologram.spot_kxy),
            "wrapped_phase": np.asarray(phi),
        }
    )

    return PatternResult(
        phase=corrected_phase,
        expected_farfield=expected_farfield,
        target_descriptor=descriptor,
        artifacts=artifacts,
    )


def _save_pattern_result(pattern_result: PatternResult, output_root: Path, pattern_name: str):
    """Persist pattern metadata and optional debug arrays under stable filenames."""
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / f"{pattern_name}-phase.npy", pattern_result.phase)

    if pattern_result.expected_farfield is not None:
        np.save(output_root / f"{pattern_name}-expected-farfield.npy", pattern_result.expected_farfield)

    descriptor = pattern_result.target_descriptor
    if not {"spots", "centers", "radii"}.issubset(descriptor.keys()):
        raise ValueError("target_descriptor must include spots/centers/radii entries.")
    np.savez(
        output_root / f"{pattern_name}-target-descriptor.npz",
        spots=np.asarray(descriptor["spots"], dtype=float),
        centers=np.asarray(descriptor["centers"], dtype=float),
        radii=np.asarray(descriptor["radii"], dtype=float),
    )

    for artifact_name, artifact_data in pattern_result.artifacts.items():
        if artifact_data is None:
            continue
        np.save(output_root / f"{pattern_name}-artifact-{artifact_name}.npy", np.asarray(artifact_data))


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
    parser.add_argument("--lg-l", type=int, default=3)
    parser.add_argument("--lg-p", type=int, default=0)

    # Single gaussian spot params.
    parser.add_argument("--single-kx", type=float, default=0.0)
    parser.add_argument("--single-ky", type=float, default=0.0)

    # Two gaussian spot params.
    parser.add_argument("--double-center-kx", type=float, default=0.0)
    parser.add_argument("--double-center-ky", type=float, default=0.0)
    parser.add_argument("--double-sep-kxy", type=float, default=0.02, help="Separation in kxy units")

    # Lattice params.
    parser.add_argument("--lattice-nx", type=int, default=5)
    parser.add_argument("--lattice-ny", type=int, default=5)
    parser.add_argument("--lattice-pitch-x", type=float, default=0.01)
    parser.add_argument("--lattice-pitch-y", type=float, default=0.01)
    parser.add_argument("--lattice-center-kx", type=float, default=0.0)
    parser.add_argument("--lattice-center-ky", type=float, default=0.0)

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
    parser.add_argument(
        "--save-pattern-root",
        default="",
        help="Optional directory for persisting generated pattern phase/descriptor/artifacts",
    )

    args = parser.parse_args()

    slm = Holoeye(preselect="index:0")
    deep = load_phase_lut(Path(args.lut_file), args.lut_key)
    pattern_result = build_pattern(args, slm, deep)
    slm.set_phase(pattern_result.phase, settle=True)
    print(f"Pattern '{args.pattern}' displayed on SLM")

    if args.save_pattern_root:
        save_root = Path(args.save_pattern_root)
        _save_pattern_result(pattern_result, save_root, args.pattern)
        print(f"Saved pattern artifacts to {save_root.resolve()}")

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
