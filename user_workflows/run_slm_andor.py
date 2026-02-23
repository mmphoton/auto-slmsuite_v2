"""Backward-compatible wrapper around the new workflow CLI commands."""

from __future__ import annotations

import argparse
import pprint
import sys
import time
import warnings

import numpy as np
import scipy.io
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.holoeye import Holoeye
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram
from slmsuite.holography.toolbox import phase
from slmsuite.holography.toolbox.phase import blaze

from user_workflows.andor_camera import AndorConnectionConfig, PylablibAndorCamera
from user_workflows.calibration_io import assert_required_calibration_files
from user_workflows.config import dump_yaml, load_workflow_config
from user_workflows.config.schema import WorkflowConfig


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


def build_pattern(config: WorkflowConfig, slm, deep):
    """Build one of several user-selectable analytical pattern families."""
    params = config.pattern.family_params

    if config.pattern.pattern_type == "laguerre-gaussian":
        lg_phase = phase.laguerre_gaussian(slm, l=params["lg_l"], p=params["lg_p"])
        phi = np.mod(lg_phase + blaze(grid=slm, vector=(params["blaze_kx"], params["blaze_ky"])), 2 * np.pi)
        return _depth_correct(phi, deep)

    shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)

    if config.pattern.pattern_type == "single-gaussian":
        spot_kxy = np.array([[params["single_kx"]], [params["single_ky"]]])
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=slm)
    elif config.pattern.pattern_type == "double-gaussian":
        dx = float(params["double_sep_kxy"]) / 2.0
        spot_kxy = np.array(
            [
                [params["double_center_kx"] - dx, params["double_center_kx"] + dx],
                [params["double_center_ky"], params["double_center_ky"]],
            ]
        )
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=slm)
    elif config.pattern.pattern_type == "gaussian-lattice":
        hologram = SpotHologram.make_rectangular_array(
            shape,
            array_shape=(params["lattice_nx"], params["lattice_ny"]),
            array_pitch=(params["lattice_pitch_x"], params["lattice_pitch_y"]),
            array_center=(params["lattice_center_kx"], params["lattice_center_ky"]),
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
        raise ValueError(f"Unknown pattern '{config.pattern.pattern_type}'")

    hologram.optimize(
        method=params["holo_method"],
        maxiter=params["holo_maxiter"],
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


def run_feedback(fs: FourierSLM, method: str, iterations: int):
    img = fs.cam.get_image()
    target_ij = img.astype(float)
    peak = np.nanmax(target_ij)
    target_ij /= peak if peak > 0 else 1.0

    holo = FeedbackHologram(shape=fs.slm.shape, target_ij=target_ij, cameraslm=fs)
    holo.optimize(
        method=method,
        feedback="experimental",
        maxiter=int(iterations),
        stat_groups=["experimental_ij", "computational"],
    )
    fs.slm.set_phase(holo.get_phase(include_propagation=True), settle=True)


def _apply_legacy_overrides(args, config: WorkflowConfig) -> WorkflowConfig:
    legacy_map = {
        "pattern": "pattern.pattern_type",
        "lut_file": "lut_file",
        "lut_key": "lut_key",
        "blaze_kx": "pattern.family_params.blaze_kx",
        "blaze_ky": "pattern.family_params.blaze_ky",
        "lg_l": "pattern.family_params.lg_l",
        "lg_p": "pattern.family_params.lg_p",
        "single_kx": "pattern.family_params.single_kx",
        "single_ky": "pattern.family_params.single_ky",
        "double_center_kx": "pattern.family_params.double_center_kx",
        "double_center_ky": "pattern.family_params.double_center_ky",
        "double_sep_kxy": "pattern.family_params.double_sep_kxy",
        "lattice_nx": "pattern.family_params.lattice_nx",
        "lattice_ny": "pattern.family_params.lattice_ny",
        "lattice_pitch_x": "pattern.family_params.lattice_pitch_x",
        "lattice_pitch_y": "pattern.family_params.lattice_pitch_y",
        "lattice_center_kx": "pattern.family_params.lattice_center_kx",
        "lattice_center_ky": "pattern.family_params.lattice_center_ky",
        "holo_method": "pattern.family_params.holo_method",
        "holo_maxiter": "pattern.family_params.holo_maxiter",
        "use_camera": "hardware.use_camera",
        "camera_serial": "hardware.camera_serial",
        "exposure_s": "hardware.exposure_s",
        "frames": "hardware.frames",
        "feedback": "feedback.enabled",
        "feedback_iters": "feedback.maxiter",
        "calibration_root": "calibration.paths.root",
        "save_frames": "output.save_options.frames_path",
    }

    raw = config.to_dict()
    for arg_name, config_path in legacy_map.items():
        value = getattr(args, arg_name)
        if value is None:
            continue
        warnings.warn(
            f"CLI flag '--{arg_name.replace('_', '-')}' is deprecated; use --config/--set ({config_path}) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        cursor = raw
        parts = config_path.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value

    if raw["output"]["save_options"].get("frames_path"):
        raw["output"]["save_options"]["save_frames"] = True

    updated = WorkflowConfig(
        lut_file=raw["lut_file"],
        lut_key=raw["lut_key"],
        hardware=type(config.hardware)(**raw["hardware"]),
        pattern=type(config.pattern)(**raw["pattern"]),
        feedback=type(config.feedback)(**raw["feedback"]),
        output=type(config.output)(**raw["output"]),
        calibration=type(config.calibration)(**raw["calibration"]),
    )
    updated.validate()
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--set", action="append", default=[], help="Override config values via dotted key=value")

    parser.add_argument(
        "--pattern",
        default=None,
        choices=["single-gaussian", "double-gaussian", "gaussian-lattice", "laguerre-gaussian"],
        help="Pattern family to generate on SLM",
    )
    parser.add_argument("--lut-file", default=None)
    parser.add_argument("--lut-key", default=None)
    parser.add_argument("--blaze-kx", type=float, default=None)
    parser.add_argument("--blaze-ky", type=float, default=None)
    parser.add_argument("--lg-l", type=int, default=None)
    parser.add_argument("--lg-p", type=int, default=None)
    parser.add_argument("--single-kx", type=float, default=None)
    parser.add_argument("--single-ky", type=float, default=None)
    parser.add_argument("--double-center-kx", type=float, default=None)
    parser.add_argument("--double-center-ky", type=float, default=None)
    parser.add_argument("--double-sep-kxy", type=float, default=None, help="Separation in kxy units")
    parser.add_argument("--lattice-nx", type=int, default=None)
    parser.add_argument("--lattice-ny", type=int, default=None)
    parser.add_argument("--lattice-pitch-x", type=float, default=None)
    parser.add_argument("--lattice-pitch-y", type=float, default=None)
    parser.add_argument("--lattice-center-kx", type=float, default=None)
    parser.add_argument("--lattice-center-ky", type=float, default=None)
    parser.add_argument("--holo-method", default=None)
    parser.add_argument("--holo-maxiter", type=int, default=None)

    parser.add_argument("--use-camera", dest="use_camera", action="store_const", const=True, default=None)
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--exposure-s", type=float, default=None)
    parser.add_argument("--frames", type=int, default=None, help="Number of full frames to acquire")
    parser.add_argument("--feedback", dest="feedback", action="store_const", const=True, default=None)
    parser.add_argument("--feedback-iters", type=int, default=None)
    parser.add_argument("--calibration-root", default=None)
    parser.add_argument("--save-frames", default=None, help="Optional .npy output path for acquired frames")
    args = parser.parse_args()

    config = load_workflow_config(config_path=args.config, overrides=args.set)
    config = _apply_legacy_overrides(args, config)

    run_dir = config.output_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "resolved_config.yaml"
    dump_yaml(config.to_dict(), resolved_path)

    print("Resolved config:")
    pprint.pprint(config.to_dict(), sort_dicts=False)
    print(f"Saved resolved config: {resolved_path.resolve()}")

    slm = Holoeye(preselect=config.hardware.slm_preselect)
    deep = load_phase_lut(Path(config.lut_file), config.lut_key)
    pattern = build_pattern(config, slm, deep)
    slm.set_phase(pattern, settle=True)
    print(f"Pattern '{config.pattern.pattern_type}' displayed on SLM")

    if not config.hardware.use_camera:
        hold_until_interrupt(slm)
        return

    calibration_paths = assert_required_calibration_files(config.calibration.paths["root"])

    cam = PylablibAndorCamera(
        AndorConnectionConfig(
            camera_serial=config.hardware.camera_serial,
            exposure_s=config.hardware.exposure_s,
            shutter_mode=config.hardware.shutter_mode,
        ),
        verbose=True,
    )

    fs = FourierSLM(cam, slm)
    fs.load_calibration("fourier", str(calibration_paths["fourier"]))
    fs.load_calibration("wavefront_superpixel", str(calibration_paths["wavefront_superpixel"]))
    fs.slm.source["amplitude"] = np.load(calibration_paths["source_amplitude"])

    if config.feedback.enabled:
        run_feedback(fs, method=config.feedback.method, iterations=config.feedback.maxiter)

    frames = [cam.get_image() for _ in range(max(1, config.hardware.frames))]
    frames = np.asarray(frames)
    print(f"Acquired {frames.shape[0]} Andor full-frame image(s): shape={frames.shape[1:]}")

    if config.output.save_options.get("save_frames"):
        rel = config.output.save_options.get("frames_path") or "frames.npy"
        out = run_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, frames)
        output.register_file(out, "legacy_frame_stack")
        print(f"Saved frames to {out.resolve()}")

    hold_until_interrupt(slm, cam=cam)


if __name__ == "__main__":
    main()
