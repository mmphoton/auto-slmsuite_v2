"""Backward-compatible wrapper around the new workflow CLI commands."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slmsuite.hardware.cameras.andor_idus import AndorIDus
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.holoeye import Holoeye
from slmsuite.holography.algorithms import FeedbackHologram

from user_workflows.calibration_io import assert_required_calibration_files
from user_workflows.patterns import get_pattern
from user_workflows.patterns.depth_correction import apply_depth_correction


def load_phase_lut(path: Path, key: str = "deep") -> np.ndarray:
    mat = scipy.io.loadmat(path)
    if key not in mat:
        raise ValueError(f"LUT file must contain variable '{key}'")
    deep = np.asarray(mat[key]).squeeze()
    if deep.ndim != 1:
        raise ValueError(f"Expected 1D LUT, got {deep.shape}")
    return deep


def build_pattern(args, slm, deep):
    """Build one of several user-selectable analytical pattern families."""
    pattern_builder = get_pattern(args.pattern)
    result = pattern_builder.build(args, slm)
    return apply_depth_correction(result.phase, deep), result.metadata


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
    parser.add_argument(
        "--pattern",
        default=None,
        choices=["single-gaussian", "double-gaussian", "gaussian-lattice", "laguerre-gaussian"],
        help="Pattern family to generate on SLM",
    )
    parser.add_argument(
        "--pattern-config",
        default="",
        help="Optional JSON config with {'pattern': <name>, 'params': {..}}",
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

    # Pattern selection + easy parameter knobs.
    parser.add_argument(
        "--pattern",
        default="laguerre-gaussian",
        choices=list_patterns(),
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
    parser.add_argument("--feedback", action="store_true", help="Run experimental feedback optimization")
    parser.add_argument("--feedback-iters", type=int, default=10)
    parser.add_argument("--calibration-root", default="user_workflows/calibrations")
    parser.add_argument("--save-frames", default="", help="Optional legacy .npy output path for acquired frames")
    add_naming_args(parser)

    slm = Holoeye(preselect="index:0")
    deep = load_phase_lut(Path(args.lut_file), args.lut_key)
    pattern, _metadata = build_pattern(args, slm, deep)
    slm.set_phase(pattern, settle=True)
    print(f"Pattern '{args.pattern}' displayed on SLM")

    camera_mode = "andor" if args.use_camera else "none"
    output = OutputManager(
        config_from_args(args),
        pattern=args.pattern,
        camera=camera_mode,
        metadata={"workflow": "run_slm_andor"},
    )

    try:
        pattern_params_from_flat_dict(args.pattern, vars(args))
    except PatternValidationError as exc:
        parser.error(str(exc))

    slm = Holoeye(preselect="index:0")
    deep = load_phase_lut(Path(args.lut_file), args.lut_key)
    pattern_name, result = build_pattern(args, slm, deep)
    slm.set_phase(result.phase, settle=True)
    print(f"Pattern '{pattern_name}' displayed on SLM")
    print(f"Pattern metadata: {result.metadata}")

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
