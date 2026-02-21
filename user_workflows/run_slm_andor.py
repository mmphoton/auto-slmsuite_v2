"""Generate configurable SLM analytical patterns with optional Andor iDus acquisition/feedback."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import scipy.io

from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram
from slmsuite.holography.toolbox import phase
from slmsuite.holography.toolbox.phase import blaze

from user_workflows.calibration_io import assert_required_calibration_files
from user_workflows.simulation.sim_factory import build_simulated_fourier_slm


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
    f_lut = deep[idx]
    corrected = (phi - np.pi) * f_lut + np.pi
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
        spot_amp = np.array([0.7, 0.3]) if args.simulate and args.simulation_scenario == "two-spot-imbalance" else None
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=slm, spot_amp=spot_amp)
    elif args.pattern == "gaussian-lattice":
        hologram = SpotHologram.make_rectangular_array(
            shape,
            array_shape=(args.lattice_nx, args.lattice_ny),
            array_pitch=(args.lattice_pitch_x, args.lattice_pitch_y),
            array_center=(args.lattice_center_kx, args.lattice_center_ky),
            basis="kxy",
            cameraslm=slm,
        )
        if args.simulate and args.simulation_scenario == "n-spot-lattice-nonuniform":
            n = args.lattice_nx * args.lattice_ny
            lattice_amp = np.linspace(1.25, 0.75, n)
            lattice_amp /= np.linalg.norm(lattice_amp)
            hologram.set_target(new_target=lattice_amp, reset_weights=True)
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


def _save_checkpoint(checkpoint_dir: Path, iteration: int, phase_data: np.ndarray, frame: np.ndarray, save_farfield: bool):
    np.save(checkpoint_dir / f"phase_iter_{iteration:04d}.npy", phase_data)
    np.save(checkpoint_dir / f"cam_iter_{iteration:04d}.npy", frame)
    if save_farfield:
        farfield = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phase_data))))
        np.save(checkpoint_dir / f"farfield_iter_{iteration:04d}.npy", farfield)


def _print_summary_table(metrics):
    if not metrics:
        print("No telemetry metrics recorded.")
        return
    best = max(metrics, key=lambda m: m.uniformity_min_max)
    final = metrics[-1]
    print("\nFeedback Summary")
    print("iter | uniformity(min/max) | cv | objective | elapsed(s)")
    print("-----+----------------------+----+-----------+----------")
    print(
        f"best {best.iteration:4d} | {best.uniformity_min_max:20.6f} | {best.coefficient_of_variation:0.6f}"
        f" | {best.objective_value:0.6f} | {best.elapsed_time_s:0.2f}"
    )
    print(
        f"final{final.iteration:4d} | {final.uniformity_min_max:20.6f} | {final.coefficient_of_variation:0.6f}"
        f" | {final.objective_value:0.6f} | {final.elapsed_time_s:0.2f}"
    )


def run_feedback(fs: FourierSLM, args, run_dir: Path):
    logger = logging.getLogger("feedback")
    telemetry = FeedbackTelemetry(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stop_cfg = StopConfig(
        target_uniformity=args.target_uniformity,
        max_no_improvement_iters=args.max_no_improvement_iters,
        min_relative_improvement=args.min_relative_improvement,
        max_runtime_s=args.max_runtime_s,
    )

    img = fs.cam.get_image()
    target_ij = img.astype(float)
    peak = np.nanmax(target_ij)
    target_ij /= peak if peak > 0 else 1.0

    holo = FeedbackHologram(shape=fs.slm.shape, target_ij=target_ij, cameraslm=fs)

    best_uniformity = -np.inf
    no_improvement_iters = 0
    start_time = time.time()

    for iteration in range(1, int(args.feedback_iters) + 1):
        holo.optimize(
            method="WGS-Kim",
            feedback="experimental",
            maxiter=1,
            stat_groups=["experimental_ij", "computational"],
        )
        phase_data = holo.get_phase(include_propagation=True)
        fs.slm.set_phase(phase_data, settle=True)
        frame = fs.cam.get_image()

        spots = extract_spot_intensities(frame)
        metric = metric_from_spots(iteration=iteration, spot_values=spots, elapsed_time_s=time.time() - start_time)
        telemetry.record(metric)

        rel_improvement = (
            (metric.uniformity_min_max - best_uniformity) / max(abs(best_uniformity), 1e-12)
            if np.isfinite(best_uniformity)
            else np.inf
        )

        if rel_improvement >= stop_cfg.min_relative_improvement:
            best_uniformity = metric.uniformity_min_max
            no_improvement_iters = 0
            logger.info(
                "iter=%s improvement accepted, uniformity=%.6f rel_improvement=%.6f",
                iteration,
                metric.uniformity_min_max,
                rel_improvement,
            )
        else:
            no_improvement_iters += 1
            logger.warning(
                "iter=%s no significant improvement, uniformity=%.6f rel_improvement=%.6f no_improvement_iters=%s",
                iteration,
                metric.uniformity_min_max,
                rel_improvement,
                no_improvement_iters,
            )

        _save_checkpoint(checkpoint_dir, iteration, phase_data, frame, args.save_farfield)

        stop, reason = should_stop(
            telemetry.metrics,
            stop_cfg=stop_cfg,
            no_improvement_iters=no_improvement_iters,
            start_time_s=start_time,
            now_s=time.time(),
        )
        if stop:
            logger.info("Stopping early at iter=%s due to %s", iteration, reason)
            break

    csv_path, json_path = telemetry.save()
    logger.info("Saved telemetry metrics to %s and %s", csv_path, json_path)
    _print_summary_table(telemetry.metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default="laguerre-gaussian",
        choices=["single-gaussian", "double-gaussian", "gaussian-lattice", "laguerre-gaussian"],
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
    parser.add_argument("--double-sep-kxy", type=float, default=0.02)
    parser.add_argument("--lattice-nx", type=int, default=5)
    parser.add_argument("--lattice-ny", type=int, default=5)
    parser.add_argument("--lattice-pitch-x", type=float, default=0.01)
    parser.add_argument("--lattice-pitch-y", type=float, default=0.01)
    parser.add_argument("--lattice-center-kx", type=float, default=0.0)
    parser.add_argument("--lattice-center-ky", type=float, default=0.0)
    parser.add_argument("--holo-method", default="WGS-Kim")
    parser.add_argument("--holo-maxiter", type=int, default=30)
    parser.add_argument("--use-camera", action="store_true")
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--exposure-s", type=float, default=0.03)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--feedback", action="store_true")
    parser.add_argument("--feedback-iters", type=int, default=10)
    parser.add_argument("--target-uniformity", type=float, default=0.95)
    parser.add_argument("--max-no-improvement-iters", type=int, default=5)
    parser.add_argument("--min-relative-improvement", type=float, default=1e-3)
    parser.add_argument("--max-runtime-s", type=float, default=120.0)
    parser.add_argument("--run-output-dir", default="feedback_runs/latest")
    parser.add_argument("--save-farfield", action="store_true")
    parser.add_argument("--calibration-root", default="user_workflows/calibrations")
    parser.add_argument("--save-frames", default="", help="Optional .npy output path")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--simulation-scenario",
        default="two-spot-imbalance",
        choices=["two-spot-imbalance", "n-spot-lattice-nonuniform", "high-noise-failure"],
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    deep = load_phase_lut(Path(args.lut_file), args.lut_key)

    if args.simulate:
        fs = build_simulated_fourier_slm(seed=args.seed, scenario=args.simulation_scenario)
        fs.cam.set_exposure(args.exposure_s)
    else:
        from slmsuite.hardware.slms.holoeye import Holoeye
        from slmsuite.hardware.cameras.andor_idus import AndorIDus

        slm = Holoeye(preselect="index:0")
        if not args.use_camera:
            pattern = build_pattern(args, slm, deep)
            slm.set_phase(pattern, settle=True)
            print(f"Pattern '{args.pattern}' displayed on SLM")
            hold_until_interrupt(slm)
            return

        calibration_paths = assert_required_calibration_files(args.calibration_root)
        cam = AndorIDus(serial=args.camera_serial, target_temperature_c=-65, shutter_mode="auto", verbose=True)
        cam.set_exposure(args.exposure_s)
        fs = FourierSLM(cam, slm)
        fs.load_calibration("fourier", str(calibration_paths["fourier"]))
        fs.load_calibration("wavefront_superpixel", str(calibration_paths["wavefront_superpixel"]))
        fs.slm.source["amplitude"] = np.load(calibration_paths["source_amplitude"])

    pattern = build_pattern(args, fs.slm, deep)
    fs.slm.set_phase(pattern, settle=True)
    print(f"Pattern '{args.pattern}' displayed on SLM")

    if args.feedback:
        logger.info("Starting feedback loop")
        run_feedback(fs, args=args, run_dir=run_dir)

    frames = np.asarray([fs.cam.get_image() for _ in range(max(1, args.frames))])
    print(f"Acquired {frames.shape[0]} full-frame image(s): shape={frames.shape[1:]}")

    out = Path(args.save_frames) if args.save_frames else Path(args.calibration_root) / "andor_frames.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, frames)
    print(f"Saved frames to {out.resolve()}")

    hold_until_interrupt(fs.slm, cam=fs.cam)


if __name__ == "__main__":
    main()
