"""Experimental feedback optimization command."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from user_workflows.commands.acquire import create_fourier_slm
from user_workflows.commands.pattern import build_pattern, hold_until_interrupt, load_phase_lut


def add_feedback_args(parser: argparse.ArgumentParser):
    parser.add_argument("--feedback-iters", type=int, default=10)


def _spot_vectors_for_args(args):
    if args.pattern == "single-gaussian":
        return np.array([[args.single_kx], [args.single_ky]], dtype=float)

    if args.pattern == "double-gaussian":
        dx = float(args.double_sep_kxy) / 2.0
        return np.array(
            [
                [args.double_center_kx - dx, args.double_center_kx + dx],
                [args.double_center_ky, args.double_center_ky],
            ],
            dtype=float,
        )

    if args.pattern == "gaussian-lattice":
        x_offsets = (np.arange(int(args.lattice_nx), dtype=float) - 0.5 * (int(args.lattice_nx) - 1.0)) * float(args.lattice_pitch_x)
        y_offsets = (np.arange(int(args.lattice_ny), dtype=float) - 0.5 * (int(args.lattice_ny) - 1.0)) * float(args.lattice_pitch_y)
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")
        x = xx.ravel() + float(args.lattice_center_kx)
        y = yy.ravel() + float(args.lattice_center_ky)
        return np.vstack((x, y))

    return None


def run_feedback_optimization(fs, args, iterations: int):
    from slmsuite.holography.algorithms import FeedbackHologram, SpotHologram

    spot_vectors = _spot_vectors_for_args(args)
    if spot_vectors is not None:
        holo = SpotHologram(shape=fs.slm.shape, spot_vectors=spot_vectors, basis="kxy", cameraslm=fs)
        holo.optimize(
            method="WGS-Kim",
            feedback="experimental_spot",
            maxiter=int(iterations),
            stat_groups=["experimental_spot", "computational_spot"],
        )
    else:
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

    fs.slm.set_phase(holo.get_phase(include_propagation=True), settle=True)


def run_feedback(args):
    deep = None
    if args.use_phase_depth_correction:
        lut_path = Path(args.lut_file)
        deep = load_phase_lut(lut_path, args.lut_key)

    if args.dry_run:
        print("[dry-run] feedback config validated. Hardware will not be touched.")
        return

    fs = create_fourier_slm(args)
    try:
        fs.slm.set_phase(build_pattern(args, fs.slm, deep), settle=True)
        if "fourier" not in fs.calibrations:
            raise RuntimeError(
                "Missing Fourier calibration required for camera feedback. "
                "Run calibration first or set --calibration-root to a directory containing fourier-calibration.h5."
            )
        run_feedback_optimization(fs, args, iterations=args.feedback_iters)
        print(f"Feedback optimization completed ({args.feedback_iters} iterations).")
        hold_until_interrupt(fs.slm)
    finally:
        fs.cam.close()
        fs.slm.close()
