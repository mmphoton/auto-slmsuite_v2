"""Experimental feedback optimization command."""

from __future__ import annotations

import argparse
from pathlib import Path

from user_workflows.commands.acquire import create_fourier_slm
from user_workflows.commands.pattern import build_pattern, hold_until_interrupt, load_phase_lut


def add_feedback_args(parser: argparse.ArgumentParser):
    parser.add_argument("--feedback-iters", type=int, default=10)


def run_feedback_optimization(fs, iterations: int):
    from slmsuite.holography.algorithms import FeedbackHologram

    img = fs.cam.get_image()
    target_ij = img.astype(float)
    peak = target_ij.max()
    target_ij /= peak if peak > 0 else 1.0

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
        run_feedback_optimization(fs, iterations=args.feedback_iters)
        print(f"Feedback optimization completed ({args.feedback_iters} iterations).")
        hold_until_interrupt(fs.slm)
    finally:
        fs.cam.close()
        fs.slm.close()
