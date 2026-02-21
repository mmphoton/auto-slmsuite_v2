"""Backward-compatible wrapper around the new workflow CLI commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slmsuite.hardware.cameras.andor_idus import AndorIDus
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.holoeye import Holoeye
from slmsuite.holography.algorithms import FeedbackHologram

from user_workflows.calibration_io import assert_required_calibration_files
from user_workflows.patterns import get_pattern, list_patterns


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


def build_pattern(config, slm, deep):
    """Build a registered user-selectable analytical pattern."""
    pattern_factory = get_pattern(config.pattern.name)
    return pattern_factory(config, slm, deep, _depth_correct)


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
    add_pattern_args(parser)
    add_acquire_args(parser)
    add_feedback_args(parser)

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
    parser.add_argument("--feedback", action="store_true", help="Run experimental feedback optimization")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and file paths without touching hardware")
    return parser

    slm = Holoeye(preselect="index:0")
    deep = load_phase_lut(Path(args.lut_file), args.lut_key)
    config = argparse.Namespace(pattern=argparse.Namespace(name=args.pattern), args=args)
    pattern = build_pattern(config, slm, deep)
    slm.set_phase(pattern, settle=True)
    print(f"Pattern '{args.pattern}' displayed on SLM")

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.feedback and not args.use_camera:
        parser.error("--feedback requires --use-camera")

    if not args.use_camera:
        run_pattern(args)
    elif args.feedback:
        run_feedback(args)
    else:
        run_acquire(args)


if __name__ == "__main__":
    main()
