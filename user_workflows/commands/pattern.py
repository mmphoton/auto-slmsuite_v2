"""Pattern generation and SLM display command."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io


def load_phase_lut(path: Path, key: str = "deep") -> np.ndarray:
    if path.suffix.lower() == ".npy":
        deep = np.asarray(np.load(path)).squeeze()
    else:
        mat = scipy.io.loadmat(path)
        if key not in mat:
            raise ValueError(f"LUT file must contain variable '{key}'")
        deep = np.asarray(mat[key]).squeeze()
    if deep.ndim != 1:
        raise ValueError(f"Expected 1D LUT, got {deep.shape}")
    return deep


def depth_correct(phi, deep):
    idx = np.clip(np.rint((phi / (2 * np.pi)) * (deep.size - 1)), 0, deep.size - 1).astype(int)
    factors = deep[idx]
    corrected = (phi - np.pi) * factors + np.pi
    return np.mod(corrected, 2 * np.pi)


def build_pattern(args, slm, deep):
    from slmsuite.holography.algorithms import SpotHologram
    from slmsuite.holography.toolbox import phase
    from slmsuite.holography.toolbox.phase import blaze

    if args.pattern == "laguerre-gaussian":
        lg_phase = phase.laguerre_gaussian(slm, l=args.lg_l, p=args.lg_p)
        phi = np.mod(lg_phase + blaze(grid=slm, vector=(args.blaze_kx, args.blaze_ky)), 2 * np.pi)
        return depth_correct(phi, deep)

    shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)

    if args.pattern == "single-gaussian":
        spot_kxy = np.array([[args.single_kx], [args.single_ky]])
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=slm)
    elif args.pattern == "double-gaussian":
        dx = float(args.double_sep_kxy) / 2.0
        spot_kxy = np.array([
            [args.double_center_kx - dx, args.double_center_kx + dx],
            [args.double_center_ky, args.double_center_ky],
        ])
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

    hologram.optimize(method=args.holo_method, maxiter=args.holo_maxiter, feedback="computational", stat_groups=["computational"])
    phi = np.mod(hologram.get_phase(), 2 * np.pi)
    return depth_correct(phi, deep)


def add_pattern_args(parser: argparse.ArgumentParser):
    parser.add_argument("--pattern", default="laguerre-gaussian", choices=["single-gaussian", "double-gaussian", "gaussian-lattice", "laguerre-gaussian"])
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


def run_pattern(args):
    lut_path = Path(args.lut_file)
    if not lut_path.exists():
        raise FileNotFoundError(
            f"LUT file not found at '{lut_path}'. Fix: pass --lut-file or run `python user_workflows/cli.py workflow doctor --lut-file {lut_path}`"
        )
    deep = load_phase_lut(lut_path, args.lut_key)
    if args.dry_run:
        print(f"[dry-run] LUT and pattern settings validated for '{args.pattern}'.")
        return

    from slmsuite.hardware.slms.holoeye import Holoeye

    slm = Holoeye(preselect="index:0")
    pattern = build_pattern(args, slm, deep)
    slm.set_phase(pattern, settle=True)
    print(f"Pattern '{args.pattern}' displayed on SLM")
    hold_until_interrupt(slm)
