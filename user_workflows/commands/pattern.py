"""Pattern generation and SLM display command."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

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




class _SpotHologramHardwareShim:
    """Back-compat proxy exposing both SLM-like and CameraSLM-like attributes."""

    def __init__(self, slm):
        self.slm = slm
        if hasattr(slm, "pitch"):
            self.pitch = slm.pitch

    def __getattr__(self, name):
        return getattr(self.slm, name)


def _spot_hologram_cameraslm_arg(slm):
    """Deprecated helper kept for backward compatibility with older call sites."""
    return slm if hasattr(slm, "slm") else _SpotHologramHardwareShim(slm)


def _as_spot_hologram_inputs(slm, shape, spot_kxy):
    """Return (spot_vectors, basis, cameraslm) for SpotHologram construction.

    If a CameraSLM-like object is available (has `.slm`), keep kxy basis.
    Otherwise convert user kxy vectors to knm using the raw SLM hardware so
    SpotHologram does not require CameraSLM-only metadata.
    """
    from slmsuite.holography import toolbox

    cameraslm_arg = _spot_hologram_cameraslm_arg(slm)
    if hasattr(slm, "slm"):
        return np.asarray(spot_kxy, dtype=float), "kxy", cameraslm_arg

    spot_knm = toolbox.convert_vector(
        np.asarray(spot_kxy, dtype=float),
        from_units="kxy",
        to_units="knm",
        hardware=slm,
        shape=shape,
    )
    return np.asarray(spot_knm, dtype=float), "knm", None


def _spot_hologram_shape(slm):
    """Use native SLM resolution for display-compatible phase arrays."""
    return tuple(int(v) for v in slm.shape)


def _build_lattice_spot_kxy(args):
    x_offsets = (np.arange(int(args.lattice_nx), dtype=float) - 0.5 * (int(args.lattice_nx) - 1.0)) * float(args.lattice_pitch_x)
    y_offsets = (np.arange(int(args.lattice_ny), dtype=float) - 0.5 * (int(args.lattice_ny) - 1.0)) * float(args.lattice_pitch_y)
    xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")
    x = xx.ravel() + float(args.lattice_center_kx)
    y = yy.ravel() + float(args.lattice_center_ky)
    return np.vstack((x, y))


def _expand_spots_with_radius(spot_kxy, radius_kxy, points=12):
    """Approximate finite spot radius by a filled disk cloud with smooth weights."""
    base = np.asarray(spot_kxy, dtype=float)
    radius = float(radius_kxy)
    if radius <= 0:
        return base, None

    n_theta = max(6, int(points))
    n_rings = 4
    sigma = max(radius / 2.0, 1e-12)

    expanded = []
    weights = []
    for idx in range(base.shape[1]):
        center = base[:, idx:idx+1]
        expanded.append(center)
        weights.append(1.0)

        for ring in range(1, n_rings + 1):
            r = radius * (ring / n_rings)
            count = n_theta * ring
            thetas = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False, dtype=float)
            offsets = np.vstack((np.cos(thetas), np.sin(thetas))) * r
            expanded.append(center + offsets)
            weights.extend([float(np.exp(-0.5 * (r / sigma) ** 2))] * count)

    cloud = np.hstack(expanded)
    amp = np.asarray(weights, dtype=float)
    amp = amp / np.linalg.norm(amp)
    return cloud, amp


def build_pattern(args, slm, deep):
    from slmsuite.holography.algorithms import SpotHologram
    from slmsuite.holography.toolbox import phase
    from slmsuite.holography.toolbox.phase import blaze

    if args.pattern == "laguerre-gaussian":
        lg_phase = phase.laguerre_gaussian(slm, l=args.lg_l, p=args.lg_p, w=args.lg_radius_w)
        phi = np.mod(lg_phase + blaze(grid=slm, vector=(args.blaze_kx, args.blaze_ky)), 2 * np.pi)
        return depth_correct(phi, deep)

    shape = _spot_hologram_shape(slm)
    spot_amp = None

    if args.pattern == "single-gaussian":
        spot_kxy = np.array([[args.single_kx], [args.single_ky]], dtype=float)
        spot_kxy, spot_amp = _expand_spots_with_radius(spot_kxy, args.single_radius_kxy, points=args.spot_radius_points)
    elif args.pattern == "double-gaussian":
        dx = float(args.double_sep_kxy) / 2.0
        spot_kxy = np.array(
            [
                [args.double_center_kx - dx, args.double_center_kx + dx],
                [args.double_center_ky, args.double_center_ky],
            ],
            dtype=float,
        )
        spot_kxy, spot_amp = _expand_spots_with_radius(spot_kxy, args.double_radius_kxy, points=args.spot_radius_points)
    elif args.pattern == "gaussian-lattice":
        spot_kxy = _build_lattice_spot_kxy(args)
        spot_kxy, spot_amp = _expand_spots_with_radius(spot_kxy, args.lattice_radius_kxy, points=args.spot_radius_points)
    else:
        raise ValueError(f"Unknown pattern '{args.pattern}'")

    spot_vectors, basis, cameraslm_arg = _as_spot_hologram_inputs(slm, shape, spot_kxy)
    hologram = SpotHologram(shape, spot_vectors=spot_vectors, basis=basis, cameraslm=cameraslm_arg, spot_amp=spot_amp)
    hologram.optimize(method=args.holo_method, maxiter=args.holo_maxiter, feedback="computational", stat_groups=["computational"])
    # Apply an optional global blaze to all non-LG spot families as well.
    # This keeps CLI behavior consistent when users set --blaze-kx/--blaze-ky
    # expecting a first-order offset independent of the selected family.
    phi = np.mod(hologram.get_phase() + blaze(grid=slm, vector=(args.blaze_kx, args.blaze_ky)), 2 * np.pi)
    return depth_correct(phi, deep)


def add_pattern_args(parser: argparse.ArgumentParser):
    parser.add_argument("--pattern", default="laguerre-gaussian", choices=["single-gaussian", "double-gaussian", "gaussian-lattice", "laguerre-gaussian"])
    parser.add_argument("--lut-file", default="deep_1024.mat")
    parser.add_argument("--lut-key", default="deep")
    parser.add_argument("--blaze-kx", type=float, default=0.0)
    parser.add_argument("--blaze-ky", type=float, default=0.02)
    parser.add_argument("--lg-l", type=int, default=3)
    parser.add_argument("--lg-p", type=int, default=0)
    parser.add_argument("--lg-radius-w", type=float, default=None, help="Laguerre-Gaussian source radius parameter w.")
    parser.add_argument("--single-kx", type=float, default=0.0)
    parser.add_argument("--single-ky", type=float, default=0.0)
    parser.add_argument("--single-radius-kxy", type=float, default=0.0)
    parser.add_argument("--double-center-kx", type=float, default=0.0)
    parser.add_argument("--double-center-ky", type=float, default=0.0)
    parser.add_argument("--double-sep-kxy", type=float, default=0.02)
    parser.add_argument(
        "--double-radius-kxy",
        type=float,
        default=0.0,
        help="Effective radius in k-space for each spot (0 keeps diffraction-limited points).",
    )
    parser.add_argument("--lattice-radius-kxy", type=float, default=0.0)
    parser.add_argument(
        "--spot-radius-points",
        type=int,
        default=12,
        help="Number of points on circular ring used to approximate non-zero spot radius.",
    )
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
            f"LUT file not found at '{lut_path}'. Fix: pass --lut-file with a valid LUT path."
        )
    deep = load_phase_lut(lut_path, args.lut_key)

    if args.dry_run:
        print(f"[dry-run] pattern settings validated for '{args.pattern}'.")
        return

    from slmsuite.hardware.slms.holoeye import Holoeye

    slm = Holoeye(preselect="index:0")
    pattern = build_pattern(args, slm, deep)
    slm.set_phase(pattern, settle=True)
    print(f"Pattern '{args.pattern}' displayed on SLM")
    hold_until_interrupt(slm)
