"""Built-in user workflow patterns."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from slmsuite.holography.algorithms import SpotHologram
from slmsuite.holography import toolbox
from slmsuite.holography.toolbox import phase
from slmsuite.holography.toolbox.phase import blaze

from user_workflows.patterns.registry import register_pattern


def _spot_inputs_from_kxy(slm, shape, spot_kxy):
    if hasattr(slm, "slm"):
        return np.asarray(spot_kxy, dtype=float), "kxy", slm

    spot_knm = toolbox.convert_vector(
        np.asarray(spot_kxy, dtype=float),
        from_units="kxy",
        to_units="knm",
        hardware=slm,
        shape=shape,
    )
    return np.asarray(spot_knm, dtype=float), "knm", None


def _build_lattice_spot_kxy(args):
    x_offsets = (np.arange(int(args.lattice_nx), dtype=float) - 0.5 * (int(args.lattice_nx) - 1.0)) * float(args.lattice_pitch_x)
    y_offsets = (np.arange(int(args.lattice_ny), dtype=float) - 0.5 * (int(args.lattice_ny) - 1.0)) * float(args.lattice_pitch_y)
    xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")
    return np.vstack((xx.ravel() + float(args.lattice_center_kx), yy.ravel() + float(args.lattice_center_ky)))


@register_pattern(name="laguerre-gaussian")
def laguerre_gaussian(config, slm, deep, depth_correct):
    args = config.args
    lg_phase = phase.laguerre_gaussian(slm, l=args.lg_l, p=args.lg_p)
    phi = np.mod(lg_phase + blaze(grid=slm, vector=(args.blaze_kx, args.blaze_ky)), 2 * np.pi)
    return depth_correct(phi, deep)


def _optimize_spot_hologram(config, slm, deep, spot_kxy, depth_correct):
    args = config.args
    shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)
    spot_vectors, basis, cameraslm = _spot_inputs_from_kxy(slm, shape, spot_kxy)
    hologram = SpotHologram(shape, spot_vectors=spot_vectors, basis=basis, cameraslm=cameraslm)
    hologram.optimize(
        method=args.holo_method,
        maxiter=args.holo_maxiter,
        feedback="computational",
        stat_groups=["computational"],
    )
    phi = np.mod(hologram.get_phase(), 2 * np.pi)
    return depth_correct(phi, deep)


@register_pattern(name="single-gaussian")
def single_gaussian(config, slm, deep, depth_correct):
    args = config.args
    spot_kxy = np.array([[args.single_kx], [args.single_ky]], dtype=float)
    return _optimize_spot_hologram(config, slm, deep, spot_kxy, depth_correct)


@register_pattern(name="double-gaussian")
def double_gaussian(config, slm, deep, depth_correct):
    args = config.args
    dx = float(args.double_sep_kxy) / 2.0
    spot_kxy = np.array(
        [
            [args.double_center_kx - dx, args.double_center_kx + dx],
            [args.double_center_ky, args.double_center_ky],
        ],
        dtype=float,
    )
    return _optimize_spot_hologram(config, slm, deep, spot_kxy, depth_correct)


@register_pattern(name="gaussian-lattice")
def gaussian_lattice(config, slm, deep, depth_correct):
    return _optimize_spot_hologram(config, slm, deep, _build_lattice_spot_kxy(config.args), depth_correct)
