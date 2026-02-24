"""Built-in user workflow patterns."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from slmsuite.holography.algorithms import SpotHologram
from slmsuite.holography.toolbox import phase
from slmsuite.holography.toolbox.phase import blaze

from user_workflows.patterns.registry import register_pattern


def _spot_hologram_cameraslm_arg(slm):
    return slm if hasattr(slm, "slm") else SimpleNamespace(slm=slm)


@register_pattern(name="laguerre-gaussian")
def laguerre_gaussian(config, slm, deep, depth_correct):
    args = config.args
    lg_phase = phase.laguerre_gaussian(slm, l=args.lg_l, p=args.lg_p)
    phi = np.mod(lg_phase + blaze(grid=slm, vector=(args.blaze_kx, args.blaze_ky)), 2 * np.pi)
    return depth_correct(phi, deep)


def _optimize_spot_hologram(config, slm, deep, hologram, depth_correct):
    args = config.args
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
    shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)
    cameraslm_arg = _spot_hologram_cameraslm_arg(slm)
    spot_kxy = np.array([[args.single_kx], [args.single_ky]])
    hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=cameraslm_arg)
    return _optimize_spot_hologram(config, slm, deep, hologram, depth_correct)


@register_pattern(name="double-gaussian")
def double_gaussian(config, slm, deep, depth_correct):
    args = config.args
    shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)
    cameraslm_arg = _spot_hologram_cameraslm_arg(slm)
    dx = float(args.double_sep_kxy) / 2.0
    spot_kxy = np.array(
        [
            [args.double_center_kx - dx, args.double_center_kx + dx],
            [args.double_center_ky, args.double_center_ky],
        ]
    )
    hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=cameraslm_arg)
    return _optimize_spot_hologram(config, slm, deep, hologram, depth_correct)


@register_pattern(name="gaussian-lattice")
def gaussian_lattice(config, slm, deep, depth_correct):
    args = config.args
    shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)
    cameraslm_arg = _spot_hologram_cameraslm_arg(slm)
    hologram = SpotHologram.make_rectangular_array(
        shape,
        array_shape=(args.lattice_nx, args.lattice_ny),
        array_pitch=(args.lattice_pitch_x, args.lattice_pitch_y),
        array_center=(args.lattice_center_kx, args.lattice_center_ky),
        basis="kxy",
        cameraslm=cameraslm_arg,
    )
    return _optimize_spot_hologram(config, slm, deep, hologram, depth_correct)
