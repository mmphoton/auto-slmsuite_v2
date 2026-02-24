"""Rectangular lattice of Gaussian-like spots pattern."""

from __future__ import annotations

import numpy as np

from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import SpotHologram

from user_workflows.patterns.base import BasePattern, PatternResult, register_pattern


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


def _spot_hologram_shape(slm):
    """Use native SLM resolution for display-compatible phase arrays."""
    return tuple(int(v) for v in slm.shape)


def _build_lattice_spot_kxy(args):
    x_offsets = (np.arange(int(args.lattice_nx), dtype=float) - 0.5 * (int(args.lattice_nx) - 1.0)) * float(args.lattice_pitch_x)
    y_offsets = (np.arange(int(args.lattice_ny), dtype=float) - 0.5 * (int(args.lattice_ny) - 1.0)) * float(args.lattice_pitch_y)
    xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")
    return np.vstack((xx.ravel() + float(args.lattice_center_kx), yy.ravel() + float(args.lattice_center_ky)))


@register_pattern
class GaussianLatticePattern(BasePattern):
    name = "gaussian-lattice"

    def build(self, args, slm) -> PatternResult:
        shape = _spot_hologram_shape(slm)
        spot_kxy = _build_lattice_spot_kxy(args)
        spot_vectors, basis, cameraslm = _spot_inputs_from_kxy(slm, shape, spot_kxy)
        hologram = SpotHologram(shape, spot_vectors=spot_vectors, basis=basis, cameraslm=cameraslm)
        hologram.optimize(
            method=args.holo_method,
            maxiter=args.holo_maxiter,
            feedback="computational",
            stat_groups=["computational"],
        )
        phase = np.mod(hologram.get_phase(), 2 * np.pi)
        return PatternResult(
            phase=phase,
            metadata={
                "pattern": self.name,
                "array_shape": [args.lattice_nx, args.lattice_ny],
                "array_pitch": [args.lattice_pitch_x, args.lattice_pitch_y],
                "array_center": [args.lattice_center_kx, args.lattice_center_ky],
                "padded_shape": shape,
                "holo_method": args.holo_method,
                "holo_maxiter": args.holo_maxiter,
            },
        )
