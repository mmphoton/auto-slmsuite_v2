"""Two Gaussian-like spots pattern."""

from __future__ import annotations

from types import SimpleNamespace

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


@register_pattern
class DoubleGaussianPattern(BasePattern):
    name = "double-gaussian"

    def build(self, args, slm) -> PatternResult:
        shape = _spot_hologram_shape(slm)
        dx = float(args.double_sep_kxy) / 2.0
        spot_kxy = np.array(
            [
                [args.double_center_kx - dx, args.double_center_kx + dx],
                [args.double_center_ky, args.double_center_ky],
            ],
            dtype=float,
        )
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
                "spot_kxy": spot_kxy.tolist(),
                "separation_kxy": args.double_sep_kxy,
                "padded_shape": shape,
                "holo_method": args.holo_method,
                "holo_maxiter": args.holo_maxiter,
            },
        )
