"""Single Gaussian-like spot pattern."""

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


@register_pattern
class SingleGaussianPattern(BasePattern):
    name = "single-gaussian"

    def build(self, args, slm) -> PatternResult:
        shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)
        spot_kxy = np.array([[args.single_kx], [args.single_ky]], dtype=float)
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
                "padded_shape": shape,
                "holo_method": args.holo_method,
                "holo_maxiter": args.holo_maxiter,
            },
        )
