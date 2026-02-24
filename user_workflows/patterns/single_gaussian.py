"""Single Gaussian-like spot pattern."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from slmsuite.holography.algorithms import SpotHologram

from user_workflows.patterns.base import BasePattern, PatternResult, register_pattern


def _spot_hologram_cameraslm_arg(slm):
    return slm if hasattr(slm, "slm") else SimpleNamespace(slm=slm)


@register_pattern
class SingleGaussianPattern(BasePattern):
    name = "single-gaussian"

    def build(self, args, slm) -> PatternResult:
        shape = SpotHologram.get_padded_shape(slm, padding_order=1, square_padding=True)
        cameraslm_arg = _spot_hologram_cameraslm_arg(slm)
        spot_kxy = np.array([[args.single_kx], [args.single_ky]])
        hologram = SpotHologram(shape, spot_vectors=spot_kxy, basis="kxy", cameraslm=cameraslm_arg)
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
