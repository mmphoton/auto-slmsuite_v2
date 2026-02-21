"""Laguerre-Gaussian phase mode pattern with optional blaze."""

from __future__ import annotations

import numpy as np

from slmsuite.holography.toolbox import phase
from slmsuite.holography.toolbox.phase import blaze

from user_workflows.patterns.base import BasePattern, PatternResult, register_pattern


@register_pattern
class LaguerreGaussianPattern(BasePattern):
    name = "laguerre-gaussian"

    def build(self, args, slm) -> PatternResult:
        lg_phase = phase.laguerre_gaussian(slm, l=args.lg_l, p=args.lg_p)
        blaze_phase = blaze(grid=slm, vector=(args.blaze_kx, args.blaze_ky))
        pattern_phase = np.mod(lg_phase + blaze_phase, 2 * np.pi)
        return PatternResult(
            phase=pattern_phase,
            metadata={
                "pattern": self.name,
                "lg_l": args.lg_l,
                "lg_p": args.lg_p,
                "blaze_vector": [args.blaze_kx, args.blaze_ky],
            },
        )
