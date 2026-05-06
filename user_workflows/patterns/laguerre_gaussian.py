"""Laguerre-Gaussian phase mode pattern with optional blaze."""

from __future__ import annotations

import numpy as np

from slmsuite.holography.algorithms import Hologram
from slmsuite.holography.toolbox import phase
from slmsuite.holography.toolbox.phase import blaze

from user_workflows.commands.pattern import _build_laguerre_gaussian_target
from user_workflows.patterns.base import BasePattern, PatternResult, register_pattern


@register_pattern
class LaguerreGaussianPattern(BasePattern):
    name = "laguerre-gaussian"

    def build(self, args, slm) -> PatternResult:
        shape = tuple(int(v) for v in slm.shape)
        radius_w = getattr(args, "lg_radius_w", None)
        lg_phase = phase.laguerre_gaussian(slm, l=args.lg_l, p=args.lg_p, w=radius_w)
        target = _build_laguerre_gaussian_target(shape, args.lg_l, args.lg_p, radius_w)
        hologram = Hologram(target=target, phase=lg_phase, slm_shape=shape)
        hologram.optimize(
            method=getattr(args, "holo_method", "WGS-Kim"),
            maxiter=getattr(args, "holo_maxiter", 30),
            feedback="computational",
            stat_groups=["computational"],
        )
        blaze_phase = blaze(grid=slm, vector=(args.blaze_kx, args.blaze_ky))
        pattern_phase = np.mod(hologram.get_phase() + blaze_phase, 2 * np.pi)
        return PatternResult(
            phase=pattern_phase,
            metadata={
                "pattern": self.name,
                "lg_l": args.lg_l,
                "lg_p": args.lg_p,
                "lg_radius_w": radius_w,
                "holo_method": getattr(args, "holo_method", "WGS-Kim"),
                "holo_maxiter": getattr(args, "holo_maxiter", 30),
                "blaze_vector": [args.blaze_kx, args.blaze_ky],
            },
        )
