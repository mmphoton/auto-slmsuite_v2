"""Schema-driven pattern generation facade."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np

from user_workflows.patterns.schemas import PATTERN_PARAM_SCHEMAS, pattern_params_from_flat_dict


class PatternService:
    def available_patterns(self) -> list[str]:
        return sorted(PATTERN_PARAM_SCHEMAS.keys())

    def schema_for(self, pattern_name: str) -> dict[str, Any]:
        schema = PATTERN_PARAM_SCHEMAS[pattern_name]
        return {f.name: str(f.type) for f in schema.__dataclass_fields__.values()}

    def defaults_for(self, pattern_name: str) -> dict[str, Any]:
        params = PATTERN_PARAM_SCHEMAS[pattern_name]()
        return asdict(params)

    def generate(self, pattern_name: str, params: Mapping[str, Any], shape: tuple[int, int] = (128, 128)) -> np.ndarray:
        parsed = pattern_params_from_flat_dict(pattern_name, params)
        p = asdict(parsed)
        y, x = np.indices(shape)
        xx = (x - shape[1] / 2) / shape[1]
        yy = (y - shape[0] / 2) / shape[0]
        if pattern_name == "single-gaussian":
            phase = 2 * np.pi * (p["kx"] * xx + p["ky"] * yy)
        elif pattern_name == "double-gaussian":
            phase = 2 * np.pi * ((p["center_kx"] + p["sep_kxy"] / 2) * xx + p["center_ky"] * yy)
            phase += 2 * np.pi * ((p["center_kx"] - p["sep_kxy"] / 2) * xx + p["center_ky"] * yy)
        elif pattern_name == "gaussian-lattice":
            phase = 2 * np.pi * (p["center_kx"] * xx + p["center_ky"] * yy)
            phase += np.sin(2 * np.pi * p["pitch_x"] * x) + np.sin(2 * np.pi * p["pitch_y"] * y)
        elif pattern_name == "laguerre-gaussian":
            theta = np.arctan2(yy, xx)
            r = np.sqrt(xx**2 + yy**2)
            phase = p["l"] * theta + p["p"] * r
        else:
            raise KeyError(f"Unsupported pattern {pattern_name}")
        return np.mod(phase, 2 * np.pi)
