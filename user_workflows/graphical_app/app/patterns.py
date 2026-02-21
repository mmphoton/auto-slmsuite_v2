"""Schema-driven pattern generation facade."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Mapping, get_type_hints

import numpy as np

from user_workflows.patterns.schemas import PATTERN_PARAM_SCHEMAS, pattern_params_from_flat_dict


_FIELD_CONSTRAINTS: dict[str, dict[str, dict[str, Any]]] = {
    "single-gaussian": {
        "kx": {"range": [-1.0, 1.0]},
        "ky": {"range": [-1.0, 1.0]},
    },
    "double-gaussian": {
        "center_kx": {"range": [-1.0, 1.0]},
        "center_ky": {"range": [-1.0, 1.0]},
        "sep_kxy": {"range": [0.0, None], "exclusive_min": True},
    },
    "gaussian-lattice": {
        "nx": {"range": [1, None]},
        "ny": {"range": [1, None]},
        "pitch_x": {"range": [0.0, None], "exclusive_min": True},
        "pitch_y": {"range": [0.0, None], "exclusive_min": True},
        "center_kx": {"range": [-1.0, 1.0]},
        "center_ky": {"range": [-1.0, 1.0]},
    },
    "laguerre-gaussian": {
        "l": {"range": [-50, 50]},
        "p": {"range": [0, None]},
    },
}


class PatternService:
    def available_patterns(self) -> list[str]:
        return sorted(PATTERN_PARAM_SCHEMAS.keys())

    def schema_for(self, pattern_name: str) -> dict[str, Any]:
        schema_cls = PATTERN_PARAM_SCHEMAS[pattern_name]
        type_hints = get_type_hints(schema_cls)
        descriptions = getattr(schema_cls, "FIELD_DESCRIPTIONS", {})
        constraints = _FIELD_CONSTRAINTS.get(pattern_name, {})

        metadata: dict[str, Any] = {"pattern": pattern_name, "parameters": []}
        for field in fields(schema_cls):
            param_type = type_hints.get(field.name, field.type)
            metadata["parameters"].append(
                {
                    "name": field.name,
                    "type": getattr(param_type, "__name__", str(param_type)),
                    "default": field.default,
                    "range": constraints.get(field.name, {}).get("range"),
                    "options": constraints.get(field.name, {}).get("options"),
                    "exclusive_min": constraints.get(field.name, {}).get("exclusive_min", False),
                    "help": descriptions.get(field.name, ""),
                }
            )
        return metadata

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
