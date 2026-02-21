"""Calibration/alignment tools with profile compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from user_workflows.graphical_app.persistence.store import PersistenceStore


@dataclass
class CalibrationProfile:
    name: str
    mode: str
    slm_model: str
    camera_model: str
    matrix: list[list[float]]


class CalibrationTools:
    def __init__(self) -> None:
        self.store = PersistenceStore()

    def before_after_metrics(self, before: np.ndarray, after: np.ndarray) -> Dict[str, float]:
        before_rmse = float(np.sqrt(np.mean(before**2)))
        after_rmse = float(np.sqrt(np.mean(after**2)))
        return {
            "before_rmse": before_rmse,
            "after_rmse": after_rmse,
            "rmse_delta": after_rmse - before_rmse,
            "rmse_improvement": before_rmse - after_rmse,
        }

    def save_profile(self, profile: CalibrationProfile, path: Path) -> None:
        self.store.save_json(path, profile.__dict__)

    def load_profile(self, path: Path) -> Dict[str, Any]:
        return self.store.load_json(path)

    def validate_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        required = ("name", "mode", "slm_model", "camera_model", "matrix")
        missing = [field for field in required if field not in profile]
        matrix = profile.get("matrix")
        matrix_valid = isinstance(matrix, list) and all(isinstance(row, list) for row in matrix) and len(matrix) > 0
        is_valid = not missing and matrix_valid
        return {
            "valid": is_valid,
            "missing_fields": missing,
            "matrix_valid": matrix_valid,
            "summary": {
                "name": profile.get("name", ""),
                "mode": profile.get("mode", ""),
                "slm_model": profile.get("slm_model", ""),
                "camera_model": profile.get("camera_model", ""),
                "matrix_shape": [len(matrix), len(matrix[0])] if matrix_valid and matrix and matrix[0] else [0, 0],
            },
        }

    def compatible(self, profile: Dict[str, Any], mode: str, slm_model: str, camera_model: str) -> bool:
        return profile.get("mode") == mode and profile.get("slm_model") == slm_model and profile.get("camera_model") == camera_model

    def compatibility_report(self, profile: Dict[str, Any], mode: str, slm_model: str, camera_model: str) -> Dict[str, Any]:
        checks = {
            "mode": profile.get("mode") == mode,
            "slm_model": profile.get("slm_model") == slm_model,
            "camera_model": profile.get("camera_model") == camera_model,
        }
        return {
            "compatible": all(checks.values()),
            "checks": checks,
            "expected": {"mode": mode, "slm_model": slm_model, "camera_model": camera_model},
            "actual": {
                "mode": profile.get("mode"),
                "slm_model": profile.get("slm_model"),
                "camera_model": profile.get("camera_model"),
            },
        }

    def apply_profile(self, profile: Dict[str, Any], before_frame: np.ndarray) -> Dict[str, Any]:
        matrix = np.asarray(profile.get("matrix", []), dtype=float)
        if matrix.size == 0:
            matrix = np.eye(2, dtype=float)
        scale = float(np.mean(np.abs(matrix))) if matrix.size else 1.0
        adjusted = np.asarray(before_frame, dtype=float) / max(scale, 1e-6)
        after_frame = np.clip(adjusted, 0.0, 1.0)
        metrics = self.before_after_metrics(before_frame, after_frame)
        return {
            "applied_profile": dict(profile),
            "metrics": metrics,
            "scale_factor": scale,
        }
