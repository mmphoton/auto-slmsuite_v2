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
        return {
            "before_rmse": float(np.sqrt(np.mean(before**2))),
            "after_rmse": float(np.sqrt(np.mean(after**2))),
        }

    def save_profile(self, profile: CalibrationProfile, path: Path) -> None:
        self.store.save_json(path, profile.__dict__)

    def load_profile(self, path: Path) -> Dict[str, Any]:
        return self.store.load_json(path)

    def compatible(self, profile: Dict[str, Any], mode: str, slm_model: str, camera_model: str) -> bool:
        return profile.get("mode") == mode and profile.get("slm_model") == slm_model and profile.get("camera_model") == camera_model
