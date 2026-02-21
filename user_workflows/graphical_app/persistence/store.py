"""Persistence layer for layouts, sessions, recipes, and naming templates."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from user_workflows.graphical_app.app.state import AppState


DEFAULT_LAYOUT_MODEL: Dict[str, Any] = {
    "window_geometry": "1400x900",
    "columns": {
        "left": ["Device", "Optimization", "Logs", "Session"],
        "center": ["SLM", "Plots"],
        "right": ["Camera", "Calibration"],
    },
    "visibility": {
        "Device": True,
        "SLM": True,
        "Camera": True,
        "Plots": True,
        "Optimization": True,
        "Calibration": True,
        "Logs": True,
        "Session": True,
    },
    "sashes": {"main": [400, 1000]},
    "popout_plots": [],
}


PRESET_LAYOUT_MODELS: Dict[str, Dict[str, Any]] = {
    "Acquisition": {
        **DEFAULT_LAYOUT_MODEL,
        "columns": {
            "left": ["Device", "Camera", "Session"],
            "center": ["SLM", "Plots"],
            "right": ["Optimization", "Calibration", "Logs"],
        },
        "visibility": {
            **DEFAULT_LAYOUT_MODEL["visibility"],
            "Optimization": False,
            "Calibration": False,
        },
    },
    "Optimization": {
        **DEFAULT_LAYOUT_MODEL,
        "columns": {
            "left": ["Device", "SLM", "Session"],
            "center": ["Plots", "Optimization"],
            "right": ["Camera", "Calibration", "Logs"],
        },
        "visibility": {
            **DEFAULT_LAYOUT_MODEL["visibility"],
            "Calibration": False,
        },
    },
    "Calibration": {
        **DEFAULT_LAYOUT_MODEL,
        "columns": {
            "left": ["Device", "Camera", "Session"],
            "center": ["Calibration", "Plots"],
            "right": ["SLM", "Optimization", "Logs"],
        },
        "visibility": {
            **DEFAULT_LAYOUT_MODEL["visibility"],
            "Optimization": False,
        },
    },
}


class PersistenceStore:
    def sanitize_token(self, value: Any) -> str:
        token = str(value).strip().replace(" ", "-")
        token = re.sub(r"[^A-Za-z0-9._-]+", "-", token)
        token = token.strip("-_")
        return token or "na"

    def save_json(self, path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(data), indent=2))

    def load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def save_array(self, path: Path, data: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data)

    def render_name(self, template: str, **tokens: str) -> str:
        merged = {
            "date": datetime.utcnow().strftime("%Y%m%d"),
            "time": datetime.utcnow().strftime("%H%M%S"),
            **{key: self.sanitize_token(val) for key, val in tokens.items()},
        }
        rendered = template.format(**merged)
        return self.sanitize_token(rendered)

    def resolve_path(self, candidate: Path, collision_policy: str) -> Path:
        policy = (collision_policy or "increment").strip().lower()
        if policy == "overwrite" or not candidate.exists():
            return candidate
        if policy == "error":
            raise FileExistsError(f"Output file already exists: {candidate}")
        # default: increment
        idx = 1
        while True:
            attempt = candidate.with_name(f"{candidate.stem}_{idx:03d}{candidate.suffix}")
            if not attempt.exists():
                return attempt
            idx += 1

    def snapshot_session(self, state: AppState, path: Path, software_version: str = "0.1.0") -> None:
        payload = asdict(state)
        payload["software_version"] = software_version
        payload["metadata_snapshot"] = {
            "software_version": software_version,
            "mode": state.mode.value,
            "devices": state.device_status,
            "camera_settings": dict(state.settings_snapshots.camera),
            "blaze": asdict(state.blaze),
            "calibration": dict(state.settings_snapshots.calibration),
            "optimizer": dict(state.settings_snapshots.optimizer),
        }
        self.save_json(path, payload)

    def default_layout_model(self) -> Dict[str, Any]:
        return copy.deepcopy(DEFAULT_LAYOUT_MODEL)

    def preset_layout_model(self, preset_name: str) -> Dict[str, Any]:
        base = PRESET_LAYOUT_MODELS.get(preset_name)
        if base is None:
            return self.default_layout_model()
        return copy.deepcopy(base)

    def save_layout_model(self, path: Path, model: Mapping[str, Any]) -> None:
        self.save_json(path, model)

    def load_layout_model(self, path: Path) -> Dict[str, Any]:
        payload = self.load_json(path)
        if not payload:
            return self.default_layout_model()

        merged = self.default_layout_model()
        merged.update({k: v for k, v in payload.items() if k not in {"columns", "visibility", "sashes"}})
        merged["columns"].update(payload.get("columns", {}))
        merged["visibility"].update(payload.get("visibility", {}))
        merged["sashes"].update(payload.get("sashes", {}))
        return merged
