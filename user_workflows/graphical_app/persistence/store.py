"""Persistence layer for layouts, sessions, recipes, and naming templates."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from user_workflows.graphical_app.app.state import AppState


class PersistenceStore:
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
            **tokens,
        }
        return template.format(**merged)

    def snapshot_session(self, state: AppState, path: Path, software_version: str = "0.1.0") -> None:
        payload = asdict(state)
        payload["software_version"] = software_version
        self.save_json(path, payload)
