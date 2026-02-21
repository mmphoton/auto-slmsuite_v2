"""Centralized application state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Mode(str, Enum):
    SIMULATION = "simulation"
    HARDWARE = "hardware"


@dataclass
class RunMetadata:
    run_id: str
    mode: Mode
    device_snapshot: Dict[str, Any]
    parameters: Dict[str, Any]
    calibration_profile: Optional[str] = None
    optimizer: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppState:
    mode: Mode = Mode.SIMULATION
    device_status: Dict[str, str] = field(default_factory=lambda: {"slm": "disconnected", "camera": "disconnected"})
    session_name: str = "default"
    output_directory: str = "user_workflows/output"
    naming_template: str = "{date}_{session}_{run_id}_{artifact}"
    collision_policy: str = "increment"
    plots: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recipes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_run: Optional[RunMetadata] = None
    queued_patterns: List[str] = field(default_factory=list)
    notifications: List[str] = field(default_factory=list)

    def notify(self, message: str) -> None:
        self.notifications.append(message)
