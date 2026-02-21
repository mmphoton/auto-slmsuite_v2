"""Centralized application state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Mode(str, Enum):
    SIMULATION = "simulation"
    HARDWARE = "hardware"


class DeviceConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    BUSY = "busy"
    ERROR = "error"


class LogLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class DeviceError:
    message: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class DeviceStatus:
    state: DeviceConnectionState = DeviceConnectionState.DISCONNECTED
    last_error: Optional[DeviceError] = None


@dataclass
class WorkflowStatus:
    active_mode: Mode = Mode.SIMULATION
    active_workflow: str = "idle"


@dataclass
class LogEntry:
    timestamp: str
    level: LogLevel
    message: str
    source: str = "controller"


@dataclass
class ProgressState:
    task_name: Optional[str] = None
    current: int = 0
    total: int = 0
    is_active: bool = False
    message: str = ""


@dataclass
class OutputConfig:
    folder: str = "user_workflows/output"
    session_name: str = "default"
    naming_template: str = "{date}_{session}_{run_id}_{artifact}"
    collision_policy: str = "increment"


@dataclass
class SettingsSnapshots:
    calibration: Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)
    plots: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RunMetadata:
    run_id: str
    mode: Mode
    device_snapshot: Dict[str, str]
    parameters: Dict[str, Any]
    calibration_profile: Optional[str] = None
    optimizer: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppState:
    workflow: WorkflowStatus = field(default_factory=WorkflowStatus)
    devices: Dict[str, DeviceStatus] = field(
        default_factory=lambda: {
            "slm": DeviceStatus(),
            "camera": DeviceStatus(),
        }
    )
    output: OutputConfig = field(default_factory=OutputConfig)
    settings_snapshots: SettingsSnapshots = field(default_factory=SettingsSnapshots)
    recipes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_run: Optional[RunMetadata] = None
    queued_patterns: List[str] = field(default_factory=list)
    notifications: List[str] = field(default_factory=list)
    logs: List[LogEntry] = field(default_factory=list)
    progress: ProgressState = field(default_factory=ProgressState)
    camera_telemetry: Dict[str, Any] = field(default_factory=dict)
    camera_temperature_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "warning_c": -55.0,
            "critical_c": -45.0,
        }
    )

    @property
    def mode(self) -> Mode:
        return self.workflow.active_mode

    @mode.setter
    def mode(self, mode: Mode) -> None:
        self.workflow.active_mode = mode

    @property
    def session_name(self) -> str:
        return self.output.session_name

    @property
    def output_directory(self) -> str:
        return self.output.folder

    @property
    def naming_template(self) -> str:
        return self.output.naming_template

    @property
    def collision_policy(self) -> str:
        return self.output.collision_policy

    @property
    def plots(self) -> Dict[str, Dict[str, Any]]:
        return self.settings_snapshots.plots

    @property
    def device_status(self) -> Dict[str, str]:
        return {name: status.state.value for name, status in self.devices.items()}

    def set_device_state(self, device: str, state: DeviceConnectionState, error: Optional[DeviceError] = None) -> None:
        status = self.devices.setdefault(device, DeviceStatus())
        status.state = state
        status.last_error = error

    def notify(self, message: str) -> None:
        self.notifications.append(message)

    def add_log(self, level: LogLevel, message: str, source: str = "controller") -> None:
        self.logs.append(
            LogEntry(
                timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                level=level,
                message=message,
                source=source,
            )
        )

    def update_camera_telemetry(self, telemetry: Optional[Dict[str, Any]]) -> None:
        payload = dict(telemetry or {})
        if "temperature_c" in payload and payload["temperature_c"] is not None:
            temperature = float(payload["temperature_c"])
            payload["temperature_status"] = "ok"
            if temperature >= self.camera_temperature_thresholds["critical_c"]:
                payload["temperature_status"] = "critical"
            elif temperature >= self.camera_temperature_thresholds["warning_c"]:
                payload["temperature_status"] = "warning"
        elif "temperature_status" not in payload:
            payload["temperature_status"] = "unknown"
        self.camera_telemetry = payload
