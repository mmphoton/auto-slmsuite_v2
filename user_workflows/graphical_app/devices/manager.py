"""Unified device manager with explicit lifecycle controls and transition safety."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from user_workflows.graphical_app.app.state import AppState, Mode
from user_workflows.graphical_app.devices.adapters import HardwareCamera, HardwareSLM, SimulatedCamera, SimulatedSLM


class DeviceState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class DeviceManager:
    state: AppState

    def __post_init__(self) -> None:
        self.slm = SimulatedSLM()
        self.camera = SimulatedCamera(self.slm)

    def set_mode(self, mode: Mode) -> None:
        self.safe_stop()
        self.release_both()
        self.state.mode = mode
        if mode == Mode.SIMULATION:
            self.slm = SimulatedSLM()
            self.camera = SimulatedCamera(self.slm)
        else:
            self.slm = HardwareSLM()
            self.camera = HardwareCamera(self.slm)

    def discover(self) -> dict[str, str]:
        return {"slm": "available", "camera": "available"}

    def connect(self) -> None:
        self.slm.connect()
        self.camera.connect()
        self.state.device_status.update({"slm": DeviceState.CONNECTED.value, "camera": DeviceState.CONNECTED.value})

    def reconnect(self) -> None:
        self.release_both()
        self.connect()

    def release_slm(self) -> None:
        self.slm.disconnect()
        self.state.device_status["slm"] = DeviceState.DISCONNECTED.value

    def release_camera(self) -> None:
        self.camera.disconnect()
        self.state.device_status["camera"] = DeviceState.DISCONNECTED.value

    def release_both(self) -> None:
        self.release_slm()
        self.release_camera()

    def keep_both_active(self) -> None:
        if self.state.device_status["slm"] != DeviceState.CONNECTED.value:
            self.slm.connect()
            self.state.device_status["slm"] = DeviceState.CONNECTED.value
        if self.state.device_status["camera"] != DeviceState.CONNECTED.value:
            self.camera.connect()
            self.state.device_status["camera"] = DeviceState.CONNECTED.value

    def safe_stop(self) -> None:
        if self.state.active_run is not None:
            self.state.notify("Safe-stop triggered: active run halted before transition.")
            self.state.active_run = None
