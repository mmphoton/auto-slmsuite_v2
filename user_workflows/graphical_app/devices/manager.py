"""Unified device manager with explicit lifecycle controls and transition safety."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from user_workflows.graphical_app.app.state import AppState, DeviceConnectionState, DeviceError, Mode
from user_workflows.graphical_app.devices.adapters import HardwareCamera, HardwareSLM, SimulatedCamera, SimulatedSLM


class DeviceState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    BUSY = "busy"
    ERROR = "error"


class LifecycleAction(str, Enum):
    DISCOVER = "discover"
    CONNECT = "connect"
    RECONNECT = "reconnect"
    RELEASE = "release"
    SAFE_STOP = "safe_stop"


@dataclass
class DeviceManager:
    state: AppState
    _lifecycle_stage: str = field(init=False, default="disconnected")

    def __post_init__(self) -> None:
        self.slm = SimulatedSLM()
        self.camera = SimulatedCamera(self.slm)

    def _ensure_transition(self, action: LifecycleAction) -> None:
        allowed: Dict[str, set[LifecycleAction]] = {
            "disconnected": {LifecycleAction.DISCOVER, LifecycleAction.CONNECT, LifecycleAction.RELEASE, LifecycleAction.SAFE_STOP},
            "discovered": {LifecycleAction.DISCOVER, LifecycleAction.CONNECT, LifecycleAction.RELEASE, LifecycleAction.SAFE_STOP},
            "connected": {LifecycleAction.DISCOVER, LifecycleAction.RECONNECT, LifecycleAction.RELEASE, LifecycleAction.SAFE_STOP},
            "safe_stopped": {LifecycleAction.DISCOVER, LifecycleAction.CONNECT, LifecycleAction.RELEASE, LifecycleAction.SAFE_STOP},
            "error": {LifecycleAction.DISCOVER, LifecycleAction.RECONNECT, LifecycleAction.RELEASE, LifecycleAction.SAFE_STOP},
        }
        if action not in allowed.get(self._lifecycle_stage, set()):
            raise RuntimeError(f"Invalid device transition: stage={self._lifecycle_stage}, action={action.value}")

    def _result(self, device: str, action: str, connected: bool, reason: str | None = None) -> Dict[str, Any]:
        return {
            "device": device,
            "action": action,
            "connected": connected,
            "failed": not connected,
            "reason": reason,
            "state": self.state.device_status.get(device, DeviceState.DISCONNECTED.value),
        }

    def _set_error(self, device: str, action: str, exc: Exception) -> Dict[str, Any]:
        error = DeviceError(message=str(exc), code=type(exc).__name__, details={"action": action})
        self.state.set_device_state(device, DeviceConnectionState.ERROR, error=error)
        self._lifecycle_stage = "error"
        return self._result(device, action, connected=False, reason=str(exc))

    def _refresh_telemetry(self) -> Dict[str, Any]:
        try:
            telemetry = self.camera.telemetry()
        except Exception as exc:
            telemetry = {
                "error": str(exc),
                "temperature_c": None,
                "temperature_status": "unknown",
            }
        self.state.update_camera_telemetry(telemetry)
        return self.state.camera_telemetry

    def _connect_device(self, device: str, adapter: Any) -> Dict[str, Any]:
        self.state.set_device_state(device, DeviceConnectionState.CONNECTING)
        try:
            adapter.connect()
            self.state.set_device_state(device, DeviceConnectionState.CONNECTED)
            return self._result(device, "connect", connected=True)
        except Exception as exc:
            return self._set_error(device, "connect", exc)

    def set_mode(self, mode: Mode) -> Dict[str, Any]:
        safe_stop = self.safe_stop()
        release = self.release_both()
        self.state.mode = mode
        if mode == Mode.SIMULATION:
            self.slm = SimulatedSLM()
            self.camera = SimulatedCamera(self.slm)
        else:
            self.slm = HardwareSLM()
            self.camera = HardwareCamera(self.slm)
        self._lifecycle_stage = "disconnected"
        self._refresh_telemetry()
        return {"mode": mode.value, "safe_stop": safe_stop, "release": release, "telemetry": self.state.camera_telemetry}

    def discover(self) -> dict[str, Any]:
        self._ensure_transition(LifecycleAction.DISCOVER)
        self._lifecycle_stage = "discovered"
        self._refresh_telemetry()
        return {
            "slm": self._result("slm", "discover", connected=self.state.device_status["slm"] == DeviceState.CONNECTED.value),
            "camera": self._result("camera", "discover", connected=self.state.device_status["camera"] == DeviceState.CONNECTED.value),
            "telemetry": self.state.camera_telemetry,
        }

    def connect(self) -> Dict[str, Any]:
        self._ensure_transition(LifecycleAction.CONNECT)
        results: Dict[str, Any] = {
            "slm": self._connect_device("slm", self.slm),
            "camera": self._connect_device("camera", self.camera),
        }
        self._lifecycle_stage = "connected" if all(r["connected"] for k, r in results.items() if k in ("slm", "camera")) else "error"
        results["telemetry"] = self._refresh_telemetry()
        return results

    def reconnect(self) -> Dict[str, Any]:
        self._ensure_transition(LifecycleAction.RECONNECT)
        release = self.release_both()
        self._lifecycle_stage = "discovered"
        connect = self.connect()
        return {"release": release, "connect": connect, "telemetry": self.state.camera_telemetry}

    def release_slm(self) -> Dict[str, Any]:
        self._ensure_transition(LifecycleAction.RELEASE)
        try:
            self.slm.disconnect()
            self.state.set_device_state("slm", DeviceConnectionState.DISCONNECTED)
            if self.state.device_status["camera"] != DeviceState.CONNECTED.value:
                self._lifecycle_stage = "disconnected"
            return self._result("slm", "release", connected=False)
        except Exception as exc:
            return self._set_error("slm", "release", exc)

    def release_camera(self) -> Dict[str, Any]:
        self._ensure_transition(LifecycleAction.RELEASE)
        try:
            self.camera.disconnect()
            self.state.set_device_state("camera", DeviceConnectionState.DISCONNECTED)
            if self.state.device_status["slm"] != DeviceState.CONNECTED.value:
                self._lifecycle_stage = "disconnected"
            payload = self._result("camera", "release", connected=False)
            payload["telemetry"] = self._refresh_telemetry()
            return payload
        except Exception as exc:
            return self._set_error("camera", "release", exc)

    def release_both(self) -> Dict[str, Any]:
        self._ensure_transition(LifecycleAction.RELEASE)
        slm = self.release_slm()
        camera = self.release_camera()
        self._lifecycle_stage = "disconnected"
        return {"slm": slm, "camera": camera, "telemetry": self.state.camera_telemetry}

    def keep_both_active(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        results["slm"] = self._result("slm", "keep_active", connected=True) if self.state.device_status["slm"] == DeviceState.CONNECTED.value else self._connect_device("slm", self.slm)
        results["camera"] = self._result("camera", "keep_active", connected=True) if self.state.device_status["camera"] == DeviceState.CONNECTED.value else self._connect_device("camera", self.camera)
        self._lifecycle_stage = "connected" if results["slm"]["connected"] and results["camera"]["connected"] else "error"
        results["telemetry"] = self._refresh_telemetry()
        return results

    def safe_stop(self) -> Dict[str, Any]:
        self._ensure_transition(LifecycleAction.SAFE_STOP)
        halted = self.state.active_run is not None
        if halted:
            self.state.notify("Safe-stop triggered: active run halted before transition.")
            self.state.active_run = None
        self._lifecycle_stage = "safe_stopped" if halted else self._lifecycle_stage
        self._refresh_telemetry()
        return {"halted": halted, "telemetry": self.state.camera_telemetry}
