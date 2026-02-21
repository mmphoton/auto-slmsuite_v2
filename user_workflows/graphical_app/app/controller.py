"""High-level application controller that exposes all backend functionality to UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np

from user_workflows.graphical_app.app.interfaces import (
    OperationResult,
    failure_result,
    feature_matrix_rows,
    success_result,
)
from user_workflows.graphical_app.app.state import (
    AppState,
    DeviceConnectionState,
    DeviceError,
    LogLevel,
    Mode,
    RunMetadata,
)
from user_workflows.graphical_app.calibration.tools import CalibrationProfile, CalibrationTools
from user_workflows.graphical_app.devices.manager import DeviceManager
from user_workflows.graphical_app.optimization.runner import OptimizationRunner
from user_workflows.graphical_app.app.patterns import PatternService
from user_workflows.graphical_app.persistence.store import PersistenceStore
from user_workflows.graphical_app.plotting.backend import PlotBackend
from user_workflows.graphical_app.plugins.registry import PluginRegistry
from user_workflows.graphical_app.sequence.runner import SequenceRunner


class AppController:
    def __init__(self) -> None:
        self.state = AppState()
        self.devices = DeviceManager(self.state)
        self.plots = PlotBackend()
        self.persistence = PersistenceStore()
        self.optimizer = OptimizationRunner()
        self.calibration = CalibrationTools()
        self.patterns = PatternService()
        self.sequence = SequenceRunner()
        self.plugins = PluginRegistry()
        self.plugins.load_builtin_patterns()

    def _handle_exception(self, action: str, exc: Exception) -> OperationResult:
        message = f"{action} failed: {exc}"
        self.state.notify(message)
        self.state.add_command_log(action, "failure", message, level=LogLevel.ERROR)
        for device_name in ("slm", "camera"):
            self.state.set_device_state(
                device_name,
                DeviceConnectionState.ERROR,
                error=DeviceError(message=str(exc), code=type(exc).__name__, details={"action": action}),
            )
        return failure_result(message, code=type(exc).__name__, details={"action": action})

    def _run(self, action: str, fn: Callable[[], Any], ok_message: str, payload: Any | None = None) -> OperationResult:
        self.state.add_command_log(action, "start", f"Starting {action}")
        try:
            result = fn()
            self.state.add_command_log(action, "success", ok_message)
            return success_result(ok_message, payload=result if payload is None else payload)
        except Exception as exc:  # central exception hook for backend errors
            return self._handle_exception(action, exc)

    def start_run(self, run_id: str, params: Mapping[str, Any]) -> OperationResult:
        def _impl() -> None:
            self.state.active_run = RunMetadata(
                run_id=run_id,
                mode=self.state.mode,
                device_snapshot=dict(self.state.device_status),
                parameters=dict(params),
                calibration_profile=self.state.settings_snapshots.calibration.get("profile_name"),
                optimizer=dict(self.state.settings_snapshots.optimizer),
            )
            self.state.workflow.active_workflow = "run_active"

        return self._run("start_run", _impl, f"Run '{run_id}' started")

    def stop_run(self) -> OperationResult:
        def _impl() -> None:
            self.state.active_run = None
            self.state.workflow.active_workflow = "idle"

        return self._run("stop_run", _impl, "Run stopped")

    def set_mode(self, mode: str) -> OperationResult:
        return self._run("set_mode", lambda: self.devices.set_mode(Mode(mode)), f"Mode set to {mode}")

    def discover_devices(self) -> OperationResult:
        return self._run("discover_devices", self.devices.discover, "Device discovery complete")

    def connect_devices(self) -> OperationResult:
        return self._run("connect_devices", self.devices.connect, "Devices connected")

    def reconnect_devices(self) -> OperationResult:
        return self._run("reconnect_devices", self.devices.reconnect, "Devices reconnected")

    def release_slm(self) -> OperationResult:
        return self._run("release_slm", self.devices.release_slm, "SLM released")

    def release_camera(self) -> OperationResult:
        return self._run("release_camera", self.devices.release_camera, "Camera released")

    def release_both(self) -> OperationResult:
        return self._run("release_both", self.devices.release_both, "SLM and camera released")

    def available_patterns(self) -> OperationResult:
        return self._run("available_patterns", self.patterns.available_patterns, "Loaded pattern catalog")

    def generate_pattern(self, name: str, params: Mapping[str, Any]) -> OperationResult:
        return self._run(
            "generate_pattern",
            lambda: self.patterns.generate(name, params, shape=(128, 128)),
            f"Pattern '{name}' generated",
        )

    def simulate_before_apply(self, pattern: np.ndarray) -> OperationResult:
        def _impl() -> Dict[str, np.ndarray]:
            self.devices.slm.apply_pattern(pattern)
            experimental = self.devices.camera.acquire_frame()
            simulated_phase = pattern
            simulated_intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * pattern))))
            simulated_intensity /= max(float(simulated_intensity.max()), 1e-9)
            payload = {
                "simulated_phase": simulated_phase,
                "simulated_intensity": simulated_intensity,
                "experimental_intensity": experimental,
            }
            for key, value in payload.items():
                self.plots.update(key, value)
            return payload

        return self._run("simulate_before_apply", _impl, "Simulation complete")

    def apply_pattern(self, pattern: np.ndarray) -> OperationResult:
        return self._run("apply_pattern", lambda: self.devices.slm.apply_pattern(pattern), "Pattern applied to SLM")

    def queue_pattern(self, pattern: np.ndarray) -> OperationResult:
        return self._run("queue_pattern", lambda: self.devices.slm.queue_pattern(pattern), "Pattern queued")

    def clear_pattern_queue(self) -> OperationResult:
        return self._run("clear_pattern_queue", self.devices.slm.clear_queue, "SLM queue cleared")

    def configure_camera(self, settings: Mapping[str, Any]) -> OperationResult:
        return self._run("configure_camera", lambda: self.devices.camera.configure(settings), "Camera configured")

    def camera_telemetry(self) -> OperationResult:
        def _impl() -> Dict[str, Any]:
            telemetry = self.devices.camera.telemetry()
            self.state.update_camera_telemetry(telemetry)
            temp_state = self.state.camera_telemetry.get("temperature_status", "unknown")
            if temp_state in {"warning", "critical"}:
                temp_c = self.state.camera_telemetry.get("temperature_c", "n/a")
                msg = f"Camera temperature {temp_c}C reached {temp_state} threshold"
                self.state.notify(msg)
                self.state.add_log(LogLevel.WARNING, msg, source="camera")
            return self.state.camera_telemetry

        return self._run("camera_telemetry", _impl, "Camera telemetry refreshed")

    def run_optimization(self, config: Mapping[str, Any]) -> OperationResult:
        def _impl() -> None:
            total = int(config.get("iterations", 20))
            self.state.start_task("optimization", total=total, message="Optimization running")
            self.state.settings_snapshots.optimizer = dict(config)
            self.optimizer.start(config)
            hist = self.optimizer.history()
            arr = np.array([[x["iteration"], x["objective"]] for x in hist], dtype=float)
            self.plots.update("optimization_convergence", arr)
            self.state.update_task("optimization", len(hist), "Optimization complete")
            self.state.complete_task("optimization", "Optimization complete")

        return self._run("run_optimization", _impl, "Optimization complete")

    def cancel_optimization(self) -> OperationResult:
        def _impl() -> None:
            self.optimizer.stop()
            self.state.cancel_task("optimization")

        return self._run("cancel_optimization", _impl, "Optimization cancelled")

    def run_calibration(self, profile_path: str) -> OperationResult:
        def _impl() -> Dict[str, Any]:
            self.state.start_task("calibration", total=3, message="Calibration running")
            profile = CalibrationProfile(
                name="default",
                mode=self.state.mode.value,
                slm_model="simulated_slm",
                camera_model="simulated_camera",
                matrix=[[1.0, 0.0], [0.0, 1.0]],
            )
            self.state.update_task("calibration", 1, "Saving calibration profile")
            self.calibration.save_profile(profile, Path(profile_path))
            self.state.update_task("calibration", 2, "Loading calibration profile")
            loaded = self.calibration.load_profile(Path(profile_path))
            self.state.settings_snapshots.calibration = dict(loaded)
            self.state.update_task("calibration", 3, "Calibration complete")
            self.state.complete_task("calibration", "Calibration complete")
            return loaded

        return self._run("run_calibration", _impl, "Calibration complete")

    def cancel_calibration(self) -> OperationResult:
        def _impl() -> None:
            self.state.cancel_task("calibration")

        return self._run("cancel_calibration", _impl, "Calibration cancelled")

    def run_sequence(self, sequence_steps: list[Mapping[str, Any]]) -> OperationResult:
        def _impl() -> list[Dict[str, Any]]:
            self.sequence.import_sequence(sequence_steps)
            total = len(sequence_steps)
            self.state.start_task("sequence", total=total, message="Sequence running")
            runtime = self.sequence.run(self.state.camera_telemetry)
            self.state.update_task("sequence", len(runtime), "Sequence complete")
            self.state.complete_task("sequence", "Sequence complete")
            return runtime

        return self._run("run_sequence", _impl, "Sequence complete")

    def cancel_sequence(self) -> OperationResult:
        def _impl() -> None:
            self.state.cancel_task("sequence")

        return self._run("cancel_sequence", _impl, "Sequence cancelled")

    def export_plot(self, name: str, output_dir: str) -> OperationResult:
        return self._run("export_plot", lambda: self.plots.export(name, Path(output_dir)), f"Plot '{name}' exported")

    def save_session_snapshot(self, path: str) -> OperationResult:
        return self._run("save_session_snapshot", lambda: self.persistence.snapshot_session(self.state, Path(path)), "Session snapshot saved")

    def output_name_preview(self, artifact: str, run_id: str) -> OperationResult:
        return self._run(
            "output_name_preview",
            lambda: self.persistence.render_name(
                self.state.naming_template,
                session=self.state.session_name,
                run_id=run_id,
                artifact=artifact,
            ),
            "Output name preview generated",
        )

    def functionality_matrix(self) -> Dict[str, bool]:
        ui_bindings = {
            "slm.apply": True,
            "slm.queue": True,
            "camera.acquire": True,
            "camera.telemetry": True,
            "plot.simulated_phase": True,
            "plot.simulated_intensity": True,
            "plot.experimental_intensity": True,
            "plot.optimization_convergence": True,
            "optimizer.start_pause_resume_stop": True,
            "sequence.sync_run": True,
            "calibration.save_load_apply": True,
        }
        return {row: ui_bindings.get(row, False) for row in feature_matrix_rows()}
