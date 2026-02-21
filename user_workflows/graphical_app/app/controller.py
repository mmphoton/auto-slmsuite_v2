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
from user_workflows.graphical_app.devices.camera_settings import camera_settings_schema, parse_camera_settings
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


    def _compose_pattern_with_blaze(self, base_pattern: np.ndarray) -> np.ndarray:
        composed = np.asarray(base_pattern, dtype=float)
        if not self.state.blaze.enabled:
            return np.mod(composed, 2 * np.pi)

        ny, nx = composed.shape
        y, x = np.indices((ny, nx))
        xx = (x - nx / 2) / nx
        yy = (y - ny / 2) / ny
        blaze_phase = 2 * np.pi * (self.state.blaze.kx * xx + self.state.blaze.ky * yy)
        composed = composed + blaze_phase

        if self.state.blaze.offset is not None:
            composed = composed + float(self.state.blaze.offset)
        if self.state.blaze.scale is not None:
            composed = composed * float(self.state.blaze.scale)

        return np.mod(composed, 2 * np.pi)

    def configure_blaze(self, settings: Mapping[str, Any]) -> OperationResult:
        def _impl() -> dict[str, Any]:
            enabled = bool(settings.get("enabled", False))
            kx = float(settings.get("kx", 0.0))
            ky = float(settings.get("ky", 0.0))
            if not -1.0 <= kx <= 1.0:
                raise ValueError("blaze kx must be within [-1.0, 1.0]")
            if not -1.0 <= ky <= 1.0:
                raise ValueError("blaze ky must be within [-1.0, 1.0]")

            offset_raw = settings.get("offset")
            scale_raw = settings.get("scale")
            offset = None if offset_raw in (None, "") else float(offset_raw)
            scale = None if scale_raw in (None, "") else float(scale_raw)
            if scale is not None and scale <= 0:
                raise ValueError("blaze scale must be > 0 when provided")

            self.state.blaze.enabled = enabled
            self.state.blaze.kx = kx
            self.state.blaze.ky = ky
            self.state.blaze.offset = offset
            self.state.blaze.scale = scale
            return {
                "enabled": enabled,
                "kx": kx,
                "ky": ky,
                "offset": offset,
                "scale": scale,
            }

        return self._run("configure_blaze", _impl, "Blaze settings updated")

    def start_run(self, run_id: str, params: Mapping[str, Any]) -> OperationResult:
        def _impl() -> None:
            self.state.active_run = RunMetadata(
                run_id=run_id,
                mode=self.state.mode,
                device_snapshot=dict(self.state.device_status),
                parameters=dict(params),
                calibration_profile=self.state.settings_snapshots.calibration.get("profile_name"),
                optimizer=dict(self.state.settings_snapshots.optimizer),
                camera_settings=dict(self.state.settings_snapshots.camera),
                blaze={
                    "enabled": self.state.blaze.enabled,
                    "kx": self.state.blaze.kx,
                    "ky": self.state.blaze.ky,
                    "offset": self.state.blaze.offset,
                    "scale": self.state.blaze.scale,
                },
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

    def _refresh_simulated_plots(self, composed_pattern: np.ndarray) -> Dict[str, np.ndarray]:
        simulated_phase = composed_pattern
        simulated_intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * composed_pattern))))
        simulated_intensity /= max(float(simulated_intensity.max()), 1e-9)
        payload = {
            "simulated_phase": simulated_phase,
            "simulated_intensity": simulated_intensity,
        }
        for key, value in payload.items():
            self.plots.update(key, value)
        return payload

    def _wgs_target_intensity(self, phase: np.ndarray) -> np.ndarray:
        target = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phase))))
        target /= max(float(target.max()), 1e-9)
        return target

    def _optimization_feedback(self, candidate_phase: np.ndarray, _iteration: int) -> np.ndarray:
        self.devices.slm.apply_pattern(candidate_phase)
        if self.state.mode == Mode.HARDWARE:
            return self.devices.camera.acquire_frame()
        return self._wgs_target_intensity(candidate_phase)

    def simulate_before_apply(self, pattern: np.ndarray) -> OperationResult:
        def _impl() -> Dict[str, np.ndarray]:
            composed_pattern = self._compose_pattern_with_blaze(pattern)
            self.devices.slm.apply_pattern(composed_pattern)
            experimental = self.devices.camera.acquire_frame()
            payload = self._refresh_simulated_plots(composed_pattern)
            payload["experimental_intensity"] = experimental
            self.plots.update("experimental_intensity", experimental)
            return payload

        return self._run("simulate_before_apply", _impl, "Simulation complete")

    def apply_pattern(self, pattern: np.ndarray) -> OperationResult:
        def _impl() -> np.ndarray:
            composed = self._compose_pattern_with_blaze(pattern)
            self.devices.slm.apply_pattern(composed)
            self._refresh_simulated_plots(composed)
            return composed

        return self._run(
            "apply_pattern",
            _impl,
            "Pattern applied to SLM",
        )

    def queue_pattern(self, pattern: np.ndarray) -> OperationResult:
        return self._run(
            "queue_pattern",
            lambda: self.devices.slm.queue_pattern(self._compose_pattern_with_blaze(pattern)),
            "Pattern queued",
        )

    def clear_pattern_queue(self) -> OperationResult:
        return self._run("clear_pattern_queue", self.devices.slm.clear_queue, "SLM queue cleared")

    def camera_settings_schema(self) -> OperationResult:
        return self._run("camera_settings_schema", camera_settings_schema, "Camera settings schema loaded")

    def configure_camera(self, settings: Mapping[str, Any]) -> OperationResult:
        def _impl() -> Dict[str, Any]:
            parsed = parse_camera_settings(settings)
            payload = parsed.to_payload()
            self.devices.camera.configure(payload)
            self.state.settings_snapshots.camera = dict(payload)
            return payload

        return self._run("configure_camera", _impl, "Camera configured")

    def read_camera_settings(self) -> OperationResult:
        return self._run(
            "read_camera_settings",
            lambda: dict(self.state.settings_snapshots.camera),
            "Read current camera settings",
        )

    def camera_telemetry(self) -> OperationResult:
        def _impl() -> Dict[str, Any]:
            telemetry = self.devices.camera.telemetry()
            self.state.update_camera_telemetry(telemetry)
            temp_state = self.state.camera_telemetry.get("temperature_status", "unknown")
            temp_c = self.state.camera_telemetry.get("temperature_c", "n/a")
            self.state.add_log(LogLevel.INFO, f"Camera temperature update: {temp_c}C ({temp_state})", source="camera")
            if temp_state in {"warning", "critical"}:
                msg = f"Camera temperature {temp_c}C reached {temp_state} threshold"
                self.state.notify(msg)
                self.state.add_log(LogLevel.WARNING, msg, source="camera")
            return self.state.camera_telemetry

        return self._run("camera_telemetry", _impl, "Camera telemetry refreshed")

    def run_optimization(self, config: Mapping[str, Any]) -> OperationResult:
        def _impl() -> Dict[str, Any]:
            wgs = dict(config.get("wgs", {})) if isinstance(config.get("wgs"), Mapping) else {}
            total = int(wgs.get("max_iterations", config.get("iterations", 20)))
            self.state.start_task("optimization", total=total, message="Optimization running")
            self.state.settings_snapshots.optimizer = dict(config)

            initial_phase = self.devices.slm.active_pattern if hasattr(self.devices.slm, "active_pattern") else np.zeros((128, 128), dtype=float)
            target = self._wgs_target_intensity(initial_phase)
            self.plots.update("optimization_phase_before", initial_phase)
            self.plots.update("optimization_intensity_before", self._wgs_target_intensity(initial_phase))

            self.optimizer.start(
                config,
                initial_phase=initial_phase,
                target_intensity=target,
                feedback_provider=self._optimization_feedback,
            )
            self.optimizer.run_to_completion()
            hist = self.optimizer.history()
            arr = np.array([[x["iteration"], x["objective"]] for x in hist], dtype=float) if hist else np.zeros((0, 2), dtype=float)
            self.plots.update("optimization_convergence", arr)

            final_phase = self.optimizer.current_phase()
            if final_phase is not None:
                self.plots.update("optimization_phase_after", final_phase)
                self.plots.update("optimization_intensity_after", self._wgs_target_intensity(final_phase))

            progress = dict(self.optimizer.progress())
            self.state.update_task("optimization", int(progress.get("iteration", len(hist))), "Optimization complete")
            self.state.complete_task("optimization", "Optimization complete")
            return {"history": hist, "progress": progress}

        return self._run("run_optimization", _impl, "Optimization complete")

    def start_optimization(self, config: Mapping[str, Any]) -> OperationResult:
        return self.run_optimization(config)

    def pause_optimization(self) -> OperationResult:
        def _impl() -> Dict[str, Any]:
            self.optimizer.pause()
            progress = dict(self.optimizer.progress())
            self.state.update_task("optimization", int(progress.get("iteration", 0)), "Optimization paused")
            return progress

        return self._run("pause_optimization", _impl, "Optimization paused")

    def resume_optimization(self) -> OperationResult:
        def _impl() -> Dict[str, Any]:
            self.optimizer.resume()
            self.optimizer.run_to_completion()
            hist = self.optimizer.history()
            arr = np.array([[x["iteration"], x["objective"]] for x in hist], dtype=float) if hist else np.zeros((0, 2), dtype=float)
            self.plots.update("optimization_convergence", arr)
            progress = dict(self.optimizer.progress())
            self.state.update_task("optimization", int(progress.get("iteration", len(hist))), "Optimization resumed")
            if not progress.get("running", False):
                self.state.complete_task("optimization", "Optimization complete")
            return progress

        return self._run("resume_optimization", _impl, "Optimization resumed")

    def cancel_optimization(self) -> OperationResult:
        def _impl() -> None:
            self.optimizer.stop()
            self.state.cancel_task("optimization")

        return self._run("cancel_optimization", _impl, "Optimization cancelled")

    def stop_optimization(self) -> OperationResult:
        return self.cancel_optimization()

    def optimization_progress(self) -> OperationResult:
        return self._run("optimization_progress", lambda: dict(self.optimizer.progress()), "Optimization progress fetched")

    def export_optimization_history(self, out_path: str) -> OperationResult:
        def _impl() -> Dict[str, Any]:
            run_meta = {}
            if self.state.active_run is not None:
                run_meta = {
                    "run_id": self.state.active_run.run_id,
                    "mode": self.state.active_run.mode.value,
                    "parameters": dict(self.state.active_run.parameters),
                }
            out = Path(out_path)
            self.optimizer.export_history(out, run_metadata=run_meta)
            return {"csv": str(out), "json": str(out.with_suffix(".json"))}

        return self._run("export_optimization_history", _impl, "Optimization history exported")

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

    def configure_plot(self, name: str, settings: Mapping[str, Any]) -> OperationResult:
        return self._run("configure_plot", lambda: self.plots.configure(name, settings), f"Plot '{name}' configured")

    def zoom_plot(self, name: str, factor: float) -> OperationResult:
        return self._run("zoom_plot", lambda: self.plots.zoom(name, factor), f"Plot '{name}' zoom updated")

    def pan_plot(self, name: str, dx_fraction: float, dy_fraction: float) -> OperationResult:
        return self._run("pan_plot", lambda: self.plots.pan(name, dx_fraction, dy_fraction), f"Plot '{name}' pan updated")

    def reset_plot(self, name: str) -> OperationResult:
        return self._run("reset_plot", lambda: self.plots.reset_view(name), f"Plot '{name}' reset")

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
            "plot.optimization_phase_before_after": True,
            "plot.optimization_intensity_before_after": True,
            "optimizer.start_pause_resume_stop": True,
            "sequence.sync_run": True,
            "calibration.save_load_apply": True,
        }
        return {row: ui_bindings.get(row, False) for row in feature_matrix_rows()}
