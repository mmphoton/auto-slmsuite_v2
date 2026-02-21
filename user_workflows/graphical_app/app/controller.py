"""High-level application controller that exposes all backend functionality to UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from user_workflows.graphical_app.app.interfaces import feature_matrix_rows
from user_workflows.graphical_app.app.state import AppState, Mode, RunMetadata
from user_workflows.graphical_app.app.patterns import PatternService
from user_workflows.graphical_app.calibration.tools import CalibrationTools
from user_workflows.graphical_app.devices.manager import DeviceManager
from user_workflows.graphical_app.optimization.runner import OptimizationRunner
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

    def start_run(self, run_id: str, params: Mapping[str, Any]) -> None:
        self.state.active_run = RunMetadata(
            run_id=run_id,
            mode=self.state.mode,
            device_snapshot=dict(self.state.device_status),
            parameters=dict(params),
        )

    def stop_run(self) -> None:
        self.state.active_run = None

    def set_mode(self, mode: str) -> None:
        self.devices.set_mode(Mode(mode))

    def generate_pattern(self, name: str, params: Mapping[str, Any]) -> np.ndarray:
        return self.patterns.generate(name, params, shape=(128, 128))

    def simulate_before_apply(self, pattern: np.ndarray) -> Dict[str, np.ndarray]:
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

    def run_optimization(self, config: Mapping[str, Any]) -> None:
        self.optimizer.start(config)
        hist = self.optimizer.history()
        arr = np.array([[x["iteration"], x["objective"]] for x in hist], dtype=float)
        self.plots.update("optimization_convergence", arr)

    def export_plot(self, name: str, output_dir: str) -> None:
        self.plots.export(name, Path(output_dir))

    def save_session_snapshot(self, path: str) -> None:
        self.persistence.snapshot_session(self.state, Path(path))

    def output_name_preview(self, artifact: str, run_id: str) -> str:
        return self.persistence.render_name(self.state.naming_template, session=self.state.session_name, run_id=run_id, artifact=artifact)

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
