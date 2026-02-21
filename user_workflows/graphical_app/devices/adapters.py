"""Simulation and hardware adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np

from user_workflows.graphical_app.app.interfaces import CameraInterface, SLMInterface


class HardwareIntegrationError(NotImplementedError):
    """Raised when hardware adapters are invoked without an integration backend."""


@dataclass
class SimulationSettings:
    noise_sigma: float = 0.02
    blur_sigma: float = 1.2
    aberration_strength: float = 0.1
    latency_ms: float = 5.0


class SimulatedSLM(SLMInterface):
    def __init__(self) -> None:
        self.connected = False
        self.active_pattern = np.zeros((128, 128))
        self.queue = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def apply_pattern(self, pattern: np.ndarray) -> None:
        self.active_pattern = pattern

    def queue_pattern(self, pattern: np.ndarray) -> None:
        self.queue.append(pattern)

    def clear_queue(self) -> None:
        self.queue.clear()


class SimulatedCamera(CameraInterface):
    def __init__(self, slm: SimulatedSLM, settings: SimulationSettings | None = None) -> None:
        self.connected = False
        self.slm = slm
        self.settings = settings or SimulationSettings()
        self._camera_settings: Dict[str, Any] = {
            "exposure_ms": 10,
            "gain": 1.0,
            "roi": [0, 0, 128, 128],
            "binning": 1,
            "trigger": "internal",
            "fps": 30,
            "acquisition_mode": "single",
        }

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def configure(self, settings: Mapping[str, Any]) -> None:
        self._camera_settings.update(dict(settings))

    def acquire_frame(self) -> np.ndarray:
        phase = self.slm.active_pattern
        intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phase))))
        intensity /= max(float(intensity.max()), 1e-9)
        noise = np.random.normal(0, self.settings.noise_sigma, intensity.shape)
        return np.clip(intensity + noise, 0, 1)

    def telemetry(self) -> Dict[str, Any]:
        return {
            "temperature_c": -64.0 + float(np.random.normal(0.0, 0.4)),
            "settings": dict(self._camera_settings),
        }


class HardwareSLM(SLMInterface):
    """Explicit hardware adapter; wire to a real backend before hardware mode use."""

    def __init__(self) -> None:
        self.connected = False

    def connect(self) -> None:
        raise HardwareIntegrationError("Hardware SLM integration is not implemented.")

    def disconnect(self) -> None:
        self.connected = False

    def apply_pattern(self, pattern: np.ndarray) -> None:
        raise HardwareIntegrationError("Hardware SLM integration is not implemented.")

    def queue_pattern(self, pattern: np.ndarray) -> None:
        raise HardwareIntegrationError("Hardware SLM integration is not implemented.")

    def clear_queue(self) -> None:
        raise HardwareIntegrationError("Hardware SLM integration is not implemented.")


class HardwareCamera(CameraInterface):
    """Explicit hardware adapter; wire to a real backend before hardware mode use."""

    def __init__(self, _slm: SLMInterface) -> None:
        self.connected = False

    def connect(self) -> None:
        raise HardwareIntegrationError("Hardware camera integration is not implemented.")

    def disconnect(self) -> None:
        self.connected = False

    def configure(self, settings: Mapping[str, Any]) -> None:
        raise HardwareIntegrationError("Hardware camera integration is not implemented.")

    def acquire_frame(self) -> np.ndarray:
        raise HardwareIntegrationError("Hardware camera integration is not implemented.")

    def telemetry(self) -> Dict[str, Any]:
        raise HardwareIntegrationError("Hardware camera integration is not implemented.")
