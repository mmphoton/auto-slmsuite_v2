"""Optimization runner with WGS support, lifecycle controls, and history export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

import numpy as np

from user_workflows.graphical_app.app.interfaces import OptimizationRunnerInterface


@dataclass
class OptimizationRunner(OptimizationRunnerInterface):
    _history: List[Dict[str, float]] = field(default_factory=list)
    _running: bool = False
    _paused: bool = False
    _config: Dict[str, Any] = field(default_factory=dict)
    _phase: np.ndarray | None = None
    _target_intensity: np.ndarray | None = None
    _feedback_provider: Callable[[np.ndarray, int], np.ndarray] | None = None
    _iteration: int = 0
    _max_iterations: int = 0
    _gain: float = 0.2
    _last_feedback: np.ndarray | None = None

    def start(
        self,
        config: Mapping[str, Any],
        initial_phase: np.ndarray,
        target_intensity: np.ndarray,
        feedback_provider: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ) -> None:
        self._history.clear()
        self._config = dict(config)
        self._running = True
        self._paused = False
        self._phase = np.asarray(initial_phase, dtype=float).copy()
        self._target_intensity = np.asarray(target_intensity, dtype=float).copy()
        self._feedback_provider = feedback_provider
        self._iteration = 0

        wgs = dict(config.get("wgs", {})) if isinstance(config.get("wgs"), Mapping) else {}
        requested = wgs.get("max_iterations", config.get("iterations", 20))
        self._max_iterations = max(1, int(requested))
        self._gain = float(wgs.get("gain", config.get("gain", 0.2)))

    def _simulated_intensity(self, phase: np.ndarray) -> np.ndarray:
        intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phase))))
        intensity /= max(float(intensity.max()), 1e-9)
        return intensity

    def step(self) -> bool:
        if not self._running or self._paused:
            return False
        if self._phase is None or self._target_intensity is None:
            return False
        if self._iteration >= self._max_iterations:
            self._running = False
            return False

        if self._feedback_provider is None:
            measured = self._simulated_intensity(self._phase)
        else:
            measured = np.asarray(self._feedback_provider(self._phase, self._iteration), dtype=float)
        measured = measured / max(float(np.max(measured)), 1e-9)
        self._last_feedback = measured

        error = measured - self._target_intensity
        grad = np.real(np.fft.ifft2(np.fft.ifftshift(error)))
        grad = grad[: self._phase.shape[0], : self._phase.shape[1]]
        self._phase = np.mod(self._phase - self._gain * grad, 2 * np.pi)

        objective = float(np.mean(np.square(error)))
        regularization = float(np.mean(np.square(grad)))
        self._history.append(
            {
                "iteration": float(self._iteration),
                "objective": objective,
                "component_regularization": regularization,
                "feedback_mean": float(np.mean(measured)),
            }
        )
        self._iteration += 1

        if self._iteration >= self._max_iterations:
            self._running = False
        return True

    def run_to_completion(self) -> None:
        while self._running and not self._paused:
            if not self.step():
                break

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._running = False

    def progress(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "paused": self._paused,
            "iteration": self._iteration,
            "max_iterations": self._max_iterations,
            "percent": 100.0 * self._iteration / max(self._max_iterations, 1),
        }

    def current_phase(self) -> np.ndarray | None:
        return None if self._phase is None else self._phase.copy()

    def last_feedback(self) -> np.ndarray | None:
        return None if self._last_feedback is None else self._last_feedback.copy()

    def history(self) -> List[Mapping[str, float]]:
        return list(self._history)

    def export_history(self, out: Path, run_metadata: Mapping[str, Any] | None = None) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = np.array([[h["iteration"], h["objective"], h["component_regularization"], h["feedback_mean"]] for h in self._history])
        np.savetxt(out, arr, delimiter=",", header="iteration,objective,component_regularization,feedback_mean", comments="")

        sidecar = out.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "config": self._config,
                    "progress": self.progress(),
                    "history_points": len(self._history),
                    "run_metadata": dict(run_metadata or {}),
                },
                indent=2,
            )
        )
