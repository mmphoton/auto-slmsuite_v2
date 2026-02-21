"""Optimization runner with WGS support, lifecycle controls, and history export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

import numpy as np

from user_workflows.graphical_app.app.interfaces import OptimizationRunnerInterface
from user_workflows.graphical_app.optimization.targets import TargetDefinition


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
    _target_definition: TargetDefinition | None = None
    _ratio_mode: str = "simulation"
    _objective_weights: Dict[str, float] = field(default_factory=lambda: {"intensity": 1.0, "ratio": 0.0, "regularization": 0.05})

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
        self._ratio_mode = str(config.get("ratio_mode", "simulation"))

        weights = dict(config.get("objective_weights", {})) if isinstance(config.get("objective_weights"), Mapping) else {}
        self._objective_weights = {
            "intensity": float(weights.get("intensity", 1.0)),
            "ratio": float(weights.get("ratio", 0.0)),
            "regularization": float(weights.get("regularization", 0.05)),
        }
        self._target_definition = TargetDefinition.from_config(config, fallback_shape=self._phase.shape)

    def _simulated_intensity(self, phase: np.ndarray) -> np.ndarray:
        intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phase))))
        intensity /= max(float(intensity.max()), 1e-9)
        return intensity

    def _measured_ratios(self, measured: np.ndarray) -> np.ndarray:
        if self._target_definition is None:
            return np.ones((1,), dtype=float)

        h, w = measured.shape
        samples: list[float] = []
        for x, y in self._target_definition.beam_positions:
            xi = int(np.clip(round(x), 0, max(w - 1, 0)))
            yi = int(np.clip(round(y), 0, max(h - 1, 0)))
            samples.append(float(measured[yi, xi]))

        vec = np.asarray(samples, dtype=float)
        denom = float(np.sum(vec))
        if denom <= 1e-12:
            return np.zeros_like(vec)
        return vec / denom

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

        intensity_error = measured - self._target_intensity
        grad = np.real(np.fft.ifft2(np.fft.ifftshift(intensity_error)))
        grad = grad[: self._phase.shape[0], : self._phase.shape[1]]

        desired = np.asarray(self._target_definition.desired_ratios if self._target_definition else [1.0], dtype=float)
        measured_ratios = self._measured_ratios(measured)
        n = min(len(desired), len(measured_ratios))
        ratio_error = measured_ratios[:n] - desired[:n] if n else np.zeros((1,), dtype=float)
        ratio_mse = float(np.mean(np.square(ratio_error))) if ratio_error.size else 0.0
        ratio_scalar = float(np.mean(ratio_error)) if ratio_error.size else 0.0

        weighted_error = self._objective_weights["intensity"] * intensity_error + self._objective_weights["ratio"] * ratio_scalar
        weighted_grad = np.real(np.fft.ifft2(np.fft.ifftshift(weighted_error)))
        weighted_grad = weighted_grad[: self._phase.shape[0], : self._phase.shape[1]]
        self._phase = np.mod(self._phase - self._gain * weighted_grad, 2 * np.pi)

        intensity_mse = float(np.mean(np.square(intensity_error)))
        regularization = float(np.mean(np.square(grad)))
        objective = (
            self._objective_weights["intensity"] * intensity_mse
            + self._objective_weights["ratio"] * ratio_mse
            + self._objective_weights["regularization"] * regularization
        )
        self._history.append(
            {
                "iteration": float(self._iteration),
                "objective": objective,
                "component_intensity": intensity_mse,
                "component_ratio": ratio_mse,
                "component_regularization": regularization,
                "feedback_mean": float(np.mean(measured)),
                "ratio_mode_camera": 1.0 if self._ratio_mode == "camera" else 0.0,
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
        arr = np.array(
            [
                [
                    h["iteration"],
                    h["objective"],
                    h["component_intensity"],
                    h["component_ratio"],
                    h["component_regularization"],
                    h["feedback_mean"],
                ]
                for h in self._history
            ]
        )
        np.savetxt(
            out,
            arr,
            delimiter=",",
            header="iteration,objective,component_intensity,component_ratio,component_regularization,feedback_mean",
            comments="",
        )

        sidecar = out.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "config": self._config,
                    "progress": self.progress(),
                    "history_points": len(self._history),
                    "run_metadata": dict(run_metadata or {}),
                    "target_definition": None if self._target_definition is None else self._target_definition.to_payload(),
                    "ratio_mode": self._ratio_mode,
                    "objective_weights": dict(self._objective_weights),
                },
                indent=2,
            )
        )
