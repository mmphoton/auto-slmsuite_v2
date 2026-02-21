"""Optimization runner with start/pause/resume/stop and history export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from user_workflows.graphical_app.app.interfaces import OptimizationRunnerInterface


@dataclass
class OptimizationRunner(OptimizationRunnerInterface):
    _history: List[Dict[str, float]] = field(default_factory=list)
    _running: bool = False
    _paused: bool = False

    def start(self, config: Mapping[str, Any]) -> None:
        self._history.clear()
        self._running = True
        self._paused = False
        value = float(config.get("initial", 1.0))
        for idx in range(int(config.get("iterations", 20))):
            if not self._running or self._paused:
                break
            value *= 0.9
            self._history.append({"iteration": float(idx), "objective": value, "component_regularization": value * 0.1})

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._running = False

    def history(self) -> List[Mapping[str, float]]:
        return list(self._history)

    def export_history(self, out: Path) -> None:
        arr = np.array([[h["iteration"], h["objective"], h["component_regularization"]] for h in self._history])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, arr, delimiter=",", header="iteration,objective,component_regularization", comments="")
