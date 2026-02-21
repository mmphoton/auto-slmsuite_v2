"""Core interfaces for the graphical application layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence

import numpy as np




@dataclass
class OperationError:
    code: str
    details: Dict[str, Any] | None = None


@dataclass
class OperationResult:
    success: bool
    message: str
    payload: Any | None = None
    error: OperationError | None = None


def success_result(message: str, payload: Any | None = None) -> OperationResult:
    return OperationResult(success=True, message=message, payload=payload)


def failure_result(message: str, code: str = "operation_error", details: Dict[str, Any] | None = None) -> OperationResult:
    return OperationResult(success=False, message=message, error=OperationError(code=code, details=details))


@dataclass
class PlotExport:
    image_path: Path
    data_path: Path
    metadata_path: Path


class SLMInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def apply_pattern(self, pattern: np.ndarray) -> None: ...

    @abstractmethod
    def queue_pattern(self, pattern: np.ndarray) -> None: ...

    @abstractmethod
    def clear_queue(self) -> None: ...


class CameraInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def configure(self, settings: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def acquire_frame(self) -> np.ndarray: ...

    @abstractmethod
    def telemetry(self) -> Dict[str, Any]: ...


class PatternGeneratorInterface(ABC):
    @abstractmethod
    def available_patterns(self) -> Sequence[str]: ...

    @abstractmethod
    def schema_for(self, pattern_name: str) -> Mapping[str, Any]: ...

    @abstractmethod
    def generate(self, pattern_name: str, params: Mapping[str, Any]) -> np.ndarray: ...


class OptimizationRunnerInterface(ABC):
    @abstractmethod
    def start(self, config: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def resume(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def history(self) -> List[Mapping[str, float]]: ...


class PlotBackendInterface(ABC):
    @abstractmethod
    def update(self, plot_name: str, data: np.ndarray) -> None: ...

    @abstractmethod
    def configure(self, plot_name: str, settings: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def export(self, plot_name: str, output_dir: Path) -> PlotExport: ...


class PersistenceInterface(Protocol):
    def save_json(self, path: Path, data: Mapping[str, Any]) -> None: ...

    def load_json(self, path: Path) -> Dict[str, Any]: ...

    def save_array(self, path: Path, data: np.ndarray) -> None: ...


class PluginInterface(Protocol):
    name: str
    kind: str

    def schema(self) -> Mapping[str, Any]: ...

    def run(self, **kwargs: Any) -> Any: ...


def feature_matrix_rows() -> Iterable[str]:
    return [
        "slm.apply",
        "slm.queue",
        "camera.acquire",
        "camera.telemetry",
        "plot.simulated_phase",
        "plot.simulated_intensity",
        "plot.experimental_intensity",
        "plot.optimization_convergence",
        "optimizer.start_pause_resume_stop",
        "sequence.sync_run",
        "calibration.save_load_apply",
    ]
