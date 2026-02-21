"""Core abstractions for building reusable user workflow patterns.

This module intentionally does not depend on CLI parsing so pattern builders can
be imported and tested in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class PatternContext:
    """Shared runtime dependencies and solver settings for pattern generation."""

    slm: Any
    cameraslm: Any | None = None

    # Optional depth/LUT correction inputs.
    phase_lut: np.ndarray | None = None
    depth_correction: Any | None = None

    # Shared solver configuration.
    holo_method: str = "WGS-Kim"
    holo_maxiter: int = 30
    feedback: str = "computational"
    stat_groups: tuple[str, ...] = ("computational",)


@dataclass(slots=True)
class PatternResult:
    """Output payload for generated patterns."""

    phase: np.ndarray
    spot_vectors: np.ndarray | None = None
    expected_farfield: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class BasePattern(ABC):
    """Abstract interface that all workflow patterns should implement."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the registry key for this pattern."""

    @classmethod
    @abstractmethod
    def schema(cls) -> Any:
        """Return a parameter schema/model for validating pattern inputs."""

    @abstractmethod
    def build(self, context: PatternContext, params: Any) -> PatternResult:
        """Build a phase pattern from runtime context and validated parameters."""
