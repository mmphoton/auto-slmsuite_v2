"""Pattern interfaces and registration for SLM analytical pattern generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PatternResult:
    """Container returned by pattern builders."""

    phase: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePattern(ABC):
    """Base class for all user-selectable SLM pattern families."""

    name: str

    @abstractmethod
    def build(self, args, slm) -> PatternResult:
        """Return a phase pattern and useful metadata."""


PATTERN_REGISTRY: dict[str, type[BasePattern]] = {}


def register_pattern(cls: type[BasePattern]) -> type[BasePattern]:
    """Register a pattern class by its canonical CLI name."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"Pattern class {cls.__name__} must define 'name'")
    PATTERN_REGISTRY[cls.name] = cls
    return cls


def get_pattern(name: str) -> BasePattern:
    try:
        return PATTERN_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown pattern '{name}'") from exc
