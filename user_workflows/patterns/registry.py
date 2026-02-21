"""Pattern registry for runtime-selectable SLM workflows."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

import numpy as np


class PatternRegistryError(ValueError):
    """Raised when registry operations are invalid."""


@dataclass
class _Entry:
    builder: Callable[[dict[str, Any], Any], Any]
    schema: dict[str, Any]


class PatternRegistry:
    """In-memory registry of pattern builders and parameter schemas."""

    def __init__(self) -> None:
        self._patterns: dict[str, _Entry] = {}

    def register(self, *, name: str, builder: Callable[[dict[str, Any], Any], Any], schema: dict[str, Any]) -> None:
        if name in self._patterns:
            raise PatternRegistryError(f"Pattern '{name}' is already registered.")
        self._patterns[name] = _Entry(builder=builder, schema=schema)

    def list_patterns(self) -> list[str]:
        return sorted(self._patterns.keys())

    def get(self, name: str) -> _Entry:
        if name not in self._patterns:
            available = ", ".join(self.list_patterns()) or "<none>"
            raise PatternRegistryError(f"Unknown pattern '{name}'. Registered patterns: {available}.")
        return self._patterns[name]

    def validate(self, name: str, params: dict[str, Any]) -> None:
        entry = self.get(name)
        _validate_against_schema(params, entry.schema)

    def default_params(self, name: str) -> dict[str, Any]:
        entry = self.get(name)
        properties = entry.schema.get("properties", {})
        return {
            key: spec["default"]
            for key, spec in properties.items()
            if isinstance(spec, dict) and "default" in spec
        }

    def build(self, name: str, params: dict[str, Any], context: Any):
        self.validate(name, params)
        phase = np.asarray(self.get(name).builder(params, context), dtype=float)
        return np.mod(phase, 2 * np.pi)


_PATTERNS: dict[str, object] = {}
_LAZY_PATTERNS: dict[str, str] = {}
registry = PatternRegistry()


def _ensure_unique(name: str):
    if name in _PATTERNS:
        raise PatternRegistryError(
            f"Pattern '{name}' is already registered with {_PATTERNS[name]!r}; duplicate registration is not allowed."
        )
    if name in _LAZY_PATTERNS:
        raise PatternRegistryError(
            f"Pattern '{name}' is already registered for lazy import as '{_LAZY_PATTERNS[name]}'; duplicate registration is not allowed."
        )


def register_pattern(cls_or_factory=None, *, name: str | None = None):
    """Register a pattern class/factory by name.

    Can be used as a decorator or as a helper function.
    """

    def _register(target):
        pattern_name = name or getattr(target, "pattern_name", None) or target.__name__
        _ensure_unique(pattern_name)
        _PATTERNS[pattern_name] = target
        return target

    if cls_or_factory is None:
        return _register
    return _register(cls_or_factory)


def register_lazy_pattern(name: str, import_path: str):
    """Register a lazily-imported pattern using ``module:attribute``."""
    if ":" not in import_path:
        raise PatternRegistryError(
            f"Lazy pattern '{name}' has invalid import path '{import_path}'; expected 'module:attribute'."
        )
    _ensure_unique(name)
    _LAZY_PATTERNS[name] = import_path


def _materialize_lazy(name: str):
    import_path = _LAZY_PATTERNS.pop(name)
    module_name, attr_name = import_path.split(":", maxsplit=1)
    module = import_module(module_name)
    target = getattr(module, attr_name)
    _PATTERNS[name] = target


def get_pattern(name: str):
    """Fetch a registered pattern class/factory by name."""
    if name in _LAZY_PATTERNS:
        _materialize_lazy(name)

    if name not in _PATTERNS:
        available = ", ".join(list_patterns()) or "<none>"
        raise PatternRegistryError(f"Unknown pattern '{name}'. Registered patterns: {available}.")
    return _PATTERNS[name]


def list_patterns():
    """List all registered pattern names, including lazily registered patterns."""
    return sorted({*_PATTERNS.keys(), *_LAZY_PATTERNS.keys()})


def get_default_registry() -> PatternRegistry:
    return registry


def _validate_against_schema(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    if schema.get("type") != "object":
        raise PatternRegistryError("Only object schemas are supported.")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for key in required:
        if key not in payload:
            raise PatternRegistryError(f"Missing required parameter '{key}'.")

    if not schema.get("additionalProperties", True):
        extra = set(payload) - set(properties)
        if extra:
            raise PatternRegistryError(f"Unknown parameters: {', '.join(sorted(extra))}.")

    for key, value in payload.items():
        rules = properties.get(key)
        if not isinstance(rules, dict):
            continue
        expected_type = rules.get("type")
        if expected_type == "number" and not isinstance(value, (int, float)):
            raise PatternRegistryError(f"'{key}' must be a number.")

        minimum = rules.get("minimum")
        maximum = rules.get("maximum")
        if minimum is not None and value < minimum:
            raise PatternRegistryError(f"'{key}' must be >= {minimum}.")
        if maximum is not None and value > maximum:
            raise PatternRegistryError(f"'{key}' must be <= {maximum}.")


registry.register(
    name="flat-phase",
    builder=lambda _params, context: np.zeros(context.shape, dtype=float),
    schema={"type": "object", "properties": {}, "additionalProperties": False},
)
