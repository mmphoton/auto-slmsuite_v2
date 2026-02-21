"""Pattern registry for runtime-selectable SLM workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict

_PATTERNS: Dict[str, object] = {}
_LAZY_PATTERNS: Dict[str, str] = {}


class PatternRegistryError(ValueError):
    """Raised when registry operations are invalid."""


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
