"""Typed schemas for user-selectable analytical SLM pattern parameters."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any, ClassVar, Dict, Mapping, Type, Union, get_args, get_origin, get_type_hints


class PatternValidationError(ValueError):
    """Raised when flat pattern parameters cannot be parsed or validated."""


@dataclass(frozen=True)
class SingleGaussianParams:
    """Single focused Gaussian-like spot in k-space."""

    kx: float = 0.0
    ky: float = 0.0

    FIELD_DESCRIPTIONS: ClassVar[Dict[str, str]] = {
        "kx": "Spot kx coordinate in normalized k-space units.",
        "ky": "Spot ky coordinate in normalized k-space units.",
    }

    def __post_init__(self) -> None:
        _validate_range("kx", self.kx, -1.0, 1.0)
        _validate_range("ky", self.ky, -1.0, 1.0)


@dataclass(frozen=True)
class DoubleGaussianParams:
    """Two Gaussian-like spots separated along kx around a center point."""

    center_kx: float = 0.0
    center_ky: float = 0.0
    sep_kxy: float = 0.02

    FIELD_DESCRIPTIONS: ClassVar[Dict[str, str]] = {
        "center_kx": "Center kx coordinate in normalized k-space units.",
        "center_ky": "Center ky coordinate in normalized k-space units.",
        "sep_kxy": "Distance between two spots in normalized k-space units (must be > 0).",
    }

    def __post_init__(self) -> None:
        _validate_range("center_kx", self.center_kx, -1.0, 1.0)
        _validate_range("center_ky", self.center_ky, -1.0, 1.0)
        _validate_positive("sep_kxy", self.sep_kxy)


@dataclass(frozen=True)
class GaussianLatticeParams:
    """Rectangular lattice of Gaussian-like spots in k-space."""

    nx: int = 5
    ny: int = 5
    pitch_x: float = 0.01
    pitch_y: float = 0.01
    center_kx: float = 0.0
    center_ky: float = 0.0

    FIELD_DESCRIPTIONS: ClassVar[Dict[str, str]] = {
        "nx": "Number of lattice sites along x (integer >= 1).",
        "ny": "Number of lattice sites along y (integer >= 1).",
        "pitch_x": "Lattice pitch along x in normalized k-space units (must be > 0).",
        "pitch_y": "Lattice pitch along y in normalized k-space units (must be > 0).",
        "center_kx": "Lattice center kx coordinate in normalized k-space units.",
        "center_ky": "Lattice center ky coordinate in normalized k-space units.",
    }

    def __post_init__(self) -> None:
        _validate_int_min("nx", self.nx, 1)
        _validate_int_min("ny", self.ny, 1)
        _validate_positive("pitch_x", self.pitch_x)
        _validate_positive("pitch_y", self.pitch_y)
        _validate_range("center_kx", self.center_kx, -1.0, 1.0)
        _validate_range("center_ky", self.center_ky, -1.0, 1.0)


@dataclass(frozen=True)
class LaguerreGaussianParams:
    """Laguerre-Gaussian phase mode parameters."""

    l: int = 3
    p: int = 0

    FIELD_DESCRIPTIONS: ClassVar[Dict[str, str]] = {
        "l": "Azimuthal index l (integer, |l| <= 50).",
        "p": "Radial index p (integer >= 0).",
    }

    def __post_init__(self) -> None:
        if abs(self.l) > 50:
            raise PatternValidationError("'l' must satisfy |l| <= 50")
        _validate_int_min("p", self.p, 0)


PatternParams = Union[
    SingleGaussianParams,
    DoubleGaussianParams,
    GaussianLatticeParams,
    LaguerreGaussianParams,
]

PATTERN_PARAM_SCHEMAS: Dict[str, Type[PatternParams]] = {
    "single-gaussian": SingleGaussianParams,
    "double-gaussian": DoubleGaussianParams,
    "gaussian-lattice": GaussianLatticeParams,
    "laguerre-gaussian": LaguerreGaussianParams,
}

PATTERN_DESCRIPTIONS: Dict[str, str] = {
    "single-gaussian": "Single focused Gaussian-like spot.",
    "double-gaussian": "Two Gaussian-like spots separated in k-space.",
    "gaussian-lattice": "Rectangular lattice of Gaussian-like spots.",
    "laguerre-gaussian": "Laguerre-Gaussian phase mode.",
}

_PATTERN_PREFIXES: Dict[str, str] = {
    "single-gaussian": "single_",
    "double-gaussian": "double_",
    "gaussian-lattice": "lattice_",
    "laguerre-gaussian": "lg_",
}

def pattern_params_from_flat_dict(pattern: str, values: Mapping[str, Any]) -> PatternParams:
    """Convert flat CLI/config values into a validated typed pattern schema."""
    schema_cls = PATTERN_PARAM_SCHEMAS.get(pattern)
    if schema_cls is None:
        supported = ", ".join(sorted(PATTERN_PARAM_SCHEMAS))
        raise PatternValidationError(f"Unknown pattern '{pattern}'. Supported: {supported}")

    prefix = _PATTERN_PREFIXES[pattern]
    normalized = {str(k).replace("-", "_"): v for k, v in dict(values).items()}

    kwargs: Dict[str, Any] = {}
    errors = []

    type_hints = get_type_hints(schema_cls)
    for field in fields(schema_cls):
        name = field.name
        raw = None
        has_value = False

        if name in normalized:
            raw = normalized[name]
            has_value = True
        elif f"{prefix}{name}" in normalized:
            raw = normalized[f"{prefix}{name}"]
            has_value = True

        if has_value:
            try:
                kwargs[name] = _coerce_type(raw, type_hints.get(name, field.type))
            except PatternValidationError as exc:
                errors.append(f"{name}: {exc}")
        elif field.default is not MISSING:
            kwargs[name] = field.default
        else:
            errors.append(f"{name}: required value is missing")

    if errors:
        raise PatternValidationError(
            f"Invalid parameters for '{pattern}': " + "; ".join(errors)
        )

    try:
        return schema_cls(**kwargs)
    except PatternValidationError:
        raise
    except Exception as exc:
        raise PatternValidationError(f"Invalid parameters for '{pattern}': {exc}") from exc


def pattern_field_descriptions(pattern: str) -> Dict[str, str]:
    """Return human-readable parameter descriptions for CLI/help generation."""
    schema_cls = PATTERN_PARAM_SCHEMAS[pattern]
    return dict(schema_cls.FIELD_DESCRIPTIONS)


def _coerce_type(raw: Any, expected_type: Any) -> Any:
    if raw is None:
        raise PatternValidationError("value cannot be None")

    origin = get_origin(expected_type)
    if origin is Union:
        for option in get_args(expected_type):
            if option is type(None):
                continue
            try:
                return _coerce_type(raw, option)
            except PatternValidationError:
                continue
        raise PatternValidationError(f"expected one of {get_args(expected_type)}, got {raw!r}")

    if expected_type is int:
        if isinstance(raw, bool):
            raise PatternValidationError("expected int, got bool")
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise PatternValidationError(f"expected int, got {raw!r}") from exc

    if expected_type is float:
        if isinstance(raw, bool):
            raise PatternValidationError("expected float, got bool")
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise PatternValidationError(f"expected float, got {raw!r}") from exc

    return raw


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise PatternValidationError(f"'{name}' must be > 0 (got {value})")


def _validate_range(name: str, value: float, minimum: float, maximum: float) -> None:
    if value < minimum or value > maximum:
        raise PatternValidationError(
            f"'{name}' must be in [{minimum}, {maximum}] (got {value})"
        )


def _validate_int_min(name: str, value: int, minimum: int) -> None:
    if not isinstance(value, int):
        raise PatternValidationError(f"'{name}' must be an integer")
    if value < minimum:
        raise PatternValidationError(f"'{name}' must be >= {minimum} (got {value})")
