"""Composite pattern utilities for combining multiple pattern generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PatternCallable = Callable[..., Any]


@dataclass(frozen=True)
class CompositeResult:
    """Container for composite phase output and diagnostics metadata."""

    phase: np.ndarray
    metadata: dict[str, Any]


class CompositePattern:
    """Compose an ordered list of child pattern generators.

    Parameters
    ----------
    children:
        Ordered sequence of child pattern callables. Each callable should return
        either a phase ndarray or a mapping with at least a ``phase`` key.
    mode:
        Composition mode. Supported values:
          - ``phase_add_wrap``: sum child phases and wrap in ``[0, 2π)``.
          - ``weighted_sum``: weighted linear sum of child phases.
    weights:
        Optional weights for ``weighted_sum``. If omitted, uniform weights are used.
    spot_union:
        If ``True``, include an optional union of child spot objectives in metadata.
    """

    SUPPORTED_MODES = {"phase_add_wrap", "weighted_sum"}

    def __init__(
        self,
        children: Sequence[PatternCallable],
        mode: str = "phase_add_wrap",
        weights: Sequence[float] | None = None,
        spot_union: bool = False,
    ) -> None:
        if not children:
            raise ValueError("CompositePattern requires at least one child pattern")
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported composition mode '{mode}'")

        self.children = list(children)
        self.mode = mode
        self.spot_union = bool(spot_union)
        self.weights = self._resolve_weights(weights)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        resolver: Callable[[str, Mapping[str, Any]], PatternCallable],
    ) -> "CompositePattern":
        """Build a composite pattern from config.

        Expected config schema::

            {
              "pattern": "composite",
              "mode": "phase_add_wrap",
              "spot_union": false,
              "children": [
                "laguerre-gaussian",
                {"pattern": "gaussian-lattice", "lattice_nx": 6}
              ]
            }
        """

        if config.get("pattern") != "composite":
            raise ValueError("Composite config must use pattern='composite'")

        child_patterns: list[PatternCallable] = []
        for child in config.get("children", []):
            if isinstance(child, str):
                child_cfg: dict[str, Any] = {"pattern": child}
            elif isinstance(child, Mapping):
                if "pattern" not in child:
                    raise ValueError("Child config mapping must include a 'pattern' key")
                child_cfg = dict(child)
            else:
                raise TypeError("Child entries must be pattern names or config mappings")
            child_patterns.append(resolver(str(child_cfg["pattern"]), child_cfg))

        return cls(
            children=child_patterns,
            mode=str(config.get("mode", "phase_add_wrap")),
            weights=config.get("weights"),
            spot_union=bool(config.get("spot_union", False)),
        )

    def _resolve_weights(self, weights: Sequence[float] | None) -> np.ndarray:
        n_children = len(self.children)
        if self.mode != "weighted_sum":
            return np.ones(n_children, dtype=float)

        if weights is None:
            return np.full(n_children, 1.0 / n_children, dtype=float)

        arr = np.asarray(weights, dtype=float)
        if arr.shape != (n_children,):
            raise ValueError(
                f"weights must have {n_children} entries for {n_children} children, got {arr.shape}"
            )
        return arr

    def __call__(self, *args: Any, **kwargs: Any) -> CompositeResult:
        child_outputs = [self._coerce_child_output(child(*args, **kwargs)) for child in self.children]
        phases = [output["phase"] for output in child_outputs]
        phase = self._compose_phase(phases)

        metadata: dict[str, Any] = {
            "pattern": "composite",
            "mode": self.mode,
            "weights": self.weights.tolist(),
            "spot_union_enabled": self.spot_union,
            "children": child_outputs,
        }

        if self.spot_union:
            metadata["spot_union"] = self._compute_spot_union(child_outputs)

        return CompositeResult(phase=phase, metadata=metadata)

    def _compose_phase(self, child_phases: Sequence[np.ndarray]) -> np.ndarray:
        if self.mode == "phase_add_wrap":
            return np.mod(np.sum(child_phases, axis=0), 2 * np.pi)

        stacked = np.stack(child_phases, axis=0)
        return np.tensordot(self.weights, stacked, axes=(0, 0))

    @staticmethod
    def _coerce_child_output(raw: Any) -> dict[str, Any]:
        if isinstance(raw, Mapping):
            if "phase" not in raw:
                raise ValueError("Child mapping output must include a 'phase' key")
            coerced = dict(raw)
            coerced["phase"] = np.asarray(coerced["phase"])
            return coerced

        return {"phase": np.asarray(raw)}

    @staticmethod
    def _compute_spot_union(child_outputs: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        unions: list[np.ndarray] = []
        for child in child_outputs:
            spots = child.get("spots")
            if spots is None and isinstance(child.get("metadata"), Mapping):
                spots = child["metadata"].get("spots")
            if spots is None:
                continue
            spots_arr = np.asarray(spots, dtype=float)
            if spots_arr.ndim == 1:
                spots_arr = spots_arr.reshape(1, -1)
            unions.append(spots_arr)

        if not unions:
            return []

        stacked = np.vstack(unions)
        unique = np.unique(stacked, axis=0)
        return unique.tolist()
