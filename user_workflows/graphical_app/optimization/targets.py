"""Structured target definitions for ratio-aware optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class LatticeConfig:
    geometry: str = "square"
    spacing: float = 12.0
    rotation_deg: float = 0.0


@dataclass
class TargetDefinition:
    beam_count: int
    beam_positions: list[tuple[float, float]] = field(default_factory=list)
    desired_ratios: list[float] = field(default_factory=list)
    lattice: LatticeConfig = field(default_factory=LatticeConfig)

    @classmethod
    def from_config(cls, payload: Mapping[str, Any], fallback_shape: tuple[int, int]) -> "TargetDefinition":
        target_payload = dict(payload.get("target_definition", {})) if isinstance(payload.get("target_definition"), Mapping) else {}
        ratio_payload = dict(payload.get("ratio_targets", {})) if isinstance(payload.get("ratio_targets"), Mapping) else {}

        beam_positions_raw = target_payload.get("beam_positions")
        if not isinstance(beam_positions_raw, list) or not beam_positions_raw:
            h, w = fallback_shape
            beam_positions = [(w / 2.0, h / 2.0)]
        else:
            beam_positions = []
            for item in beam_positions_raw:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    beam_positions.append((float(item[0]), float(item[1])))

        desired_raw = ratio_payload.get("desired_ratios", target_payload.get("desired_ratios"))
        desired: list[float]
        if isinstance(desired_raw, list) and desired_raw:
            desired = [max(0.0, float(v)) for v in desired_raw]
        else:
            desired = [1.0 for _ in beam_positions]

        if len(desired) < len(beam_positions):
            desired.extend([1.0] * (len(beam_positions) - len(desired)))
        desired = desired[: len(beam_positions)]
        total = sum(desired)
        if total <= 0:
            desired = [1.0 / len(desired) for _ in desired]
        else:
            desired = [v / total for v in desired]

        lattice_payload = target_payload.get("lattice", {}) if isinstance(target_payload.get("lattice"), Mapping) else {}
        lattice = LatticeConfig(
            geometry=str(lattice_payload.get("geometry", "square")),
            spacing=float(lattice_payload.get("spacing", 12.0)),
            rotation_deg=float(lattice_payload.get("rotation_deg", 0.0)),
        )
        beam_count = int(target_payload.get("beam_count", len(beam_positions)))
        beam_count = max(1, beam_count)

        return cls(
            beam_count=beam_count,
            beam_positions=beam_positions,
            desired_ratios=desired,
            lattice=lattice,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["beam_positions"] = [[x, y] for x, y in self.beam_positions]
        return payload

