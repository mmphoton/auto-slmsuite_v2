"""Synchronized sequence editor/runner with dry-run and metadata links."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class SequenceRunner:
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def import_sequence(self, sequence: List[Mapping[str, Any]]) -> None:
        self.steps = [dict(item) for item in sequence]

    def export_sequence(self) -> List[Dict[str, Any]]:
        return list(self.steps)

    def dry_run_timing(self) -> Dict[str, float]:
        total_ms = sum(float(s.get("duration_ms", 0.0)) for s in self.steps)
        return {"steps": float(len(self.steps)), "total_duration_ms": total_ms}

    def run(self, telemetry: Mapping[str, Any]) -> List[Dict[str, Any]]:
        runtime = []
        for idx, step in enumerate(self.steps):
            runtime.append(
                {
                    "step_index": idx,
                    "status": "done",
                    "pattern": step.get("pattern"),
                    "duration_ms": step.get("duration_ms"),
                    "telemetry": dict(telemetry),
                }
            )
        return runtime
