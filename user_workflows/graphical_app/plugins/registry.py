"""Plugin and schema-driven registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from user_workflows.patterns.schemas import PATTERN_PARAM_SCHEMAS


@dataclass
class PluginRegistry:
    patterns: Dict[str, Any] = field(default_factory=dict)
    optimizers: Dict[str, Any] = field(default_factory=dict)

    def load_builtin_patterns(self) -> None:
        for name, schema in PATTERN_PARAM_SCHEMAS.items():
            self.patterns[name] = {
                "schema": {f.name: str(f.type) for f in schema.__dataclass_fields__.values()},
                "defaults": schema(),
            }

    def register_pattern_plugin(self, name: str, plugin: Mapping[str, Any]) -> None:
        self.patterns[name] = dict(plugin)

    def register_optimizer_plugin(self, name: str, plugin: Mapping[str, Any]) -> None:
        self.optimizers[name] = dict(plugin)
