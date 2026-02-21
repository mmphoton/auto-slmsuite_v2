"""Dynamic pattern parameter form renderer with validation."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Mapping


class PatternFormRenderer(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self._vars: dict[str, tk.StringVar] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    def render(self, schema: Mapping[str, Any]) -> None:
        for child in self.winfo_children():
            child.destroy()

        self._vars.clear()
        self._meta.clear()

        for row, param in enumerate(schema.get("parameters", [])):
            name = str(param["name"])
            self._meta[name] = dict(param)
            default = param.get("default", "")
            var = tk.StringVar(value=str(default))
            self._vars[name] = var

            ttk.Label(self, text=f"{name} ({param.get('type', 'value')}):").grid(row=row * 2, column=0, sticky="w", padx=2)
            entry = ttk.Entry(self, textvariable=var)
            entry.grid(row=row * 2, column=1, sticky="ew", padx=2)

            help_text = self._build_help_text(param)
            ttk.Label(self, text=help_text, foreground="#555555").grid(row=row * 2 + 1, column=0, columnspan=2, sticky="w", padx=2)

        self.columnconfigure(1, weight=1)

    def collect_values(self) -> tuple[dict[str, Any] | None, list[str]]:
        parsed: dict[str, Any] = {}
        errors: list[str] = []

        for name, var in self._vars.items():
            meta = self._meta[name]
            raw = var.get().strip()
            if raw == "":
                errors.append(f"{name}: value is required")
                continue

            expected_type = meta.get("type", "str")
            try:
                value = int(raw) if expected_type == "int" else float(raw) if expected_type == "float" else raw
            except ValueError:
                errors.append(f"{name}: expected {expected_type}, got '{raw}'")
                continue

            if meta.get("options") and value not in meta["options"]:
                errors.append(f"{name}: choose one of {meta['options']}")

            if meta.get("range") is not None:
                minimum, maximum = meta["range"]
                if minimum is not None:
                    if meta.get("exclusive_min") and value <= minimum:
                        errors.append(f"{name}: must be greater than {minimum}")
                    elif not meta.get("exclusive_min") and value < minimum:
                        errors.append(f"{name}: must be at least {minimum}")
                if maximum is not None and value > maximum:
                    errors.append(f"{name}: must be at most {maximum}")

            parsed[name] = value

        return (parsed if not errors else None, errors)

    def set_values(self, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            if name in self._vars:
                self._vars[name].set(str(value))

    def reset_to_defaults(self) -> None:
        for name, meta in self._meta.items():
            self._vars[name].set(str(meta.get("default", "")))

    def represented_fields(self) -> set[str]:
        return set(self._vars)

    @staticmethod
    def _build_help_text(param: Mapping[str, Any]) -> str:
        chunks: list[str] = []
        if param.get("help"):
            chunks.append(str(param["help"]))
        if param.get("range") is not None:
            minimum, maximum = param["range"]
            if param.get("exclusive_min"):
                chunks.append(f"Range: ({minimum}, {maximum}]")
            else:
                chunks.append(f"Range: [{minimum}, {maximum}]")
        if param.get("options"):
            chunks.append(f"Options: {param['options']}")
        return "  ".join(chunks)


def parity_check_for_schema(schema: Mapping[str, Any], represented_fields: set[str]) -> list[str]:
    expected = {str(p["name"]) for p in schema.get("parameters", [])}
    missing = sorted(expected - represented_fields)
    extra = sorted(represented_fields - expected)
    issues: list[str] = []
    if missing:
        issues.append(f"Missing controls for: {', '.join(missing)}")
    if extra:
        issues.append(f"Unexpected controls for: {', '.join(extra)}")
    return issues
