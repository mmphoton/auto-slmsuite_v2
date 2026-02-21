"""Calibration panel UI with profile file picker, validation and apply workflow."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable

from user_workflows.graphical_app.app.controller import AppController


class CalibrationWindow(ttk.Frame):
    def __init__(self, parent: tk.Misc, controller: AppController, on_status: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.controller = controller
        self.on_status = on_status

        self.profile_path_var = tk.StringVar(value="user_workflows/output/calibration_profile.json")
        self.compatibility_var = tk.StringVar(value="Compatibility: not checked")
        self.summary_var = tk.StringVar(value="Profile: n/a")
        self.metrics_var = tk.StringVar(value="Metrics: n/a")
        self.warning_var = tk.StringVar(value="")

        self._loaded_profile: dict | None = None
        self._is_compatible = False

        ttk.Label(self, text="Calibration profile path:").pack(anchor="w")
        path_row = ttk.Frame(self)
        path_row.pack(fill=tk.X)
        ttk.Entry(path_row, textvariable=self.profile_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_row, text="Browse", command=self._browse_profile).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Button(self, text="Load Profile", command=self._load_profile).pack(fill=tk.X)
        ttk.Button(self, text="Validate", command=self._validate_profile).pack(fill=tk.X)

        self.apply_btn = ttk.Button(self, text="Apply Profile", command=self._apply_profile, state=tk.DISABLED)
        self.apply_btn.pack(fill=tk.X)

        ttk.Label(self, textvariable=self.summary_var, wraplength=340, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))
        ttk.Label(self, textvariable=self.compatibility_var, wraplength=340, justify=tk.LEFT).pack(anchor="w")
        ttk.Label(self, textvariable=self.metrics_var, wraplength=340, justify=tk.LEFT).pack(anchor="w")
        ttk.Label(self, textvariable=self.warning_var, foreground="#b22222", wraplength=360, justify=tk.LEFT).pack(anchor="w", pady=(2, 0))

    def _browse_profile(self) -> None:
        selected = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if selected:
            self.profile_path_var.set(selected)

    def _load_profile(self) -> None:
        result = self.controller.load_calibration_profile(self.profile_path_var.get())
        if not result.success or not isinstance(result.payload, dict):
            self.warning_var.set(result.message)
            self.apply_btn.configure(state=tk.DISABLED)
            self.on_status(result.message)
            return

        self._loaded_profile = result.payload
        self.warning_var.set("")
        self._render_summary(result.payload)
        self.on_status(result.message)

    def _validate_profile(self) -> None:
        if self._loaded_profile is None:
            self._load_profile()
            if self._loaded_profile is None:
                return

        result = self.controller.validate_calibration_profile(self._loaded_profile)
        if not result.success or not isinstance(result.payload, dict):
            self.warning_var.set(result.message)
            self.compatibility_var.set("Compatibility: validation failed")
            self.apply_btn.configure(state=tk.DISABLED)
            self.on_status(result.message)
            return

        payload = result.payload
        summary = payload.get("summary", {})
        self.summary_var.set(
            f"Profile: {summary.get('name', 'n/a')} | mode={summary.get('mode', 'n/a')} | "
            f"slm={summary.get('slm_model', 'n/a')} | camera={summary.get('camera_model', 'n/a')}"
        )
        compatibility = payload.get("compatibility", {})
        checks = compatibility.get("checks", {})
        is_compatible = bool(compatibility.get("compatible", False)) and bool(payload.get("valid", False))
        self._is_compatible = is_compatible
        if is_compatible:
            self.compatibility_var.set(
                f"Compatibility: PASS (mode={checks.get('mode')}, slm={checks.get('slm_model')}, camera={checks.get('camera_model')})"
            )
            self.warning_var.set("")
            self.apply_btn.configure(state=tk.NORMAL)
        else:
            expected = compatibility.get("expected", {})
            actual = compatibility.get("actual", {})
            self.compatibility_var.set("Compatibility: FAIL")
            self.warning_var.set(
                "Apply blocked: calibration profile is incompatible with active mode/device identifiers. "
                f"Expected {expected}, got {actual}."
            )
            self.apply_btn.configure(state=tk.DISABLED)
        self.on_status(result.message)

    def _apply_profile(self) -> None:
        if not self._is_compatible:
            self.warning_var.set("Apply blocked: run validation and resolve compatibility mismatch first.")
            self.apply_btn.configure(state=tk.DISABLED)
            return

        result = self.controller.apply_calibration_profile(self.profile_path_var.get())
        if not result.success or not isinstance(result.payload, dict):
            self.warning_var.set(result.message)
            self.on_status(result.message)
            return

        metrics = result.payload.get("metrics", {})
        self.metrics_var.set(
            "Metrics: "
            f"before_rmse={metrics.get('before_rmse', 0.0):.4f}, "
            f"after_rmse={metrics.get('after_rmse', 0.0):.4f}, "
            f"improvement={metrics.get('rmse_improvement', 0.0):.4f}"
        )
        self.warning_var.set("")
        self.on_status(result.message)

    def _render_summary(self, profile: dict) -> None:
        name = profile.get("name") or Path(self.profile_path_var.get()).name
        matrix = profile.get("matrix")
        rows = len(matrix) if isinstance(matrix, list) else 0
        cols = len(matrix[0]) if rows and isinstance(matrix[0], list) else 0
        self.summary_var.set(
            f"Profile: {name} | mode={profile.get('mode', 'n/a')} | "
            f"slm={profile.get('slm_model', 'n/a')} | camera={profile.get('camera_model', 'n/a')} | matrix={rows}x{cols}"
        )
