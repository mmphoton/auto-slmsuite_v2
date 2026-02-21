"""Tkinter-based dockable/resizable UI exposing backend features."""

from __future__ import annotations

import base64
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np

from user_workflows.graphical_app.app.controller import AppController
from user_workflows.graphical_app.app.interfaces import OperationResult
from user_workflows.graphical_app.app.state import LogLevel
from user_workflows.graphical_app.ui.calibration_window import CalibrationWindow
from user_workflows.graphical_app.ui.pattern_form import PatternFormRenderer, parity_check_for_schema



class PlotWidget(ttk.Frame):
    def __init__(self, parent: tk.Misc, controller: AppController, plot_name: str, on_status) -> None:
        super().__init__(parent)
        self.controller = controller
        self.plot_name = plot_name
        self.on_status = on_status
        self._img: tk.PhotoImage | None = None

        self.canvas = tk.Canvas(self, width=300, height=240, background="#101010", highlightthickness=1)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, pady=2)
        for label, command in (("Zoom +", lambda: self._zoom(0.7)), ("Zoom -", lambda: self._zoom(1.3)), ("←", lambda: self._pan(-0.15, 0.0)), ("→", lambda: self._pan(0.15, 0.0)), ("↑", lambda: self._pan(0.0, -0.15)), ("↓", lambda: self._pan(0.0, 0.15)), ("Reset", self._reset)):
            ttk.Button(controls, text=label, command=command).pack(side=tk.LEFT, padx=1)

        settings = ttk.Frame(self)
        settings.pack(fill=tk.X)
        self.autoscale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings, text="Autoscale", variable=self.autoscale_var, command=self._apply_settings).grid(row=0, column=0, sticky="w")

        ttk.Label(settings, text="Scale").grid(row=0, column=1, sticky="e")
        self.scale_var = tk.StringVar(value="linear")
        ttk.Combobox(settings, textvariable=self.scale_var, values=["linear", "log"], width=8, state="readonly").grid(row=0, column=2, sticky="w")

        ttk.Label(settings, text="Colormap").grid(row=0, column=3, sticky="e")
        self.cmap_var = tk.StringVar(value="viridis")
        ttk.Combobox(settings, textvariable=self.cmap_var, values=["viridis", "gray", "plasma", "magma"], width=10, state="readonly").grid(row=0, column=4, sticky="w")

        ttk.Label(settings, text="xlim").grid(row=1, column=0, sticky="e")
        self.xmin = tk.StringVar(value="")
        self.xmax = tk.StringVar(value="")
        ttk.Entry(settings, textvariable=self.xmin, width=8).grid(row=1, column=1, sticky="w")
        ttk.Entry(settings, textvariable=self.xmax, width=8).grid(row=1, column=2, sticky="w")

        ttk.Label(settings, text="ylim").grid(row=1, column=3, sticky="e")
        self.ymin = tk.StringVar(value="")
        self.ymax = tk.StringVar(value="")
        ttk.Entry(settings, textvariable=self.ymin, width=8).grid(row=1, column=4, sticky="w")
        ttk.Button(settings, text="Apply", command=self._apply_settings).grid(row=1, column=5, padx=2)

    def _zoom(self, factor: float) -> None:
        self.on_status(self.controller.zoom_plot(self.plot_name, factor))
        self.refresh()

    def _pan(self, dx: float, dy: float) -> None:
        self.on_status(self.controller.pan_plot(self.plot_name, dx, dy))
        self.refresh()

    def _reset(self) -> None:
        self.on_status(self.controller.reset_plot(self.plot_name))
        self.refresh()

    def _parse_pair(self, a: str, b: str) -> tuple[float, float] | None:
        if not a.strip() or not b.strip():
            return None
        return (float(a), float(b))

    def _apply_settings(self) -> None:
        try:
            payload = {
                "autoscale": self.autoscale_var.get(),
                "scale": self.scale_var.get(),
                "colormap": self.cmap_var.get(),
                "xlim": self._parse_pair(self.xmin.get(), self.xmax.get()),
                "ylim": self._parse_pair(self.ymin.get(), self.ymax.get()),
            }
        except ValueError as exc:
            self.on_status(f"Plot settings error ({self.plot_name}): {exc}")
            return
        self.on_status(self.controller.configure_plot(self.plot_name, payload))
        self.refresh()

    def _sync_from_backend(self) -> None:
        model = self.controller.plots.get_plot_model(self.plot_name)
        self.autoscale_var.set(bool(model.settings.autoscale))
        self.scale_var.set(model.settings.scale)
        self.cmap_var.set(model.settings.colormap)
        xlim = model.settings.xlim
        ylim = model.settings.ylim
        self.xmin.set("" if xlim is None else f"{xlim[0]:.3f}")
        self.xmax.set("" if xlim is None else f"{xlim[1]:.3f}")
        self.ymin.set("" if ylim is None else f"{ylim[0]:.3f}")
        self.ymax.set("" if ylim is None else f"{ylim[1]:.3f}")

    def refresh(self) -> None:
        self._sync_from_backend()
        rgb = self.controller.plots.render_rgb(self.plot_name)
        if rgb.ndim != 3:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        h, w = rgb.shape[:2]
        ppm = f"P6\n{w} {h}\n255\n".encode("ascii") + rgb.astype(np.uint8).tobytes()
        self._img = tk.PhotoImage(data=base64.b64encode(ppm), format="PPM")
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._img)
        self.canvas.configure(scrollregion=(0, 0, w, h))


class MainWindow(tk.Tk):
    PAGE_TITLES: dict[str, str] = {
        "device_mode": "Device & Mode",
        "slm_patterns_blaze": "SLM Patterns & Blaze",
        "camera_telemetry": "Camera & Telemetry",
        "plot_workspace": "Plot Workspace",
        "optimization": "Optimization (WGS + Ratio)",
        "calibration": "Calibration",
        "session_output_recipes": "Session/Output/Recipes",
        "logs_diagnostics": "Logs/Diagnostics",
    }
    PANEL_TO_PAGE: dict[str, str] = {
        "Device": "device_mode",
        "SLM": "slm_patterns_blaze",
        "Camera": "camera_telemetry",
        "Plots": "plot_workspace",
        "Optimization": "optimization",
        "Ratio Targets": "optimization",
        "Calibration": "calibration",
        "Session": "session_output_recipes",
        "Logs": "logs_diagnostics",
    }
    PANEL_NAMES = ["Device", "SLM", "Camera", "Plots", "Optimization", "Ratio Targets", "Calibration", "Logs", "Session"]

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self.controller = controller
        self.store = self.controller.persistence
        self.title("SLM Suite Graphical App")
        self.layout_path = Path("user_workflows/output/gui_layout.json")
        self.pattern_presets_path = Path("user_workflows/output/pattern_presets.json")

        self.panel_frames: dict[str, ttk.LabelFrame] = {}
        self.panel_columns: dict[str, str] = {}
        self.visibility_vars = {name: tk.BooleanVar(value=True) for name in self.PANEL_NAMES}
        self.plot_popouts: dict[str, tk.Toplevel] = {}
        self.plot_widgets: dict[str, PlotWidget] = {}

        self.page_notebook = ttk.Notebook(self)
        self.page_notebook.pack(fill=tk.BOTH, expand=True)
        self.page_panels: dict[str, ttk.PanedWindow] = {}
        self.page_columns: dict[str, dict[str, ttk.Frame]] = {}
        self.page_nav_var = tk.StringVar(value=self.PAGE_TITLES["device_mode"])
        self._build_page_shell()

        self.status_var = tk.StringVar(value="Ready")
        self._build_all_panels()
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=4, pady=2)

        self.restore_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(800, self._schedule_progress_refresh)
        self.after(500, self._schedule_plot_refresh)

    def _safe_callback(self, callback_name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            message = f"{callback_name} error: {exc}"
            self.status_var.set(message)
            self.controller.state.notify(message)
            self.controller.state.add_log(LogLevel.ERROR, message, source="ui")
            self._refresh_logs()

    def _bind_safe(self, callback_name: str, fn):
        return lambda: self._safe_callback(callback_name, fn)

    def _handle_result(self, result: OperationResult) -> None:
        self.status_var.set(result.message)
        if not result.success:
            self.controller.state.notify(result.message)
        self._refresh_logs()
        self._refresh_progress_widgets()

    def _build_all_panels(self) -> None:
        self._build_device_panel()
        self._build_slm_panel()
        self._build_camera_panel()
        self._build_plot_panel()
        self._build_optimization_panel()
        self._build_ratio_targets_panel()
        self._build_calibration_panel()
        self._build_logs_panel()
        self._build_session_panel()

    def _build_page_shell(self) -> None:
        nav = ttk.Frame(self)
        nav.pack(fill=tk.X, padx=4, pady=(4, 2), before=self.page_notebook)
        ttk.Label(nav, text="Navigate:").pack(side=tk.LEFT, padx=(0, 6))
        for page_key, page_title in self.PAGE_TITLES.items():
            ttk.Button(nav, text=page_title, command=lambda key=page_key: self.select_page(key)).pack(side=tk.LEFT, padx=2)

            page = ttk.Frame(self.page_notebook)
            self.page_notebook.add(page, text=page_title)
            pane = ttk.PanedWindow(page, orient=tk.HORIZONTAL)
            pane.pack(fill=tk.BOTH, expand=True)
            left = ttk.Frame(pane)
            center = ttk.Frame(pane)
            right = ttk.Frame(pane)
            pane.add(left, weight=1)
            pane.add(center, weight=2)
            pane.add(right, weight=1)
            self.page_panels[page_key] = pane
            self.page_columns[page_key] = {"left": left, "center": center, "right": right}

        self.page_notebook.bind("<<NotebookTabChanged>>", self._sync_page_nav_state)

    def _sync_page_nav_state(self, _event: object | None = None) -> None:
        idx = self.page_notebook.index("current")
        page_title = self.page_notebook.tab(idx, "text")
        self.page_nav_var.set(page_title)

    def select_page(self, page_key: str) -> None:
        target = self.PAGE_TITLES.get(page_key)
        if target is None:
            return
        for idx in range(self.page_notebook.index("end")):
            if self.page_notebook.tab(idx, "text") == target:
                self.page_notebook.select(idx)
                self.page_nav_var.set(target)
                return

    def _create_panel(self, panel_name: str, column: str) -> ttk.LabelFrame:
        page_key = self.PANEL_TO_PAGE[panel_name]
        frame = ttk.LabelFrame(self.page_columns[page_key][column], text=f"{panel_name} Panel")
        self.panel_frames[panel_name] = frame
        self.panel_columns[panel_name] = column
        return frame

    def _build_device_panel(self) -> None:
        frm = self._create_panel("Device", "left")
        self.mode = tk.StringVar(value="simulation")
        ttk.Radiobutton(frm, text="Simulation", variable=self.mode, value="simulation", command=self._bind_safe("set_mode", self._set_mode)).pack(anchor="w")
        ttk.Radiobutton(frm, text="Hardware", variable=self.mode, value="hardware", command=self._bind_safe("set_mode", self._set_mode)).pack(anchor="w")
        ttk.Button(frm, text="Discover", command=self._bind_safe("discover_devices", lambda: self._handle_result(self.controller.discover_devices()))).pack(fill=tk.X)
        ttk.Button(frm, text="Connect", command=self._bind_safe("connect_devices", lambda: self._handle_result(self.controller.connect_devices()))).pack(fill=tk.X)
        ttk.Button(frm, text="Reconnect", command=self._bind_safe("reconnect_devices", lambda: self._handle_result(self.controller.reconnect_devices()))).pack(fill=tk.X)
        ttk.Button(frm, text="Release SLM", command=self._bind_safe("release_slm", lambda: self._handle_result(self.controller.release_slm()))).pack(fill=tk.X)
        ttk.Button(frm, text="Release Camera", command=self._bind_safe("release_camera", lambda: self._handle_result(self.controller.release_camera()))).pack(fill=tk.X)
        ttk.Button(frm, text="Release Both", command=self._bind_safe("release_both", lambda: self._handle_result(self.controller.release_both()))).pack(fill=tk.X)

    def _build_slm_panel(self) -> None:
        frm = self._create_panel("SLM", "center")
        pattern_options = self.controller.available_patterns().payload or []
        default_pattern = pattern_options[0] if pattern_options else "single-gaussian"
        self.pattern_name = tk.StringVar(value=default_pattern)
        self.pattern_combo = ttk.Combobox(frm, textvariable=self.pattern_name, values=pattern_options, state="readonly")
        self.pattern_combo.pack(fill=tk.X)
        self.pattern_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_pattern_change())

        self.pattern_form = PatternFormRenderer(frm)
        self.pattern_form.pack(fill=tk.X, pady=4)

        self.preset_name = tk.StringVar(value="default")
        ttk.Label(frm, text="Pattern preset name:").pack(anchor="w")
        ttk.Entry(frm, textvariable=self.preset_name).pack(fill=tk.X)
        ttk.Button(frm, text="Save Pattern Preset", command=self._bind_safe("save_pattern_preset", self._save_pattern_preset)).pack(fill=tk.X)
        ttk.Button(frm, text="Load Pattern Preset", command=self._bind_safe("load_pattern_preset", self._load_pattern_preset)).pack(fill=tk.X)
        ttk.Button(frm, text="Reset Parameters", command=self._bind_safe("reset_pattern_params", self._reset_pattern_params)).pack(fill=tk.X)

        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(frm, text="Blaze composition").pack(anchor="w")
        self.blaze_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Enable blaze", variable=self.blaze_enabled).pack(anchor="w")

        self.blaze_kx = tk.StringVar(value="0.0")
        ttk.Label(frm, text="blaze kx [-1,1]:").pack(anchor="w")
        ttk.Entry(frm, textvariable=self.blaze_kx).pack(fill=tk.X)

        self.blaze_ky = tk.StringVar(value="0.0")
        ttk.Label(frm, text="blaze ky [-1,1]:").pack(anchor="w")
        ttk.Entry(frm, textvariable=self.blaze_ky).pack(fill=tk.X)

        self.blaze_offset = tk.StringVar(value="")
        ttk.Label(frm, text="offset (optional rad):").pack(anchor="w")
        ttk.Entry(frm, textvariable=self.blaze_offset).pack(fill=tk.X)

        self.blaze_scale = tk.StringVar(value="")
        ttk.Label(frm, text="scale (optional > 0):").pack(anchor="w")
        ttk.Entry(frm, textvariable=self.blaze_scale).pack(fill=tk.X)

        ttk.Button(frm, text="Simulate Before Apply", command=self._bind_safe("simulate", self._simulate)).pack(fill=tk.X)
        ttk.Button(frm, text="Apply", command=self._bind_safe("apply", self._apply)).pack(fill=tk.X)
        ttk.Button(frm, text="Queue", command=self._bind_safe("queue", self._queue)).pack(fill=tk.X)
        ttk.Button(frm, text="Clear Queue", command=self._bind_safe("clear_queue", lambda: self._handle_result(self.controller.clear_pattern_queue()))).pack(fill=tk.X)
        self._on_pattern_change()

    def _build_camera_panel(self) -> None:
        frm = self._create_panel("Camera", "right")
        schema_result = self.controller.camera_settings_schema()
        schema = schema_result.payload if schema_result.success and isinstance(schema_result.payload, dict) else {}

        self.camera_form_vars: dict[str, tk.Variable] = {}
        self._camera_last_applied: dict[str, object] = dict(self.controller.state.settings_snapshots.camera)

        for field in ("exposure_ms", "gain", "roi_x", "roi_y", "roi_width", "roi_height", "fps"):
            unit = schema.get(field, {}).get("unit", "")
            ttk.Label(frm, text=f"{field} ({unit})" if unit else field).pack(anchor="w")
            var = tk.StringVar(value=str(self._camera_last_applied.get(field, schema.get(field, {}).get("minimum", ""))))
            self.camera_form_vars[field] = var
            ttk.Entry(frm, textvariable=var).pack(fill=tk.X)

        ttk.Label(frm, text="binning (px)").pack(anchor="w")
        binning_var = tk.StringVar(value=str(self._camera_last_applied.get("binning", 1)))
        self.camera_form_vars["binning"] = binning_var
        ttk.Combobox(frm, textvariable=binning_var, values=[1, 2, 4, 8], state="readonly").pack(fill=tk.X)

        ttk.Label(frm, text="trigger_mode").pack(anchor="w")
        trigger_var = tk.StringVar(value=str(self._camera_last_applied.get("trigger_mode", "internal")))
        self.camera_form_vars["trigger_mode"] = trigger_var
        ttk.Combobox(frm, textvariable=trigger_var, values=["internal", "external", "software"], state="readonly").pack(fill=tk.X)

        ttk.Label(frm, text="shutter_mode").pack(anchor="w")
        shutter_var = tk.StringVar(value=str(self._camera_last_applied.get("shutter_mode", "rolling")))
        self.camera_form_vars["shutter_mode"] = shutter_var
        ttk.Combobox(frm, textvariable=shutter_var, values=["rolling", "global"], state="readonly").pack(fill=tk.X)

        ttk.Label(frm, text="acquisition_mode").pack(anchor="w")
        acquisition_var = tk.StringVar(value=str(self._camera_last_applied.get("acquisition_mode", "single")))
        self.camera_form_vars["acquisition_mode"] = acquisition_var
        ttk.Combobox(frm, textvariable=acquisition_var, values=["single", "continuous", "kinetic"], state="readonly").pack(fill=tk.X)

        button_row = ttk.Frame(frm)
        button_row.pack(fill=tk.X, pady=2)
        ttk.Button(button_row, text="Apply", command=self._bind_safe("configure_camera", self._configure_camera)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(button_row, text="Read Current", command=self._bind_safe("read_camera", self._read_camera_settings)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(button_row, text="Revert", command=self._bind_safe("revert_camera", self._revert_camera_settings)).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self._set_camera_form(self._camera_last_applied)

        self.camera_feedback_var = tk.StringVar(value="Camera settings idle")
        ttk.Label(frm, textvariable=self.camera_feedback_var).pack(anchor="w", pady=(2, 4))

        self.temp_label = ttk.Label(frm, text="Temperature: n/a")
        self.temp_label.pack(anchor="w")
        ttk.Button(frm, text="Refresh Telemetry", command=self._bind_safe("telemetry", self._telemetry)).pack(fill=tk.X)

    def _build_plot_panel(self) -> None:
        frm = self._create_panel("Plots", "center")
        self.plot_names = ["simulated_phase", "simulated_intensity", "experimental_intensity", "optimization_convergence", "optimization_phase_before", "optimization_phase_after", "optimization_intensity_before", "optimization_intensity_after", "ratio_targets_vs_measured", "ratio_error_by_beam", "ratio_metrics"]
        self.plot_select = tk.StringVar(value=self.plot_names[0])
        ttk.Combobox(frm, textvariable=self.plot_select, values=self.plot_names, state="readonly").pack(fill=tk.X)
        ttk.Button(frm, text="Pop-out Plot", command=self._bind_safe("pop_plot", self._pop_plot)).pack(fill=tk.X)
        ttk.Button(frm, text="Export Plot", command=self._bind_safe("export_plot", self._export_plot)).pack(fill=tk.X)

        self.plot_notebook = ttk.Notebook(frm)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True, pady=4)
        for plot_name in self.plot_names:
            tab = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(tab, text=plot_name.replace("_", " ").title())
            widget = PlotWidget(tab, self.controller, plot_name, self._set_plot_status)
            widget.pack(fill=tk.BOTH, expand=True)
            self.plot_widgets[plot_name] = widget

    def _build_optimization_panel(self) -> None:
        frm = self._create_panel("Optimization", "left")
        ttk.Label(frm, text="WGS max iterations").pack(anchor="w")
        self.optimization_max_iters = tk.StringVar(value="20")
        ttk.Entry(frm, textvariable=self.optimization_max_iters).pack(fill=tk.X)

        ttk.Label(frm, text="WGS gain").pack(anchor="w")
        self.optimization_gain = tk.StringVar(value="0.2")
        ttk.Entry(frm, textvariable=self.optimization_gain).pack(fill=tk.X)

        self.optimization_settings = ttk.Entry(frm)
        self.optimization_settings.insert(0, '{"initial": 1.0}')
        self.optimization_settings.pack(fill=tk.X)

        ttk.Button(frm, text="Start", command=self._bind_safe("start_optimization", self._run_optimization)).pack(fill=tk.X)
        ttk.Button(frm, text="Pause", command=self._bind_safe("pause_optimization", lambda: self._handle_result(self.controller.pause_optimization()))).pack(fill=tk.X)
        ttk.Button(frm, text="Resume", command=self._bind_safe("resume_optimization", lambda: self._handle_result(self.controller.resume_optimization()))).pack(fill=tk.X)
        ttk.Button(frm, text="Stop", command=self._bind_safe("stop_optimization", lambda: self._handle_result(self.controller.stop_optimization()))).pack(fill=tk.X)
        ttk.Button(frm, text="Export History", command=self._bind_safe("export_optimization_history", self._export_optimization_history)).pack(fill=tk.X)

        self.optimization_progress = ttk.Progressbar(frm, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.optimization_progress.pack(fill=tk.X, pady=3)
        self.optimization_progress_label = ttk.Label(frm, text="idle")
        self.optimization_progress_label.pack(anchor="w")

    def _build_ratio_targets_panel(self) -> None:
        frm = self._create_panel("Ratio Targets", "left")
        ttk.Label(frm, text="Mode").pack(anchor="w")
        self.ratio_mode = tk.StringVar(value="simulation")
        ttk.Combobox(frm, textvariable=self.ratio_mode, values=["simulation", "camera"], state="readonly").pack(fill=tk.X)

        ttk.Label(frm, text="Beam count").pack(anchor="w")
        self.ratio_beam_count = tk.StringVar(value="3")
        ttk.Entry(frm, textvariable=self.ratio_beam_count).pack(fill=tk.X)

        ttk.Label(frm, text="Beam positions [[x,y],...]").pack(anchor="w")
        self.ratio_positions = ttk.Entry(frm)
        self.ratio_positions.insert(0, "[[32,32],[64,64],[96,96]]")
        self.ratio_positions.pack(fill=tk.X)

        ttk.Label(frm, text="Desired ratios [..]").pack(anchor="w")
        self.ratio_desired = ttk.Entry(frm)
        self.ratio_desired.insert(0, "[0.2,0.3,0.5]")
        self.ratio_desired.pack(fill=tk.X)

        ttk.Label(frm, text="Lattice geometry").pack(anchor="w")
        self.ratio_lattice_geometry = tk.StringVar(value="square")
        ttk.Combobox(frm, textvariable=self.ratio_lattice_geometry, values=["square", "hex", "line"], state="readonly").pack(fill=tk.X)

        ttk.Label(frm, text="Lattice spacing").pack(anchor="w")
        self.ratio_lattice_spacing = tk.StringVar(value="12.0")
        ttk.Entry(frm, textvariable=self.ratio_lattice_spacing).pack(fill=tk.X)

        ttk.Label(frm, text="Lattice rotation (deg)").pack(anchor="w")
        self.ratio_lattice_rotation = tk.StringVar(value="0.0")
        ttk.Entry(frm, textvariable=self.ratio_lattice_rotation).pack(fill=tk.X)

        ttk.Label(frm, text="Objective weights [intensity/ratio/regularization]").pack(anchor="w")
        self.weight_intensity = tk.StringVar(value="1.0")
        self.weight_ratio = tk.StringVar(value="0.5")
        self.weight_regularization = tk.StringVar(value="0.05")
        weight_row = ttk.Frame(frm)
        weight_row.pack(fill=tk.X)
        ttk.Entry(weight_row, textvariable=self.weight_intensity, width=8).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Entry(weight_row, textvariable=self.weight_ratio, width=8).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Entry(weight_row, textvariable=self.weight_regularization, width=8).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.ratio_metrics_var = tk.StringVar(value="ratio metrics: n/a")
        ttk.Label(frm, textvariable=self.ratio_metrics_var).pack(anchor="w", pady=(4, 0))

    def _build_calibration_panel(self) -> None:
        frm = self._create_panel("Calibration", "right")
        self.calibration_window = CalibrationWindow(frm, self.controller, self.status_var.set)
        self.calibration_window.pack(fill=tk.X)
        ttk.Button(frm, text="Cancel Calibration", command=self._bind_safe("cancel_calibration", lambda: self._handle_result(self.controller.cancel_calibration()))).pack(fill=tk.X)
        self.calibration_progress = ttk.Progressbar(frm, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.calibration_progress.pack(fill=tk.X, pady=3)

    def _build_logs_panel(self) -> None:
        frm = self._create_panel("Logs", "left")
        columns = ("timestamp", "level", "source", "message")
        self.log_tree = ttk.Treeview(frm, columns=columns, show="headings", height=10)
        for col, width in (("timestamp", 150), ("level", 90), ("source", 100), ("message", 420)):
            self.log_tree.heading(col, text=col.title())
            self.log_tree.column(col, width=width, anchor="w")
        self.log_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(frm, text="Refresh Logs", command=self._bind_safe("refresh_logs", self._refresh_logs)).pack(fill=tk.X)

    def _build_session_panel(self) -> None:
        frm = self._create_panel("Session", "right")
        ttk.Label(frm, text="Output directory").pack(anchor="w")
        self.output_dir_var = tk.StringVar(value=self.controller.state.output_directory)
        ttk.Entry(frm, textvariable=self.output_dir_var).pack(fill=tk.X)
        ttk.Button(frm, text="Browse Output Directory", command=self._bind_safe("choose_output_dir", self._choose_output_dir)).pack(fill=tk.X)

        ttk.Label(frm, text="Naming template").pack(anchor="w")
        self.naming_template_var = tk.StringVar(value=self.controller.state.naming_template)
        ttk.Entry(frm, textvariable=self.naming_template_var).pack(fill=tk.X)

        ttk.Label(frm, text="Collision policy").pack(anchor="w")
        self.collision_policy_var = tk.StringVar(value=self.controller.state.collision_policy)
        ttk.Combobox(frm, textvariable=self.collision_policy_var, values=["increment", "overwrite", "error"], state="readonly").pack(fill=tk.X)

        ttk.Label(frm, text="Run ID override").pack(anchor="w")
        self.run_override_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.run_override_var).pack(fill=tk.X)
        ttk.Button(frm, text="Apply Output Settings", command=self._bind_safe("apply_output_settings", self._apply_output_settings)).pack(fill=tk.X)

        self.name_preview_var = tk.StringVar(value="Name preview: n/a")
        ttk.Label(frm, textvariable=self.name_preview_var).pack(anchor="w")
        ttk.Button(frm, text="Preview Current Template", command=self._bind_safe("preview_naming", self._preview_naming)).pack(fill=tk.X)

        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Button(frm, text="Save Layout", command=self._bind_safe("save_layout", self.save_layout)).pack(fill=tk.X)
        ttk.Button(frm, text="Restore Layout", command=self._bind_safe("restore_layout", self.restore_layout)).pack(fill=tk.X)
        ttk.Button(frm, text="Reset Layout", command=self._bind_safe("reset_layout", self._reset_layout)).pack(fill=tk.X)
        ttk.Button(frm, text="Save Session Snapshot", command=self._bind_safe("save_snapshot", self._save_snapshot)).pack(fill=tk.X)

        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(frm, text="Sequence JSON:").pack(anchor="w")
        self.sequence_entry = ttk.Entry(frm)
        self.sequence_entry.insert(0, '[{"pattern":"single-gaussian","duration_ms":25},{"pattern":"blaze","duration_ms":30}]')
        self.sequence_entry.pack(fill=tk.X)
        ttk.Button(frm, text="Run Sequence", command=self._bind_safe("run_sequence", self._run_sequence)).pack(fill=tk.X)
        ttk.Button(frm, text="Cancel Sequence", command=self._bind_safe("cancel_sequence", lambda: self._handle_result(self.controller.cancel_sequence()))).pack(fill=tk.X)
        self.sequence_progress = ttk.Progressbar(frm, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.sequence_progress.pack(fill=tk.X, pady=3)

        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        self.arrange_panel = tk.StringVar(value=self.PANEL_NAMES[0])
        ttk.Combobox(frm, textvariable=self.arrange_panel, values=self.PANEL_NAMES, state="readonly").pack(fill=tk.X)
        self.target_column = tk.StringVar(value="left")
        ttk.Combobox(frm, textvariable=self.target_column, values=["left", "center", "right"], state="readonly").pack(fill=tk.X)
        ttk.Button(frm, text="Move To Column", command=self._bind_safe("move_to_column", self._move_selected_panel_to_column)).pack(fill=tk.X)
        ttk.Button(frm, text="Move Up", command=self._bind_safe("move_panel_up", lambda: self._move_selected_panel(-1))).pack(fill=tk.X)
        ttk.Button(frm, text="Move Down", command=self._bind_safe("move_panel_down", lambda: self._move_selected_panel(1))).pack(fill=tk.X)

        self.swap_with = tk.StringVar(value=self.PANEL_NAMES[1])
        ttk.Combobox(frm, textvariable=self.swap_with, values=self.PANEL_NAMES, state="readonly").pack(fill=tk.X)
        ttk.Button(frm, text="Swap Panels", command=self._bind_safe("swap_panels", self._swap_panels)).pack(fill=tk.X)

        ttk.Label(frm, text="Visibility").pack(anchor="w")
        for panel_name in self.PANEL_NAMES:
            ttk.Checkbutton(
                frm,
                text=panel_name,
                variable=self.visibility_vars[panel_name],
                command=self._bind_safe("apply_visibility", self._apply_visibility),
            ).pack(anchor="w")

        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Button(frm, text="Preset: Acquisition", command=self._bind_safe("preset_acquisition", lambda: self._apply_preset("Acquisition"))).pack(fill=tk.X)
        ttk.Button(frm, text="Preset: Optimization", command=self._bind_safe("preset_optimization", lambda: self._apply_preset("Optimization"))).pack(fill=tk.X)
        ttk.Button(frm, text="Preset: Calibration", command=self._bind_safe("preset_calibration", lambda: self._apply_preset("Calibration"))).pack(fill=tk.X)

    def _set_mode(self) -> None:
        self._handle_result(self.controller.set_mode(self.mode.get()))

    def _on_pattern_change(self) -> None:
        schema = self.controller.patterns.schema_for(self.pattern_name.get())
        self.pattern_form.render(schema)
        parity_issues = parity_check_for_schema(schema, self.pattern_form.represented_fields())
        if parity_issues:
            raise ValueError("; ".join(parity_issues))

    def _collect_pattern_params(self) -> dict[str, object] | None:
        params, errors = self.pattern_form.collect_values()
        if errors:
            self.status_var.set("Pattern parameter errors: " + "; ".join(errors))
            self.controller.state.notify(self.status_var.get())
            self.controller.state.add_log(LogLevel.ERROR, self.status_var.get(), source="ui")
            self._refresh_logs()
            return None
        return params

    def _collect_blaze_settings(self) -> dict[str, object] | None:
        def _optional_float(raw: str, field_name: str) -> float | None:
            value = raw.strip()
            if not value:
                return None
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"{field_name} must be a float")

        try:
            kx = float(self.blaze_kx.get().strip() or "0.0")
            ky = float(self.blaze_ky.get().strip() or "0.0")
            offset = _optional_float(self.blaze_offset.get(), "offset")
            scale = _optional_float(self.blaze_scale.get(), "scale")
        except ValueError as exc:
            self.status_var.set(f"Blaze validation error: {exc}")
            return None

        if not -1.0 <= kx <= 1.0:
            self.status_var.set("Blaze validation error: kx must be within [-1.0, 1.0]")
            return None
        if not -1.0 <= ky <= 1.0:
            self.status_var.set("Blaze validation error: ky must be within [-1.0, 1.0]")
            return None
        if scale is not None and scale <= 0:
            self.status_var.set("Blaze validation error: scale must be > 0")
            return None

        return {
            "enabled": self.blaze_enabled.get(),
            "kx": kx,
            "ky": ky,
            "offset": offset,
            "scale": scale,
        }

    def _apply_blaze_settings(self) -> bool:
        settings = self._collect_blaze_settings()
        if settings is None:
            return False
        result = self.controller.configure_blaze(settings)
        self._handle_result(result)
        return result.success

    def _simulate(self) -> None:
        params = self._collect_pattern_params()
        if params is None or not self._apply_blaze_settings():
            return
        pattern_result = self.controller.generate_pattern(self.pattern_name.get(), params)
        self._handle_result(pattern_result)
        if pattern_result.success and pattern_result.payload is not None:
            self._handle_result(self.controller.simulate_before_apply(pattern_result.payload))

    def _apply(self) -> None:
        params = self._collect_pattern_params()
        if params is None or not self._apply_blaze_settings():
            return
        pattern_result = self.controller.generate_pattern(self.pattern_name.get(), params)
        self._handle_result(pattern_result)
        if pattern_result.success and pattern_result.payload is not None:
            self._handle_result(self.controller.apply_pattern(pattern_result.payload))

    def _queue(self) -> None:
        params = self._collect_pattern_params()
        if params is None or not self._apply_blaze_settings():
            return
        pattern_result = self.controller.generate_pattern(self.pattern_name.get(), params)
        self._handle_result(pattern_result)
        if pattern_result.success and pattern_result.payload is not None:
            self._handle_result(self.controller.queue_pattern(pattern_result.payload))

    def _save_pattern_preset(self) -> None:
        params = self._collect_pattern_params()
        if params is None:
            return
        payload = self.store.load_json(self.pattern_presets_path)
        presets = payload.get("presets", {})
        presets.setdefault(self.pattern_name.get(), {})[self.preset_name.get()] = params
        self.store.save_json(self.pattern_presets_path, {"presets": presets})
        self.status_var.set(f"Saved preset '{self.preset_name.get()}' for {self.pattern_name.get()}")

    def _load_pattern_preset(self) -> None:
        payload = self.store.load_json(self.pattern_presets_path)
        preset = payload.get("presets", {}).get(self.pattern_name.get(), {}).get(self.preset_name.get())
        if preset is None:
            self.status_var.set(f"Preset '{self.preset_name.get()}' not found for {self.pattern_name.get()}")
            return
        self.pattern_form.set_values(preset)
        self.status_var.set(f"Loaded preset '{self.preset_name.get()}' for {self.pattern_name.get()}")

    def _reset_pattern_params(self) -> None:
        self.pattern_form.reset_to_defaults()
        self.status_var.set(f"Reset parameters for {self.pattern_name.get()}")

    def _collect_camera_settings(self) -> dict[str, object]:
        return {
            "exposure_ms": float(str(self.camera_form_vars["exposure_ms"].get()).strip()),
            "gain": float(str(self.camera_form_vars["gain"].get()).strip()),
            "roi_x": int(str(self.camera_form_vars["roi_x"].get()).strip()),
            "roi_y": int(str(self.camera_form_vars["roi_y"].get()).strip()),
            "roi_width": int(str(self.camera_form_vars["roi_width"].get()).strip()),
            "roi_height": int(str(self.camera_form_vars["roi_height"].get()).strip()),
            "binning": int(str(self.camera_form_vars["binning"].get()).strip()),
            "trigger_mode": str(self.camera_form_vars["trigger_mode"].get()),
            "shutter_mode": str(self.camera_form_vars["shutter_mode"].get()),
            "fps": float(str(self.camera_form_vars["fps"].get()).strip()),
            "acquisition_mode": str(self.camera_form_vars["acquisition_mode"].get()),
        }

    def _set_camera_form(self, payload: dict[str, object]) -> None:
        mapped = dict(payload)
        roi = mapped.get("roi")
        if isinstance(roi, list) and len(roi) == 4:
            mapped["roi_x"], mapped["roi_y"], mapped["roi_width"], mapped["roi_height"] = roi
        for key, var in self.camera_form_vars.items():
            if key in mapped:
                var.set(str(mapped[key]))

    def _configure_camera(self) -> None:
        result = self.controller.configure_camera(self._collect_camera_settings())
        self._handle_result(result)
        if result.success and isinstance(result.payload, dict):
            self._camera_last_applied = dict(result.payload)
            self.camera_feedback_var.set("Camera settings applied successfully")
        elif not result.success:
            self.camera_feedback_var.set(f"Failed to apply camera settings: {result.message}")

    def _read_camera_settings(self) -> None:
        result = self.controller.read_camera_settings()
        self._handle_result(result)
        if result.success and isinstance(result.payload, dict):
            self._set_camera_form(result.payload)
            self.camera_feedback_var.set("Loaded current camera settings")

    def _revert_camera_settings(self) -> None:
        self._set_camera_form(self._camera_last_applied)
        self.camera_feedback_var.set("Reverted to last applied camera settings")

    def _telemetry(self) -> None:
        telem_result = self.controller.camera_telemetry()
        self._handle_result(telem_result)
        if telem_result.success and isinstance(telem_result.payload, dict):
            temp = float(telem_result.payload.get("temperature_c", 0.0))
            temp_status = telem_result.payload.get("temperature_status", "unknown")
            warn = " ⚠" if temp_status in {"warning", "critical"} else ""
            self.temp_label.configure(text=f"Temperature: {temp:.2f} C ({temp_status}){warn}")
            if temp_status == "critical":
                self.temp_label.configure(foreground="red")
            elif temp_status == "warning":
                self.temp_label.configure(foreground="orange")
            else:
                self.temp_label.configure(foreground="black")

    def _run_optimization(self) -> None:
        config = json.loads(self.optimization_settings.get())
        config["wgs"] = {
            "max_iterations": int(self.optimization_max_iters.get()),
            "gain": float(self.optimization_gain.get()),
        }
        config["ratio_mode"] = self.ratio_mode.get()
        config["target_definition"] = {
            "beam_count": int(self.ratio_beam_count.get()),
            "beam_positions": json.loads(self.ratio_positions.get()),
            "desired_ratios": json.loads(self.ratio_desired.get()),
            "lattice": {
                "geometry": self.ratio_lattice_geometry.get(),
                "spacing": float(self.ratio_lattice_spacing.get()),
                "rotation_deg": float(self.ratio_lattice_rotation.get()),
            },
        }
        config["ratio_targets"] = {"desired_ratios": json.loads(self.ratio_desired.get())}
        config["objective_weights"] = {
            "intensity": float(self.weight_intensity.get()),
            "ratio": float(self.weight_ratio.get()),
            "regularization": float(self.weight_regularization.get()),
        }
        result = self.controller.start_optimization(config)
        self._handle_result(result)
        if result.success and isinstance(result.payload, dict):
            metrics = result.payload.get("ratio_metrics", {})
            if isinstance(metrics, dict) and metrics:
                self.ratio_metrics_var.set(
                    f"ratio metrics: MAE={metrics.get('mean_abs_error', 0.0):.4f}, max={metrics.get('max_abs_error', 0.0):.4f}"
                )

    def _export_optimization_history(self) -> None:
        run_id = self._effective_run_override()
        self._handle_result(self.controller.export_optimization_history(run_id=run_id))

    def _run_sequence(self) -> None:
        steps = json.loads(self.sequence_entry.get())
        self._handle_result(self.controller.run_sequence(steps))

    def _refresh_logs(self) -> None:
        for item in self.log_tree.get_children():
            self.log_tree.delete(item)
        for entry in self.controller.state.logs[-300:]:
            self.log_tree.insert("", tk.END, values=(entry.timestamp, entry.level.value.upper(), entry.source, entry.message))

    def _refresh_progress_widgets(self) -> None:
        for name, bar in (
            ("optimization", self.optimization_progress),
            ("calibration", self.calibration_progress),
            ("sequence", self.sequence_progress),
        ):
            progress = self.controller.state.task_progress[name]
            if progress.total > 0:
                bar.configure(maximum=progress.total)
                bar["value"] = min(progress.current, progress.total)
            else:
                bar.configure(maximum=100)
                bar["value"] = 0

        opt = self.controller.optimization_progress()
        if opt.success and isinstance(opt.payload, dict):
            payload = opt.payload
            pct = float(payload.get("percent", 0.0))
            self.optimization_progress_label.configure(
                text=f"iter {int(payload.get('iteration', 0))}/{int(payload.get('max_iterations', 0))} ({pct:.1f}%)"
            )

    def _schedule_progress_refresh(self) -> None:
        self._safe_callback("progress_refresh", self._refresh_progress_widgets)
        self.after(800, self._schedule_progress_refresh)

    def _schedule_plot_refresh(self) -> None:
        self._safe_callback("plot_refresh", self._refresh_plot_widgets)
        self.after(500, self._schedule_plot_refresh)

    def _refresh_plot_widgets(self) -> None:
        for widget in self.plot_widgets.values():
            widget.refresh()

    def _set_plot_status(self, result: OperationResult | str) -> None:
        if isinstance(result, OperationResult):
            self._handle_result(result)
        else:
            self.status_var.set(result)

    def _pop_plot(self) -> None:
        self._open_plot_popout(self.plot_select.get())

    def _open_plot_popout(self, plot_name: str, geometry: str = "700x560") -> None:
        if plot_name in self.plot_popouts and self.plot_popouts[plot_name].winfo_exists():
            self.plot_popouts[plot_name].lift()
            return

        popout = tk.Toplevel(self)
        popout.title(f"Plot: {plot_name}")
        popout.geometry(geometry)
        popout.minsize(420, 320)
        widget = PlotWidget(popout, self.controller, plot_name, self._set_plot_status)
        widget.pack(fill=tk.BOTH, expand=True)
        widget.refresh()

        def _on_close() -> None:
            self.plot_popouts.pop(plot_name, None)
            popout.destroy()

        popout.protocol("WM_DELETE_WINDOW", _on_close)
        self.plot_popouts[plot_name] = popout

    def _export_plot(self) -> None:
        run_id = self._effective_run_override()
        self._handle_result(self.controller.export_plot(self.plot_select.get(), run_id=run_id))

    def _save_snapshot(self) -> None:
        run_id = self._effective_run_override()
        self._handle_result(self.controller.save_session_snapshot(run_id=run_id))

    def _choose_output_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.output_dir_var.get() or ".")
        if selected:
            self.output_dir_var.set(selected)

    def _effective_run_override(self) -> str | None:
        value = self.run_override_var.get().strip()
        return value or None

    def _apply_output_settings(self) -> None:
        folder = self.output_dir_var.get().strip()
        if not folder:
            self.status_var.set("Output directory is required")
            return
        result = self.controller.configure_output(
            folder=folder,
            template=self.naming_template_var.get().strip(),
            collision_policy=self.collision_policy_var.get(),
        )
        self._handle_result(result)

    def _preview_naming(self) -> None:
        run_id = self._effective_run_override() or "run"
        result = self.controller.output_name_preview("preview", run_id)
        self._handle_result(result)
        if result.success:
            self.name_preview_var.set(f"Name preview: {result.payload}")

    def _move_selected_panel_to_column(self) -> None:
        panel_name = self.arrange_panel.get()
        self.panel_columns[panel_name] = self.target_column.get()
        self._repack_panels(self._current_layout_columns())

    def _move_selected_panel(self, direction: int) -> None:
        panel_name = self.arrange_panel.get()
        columns = self._current_layout_columns()
        for col_name, panel_names in columns.items():
            if panel_name in panel_names:
                idx = panel_names.index(panel_name)
                new_idx = max(0, min(len(panel_names) - 1, idx + direction))
                panel_names.insert(new_idx, panel_names.pop(idx))
                self._repack_panels(columns)
                return

    def _swap_panels(self) -> None:
        first = self.arrange_panel.get()
        second = self.swap_with.get()
        if first == second:
            return
        columns = self._current_layout_columns()
        positions = {}
        for col_name, panel_names in columns.items():
            for idx, panel_name in enumerate(panel_names):
                positions[panel_name] = (col_name, idx)
        if first not in positions or second not in positions:
            return
        col_a, idx_a = positions[first]
        col_b, idx_b = positions[second]
        columns[col_a][idx_a], columns[col_b][idx_b] = columns[col_b][idx_b], columns[col_a][idx_a]
        self.panel_columns[first], self.panel_columns[second] = col_b, col_a
        self._repack_panels(columns)

    def _apply_visibility(self) -> None:
        self._repack_panels(self._current_layout_columns())

    def _current_layout_columns(self) -> dict[str, list[str]]:
        columns: dict[str, list[str]] = {"left": [], "center": [], "right": []}
        for panel_name in self.PANEL_NAMES:
            col = self.panel_columns.get(panel_name, "left")
            columns[col].append(panel_name)
        return columns

    def _repack_panels(self, columns: dict[str, list[str]]) -> None:
        for page_columns in self.page_columns.values():
            for frame in page_columns.values():
                for child in frame.pack_slaves():
                    child.pack_forget()

        for col_name in ["left", "center", "right"]:
            for panel_name in columns.get(col_name, []):
                self.panel_columns[panel_name] = col_name
                if not self.visibility_vars[panel_name].get():
                    continue
                page_key = self.PANEL_TO_PAGE[panel_name]
                panel = self.panel_frames[panel_name]
                if panel_name == "Plots":
                    panel.pack(in_=self.page_columns[page_key][col_name], fill=tk.BOTH, expand=True, padx=6, pady=6)
                else:
                    panel.pack(in_=self.page_columns[page_key][col_name], fill=tk.X, padx=6, pady=6)

    def _capture_layout_model(self) -> dict:
        popouts = []
        for plot_name, window in list(self.plot_popouts.items()):
            if window.winfo_exists():
                popouts.append({"plot_name": plot_name, "geometry": window.geometry()})
        return {
            "window_geometry": self.geometry(),
            "columns": self._current_layout_columns(),
            "visibility": {name: var.get() for name, var in self.visibility_vars.items()},
            "current_page": self.page_nav_var.get(),
            "sashes": {
                page_key: [pane.sashpos(0), pane.sashpos(1)]
                for page_key, pane in self.page_panels.items()
            },
            "popout_plots": popouts,
        }

    def _apply_layout_model(self, model: dict) -> None:
        self.geometry(model.get("window_geometry", "1400x900"))

        for panel_name, is_visible in model.get("visibility", {}).items():
            if panel_name in self.visibility_vars:
                self.visibility_vars[panel_name].set(bool(is_visible))

        columns = model.get("columns", self.store.default_layout_model()["columns"])
        for col_name in ["left", "center", "right"]:
            for panel_name in columns.get(col_name, []):
                if panel_name in self.panel_columns:
                    self.panel_columns[panel_name] = col_name
        self._repack_panels(columns)

        for page_key, pane in self.page_panels.items():
            sashes = model.get("sashes", {}).get(page_key, [])
            if len(sashes) >= 1:
                self.after(1, lambda key=page_key, value=sashes[0]: self.page_panels[key].sashpos(0, int(value)))
            if len(sashes) >= 2:
                self.after(1, lambda key=page_key, value=sashes[1]: self.page_panels[key].sashpos(1, int(value)))

        page_title = model.get("current_page")
        if isinstance(page_title, str):
            for key, title in self.PAGE_TITLES.items():
                if title == page_title:
                    self.select_page(key)
                    break

        for window in list(self.plot_popouts.values()):
            if window.winfo_exists():
                window.destroy()
        self.plot_popouts.clear()
        for popout in model.get("popout_plots", []):
            self._open_plot_popout(popout.get("plot_name", "simulated_phase"), popout.get("geometry", "500x400"))

    def save_layout(self) -> None:
        self.store.save_layout_model(self.layout_path, self._capture_layout_model())

    def restore_layout(self) -> None:
        model = self.store.load_layout_model(self.layout_path)
        self._apply_layout_model(model)

    def _reset_layout(self) -> None:
        self._apply_layout_model(self.store.default_layout_model())

    def _apply_preset(self, preset_name: str) -> None:
        self._apply_layout_model(self.store.preset_layout_model(preset_name))

    def _on_close(self) -> None:
        self.save_layout()
        self.destroy()


def launch() -> None:
    app = MainWindow(AppController())
    app.mainloop()
