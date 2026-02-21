"""Tkinter-based dockable/resizable UI exposing backend features."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from user_workflows.graphical_app.app.controller import AppController


class MainWindow(tk.Tk):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self.controller = controller
        self.title("SLM Suite Graphical App")
        self.geometry("1400x900")
        self.layout_path = Path("user_workflows/output/gui_layout.json")

        self.main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        self.left = ttk.Frame(self.main_pane)
        self.center = ttk.Frame(self.main_pane)
        self.right = ttk.Frame(self.main_pane)
        self.main_pane.add(self.left, weight=1)
        self.main_pane.add(self.center, weight=2)
        self.main_pane.add(self.right, weight=1)

        self._build_device_panel()
        self._build_slm_panel()
        self._build_camera_panel()
        self._build_plot_panel()
        self._build_session_panel()

    def _build_device_panel(self) -> None:
        frm = ttk.LabelFrame(self.left, text="Device Manager")
        frm.pack(fill=tk.X, padx=6, pady=6)
        self.mode = tk.StringVar(value="simulation")
        ttk.Radiobutton(frm, text="Simulation", variable=self.mode, value="simulation", command=self._set_mode).pack(anchor="w")
        ttk.Radiobutton(frm, text="Hardware", variable=self.mode, value="hardware", command=self._set_mode).pack(anchor="w")
        ttk.Button(frm, text="Discover", command=self.controller.discover_devices).pack(fill=tk.X)
        ttk.Button(frm, text="Connect", command=self.controller.connect_devices).pack(fill=tk.X)
        ttk.Button(frm, text="Reconnect", command=self.controller.reconnect_devices).pack(fill=tk.X)
        ttk.Button(frm, text="Release SLM", command=self.controller.release_slm).pack(fill=tk.X)
        ttk.Button(frm, text="Release Camera", command=self.controller.release_camera).pack(fill=tk.X)
        ttk.Button(frm, text="Release Both", command=self.controller.release_both).pack(fill=tk.X)

    def _build_slm_panel(self) -> None:
        frm = ttk.LabelFrame(self.center, text="SLM Controls")
        frm.pack(fill=tk.X, padx=6, pady=6)
        pattern_options = self.controller.available_patterns().payload or []
        self.pattern_name = tk.StringVar(value="single-gaussian")
        ttk.Combobox(frm, textvariable=self.pattern_name, values=pattern_options).pack(fill=tk.X)
        self.param_entry = ttk.Entry(frm)
        self.param_entry.insert(0, '{"kx":0.0,"ky":0.01}')
        self.param_entry.pack(fill=tk.X)
        ttk.Button(frm, text="Simulate Before Apply", command=self._simulate).pack(fill=tk.X)
        ttk.Button(frm, text="Apply", command=self._apply).pack(fill=tk.X)
        ttk.Button(frm, text="Queue", command=self._queue).pack(fill=tk.X)
        ttk.Button(frm, text="Clear Queue", command=self.controller.clear_pattern_queue).pack(fill=tk.X)

    def _build_camera_panel(self) -> None:
        frm = ttk.LabelFrame(self.right, text="Camera Controls")
        frm.pack(fill=tk.X, padx=6, pady=6)
        self.camera_settings = ttk.Entry(frm)
        self.camera_settings.insert(0, '{"exposure_ms":10,"gain":1.0,"roi":[0,0,128,128],"binning":1,"trigger":"internal","fps":30,"acquisition_mode":"single"}')
        self.camera_settings.pack(fill=tk.X)
        ttk.Button(frm, text="Apply Camera Settings", command=self._configure_camera).pack(fill=tk.X)
        self.temp_label = ttk.Label(frm, text="Temperature: n/a")
        self.temp_label.pack(anchor="w")
        ttk.Button(frm, text="Refresh Telemetry", command=self._telemetry).pack(fill=tk.X)

    def _build_plot_panel(self) -> None:
        frm = ttk.LabelFrame(self.center, text="Plot Workspace")
        frm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.plot_select = tk.StringVar(value="simulated_phase")
        ttk.Combobox(frm, textvariable=self.plot_select, values=["simulated_phase", "simulated_intensity", "experimental_intensity", "optimization_convergence"]).pack(fill=tk.X)
        ttk.Button(frm, text="Pop-out Plot", command=self._pop_plot).pack(fill=tk.X)
        ttk.Button(frm, text="Export Plot", command=self._export_plot).pack(fill=tk.X)

    def _build_session_panel(self) -> None:
        frm = ttk.LabelFrame(self.left, text="Session / Reproducibility")
        frm.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(frm, text="Save Layout", command=self.save_layout).pack(fill=tk.X)
        ttk.Button(frm, text="Restore Layout", command=self.restore_layout).pack(fill=tk.X)
        ttk.Button(frm, text="Reset Layout", command=self._reset_layout).pack(fill=tk.X)
        ttk.Button(frm, text="Save Session Snapshot", command=self._save_snapshot).pack(fill=tk.X)

    def _set_mode(self) -> None:
        self.controller.set_mode(self.mode.get())

    def _simulate(self) -> None:
        params = json.loads(self.param_entry.get())
        pattern_result = self.controller.generate_pattern(self.pattern_name.get(), params)
        if pattern_result.success and pattern_result.payload is not None:
            self.controller.simulate_before_apply(pattern_result.payload)

    def _apply(self) -> None:
        params = json.loads(self.param_entry.get())
        pattern_result = self.controller.generate_pattern(self.pattern_name.get(), params)
        if pattern_result.success and pattern_result.payload is not None:
            self.controller.apply_pattern(pattern_result.payload)

    def _queue(self) -> None:
        params = json.loads(self.param_entry.get())
        pattern_result = self.controller.generate_pattern(self.pattern_name.get(), params)
        if pattern_result.success and pattern_result.payload is not None:
            self.controller.queue_pattern(pattern_result.payload)

    def _configure_camera(self) -> None:
        self.controller.configure_camera(json.loads(self.camera_settings.get()))

    def _telemetry(self) -> None:
        telem_result = self.controller.camera_telemetry()
        if telem_result.success and isinstance(telem_result.payload, dict):
            temp = telem_result.payload["temperature_c"]
            warn = " ⚠" if temp > -55 else ""
            self.temp_label.configure(text=f"Temperature: {temp:.2f} C{warn}")

    def _pop_plot(self) -> None:
        w = tk.Toplevel(self)
        w.geometry("500x400")
        ttk.Label(w, text=f"{self.plot_select.get()} (zoom/pan/reset via backend settings)").pack(fill=tk.BOTH, expand=True)

    def _export_plot(self) -> None:
        output = filedialog.askdirectory() or "user_workflows/output"
        self.controller.export_plot(self.plot_select.get(), output)

    def _save_snapshot(self) -> None:
        self.controller.save_session_snapshot("user_workflows/output/session_snapshot.json")

    def save_layout(self) -> None:
        sash = self.main_pane.sashpos(0)
        self.layout_path.parent.mkdir(parents=True, exist_ok=True)
        self.layout_path.write_text(json.dumps({"sash0": sash}))

    def restore_layout(self) -> None:
        if self.layout_path.exists():
            sash0 = json.loads(self.layout_path.read_text()).get("sash0", 300)
            self.main_pane.sashpos(0, int(sash0))

    def _reset_layout(self) -> None:
        self.main_pane.sashpos(0, 300)


def launch() -> None:
    app = MainWindow(AppController())
    app.mainloop()
