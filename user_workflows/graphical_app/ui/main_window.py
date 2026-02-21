"""Tkinter-based dockable/resizable UI exposing backend features."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from user_workflows.graphical_app.app.controller import AppController
from user_workflows.graphical_app.app.interfaces import OperationResult
from user_workflows.graphical_app.app.state import LogLevel


class MainWindow(tk.Tk):
    PANEL_NAMES = ["Device", "SLM", "Camera", "Plots", "Optimization", "Calibration", "Logs", "Session"]

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self.controller = controller
        self.store = self.controller.persistence
        self.title("SLM Suite Graphical App")
        self.layout_path = Path("user_workflows/output/gui_layout.json")

        self.panel_frames: dict[str, ttk.LabelFrame] = {}
        self.panel_columns: dict[str, str] = {}
        self.visibility_vars = {name: tk.BooleanVar(value=True) for name in self.PANEL_NAMES}
        self.plot_popouts: dict[str, tk.Toplevel] = {}

        self.main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)
        self.left = ttk.Frame(self.main_pane)
        self.center = ttk.Frame(self.main_pane)
        self.right = ttk.Frame(self.main_pane)
        self.column_frames = {"left": self.left, "center": self.center, "right": self.right}
        self.main_pane.add(self.left, weight=1)
        self.main_pane.add(self.center, weight=2)
        self.main_pane.add(self.right, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        self._build_all_panels()
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=4, pady=2)

        self.restore_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(800, self._schedule_progress_refresh)

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
        self._build_calibration_panel()
        self._build_logs_panel()
        self._build_session_panel()

    def _create_panel(self, panel_name: str, column: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(self.column_frames[column], text=f"{panel_name} Panel")
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
        self.pattern_name = tk.StringVar(value="single-gaussian")
        ttk.Combobox(frm, textvariable=self.pattern_name, values=pattern_options).pack(fill=tk.X)
        self.param_entry = ttk.Entry(frm)
        self.param_entry.insert(0, '{"kx":0.0,"ky":0.01}')
        self.param_entry.pack(fill=tk.X)
        ttk.Button(frm, text="Simulate Before Apply", command=self._bind_safe("simulate", self._simulate)).pack(fill=tk.X)
        ttk.Button(frm, text="Apply", command=self._bind_safe("apply", self._apply)).pack(fill=tk.X)
        ttk.Button(frm, text="Queue", command=self._bind_safe("queue", self._queue)).pack(fill=tk.X)
        ttk.Button(frm, text="Clear Queue", command=self._bind_safe("clear_queue", lambda: self._handle_result(self.controller.clear_pattern_queue()))).pack(fill=tk.X)

    def _build_camera_panel(self) -> None:
        frm = self._create_panel("Camera", "right")
        self.camera_settings = ttk.Entry(frm)
        self.camera_settings.insert(0, '{"exposure_ms":10,"gain":1.0,"roi":[0,0,128,128],"binning":1,"trigger":"internal","fps":30,"acquisition_mode":"single"}')
        self.camera_settings.pack(fill=tk.X)
        ttk.Button(frm, text="Apply Camera Settings", command=self._bind_safe("configure_camera", self._configure_camera)).pack(fill=tk.X)
        self.temp_label = ttk.Label(frm, text="Temperature: n/a")
        self.temp_label.pack(anchor="w")
        ttk.Button(frm, text="Refresh Telemetry", command=self._bind_safe("telemetry", self._telemetry)).pack(fill=tk.X)

    def _build_plot_panel(self) -> None:
        frm = self._create_panel("Plots", "center")
        self.plot_select = tk.StringVar(value="simulated_phase")
        ttk.Combobox(frm, textvariable=self.plot_select, values=["simulated_phase", "simulated_intensity", "experimental_intensity", "optimization_convergence"]).pack(fill=tk.X)
        ttk.Button(frm, text="Pop-out Plot", command=self._bind_safe("pop_plot", self._pop_plot)).pack(fill=tk.X)
        ttk.Button(frm, text="Export Plot", command=self._bind_safe("export_plot", self._export_plot)).pack(fill=tk.X)

    def _build_optimization_panel(self) -> None:
        frm = self._create_panel("Optimization", "left")
        self.optimization_settings = ttk.Entry(frm)
        self.optimization_settings.insert(0, '{"iterations":20}')
        self.optimization_settings.pack(fill=tk.X)
        ttk.Button(frm, text="Run Optimization", command=self._bind_safe("run_optimization", self._run_optimization)).pack(fill=tk.X)
        ttk.Button(frm, text="Cancel Optimization", command=self._bind_safe("cancel_optimization", lambda: self._handle_result(self.controller.cancel_optimization()))).pack(fill=tk.X)
        self.optimization_progress = ttk.Progressbar(frm, orient=tk.HORIZONTAL, mode="determinate", maximum=100)
        self.optimization_progress.pack(fill=tk.X, pady=3)

    def _build_calibration_panel(self) -> None:
        frm = self._create_panel("Calibration", "right")
        ttk.Label(frm, text="Calibration profile path:").pack(anchor="w")
        self.calibration_profile = ttk.Entry(frm)
        self.calibration_profile.insert(0, "user_workflows/output/calibration_profile.json")
        self.calibration_profile.pack(fill=tk.X)
        ttk.Button(frm, text="Run Calibration", command=self._bind_safe("run_calibration", self._run_calibration)).pack(fill=tk.X)
        ttk.Button(frm, text="Cancel Calibration", command=self._bind_safe("cancel_calibration", lambda: self._handle_result(self.controller.cancel_calibration()))).pack(fill=tk.X)
        ttk.Button(frm, text="Save Placeholder Profile", command=self._bind_safe("save_profile", self._save_calibration_profile)).pack(fill=tk.X)
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

    def _simulate(self) -> None:
        params = json.loads(self.param_entry.get())
        pattern_result = self.controller.generate_pattern(self.pattern_name.get(), params)
        self._handle_result(pattern_result)
        if pattern_result.success and pattern_result.payload is not None:
            self._handle_result(self.controller.simulate_before_apply(pattern_result.payload))

    def _apply(self) -> None:
        params = json.loads(self.param_entry.get())
        pattern_result = self.controller.generate_pattern(self.pattern_name.get(), params)
        self._handle_result(pattern_result)
        if pattern_result.success and pattern_result.payload is not None:
            self._handle_result(self.controller.apply_pattern(pattern_result.payload))

    def _queue(self) -> None:
        params = json.loads(self.param_entry.get())
        pattern_result = self.controller.generate_pattern(self.pattern_name.get(), params)
        self._handle_result(pattern_result)
        if pattern_result.success and pattern_result.payload is not None:
            self._handle_result(self.controller.queue_pattern(pattern_result.payload))

    def _configure_camera(self) -> None:
        self._handle_result(self.controller.configure_camera(json.loads(self.camera_settings.get())))

    def _telemetry(self) -> None:
        telem_result = self.controller.camera_telemetry()
        self._handle_result(telem_result)
        if telem_result.success and isinstance(telem_result.payload, dict):
            temp = float(telem_result.payload.get("temperature_c", 0.0))
            temp_status = telem_result.payload.get("temperature_status", "unknown")
            warn = " ⚠" if temp_status in {"warning", "critical"} else ""
            self.temp_label.configure(text=f"Temperature: {temp:.2f} C ({temp_status}){warn}")

    def _run_optimization(self) -> None:
        self._handle_result(self.controller.run_optimization(json.loads(self.optimization_settings.get())))

    def _run_calibration(self) -> None:
        self._handle_result(self.controller.run_calibration(self.calibration_profile.get()))

    def _run_sequence(self) -> None:
        steps = json.loads(self.sequence_entry.get())
        self._handle_result(self.controller.run_sequence(steps))

    def _save_calibration_profile(self) -> None:
        path = Path(self.calibration_profile.get())
        profile = {
            "name": "default",
            "mode": self.mode.get(),
            "slm_model": "simulated_slm",
            "camera_model": "simulated_camera",
            "matrix": [[1.0, 0.0], [0.0, 1.0]],
        }
        self.store.save_json(path, profile)
        self.status_var.set(f"Saved placeholder profile to {path}")

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

    def _schedule_progress_refresh(self) -> None:
        self._safe_callback("progress_refresh", self._refresh_progress_widgets)
        self.after(800, self._schedule_progress_refresh)

    def _pop_plot(self) -> None:
        self._open_plot_popout(self.plot_select.get())

    def _open_plot_popout(self, plot_name: str, geometry: str = "500x400") -> None:
        if plot_name in self.plot_popouts and self.plot_popouts[plot_name].winfo_exists():
            self.plot_popouts[plot_name].lift()
            return

        popout = tk.Toplevel(self)
        popout.title(f"Plot: {plot_name}")
        popout.geometry(geometry)
        ttk.Label(popout, text=f"{plot_name} (zoom/pan/reset via backend settings)").pack(fill=tk.BOTH, expand=True)

        def _on_close() -> None:
            self.plot_popouts.pop(plot_name, None)
            popout.destroy()

        popout.protocol("WM_DELETE_WINDOW", _on_close)
        self.plot_popouts[plot_name] = popout

    def _export_plot(self) -> None:
        output = filedialog.askdirectory() or "user_workflows/output"
        self._handle_result(self.controller.export_plot(self.plot_select.get(), output))

    def _save_snapshot(self) -> None:
        self._handle_result(self.controller.save_session_snapshot("user_workflows/output/session_snapshot.json"))

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

        for col_name in columns:
            packed_names: list[str] = []
            for child in self.column_frames[col_name].pack_slaves():
                for name, panel in self.panel_frames.items():
                    if panel is child:
                        packed_names.append(name)
                        break
            for panel_name in columns[col_name]:
                if panel_name not in packed_names:
                    packed_names.append(panel_name)
            columns[col_name] = packed_names

        return columns

    def _repack_panels(self, columns: dict[str, list[str]]) -> None:
        for col_name, frame in self.column_frames.items():
            for child in frame.pack_slaves():
                child.pack_forget()
            for panel_name in columns.get(col_name, []):
                self.panel_columns[panel_name] = col_name
                if self.visibility_vars[panel_name].get():
                    self.panel_frames[panel_name].pack(fill=tk.X, padx=6, pady=6)
        if self.visibility_vars["Plots"].get():
            self.panel_frames["Plots"].pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _capture_layout_model(self) -> dict:
        popouts = []
        for plot_name, window in list(self.plot_popouts.items()):
            if window.winfo_exists():
                popouts.append({"plot_name": plot_name, "geometry": window.geometry()})
        return {
            "window_geometry": self.geometry(),
            "columns": self._current_layout_columns(),
            "visibility": {name: var.get() for name, var in self.visibility_vars.items()},
            "sashes": {"main": [self.main_pane.sashpos(0), self.main_pane.sashpos(1)]},
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

        sashes = model.get("sashes", {}).get("main", [])
        if len(sashes) >= 1:
            self.after(1, lambda: self.main_pane.sashpos(0, int(sashes[0])))
        if len(sashes) >= 2:
            self.after(1, lambda: self.main_pane.sashpos(1, int(sashes[1])))

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
