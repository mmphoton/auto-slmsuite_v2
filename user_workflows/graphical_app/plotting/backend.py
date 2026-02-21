"""Plot backend with interactive view state, per-plot settings persistence, and export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from user_workflows.graphical_app.app.interfaces import PlotBackendInterface, PlotExport


@dataclass
class PlotSettings:
    autoscale: bool = True
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    scale: str = "linear"
    colormap: str = "viridis"


@dataclass
class PlotModel:
    name: str
    data: np.ndarray
    settings: PlotSettings


class PlotBackend(PlotBackendInterface):
    REQUIRED_PLOTS = ("simulated_phase", "simulated_intensity")

    def __init__(self) -> None:
        self.data: Dict[str, np.ndarray] = {}
        self.settings: Dict[str, PlotSettings] = {}
        for plot_name in self.REQUIRED_PLOTS:
            self.update(plot_name, np.zeros((128, 128), dtype=float))

    def _ensure_plot(self, plot_name: str) -> None:
        if plot_name not in self.data:
            self.data[plot_name] = np.zeros((1, 1), dtype=float)
        if plot_name not in self.settings:
            self.settings[plot_name] = PlotSettings()

    def _autoscale_bounds(self, data: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        height, width = data.shape[:2]
        return (0.0, max(float(width - 1), 0.0)), (0.0, max(float(height - 1), 0.0))

    def update(self, plot_name: str, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=float)
        self._ensure_plot(plot_name)
        self.data[plot_name] = arr
        settings = self.settings[plot_name]
        if settings.autoscale:
            settings.xlim, settings.ylim = self._autoscale_bounds(arr)

    def configure(self, plot_name: str, settings: Mapping[str, Any]) -> None:
        self._ensure_plot(plot_name)
        plot_settings = self.settings[plot_name]
        for key, value in dict(settings).items():
            if key in {"xlim", "ylim"} and value is not None:
                value = tuple(float(v) for v in value)
            setattr(plot_settings, key, value)
        if plot_settings.autoscale:
            plot_settings.xlim, plot_settings.ylim = self._autoscale_bounds(self.data[plot_name])

    def get_plot_model(self, plot_name: str) -> PlotModel:
        self._ensure_plot(plot_name)
        return PlotModel(name=plot_name, data=self.data[plot_name], settings=self.settings[plot_name])

    def reset_view(self, plot_name: str) -> None:
        self._ensure_plot(plot_name)
        settings = self.settings[plot_name]
        settings.autoscale = True
        settings.scale = "linear"
        settings.colormap = "viridis"
        settings.xlim, settings.ylim = self._autoscale_bounds(self.data[plot_name])

    def zoom(self, plot_name: str, factor: float) -> None:
        self._ensure_plot(plot_name)
        if factor <= 0:
            return
        settings = self.settings[plot_name]
        if settings.xlim is None or settings.ylim is None:
            settings.xlim, settings.ylim = self._autoscale_bounds(self.data[plot_name])
        x0, x1 = settings.xlim or (0.0, 1.0)
        y0, y1 = settings.ylim or (0.0, 1.0)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        new_half_x = max((x1 - x0) * factor / 2.0, 1e-6)
        new_half_y = max((y1 - y0) * factor / 2.0, 1e-6)
        settings.xlim = (cx - new_half_x, cx + new_half_x)
        settings.ylim = (cy - new_half_y, cy + new_half_y)
        settings.autoscale = False

    def pan(self, plot_name: str, dx_fraction: float, dy_fraction: float) -> None:
        self._ensure_plot(plot_name)
        settings = self.settings[plot_name]
        if settings.xlim is None or settings.ylim is None:
            settings.xlim, settings.ylim = self._autoscale_bounds(self.data[plot_name])
        x0, x1 = settings.xlim or (0.0, 1.0)
        y0, y1 = settings.ylim or (0.0, 1.0)
        span_x = x1 - x0
        span_y = y1 - y0
        settings.xlim = (x0 + span_x * dx_fraction, x1 + span_x * dx_fraction)
        settings.ylim = (y0 + span_y * dy_fraction, y1 + span_y * dy_fraction)
        settings.autoscale = False

    def _normalize_data(self, data: np.ndarray, scale: str) -> np.ndarray:
        view = np.asarray(data, dtype=float)
        if view.ndim == 1:
            view = view[None, :]
        if scale == "log":
            view = np.log10(np.clip(view, 1e-9, None))
        min_v = float(np.nanmin(view)) if view.size else 0.0
        max_v = float(np.nanmax(view)) if view.size else 1.0
        if max_v - min_v <= 1e-12:
            return np.zeros_like(view)
        return np.clip((view - min_v) / (max_v - min_v), 0.0, 1.0)

    def _apply_colormap(self, norm: np.ndarray, colormap: str) -> np.ndarray:
        if colormap == "gray":
            rgb = np.stack([norm, norm, norm], axis=-1)
        elif colormap == "plasma":
            rgb = np.stack([norm, np.sqrt(norm), 1.0 - norm * 0.5], axis=-1)
        elif colormap == "magma":
            rgb = np.stack([np.power(norm, 0.6), norm * 0.3, 1.0 - norm], axis=-1)
        else:  # viridis default
            rgb = np.stack([0.3 + 0.7 * norm, np.power(norm, 0.7), 1.0 - 0.6 * norm], axis=-1)
        return np.clip((rgb * 255).astype(np.uint8), 0, 255)

    def render_rgb(self, plot_name: str) -> np.ndarray:
        model = self.get_plot_model(plot_name)
        data = model.data
        settings = model.settings

        xlim = settings.xlim
        ylim = settings.ylim
        if xlim is not None and ylim is not None and data.ndim >= 2:
            width = data.shape[1]
            height = data.shape[0]
            x0 = int(np.clip(np.floor(xlim[0]), 0, max(width - 1, 0)))
            x1 = int(np.clip(np.ceil(xlim[1]), 0, max(width - 1, 0)))
            y0 = int(np.clip(np.floor(ylim[0]), 0, max(height - 1, 0)))
            y1 = int(np.clip(np.ceil(ylim[1]), 0, max(height - 1, 0)))
            if x1 > x0 and y1 > y0:
                data = data[y0 : y1 + 1, x0 : x1 + 1]

        norm = self._normalize_data(data, settings.scale)
        return self._apply_colormap(norm, settings.colormap)

    def export(self, plot_name: str, output_dir: Path) -> PlotExport:
        self._ensure_plot(plot_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        array = self.data[plot_name]
        rgb = self.render_rgb(plot_name)
        data_path = output_dir / f"{plot_name}.npy"
        image_path = output_dir / f"{plot_name}.ppm"
        metadata_path = output_dir / f"{plot_name}.json"
        np.save(data_path, array)
        header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii")
        image_path.write_bytes(header + rgb.tobytes())
        metadata_path.write_text(
            json.dumps(
                {
                    "plot_name": plot_name,
                    "data_shape": list(array.shape),
                    "settings": asdict(self.settings[plot_name]),
                },
                indent=2,
            )
        )
        return PlotExport(image_path=image_path, data_path=data_path, metadata_path=metadata_path)
