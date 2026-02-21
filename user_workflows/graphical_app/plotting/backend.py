"""Plot backend with export and per-plot settings persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from user_workflows.graphical_app.app.interfaces import PlotBackendInterface, PlotExport


class PlotBackend(PlotBackendInterface):
    def __init__(self) -> None:
        self.data: Dict[str, np.ndarray] = {}
        self.settings: Dict[str, Dict[str, Any]] = {}

    def update(self, plot_name: str, data: np.ndarray) -> None:
        self.data[plot_name] = np.asarray(data)

    def configure(self, plot_name: str, settings: Mapping[str, Any]) -> None:
        self.settings.setdefault(plot_name, {}).update(dict(settings))

    def export(self, plot_name: str, output_dir: Path) -> PlotExport:
        output_dir.mkdir(parents=True, exist_ok=True)
        array = self.data.get(plot_name, np.zeros((1, 1)))
        data_path = output_dir / f"{plot_name}.npy"
        image_path = output_dir / f"{plot_name}.txt"
        metadata_path = output_dir / f"{plot_name}.json"
        np.save(data_path, array)
        image_path.write_text("image placeholder for GUI backend\n")
        metadata_path.write_text(json.dumps(self.settings.get(plot_name, {}), indent=2))
        return PlotExport(image_path=image_path, data_path=data_path, metadata_path=metadata_path)
