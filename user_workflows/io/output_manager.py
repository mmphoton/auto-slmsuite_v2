"""Centralized run output persistence with manifest generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from user_workflows.io.run_naming import RunNamingConfig, choose_run_directory


class OutputManager:
    """Persist run artifacts in one run directory and track them in a manifest."""

    def __init__(
        self,
        config: RunNamingConfig,
        *,
        pattern: str,
        camera: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.config = config
        self.run_dir = choose_run_directory(config, pattern=pattern, camera=camera)
        self._files: list[dict[str, str]] = []
        self._metadata = {
            "run_name": config.run_name,
            "pattern": pattern,
            "camera": camera,
            "name_template": config.name_template,
            "overwrite": config.overwrite,
            "resume": config.resume,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            self._metadata.update(metadata)

        self.run_dir.mkdir(parents=True, exist_ok=True)

    def register_file(self, path: Path, kind: str) -> Path:
        try:
            stored = str(path.relative_to(self.run_dir))
        except ValueError:
            stored = str(path.resolve())
        self._files.append({"kind": kind, "path": stored})
        return path

    def _ensure_path(self, filename: str | Path) -> Path:
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not (self.config.overwrite or self.config.resume):
            raise FileExistsError(
                f"Refusing to overwrite existing file '{path}'. Use --overwrite or --resume if desired."
            )
        return path

    def save_phase(self, phase: np.ndarray, filename: str = "phase.npy") -> Path:
        path = self._ensure_path(filename)
        np.save(path, np.asarray(phase))
        return self.register_file(path, "phase")

    def save_frame(self, frame: np.ndarray, index: int | None = None) -> Path:
        suffix = "" if index is None else f"_{index:03d}"
        path = self._ensure_path(Path("frames") / f"frame{suffix}.npy")
        np.save(path, np.asarray(frame))
        return self.register_file(path, "frame")

    def save_metrics(self, metrics: dict[str, Any], filename: str = "metrics.json") -> Path:
        path = self._ensure_path(filename)
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        return self.register_file(path, "metrics")

    def save_plot(self, figure_or_array: Any, filename: str = "plot.png") -> Path:
        path = self._ensure_path(Path("plots") / filename)

        if hasattr(figure_or_array, "savefig"):
            figure_or_array.savefig(path, dpi=150, bbox_inches="tight")
        else:
            import matplotlib.pyplot as plt

            array = np.asarray(figure_or_array)
            fig, ax = plt.subplots()
            ax.imshow(array, cmap="viridis")
            ax.set_title(filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return self.register_file(path, "plot")

    def save_manifest(self, metadata: dict[str, Any] | None = None) -> Path:
        if metadata:
            self._metadata.update(metadata)

        payload = {
            "metadata": self._metadata,
            "files": self._files,
            "file_count": len(self._files),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        path = self._ensure_path("manifest.json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return self.register_file(path, "manifest")
