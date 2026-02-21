"""Persistence helpers for diagnostic plots, arrays, and metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DiagnosticsConfig:
    """Config toggles for diagnostic outputs."""

    save_plots: bool = True
    plot_every: int = 1
    show_plots: bool = False
    save_arrays: bool = True
    save_movie: bool = False


@dataclass
class DiagnosticsSaver:
    """Coordinate saving diagnostics at common hook points.

    Hook points:
      * initial state
      * every N iterations
      * final state
    """

    root: Path
    config: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.plots_dir = self.root / "plots"
        self.arrays_dir = self.root / "arrays"
        self.metrics_dir = self.root / "metrics"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self._movie_frames: list[Path] = []

    def should_capture_iteration(self, iteration: int) -> bool:
        every = max(1, int(self.config.plot_every))
        return iteration % every == 0

    def capture_initial(self, payload: dict[str, Any]) -> None:
        self.capture(hook="initial", iteration=0, payload=payload, force=True)

    def capture_iteration(self, iteration: int, payload: dict[str, Any]) -> None:
        if self.should_capture_iteration(iteration):
            self.capture(hook="iteration", iteration=iteration, payload=payload)

    def capture_final(self, iteration: int, payload: dict[str, Any]) -> None:
        self.capture(hook="final", iteration=iteration, payload=payload, force=True)
        if self.config.save_movie:
            self._write_movie()

    def capture(self, hook: str, iteration: int, payload: dict[str, Any], force: bool = False) -> None:
        """Persist plots, arrays, and metrics for a single hook event.

        Payload schema is intentionally flexible. Suggested keys:
          * figures: dict[str, matplotlib.figure.Figure]
          * arrays: dict[str, numpy.ndarray]
          * metrics: dict[str, float | int | str | list]
          * metadata: dict[str, Any]  # additional metadata fields
        """

        metadata = {
            "run_id": self.run_id,
            "iteration": int(iteration),
            "hook": hook,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "color_limits": payload.get("color_limits", {}),
            "normalization": payload.get("normalization", {}),
        }
        metadata.update(payload.get("metadata", {}))

        stem = f"{hook}_iter_{int(iteration):05d}"

        if self.config.save_plots or force:
            figures = payload.get("figures", {})
            self._save_figures(figures, stem)

        if self.config.save_arrays or force:
            arrays = payload.get("arrays", {})
            self._save_arrays(arrays, stem)

        metrics = payload.get("metrics", {})
        self._save_metrics(metrics, metadata, stem)

        if self.config.show_plots:
            plt.show(block=False)

    def _save_figures(self, figures: dict[str, Any], stem: str) -> None:
        for name, fig in figures.items():
            out = self.plots_dir / f"{stem}_{name}.png"
            fig.savefig(out, dpi=160, bbox_inches="tight")
            if self.config.save_movie and name in {"farfield", "camera", "phase"}:
                self._movie_frames.append(out)
            plt.close(fig)

    def _save_arrays(self, arrays: dict[str, np.ndarray], stem: str) -> None:
        for name, array in arrays.items():
            np.save(self.arrays_dir / f"{stem}_{name}.npy", np.asarray(array))

    def _save_metrics(self, metrics: dict[str, Any], metadata: dict[str, Any], stem: str) -> None:
        data = {"metadata": metadata, "metrics": metrics}
        with (self.metrics_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=_json_default)

    def _write_movie(self) -> None:
        if not self._movie_frames:
            return
        try:
            import imageio.v2 as imageio
        except ImportError:
            return

        frames = [imageio.imread(frame) for frame in self._movie_frames]
        gif_path = self.plots_dir / f"{self.run_id}_iterations.gif"
        mp4_path = self.plots_dir / f"{self.run_id}_iterations.mp4"
        imageio.mimsave(gif_path, frames, fps=6)
        try:
            imageio.mimsave(mp4_path, frames, fps=6)
        except Exception:
            # MP4 support may be unavailable depending on ffmpeg availability.
            pass


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
