"""Telemetry utilities for iterative experimental feedback workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np


@dataclass
class IterationMetrics:
    iteration: int
    mean_intensity: float
    median_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    uniformity_min_max: float
    coefficient_of_variation: float
    objective_value: float
    elapsed_time_s: float


@dataclass
class StopConfig:
    target_uniformity: float = 0.95
    max_no_improvement_iters: int = 5
    min_relative_improvement: float = 1e-3
    max_runtime_s: float = 120.0


class FeedbackTelemetry:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: list[IterationMetrics] = []

    @property
    def metrics(self) -> list[IterationMetrics]:
        return list(self._metrics)

    def record(self, metric: IterationMetrics) -> None:
        self._metrics.append(metric)

    def save(self) -> tuple[Path, Path]:
        csv_path = self.output_dir / "metrics.csv"
        json_path = self.output_dir / "metrics.json"

        rows = [asdict(m) for m in self._metrics]
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            csv_path.write_text("", encoding="utf-8")

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

        return csv_path, json_path


def extract_spot_intensities(frame: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """Extract bright-pixel spot proxies from a frame."""
    arr = np.asarray(frame, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0.0])
    threshold = np.percentile(arr, percentile)
    spots = arr[arr >= threshold]
    return spots if spots.size else arr


def metric_from_spots(iteration: int, spot_values: Iterable[float], elapsed_time_s: float) -> IterationMetrics:
    spots = np.asarray(list(spot_values), dtype=float)
    if spots.size == 0:
        spots = np.array([0.0])

    mean_val = float(np.mean(spots))
    median_val = float(median(spots.tolist()))
    std_val = float(np.std(spots))
    min_val = float(np.min(spots))
    max_val = float(np.max(spots))
    uniformity = float(min_val / max_val) if max_val > 0 else 0.0
    cv = float(std_val / mean_val) if mean_val > 0 else 0.0
    objective = float(cv)

    return IterationMetrics(
        iteration=iteration,
        mean_intensity=mean_val,
        median_intensity=median_val,
        std_intensity=std_val,
        min_intensity=min_val,
        max_intensity=max_val,
        uniformity_min_max=uniformity,
        coefficient_of_variation=cv,
        objective_value=objective,
        elapsed_time_s=float(elapsed_time_s),
    )


def should_stop(
    metrics: list[IterationMetrics],
    stop_cfg: StopConfig,
    no_improvement_iters: int,
    start_time_s: float,
    now_s: float,
) -> tuple[bool, str]:
    if not metrics:
        return False, ""

    latest = metrics[-1]
    runtime = now_s - start_time_s

    if latest.uniformity_min_max >= stop_cfg.target_uniformity:
        return True, "target_uniformity_reached"

    if no_improvement_iters >= stop_cfg.max_no_improvement_iters:
        return True, "max_no_improvement_iters"

    if runtime >= stop_cfg.max_runtime_s:
        return True, "max_runtime_s"

    return False, ""
