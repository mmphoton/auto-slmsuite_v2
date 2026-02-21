"""Plot helpers for SLM diagnostics workflows.

This module centralizes plotting primitives that are commonly useful while
optimizing holograms or running camera-in-the-loop feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class PlotMetadata:
    """Metadata describing plot rendering choices for reproducibility."""

    phase_wrapped_clim: tuple[float, float] = (0.0, 2 * np.pi)
    phase_unwrapped_clim: tuple[float, float] | None = None
    farfield_norm: str = "max"
    camera_norm: str = "max"
    camera_clim: tuple[float, float] | None = None


def _normalize(image: np.ndarray, method: str = "max") -> np.ndarray:
    data = np.asarray(image, dtype=float)
    if method == "max":
        denom = np.nanmax(data)
        denom = denom if denom > 0 else 1.0
        return data / denom
    if method == "sum":
        denom = np.nansum(data)
        denom = denom if denom > 0 else 1.0
        return data / denom
    if method == "none":
        return data
    raise ValueError(f"Unsupported normalization method '{method}'.")


def plot_phase_map(
    wrapped_phase: np.ndarray,
    unwrapped_phase: np.ndarray | None = None,
    metadata: PlotMetadata | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create wrapped/unwrapped phase map views."""

    metadata = metadata or PlotMetadata()
    unwrapped = wrapped_phase if unwrapped_phase is None else unwrapped_phase

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    wrapped_im = axes[0].imshow(wrapped_phase, cmap="twilight", vmin=metadata.phase_wrapped_clim[0], vmax=metadata.phase_wrapped_clim[1])
    axes[0].set_title("Phase (wrapped)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(wrapped_im, ax=axes[0], fraction=0.046, pad=0.04)

    u_kwargs = {}
    if metadata.phase_unwrapped_clim is not None:
        u_kwargs = {
            "vmin": metadata.phase_unwrapped_clim[0],
            "vmax": metadata.phase_unwrapped_clim[1],
        }
    unwrapped_im = axes[1].imshow(unwrapped, cmap="viridis", **u_kwargs)
    axes[1].set_title("Phase (unwrapped)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(unwrapped_im, ax=axes[1], fraction=0.046, pad=0.04)

    return fig, axes


def plot_farfield(
    amplitude: np.ndarray,
    intensity: np.ndarray | None = None,
    metadata: PlotMetadata | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create expected farfield amplitude/intensity views."""

    metadata = metadata or PlotMetadata()
    amp = _normalize(amplitude, metadata.farfield_norm)
    inten = amp**2 if intensity is None else _normalize(intensity, metadata.farfield_norm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    amp_im = axes[0].imshow(amp, cmap="magma")
    axes[0].set_title("Expected farfield amplitude")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(amp_im, ax=axes[0], fraction=0.046, pad=0.04)

    int_im = axes[1].imshow(inten, cmap="inferno")
    axes[1].set_title("Expected farfield intensity")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(int_im, ax=axes[1], fraction=0.046, pad=0.04)

    return fig, axes


def plot_camera_image(
    raw: np.ndarray,
    metadata: PlotMetadata | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create raw and normalized camera image views."""

    metadata = metadata or PlotMetadata()
    normalized = _normalize(raw, metadata.camera_norm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    raw_kwargs = {}
    if metadata.camera_clim is not None:
        raw_kwargs = {"vmin": metadata.camera_clim[0], "vmax": metadata.camera_clim[1]}
    raw_im = axes[0].imshow(raw, cmap="gray", **raw_kwargs)
    axes[0].set_title("Camera image (raw)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(raw_im, ax=axes[0], fraction=0.046, pad=0.04)

    norm_im = axes[1].imshow(normalized, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Camera image (normalized)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(norm_im, ax=axes[1], fraction=0.046, pad=0.04)

    return fig, axes


def plot_spot_rois(
    image: np.ndarray,
    rois: Iterable[Mapping[str, float]],
    title: str = "Spot ROI overlays",
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay rectangular spot ROIs over an image.

    Each ROI mapping should include: x, y, width, height.
    """

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.imshow(image, cmap="gray")
    for idx, roi in enumerate(rois):
        patch = Rectangle(
            (roi["x"], roi["y"]),
            roi["width"],
            roi["height"],
            linewidth=1.5,
            edgecolor="cyan",
            facecolor="none",
        )
        ax.add_patch(patch)
        ax.text(roi["x"], roi["y"] - 2, f"ROI {idx}", color="cyan", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def plot_convergence_curves(
    iterations: np.ndarray,
    uniformity: np.ndarray | None = None,
    loss: np.ndarray | None = None,
    spot_intensities: np.ndarray | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot convergence metrics across optimization iterations."""

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    if uniformity is not None:
        axes[0].plot(iterations, uniformity, color="tab:blue")
    axes[0].set_title("Uniformity")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Uniformity")

    if loss is not None:
        axes[1].plot(iterations, loss, color="tab:red")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")

    if spot_intensities is not None:
        for idx in range(spot_intensities.shape[1]):
            axes[2].plot(iterations, spot_intensities[:, idx], alpha=0.8, label=f"Spot {idx}")
        if spot_intensities.shape[1] <= 8:
            axes[2].legend(loc="best", fontsize=8)
    axes[2].set_title("Per-spot intensity")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Intensity")

    return fig, axes
