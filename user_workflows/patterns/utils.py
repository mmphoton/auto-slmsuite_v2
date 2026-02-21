"""Shared helpers for phase-pattern workflows."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from slmsuite.holography.toolbox.phase import blaze


def apply_depth_correction(phi: np.ndarray, deep: np.ndarray) -> np.ndarray:
    """Apply a 1D LUT-based depth correction to wrapped phase data.

    Parameters
    ----------
    phi
        Phase values in radians.
    deep
        1D LUT scale vector (typically loaded from ``deep_1024.mat``).
    """
    deep = np.asarray(deep).squeeze()
    if deep.ndim != 1:
        raise ValueError(f"Expected 1D LUT, got shape {deep.shape}")

    idx = np.clip(np.rint((phi / (2 * np.pi)) * (deep.size - 1)), 0, deep.size - 1).astype(int)
    correction_factor = deep[idx]
    corrected = (phi - np.pi) * correction_factor + np.pi
    return np.mod(corrected, 2 * np.pi)


def build_spot_solve_settings(
    *,
    method: str = "WGS-Kim",
    maxiter: int = 30,
    feedback: str = "computational",
    stat_groups: Iterable[str] = ("computational",),
    **extra: Any,
) -> dict[str, Any]:
    """Build consistent ``SpotHologram.optimize(...)`` kwargs for user workflows."""
    return {
        "method": method,
        "maxiter": int(maxiter),
        "feedback": feedback,
        "stat_groups": list(stat_groups),
        **extra,
    }


def add_blaze_and_wrap(
    *,
    base_phase: np.ndarray,
    grid: Any,
    blaze_vector: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Add blaze phase to ``base_phase`` and wrap to ``[0, 2π)``."""
    return np.mod(base_phase + blaze(grid=grid, vector=blaze_vector), 2 * np.pi)


def simulate_expected_farfield(
    phase: np.ndarray,
    amplitude: np.ndarray | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Optionally estimate expected farfield intensity from nearfield phase/amplitude."""
    if amplitude is None:
        amplitude = np.ones_like(phase, dtype=float)

    nearfield = np.asarray(amplitude, dtype=float) * np.exp(1j * np.asarray(phase, dtype=float))
    farfield = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(nearfield), norm="ortho")))

    if normalize:
        peak = float(np.nanmax(farfield))
        if peak > 0:
            farfield = farfield / peak

    return farfield
