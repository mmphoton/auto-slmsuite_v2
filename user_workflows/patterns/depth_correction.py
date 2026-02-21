"""Shared utilities for phase-depth correction."""

from __future__ import annotations

import numpy as np


def apply_depth_correction(phi: np.ndarray, deep: np.ndarray) -> np.ndarray:
    """Apply LUT-based depth correction and wrap phase to [0, 2*pi)."""
    idx = np.clip(np.rint((phi / (2 * np.pi)) * (deep.size - 1)), 0, deep.size - 1).astype(int)
    depth_factor = deep[idx]
    corrected = (phi - np.pi) * depth_factor + np.pi
    return np.mod(corrected, 2 * np.pi)
