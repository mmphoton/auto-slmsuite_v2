"""Shared helpers for loading/saving calibration artifacts in user workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


REQUIRED_CALIBRATION_FILES = {
    "fourier": "fourier-calibration.h5",
    "wavefront_superpixel": "wavefront-superpixel-calibration.h5",
    "source_amplitude": "source-amplitude-corrected.npy",
}


def calibration_paths(calibration_root: str | Path) -> Dict[str, Path]:
    root = Path(calibration_root)
    return {key: root / filename for key, filename in REQUIRED_CALIBRATION_FILES.items()}


def assert_required_calibration_files(calibration_root: str | Path) -> Dict[str, Path]:
    paths = calibration_paths(calibration_root)
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required calibration artifacts. Run `python user_workflows/run_calibration.py` "
            "to generate them before running this workflow. Missing files:\n"
            f"{formatted}"
        )
    return paths
