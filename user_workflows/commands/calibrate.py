"""Calibration workflow command helpers."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import scipy.io


def load_fourier_slm(factory_path: str):
    if ":" not in factory_path:
        raise ValueError("--factory must be in 'module:function' format.")
    module_name, function_name = factory_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    return factory()


def load_phase_lut(path: Path, key: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Phase LUT file not found at '{path}'. Fix: pass --phase-lut to a valid .mat/.npy file."
        )

    if path.suffix.lower() == ".mat":
        mat = scipy.io.loadmat(path)
        if key not in mat:
            raise KeyError(f"Phase LUT '{path}' must contain variable '{key}'.")
        lut = np.asarray(mat[key]).squeeze()
    elif path.suffix.lower() == ".npy":
        lut = np.asarray(np.load(path)).squeeze()
    else:
        raise ValueError("Phase LUT must be .mat (default variable 'deep') or .npy.")

    if lut.ndim != 1:
        raise ValueError(f"Phase LUT must be 1D; got shape {lut.shape}.")
    if lut.size < 2:
        raise ValueError("Phase LUT must contain at least two entries.")
    if not np.all(np.isfinite(lut)):
        raise ValueError("Phase LUT contains NaN or inf entries.")

    return lut.astype(float)


def add_calibration_args(parser: argparse.ArgumentParser):
    parser.add_argument("--factory", required=True, help="module:function that returns FourierSLM")
    parser.add_argument("--phase-lut", required=True, help="Path to phase-depth LUT (.mat/.npy)")
    parser.add_argument("--phase-lut-key", default="deep", help="Variable name in .mat LUT file")
    parser.add_argument("--calibration-root", default="user_workflows/calibrations", help="Output directory")
    parser.add_argument("--force-fourier", action="store_true", help="Force a fresh Fourier calibration")


def run_calibration(args):
    calibration_root = Path(args.calibration_root)
    calibration_root.mkdir(parents=True, exist_ok=True)
    phase_lut = load_phase_lut(Path(args.phase_lut), args.phase_lut_key)

    if args.dry_run:
        print(f"[dry-run] calibration config is valid. Output root: {calibration_root.resolve()}")
        return

    fs = load_fourier_slm(args.factory)
    np.save(calibration_root / "phase-depth-lut.npy", phase_lut)
    fs.slm.phase_lut = phase_lut

    fourier_path = calibration_root / "fourier-calibration.h5"
    if args.force_fourier or not fourier_path.exists():
        fs.fourier_calibrate()
        fs.save_calibration("fourier", path=str(calibration_root), name="fourier-calibration")
    else:
        fs.load_calibration("fourier", str(fourier_path))

    fs.wavefront_calibrate_superpixel()
    fs.wavefront_calibration_superpixel_process()
    fs.save_calibration("wavefront_superpixel", path=str(calibration_root), name="wavefront-superpixel-calibration")

    source_phase = fs.slm.source.get("phase")
    if source_phase is not None:
        np.save(calibration_root / "source-phase-corrected.npy", source_phase)

    source_amplitude = fs.slm.source.get("amplitude")
    if source_amplitude is None:
        raise RuntimeError(
            "Wavefront processing did not produce source amplitude. Fix: rerun with `workflow calibrate --force-fourier`."
        )
    np.save(calibration_root / "source-amplitude-corrected.npy", source_amplitude)

    print("Calibration workflow completed.")
    print(f"Artifacts written under: {calibration_root.resolve()}")
