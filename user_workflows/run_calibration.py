"""Run a full SLM calibration workflow and persist outputs for subsequent scripts."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import scipy.io

from user_workflows.simulation.sim_factory import build_simulated_fourier_slm


def _load_fourier_slm(factory_path: str):
    """Load a `module:function` factory that returns an initialized FourierSLM instance."""
    if ":" not in factory_path:
        raise ValueError("--factory must be in 'module:function' format.")

    module_name, function_name = factory_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    return factory()


def _load_phase_lut(path: Path, key: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Phase LUT file not found at '{path}'. Provide --phase-lut to a .mat/.npy file."
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", default="", help="module:function that returns FourierSLM")
    parser.add_argument("--phase-lut", required=True, help="Path to phase-depth LUT (.mat/.npy)")
    parser.add_argument("--phase-lut-key", default="deep", help="Variable name in .mat LUT file")
    parser.add_argument("--calibration-root", default="user_workflows/calibrations", help="Output directory")
    parser.add_argument("--force-fourier", action="store_true", help="Force a fresh Fourier calibration")
    parser.add_argument("--simulate", action="store_true", help="Use simulated SLM/camera stack")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for simulated runs")
    parser.add_argument(
        "--simulation-scenario",
        default="two-spot-imbalance",
        choices=["two-spot-imbalance", "n-spot-lattice-nonuniform", "high-noise-failure"],
        help="Synthetic simulation case",
    )
    args = parser.parse_args()

    if not args.simulate and not args.factory:
        raise ValueError("Provide --factory for hardware runs, or use --simulate.")

    np.random.seed(args.seed)

    calibration_root = Path(args.calibration_root)
    calibration_root.mkdir(parents=True, exist_ok=True)

    if args.simulate:
        fs = build_simulated_fourier_slm(seed=args.seed, scenario=args.simulation_scenario)
    else:
        fs = _load_fourier_slm(args.factory)

    # (a) Phase-depth LUT loading/validation.
    phase_lut = _load_phase_lut(Path(args.phase_lut), args.phase_lut_key)
    np.save(calibration_root / "phase-depth-lut.npy", phase_lut)

    # Keep this attached to the SLM object for downstream scripts that read an attribute.
    fs.slm.phase_lut = phase_lut

    # (b) Fourier calibration (calibrate or load).
    fourier_path = calibration_root / "fourier-calibration.h5"
    if args.force_fourier or not fourier_path.exists():
        fs.fourier_calibrate()
        fs.save_calibration("fourier", path=str(calibration_root), name="fourier-calibration")
    else:
        fs.load_calibration("fourier", str(fourier_path))

    # (c) Wavefront superpixel calibration + processing.
    fs.wavefront_calibrate_superpixel()
    fs.wavefront_calibration_superpixel_process()

    # (d) Persist outputs.
    fs.save_calibration(
        "wavefront_superpixel",
        path=str(calibration_root),
        name="wavefront-superpixel-calibration",
    )

    source_phase = fs.slm.source.get("phase")
    if source_phase is not None:
        np.save(calibration_root / "source-phase-corrected.npy", source_phase)

    # (e) Record corrected source amplitude for WGS initialization.
    source_amplitude = fs.slm.source.get("amplitude")
    if source_amplitude is None:
        raise RuntimeError(
            "Wavefront processing did not produce `slm.source['amplitude']`; cannot initialize WGS source amplitude."
        )
    np.save(calibration_root / "source-amplitude-corrected.npy", source_amplitude)

    metadata = {
        "simulate": bool(args.simulate),
        "seed": int(args.seed),
        "simulation_scenario": args.simulation_scenario if args.simulate else "",
    }
    np.save(calibration_root / "run-metadata.npy", metadata, allow_pickle=True)

    print("Calibration workflow completed.")
    print(f"Artifacts written under: {calibration_root.resolve()}")


if __name__ == "__main__":
    main()
