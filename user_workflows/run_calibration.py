"""Backward-compatible wrapper for `workflow calibrate`."""

from __future__ import annotations

import argparse
import importlib
import pprint
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from user_workflows.config import dump_yaml, load_workflow_config
from user_workflows.config.schema import WorkflowConfig





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


def _apply_legacy_overrides(args, config: WorkflowConfig):
    raw = config.to_dict()
    legacy_map = {
        "factory": "calibration.paths.factory",
        "phase_lut": "calibration.paths.phase_lut",
        "phase_lut_key": "calibration.paths.phase_lut_key",
        "calibration_root": "calibration.paths.root",
        "force_fourier": "calibration.force_fourier",
    }
    for arg_name, path in legacy_map.items():
        value = getattr(args, arg_name)
        if value is None:
            continue
        warnings.warn(
            f"CLI flag '--{arg_name.replace('_', '-')}' is deprecated; use --config/--set ({path}) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        cursor = raw
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value

    updated = WorkflowConfig(
        lut_file=raw["lut_file"],
        lut_key=raw["lut_key"],
        hardware=type(config.hardware)(**raw["hardware"]),
        pattern=type(config.pattern)(**raw["pattern"]),
        feedback=type(config.feedback)(**raw["feedback"]),
        output=type(config.output)(**raw["output"]),
        calibration=type(config.calibration)(**raw["calibration"]),
    )
    updated.validate()
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--set", action="append", default=[], help="Override config values via dotted key=value")

    parser.add_argument("--factory", default=None, help="module:function that returns FourierSLM")
    parser.add_argument("--phase-lut", default=None, help="Path to phase-depth LUT (.mat/.npy)")
    parser.add_argument("--phase-lut-key", default=None, help="Variable name in .mat LUT file")
    parser.add_argument("--calibration-root", default=None, help="Output directory")
    parser.add_argument("--force-fourier", dest="force_fourier", action="store_const", const=True, default=None)
    args = parser.parse_args()

    config = load_workflow_config(config_path=args.config, overrides=args.set)
    config = _apply_legacy_overrides(args, config)

    calibration_root = Path(config.calibration.paths["root"])
    calibration_root.mkdir(parents=True, exist_ok=True)

    run_dir = config.output_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "resolved_config.yaml"
    dump_yaml(config.to_dict(), resolved_path)

    print("Resolved config:")
    pprint.pprint(config.to_dict(), sort_dicts=False)
    print(f"Saved resolved config: {resolved_path.resolve()}")

    factory = config.calibration.paths.get("factory", "")
    if not factory:
        raise ValueError("calibration.paths.factory must be set (or provide --factory)")
    fs = _load_fourier_slm(factory)

    phase_lut = _load_phase_lut(
        Path(config.calibration.paths["phase_lut"]),
        config.calibration.paths.get("phase_lut_key", "deep"),
    )
    np.save(calibration_root / "phase-depth-lut.npy", phase_lut)
    output.save_phase(phase_lut, filename="phase-depth-lut.npy")

    fs.slm.phase_lut = phase_lut

    fourier_path = calibration_root / "fourier-calibration.h5"
    if config.calibration.force_fourier or not fourier_path.exists():
        fs.fourier_calibrate()
        fs.save_calibration("fourier", path=str(calibration_root), name="fourier-calibration")
        output.register_file(fourier_path, "calibration")
    else:
        fs.load_calibration("fourier", str(fourier_path))

    fs.wavefront_calibrate_superpixel()
    fs.wavefront_calibration_superpixel_process()

    fs.save_calibration(
        "wavefront_superpixel",
        path=str(calibration_root),
        name="wavefront-superpixel-calibration",
    )
    output.register_file(calibration_root / "wavefront-superpixel-calibration.h5", "calibration")

    source_phase = fs.slm.source.get("phase")
    if source_phase is not None:
        np.save(calibration_root / "source-phase-corrected.npy", source_phase)
        output.save_phase(source_phase, filename="source-phase-corrected.npy")

    source_amplitude = fs.slm.source.get("amplitude")
    if source_amplitude is None:
        raise RuntimeError(
            "Wavefront processing did not produce `slm.source['amplitude']`; cannot initialize WGS source amplitude."
        )
    np.save(calibration_root / "source-amplitude-corrected.npy", source_amplitude)
    output.save_phase(source_amplitude, filename="source-amplitude-corrected.npy")

    output.save_manifest(
        {
            "calibration_root": str(calibration_root.resolve()),
            "force_fourier": bool(args.force_fourier),
        }
    )

    print("Calibration workflow completed.")
    print(f"Artifacts written under: {calibration_root.resolve()}")
    print(f"Run manifest written under: {output.run_dir.resolve()}")


if __name__ == "__main__":
    main()
