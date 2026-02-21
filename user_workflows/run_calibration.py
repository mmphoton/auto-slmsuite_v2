"""Backward-compatible wrapper for `workflow calibrate`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from user_workflows.io.output_manager import OutputManager
from user_workflows.io.run_naming import add_naming_args, config_from_args



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="module:function that returns FourierSLM")
    parser.add_argument("--phase-lut", required=True, help="Path to phase-depth LUT (.mat/.npy)")
    parser.add_argument("--phase-lut-key", default="deep", help="Variable name in .mat LUT file")
    parser.add_argument("--calibration-root", default="user_workflows/calibrations", help="Output directory")
    parser.add_argument("--force-fourier", action="store_true", help="Force a fresh Fourier calibration")
    add_naming_args(parser)
    args = parser.parse_args()

    calibration_root = Path(args.calibration_root)
    calibration_root.mkdir(parents=True, exist_ok=True)

    fs = _load_fourier_slm(args.factory)
    output = OutputManager(
        config_from_args(args),
        pattern="calibration",
        camera="none",
        metadata={"workflow": "run_calibration"},
    )

    # (a) Phase-depth LUT loading/validation.
    phase_lut = _load_phase_lut(Path(args.phase_lut), args.phase_lut_key)
    np.save(calibration_root / "phase-depth-lut.npy", phase_lut)
    output.save_phase(phase_lut, filename="phase-depth-lut.npy")

    # Keep this attached to the SLM object for downstream scripts that read an attribute.
    fs.slm.phase_lut = phase_lut

    # (b) Fourier calibration (calibrate or load).
    fourier_path = calibration_root / "fourier-calibration.h5"
    if args.force_fourier or not fourier_path.exists():
        fs.fourier_calibrate()
        fs.save_calibration("fourier", path=str(calibration_root), name="fourier-calibration")
        output.register_file(fourier_path, "calibration")
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
    output.register_file(calibration_root / "wavefront-superpixel-calibration.h5", "calibration")

    source_phase = fs.slm.source.get("phase")
    if source_phase is not None:
        np.save(calibration_root / "source-phase-corrected.npy", source_phase)
        output.save_phase(source_phase, filename="source-phase-corrected.npy")

    # (e) Record corrected source amplitude for WGS initialization.
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
