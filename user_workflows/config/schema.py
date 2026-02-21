"""Configuration schema for user workflow scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_PATTERNS = {
    "single-gaussian",
    "double-gaussian",
    "gaussian-lattice",
    "laguerre-gaussian",
}


@dataclass
class HardwareConfig:
    slm_preselect: str = "index:0"
    camera_serial: str = ""
    exposure_s: float = 0.03
    cooling_target_c: float = -65.0
    shutter_mode: str = "auto"
    use_camera: bool = False
    frames: int = 1

    def validate(self) -> None:
        if self.exposure_s <= 0:
            raise ValueError("hardware.exposure_s must be > 0")
        if self.frames < 1:
            raise ValueError("hardware.frames must be >= 1")


@dataclass
class PatternConfig:
    pattern_type: str = "laguerre-gaussian"
    family_params: dict[str, Any] = field(
        default_factory=lambda: {
            "blaze_kx": 0.0,
            "blaze_ky": 0.0045,
            "lg_l": 3,
            "lg_p": 0,
            "single_kx": 0.0,
            "single_ky": 0.0,
            "double_center_kx": 0.0,
            "double_center_ky": 0.0,
            "double_sep_kxy": 0.02,
            "lattice_nx": 5,
            "lattice_ny": 5,
            "lattice_pitch_x": 0.01,
            "lattice_pitch_y": 0.01,
            "lattice_center_kx": 0.0,
            "lattice_center_ky": 0.0,
            "holo_method": "WGS-Kim",
            "holo_maxiter": 30,
        }
    )

    def validate(self) -> None:
        if self.pattern_type not in SUPPORTED_PATTERNS:
            raise ValueError(f"pattern.pattern_type must be one of {sorted(SUPPORTED_PATTERNS)}")

        nx = int(self.family_params.get("lattice_nx", 1))
        ny = int(self.family_params.get("lattice_ny", 1))
        if nx <= 0 or ny <= 0:
            raise ValueError("pattern.family_params lattice_nx/lattice_ny must be positive")

        for key in ("single_kx", "single_ky", "double_center_kx", "double_center_ky", "lattice_center_kx", "lattice_center_ky"):
            value = float(self.family_params.get(key, 0.0))
            if abs(value) > 0.5:
                raise ValueError(f"pattern.family_params.{key} must be in [-0.5, 0.5]")

        for key in ("blaze_kx", "blaze_ky", "double_sep_kxy", "lattice_pitch_x", "lattice_pitch_y"):
            if float(self.family_params.get(key, 0.0)) < 0:
                raise ValueError(f"pattern.family_params.{key} must be nonnegative")

        if int(self.family_params.get("holo_maxiter", 0)) < 0:
            raise ValueError("pattern.family_params.holo_maxiter must be nonnegative")


@dataclass
class FeedbackConfig:
    enabled: bool = False
    method: str = "WGS-Kim"
    maxiter: int = 10
    balancing_target: float = 1.0
    stopping_thresholds: dict[str, float] = field(default_factory=lambda: {"relative_error": 1e-3})

    def validate(self) -> None:
        if self.maxiter < 0:
            raise ValueError("feedback.maxiter must be nonnegative")
        if self.balancing_target <= 0:
            raise ValueError("feedback.balancing_target must be > 0")
        for name, value in self.stopping_thresholds.items():
            if float(value) < 0:
                raise ValueError(f"feedback.stopping_thresholds.{name} must be nonnegative")


@dataclass
class OutputConfig:
    root_dir: str = "user_workflows/runs"
    run_name: str = ""
    save_options: dict[str, Any] = field(default_factory=lambda: {"save_frames": False, "frames_path": ""})
    plot_options: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.root_dir:
            raise ValueError("output.root_dir must be non-empty")


@dataclass
class CalibrationConfig:
    paths: dict[str, str] = field(
        default_factory=lambda: {
            "root": "user_workflows/calibrations",
            "phase_lut": "deep_1024.mat",
            "phase_lut_key": "deep",
            "factory": "",
        }
    )
    compatibility_policy: str = "strict"
    force_fourier: bool = False

    def validate(self) -> None:
        if "root" not in self.paths or not self.paths["root"]:
            raise ValueError("calibration.paths.root must be set")
        if self.compatibility_policy not in {"strict", "warn", "ignore"}:
            raise ValueError("calibration.compatibility_policy must be one of strict, warn, ignore")


@dataclass
class WorkflowConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    lut_file: str = "deep_1024.mat"
    lut_key: str = "deep"

    def validate(self) -> None:
        self.hardware.validate()
        self.pattern.validate()
        self.feedback.validate()
        self.output.validate()
        self.calibration.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def output_dir(self) -> Path:
        return Path(self.output.root_dir) / self.output.run_name if self.output.run_name else Path(self.output.root_dir)


__all__ = [
    "WorkflowConfig",
    "HardwareConfig",
    "PatternConfig",
    "FeedbackConfig",
    "OutputConfig",
    "CalibrationConfig",
]
