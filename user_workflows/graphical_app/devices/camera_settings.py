"""Schema and validation helpers for camera settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


TRIGGER_MODES = {"internal", "external", "software"}
SHUTTER_MODES = {"rolling", "global"}
ACQUISITION_MODES = {"single", "continuous", "kinetic"}
ALLOWED_BINNING = {1, 2, 4, 8}


@dataclass
class CameraSettings:
    exposure_ms: float = 10.0
    gain: float = 1.0
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 128
    roi_height: int = 128
    binning: int = 1
    trigger_mode: str = "internal"
    shutter_mode: str = "rolling"
    fps: float = 30.0
    acquisition_mode: str = "single"

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["roi"] = [payload.pop("roi_x"), payload.pop("roi_y"), payload.pop("roi_width"), payload.pop("roi_height")]
        return payload


def camera_settings_schema() -> dict[str, Any]:
    return {
        "exposure_ms": {"type": "float", "minimum": 0.01, "maximum": 60000, "unit": "ms"},
        "gain": {"type": "float", "minimum": 0.0, "maximum": 100.0, "unit": "x"},
        "roi_x": {"type": "int", "minimum": 0, "unit": "px"},
        "roi_y": {"type": "int", "minimum": 0, "unit": "px"},
        "roi_width": {"type": "int", "minimum": 1, "unit": "px"},
        "roi_height": {"type": "int", "minimum": 1, "unit": "px"},
        "binning": {"type": "int", "enum": sorted(ALLOWED_BINNING), "unit": "px"},
        "trigger_mode": {"type": "str", "enum": sorted(TRIGGER_MODES)},
        "shutter_mode": {"type": "str", "enum": sorted(SHUTTER_MODES)},
        "fps": {"type": "float", "minimum": 0.1, "maximum": 1000.0, "unit": "Hz"},
        "acquisition_mode": {"type": "str", "enum": sorted(ACQUISITION_MODES)},
    }


def parse_camera_settings(raw: Mapping[str, Any]) -> CameraSettings:
    roi = raw.get("roi")
    roi_x = raw.get("roi_x", roi[0] if isinstance(roi, (list, tuple)) and len(roi) == 4 else 0)
    roi_y = raw.get("roi_y", roi[1] if isinstance(roi, (list, tuple)) and len(roi) == 4 else 0)
    roi_width = raw.get("roi_width", roi[2] if isinstance(roi, (list, tuple)) and len(roi) == 4 else 128)
    roi_height = raw.get("roi_height", roi[3] if isinstance(roi, (list, tuple)) and len(roi) == 4 else 128)

    settings = CameraSettings(
        exposure_ms=float(raw.get("exposure_ms", 10.0)),
        gain=float(raw.get("gain", 1.0)),
        roi_x=int(roi_x),
        roi_y=int(roi_y),
        roi_width=int(roi_width),
        roi_height=int(roi_height),
        binning=int(raw.get("binning", 1)),
        trigger_mode=str(raw.get("trigger_mode", raw.get("trigger", "internal"))),
        shutter_mode=str(raw.get("shutter_mode", "rolling")),
        fps=float(raw.get("fps", 30.0)),
        acquisition_mode=str(raw.get("acquisition_mode", "single")),
    )
    _validate(settings)
    return settings


def _validate(settings: CameraSettings) -> None:
    if settings.exposure_ms <= 0:
        raise ValueError("exposure_ms must be > 0")
    if settings.gain < 0:
        raise ValueError("gain must be >= 0")
    if settings.roi_x < 0 or settings.roi_y < 0:
        raise ValueError("roi origin must be >= 0")
    if settings.roi_width <= 0 or settings.roi_height <= 0:
        raise ValueError("roi width and height must be > 0")
    if settings.binning not in ALLOWED_BINNING:
        raise ValueError(f"binning must be one of {sorted(ALLOWED_BINNING)}")
    if settings.trigger_mode not in TRIGGER_MODES:
        raise ValueError(f"trigger_mode must be one of {sorted(TRIGGER_MODES)}")
    if settings.shutter_mode not in SHUTTER_MODES:
        raise ValueError(f"shutter_mode must be one of {sorted(SHUTTER_MODES)}")
    if settings.fps <= 0:
        raise ValueError("fps must be > 0")
    if settings.acquisition_mode not in ACQUISITION_MODES:
        raise ValueError(f"acquisition_mode must be one of {sorted(ACQUISITION_MODES)}")
