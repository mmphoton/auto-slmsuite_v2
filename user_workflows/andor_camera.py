"""Andor camera helpers using pylablib's AndorSDK2 interface.

This follows the same connection approach used in ``Andor_example1.py``:
``from pylablib.devices import Andor`` and ``Andor.AndorSDK2Camera(idx=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AndorConnectionConfig:
    camera_serial: str = ""
    exposure_s: float = 0.03
    shutter_mode: str = "auto"


class PylablibAndorCamera:
    """Small camera adapter with the methods used by workflow code."""

    def __init__(self, config: AndorConnectionConfig, *, verbose: bool = True):
        try:
            from pylablib.devices import Andor
        except Exception as exc:
            raise ImportError("pylablib is required for Andor camera access.") from exc

        self._andor = Andor
        self._config = config
        self._verbose = bool(verbose)

        self._idx = _parse_camera_index(config.camera_serial)
        self.cam = self._andor.AndorSDK2Camera(idx=self._idx)
        self.cam.open()

        self._apply_defaults()
        self.set_exposure(config.exposure_s)

        frame = np.asarray(self.cam.snap(timeout=max(10.0, float(config.exposure_s) + 8.0)))
        self.shape = frame.shape
        self.bitdepth = 16
        self.name = f"AndorSDK2Camera_idx{self._idx}"

        if self._verbose:
            print(f"Connected to {self.name} with frame shape {self.shape}.")

    def _apply_defaults(self) -> None:
        try:
            self.cam.setup_shutter(mode=self._config.shutter_mode, ttl_mode=1)
        except Exception:
            pass
        try:
            self.cam.set_trigger_mode("int")
        except Exception:
            pass
        try:
            self.cam.set_read_mode("image")
        except Exception:
            pass

    def set_exposure(self, exposure_s: float):
        self.cam.set_exposure(float(exposure_s))

    def get_image(self):
        timeout_s = max(10.0, float(self.cam.get_exposure()) + 8.0)
        return np.asarray(self.cam.snap(timeout=timeout_s))

    def close(self):
        if self.cam is None:
            return
        try:
            self.cam.close()
        finally:
            self.cam = None


def verify_camera_discoverable(camera_serial: str = ""):
    """Open/close the Andor camera once to verify access."""
    cam = PylablibAndorCamera(AndorConnectionConfig(camera_serial=camera_serial), verbose=False)
    cam.close()


def _parse_camera_index(camera_serial: str) -> int:
    serial = (camera_serial or "").strip()
    if not serial:
        return 0
    if serial.isdigit():
        return int(serial)
    raise ValueError(
        "With pylablib AndorSDK2Camera, --camera-serial must be empty or a numeric camera index. "
        f"Got: {camera_serial!r}"
    )
