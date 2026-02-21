"""Hardware control for Andor iDus cameras via :mod:`pyAndorSDK2`."""

from __future__ import annotations

import time
import warnings
import numpy as np

from slmsuite.hardware.cameras.camera import Camera

try:
    from pyAndorSDK2 import atmcd
except ImportError:
    atmcd = None
    warnings.warn("pyAndorSDK2 not installed. Install to use Andor iDus cameras.")


class AndorIDus(Camera):
    """Andor iDus camera adapter for full-frame image acquisition."""

    sdk = None
    _DRV_SUCCESS = 20002
    _TEMP_OK_CODES = {20034, 20035, 20036, 20037, 20040}

    def __init__(
        self,
        serial="",
        pitch_um=None,
        verbose=True,
        target_temperature_c=-65,
        shutter_mode="auto",
        **kwargs,
    ):
        if atmcd is None:
            raise ImportError("pyAndorSDK2 not installed. Install to use Andor iDus cameras.")

        if AndorIDus.sdk is None:
            if verbose:
                print("Andor SDK initializing... ", end="")
            AndorIDus.sdk = atmcd.atmcd()
            self._sdk_call(AndorIDus.sdk.Initialize, "")
            if verbose:
                print("success")

        self.cam = AndorIDus.sdk

        if serial:
            self._select_camera_by_serial(serial)

        self._configure_default_acquisition()
        self._configure_shutter(shutter_mode)
        self._configure_cooling(target_temperature_c, verbose=verbose)

        width, height = self._get_detector_shape()
        bitdepth = self._get_bitdepth()

        name = self._build_name(serial)

        super().__init__(
            (width, height),
            bitdepth=bitdepth,
            pitch_um=pitch_um,
            name=name,
            **kwargs,
        )

        self.set_woi(None)

        if verbose:
            print(f"Andor iDus '{name}' ready (target {target_temperature_c}C, shutter={shutter_mode}).")

    def close(self):
        if getattr(self, "cam", None) is None:
            return
        try:
            self.cam.AbortAcquisition()
        except Exception:
            pass
        try:
            self.cam.ShutDown()
        finally:
            self.cam = None
            AndorIDus.sdk = None

    @staticmethod
    def info(verbose=True):
        if atmcd is None:
            raise ImportError("pyAndorSDK2 not installed. Install to use Andor iDus cameras.")

        sdk = atmcd.atmcd()
        cams = []
        try:
            ret = sdk.Initialize("")
            if ret != AndorIDus._DRV_SUCCESS:
                raise RuntimeError(f"Andor SDK Initialize failed with code {ret}.")

            ret, count = sdk.GetAvailableCameras()
            if ret == AndorIDus._DRV_SUCCESS:
                for i in range(int(count)):
                    r_handle, handle = sdk.GetCameraHandle(i)
                    if r_handle != AndorIDus._DRV_SUCCESS:
                        continue
                    sdk.SetCurrentCamera(handle)
                    model = "AndorIDus"
                    serial = ""
                    r_m, m = sdk.GetHeadModel()
                    if r_m == AndorIDus._DRV_SUCCESS:
                        model = str(m)
                    r_s, s = sdk.GetCameraSerialNumber()
                    if r_s == AndorIDus._DRV_SUCCESS:
                        serial = str(s)
                    cams.append(f"{model}_{serial}" if serial else model)

            if verbose:
                print("Andor cameras:")
                for c in cams:
                    print(f"'{c}'")
            return cams
        finally:
            try:
                sdk.ShutDown()
            except Exception:
                pass

    def _get_exposure_hw(self):
        ret, exposure_s, _, _ = self.cam.GetAcquisitionTimings()
        self._assert_success(ret, "GetAcquisitionTimings")
        return float(exposure_s)

    def _set_exposure_hw(self, exposure_s):
        self._sdk_call(self.cam.SetExposureTime, float(exposure_s))

    def set_woi(self, woi=None):
        width, height = self._get_detector_shape()
        if woi is None:
            x, w, y, h = 0, width, 0, height
        else:
            if len(woi) != 4:
                raise ValueError("Expected woi=(x, width, y, height).")
            x, w, y, h = [int(v) for v in woi]
            if w <= 0 or h <= 0:
                raise ValueError("WOI width/height must be positive.")
            if x < 0 or y < 0 or (x + w) > width or (y + h) > height:
                raise ValueError("WOI exceeds detector bounds.")

        # SDK is 1-indexed inclusive for image bounds.
        self._sdk_call(self.cam.SetImage, 1, 1, x + 1, x + w, y + 1, y + h)
        self.woi = (x, w, y, h)
        self.shape = (h, w)
        return self.woi

    def _get_image_hw(self, timeout_s):
        timeout_ms = int(max(1, float(timeout_s) * 1000))

        self._sdk_call(self.cam.StartAcquisition)
        wait_ret = self.cam.WaitForAcquisitionTimeOut(timeout_ms)
        if wait_ret != self._DRV_SUCCESS:
            self.cam.AbortAcquisition()
            raise TimeoutError(f"Andor WaitForAcquisitionTimeOut failed with code {wait_ret}.")

        npx = int(self.shape[0] * self.shape[1])
        ret, frame = self.cam.GetAcquiredData16(npx)
        self._assert_success(ret, "GetAcquiredData16")

        return np.asarray(frame, dtype=np.uint16).reshape(self.shape)

    def _configure_default_acquisition(self):
        self._sdk_call(self.cam.SetReadMode, 4)         # Full image mode.
        self._sdk_call(self.cam.SetAcquisitionMode, 1)  # Single scan per trigger.
        self._sdk_call(self.cam.SetTriggerMode, 0)      # Internal trigger.

    def _configure_shutter(self, shutter_mode):
        mode_map = {"auto": 0, "open": 1, "closed": 2}
        if shutter_mode not in mode_map:
            raise ValueError("shutter_mode must be one of: auto, open, closed")
        # Typical iDus usage: output TTL=1, requested mode, 0ms open/close times.
        self._sdk_call(self.cam.SetShutter, 1, mode_map[shutter_mode], 0, 0)

    def _configure_cooling(self, target_temperature_c, verbose=True):
        target = int(target_temperature_c)
        self._sdk_call(self.cam.SetTemperature, target)
        self._sdk_call(self.cam.CoolerON)

        # Brief stabilization wait; camera remains cooled for the full session.
        deadline = time.time() + 20.0
        while time.time() < deadline:
            ret, current = self.cam.GetTemperature()
            if ret in self._TEMP_OK_CODES:
                if verbose:
                    print(f"Andor cooling active at {current}C (target {target}C).")
                return
            time.sleep(0.2)

        if verbose:
            print("Andor cooling enabled; temperature still settling.")

    def _build_name(self, serial_hint):
        model = "AndorIDus"
        serial = ""
        try:
            r_m, m = self.cam.GetHeadModel()
            if r_m == self._DRV_SUCCESS:
                model = str(m)
            r_s, s = self.cam.GetCameraSerialNumber()
            if r_s == self._DRV_SUCCESS:
                serial = str(s)
        except Exception:
            serial = str(serial_hint) if serial_hint else ""
        return f"{model}_{serial}" if serial else model

    def _select_camera_by_serial(self, serial):
        ret, count = self.cam.GetAvailableCameras()
        self._assert_success(ret, "GetAvailableCameras")

        serial = str(serial)
        for idx in range(int(count)):
            r_h, handle = self.cam.GetCameraHandle(idx)
            self._assert_success(r_h, "GetCameraHandle")
            self._sdk_call(self.cam.SetCurrentCamera, handle)
            r_s, found = self.cam.GetCameraSerialNumber()
            if r_s == self._DRV_SUCCESS and str(found) == serial:
                return

        raise RuntimeError(f"Andor serial '{serial}' not found among connected cameras.")

    def _get_detector_shape(self):
        ret, width, height = self.cam.GetDetector()
        self._assert_success(ret, "GetDetector")
        return int(width), int(height)

    def _get_bitdepth(self):
        try:
            ret, n_channels = self.cam.GetNumberADChannels()
            self._assert_success(ret, "GetNumberADChannels")
            depths = []
            for i in range(int(n_channels)):
                r_d, d = self.cam.GetBitDepth(i)
                if r_d == self._DRV_SUCCESS:
                    depths.append(int(d))
            return max(depths) if depths else 16
        except Exception:
            return 16

    def _sdk_call(self, fn, *args):
        ret = fn(*args)
        self._assert_success(ret, getattr(fn, "__name__", "SDK call"))

    def _assert_success(self, ret, op_name):
        if ret != self._DRV_SUCCESS:
            raise RuntimeError(f"Andor {op_name} failed with code {ret}.")
