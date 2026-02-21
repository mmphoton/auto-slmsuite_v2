import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from user_workflows.graphical_app.app.controller import AppController
from user_workflows.graphical_app.app.state import DeviceConnectionState


def test_device_actions_return_structured_feedback_and_release_independent():
    controller = AppController()

    discover = controller.discover_devices()
    assert discover.success
    assert discover.payload["slm"]["action"] == "discover"
    assert "telemetry" in discover.payload

    connect = controller.connect_devices()
    assert connect.success
    assert connect.payload["slm"]["connected"] is True
    assert connect.payload["camera"]["connected"] is True

    slm_release = controller.release_slm()
    assert slm_release.success
    assert slm_release.payload["connected"] is False
    assert controller.state.devices["slm"].state == DeviceConnectionState.DISCONNECTED

    camera_release = controller.release_camera()
    assert camera_release.success
    assert camera_release.payload["connected"] is False
    assert "telemetry" in camera_release.payload
    assert controller.state.devices["camera"].state == DeviceConnectionState.DISCONNECTED

    release_both = controller.release_both()
    assert release_both.success
    assert release_both.payload["slm"]["connected"] is False
    assert release_both.payload["camera"]["connected"] is False


def test_hardware_mode_surfaces_not_implemented_error_and_telemetry_status():
    controller = AppController()
    controller.set_mode("hardware")

    result = controller.connect_devices()
    assert result.success
    assert result.payload["slm"]["failed"] is True
    assert "not implemented" in (result.payload["slm"]["reason"] or "").lower()
    assert result.payload["camera"]["failed"] is True
    assert controller.state.camera_telemetry["temperature_status"] == "unknown"
