from types import SimpleNamespace

from user_workflows.commands.pattern import _spot_hologram_cameraslm_arg


def test_spot_hologram_uses_none_for_raw_slm_objects():
    raw_slm = SimpleNamespace(shape=(1080, 1920))
    assert _spot_hologram_cameraslm_arg(raw_slm) is None


def test_spot_hologram_uses_cameraslm_when_available():
    camera_slm = SimpleNamespace(slm=SimpleNamespace(shape=(1080, 1920)))
    assert _spot_hologram_cameraslm_arg(camera_slm) is camera_slm
