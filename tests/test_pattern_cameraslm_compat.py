from types import SimpleNamespace

import numpy as np

from user_workflows.commands.pattern import (
    _build_lattice_spot_kxy,
    _spot_hologram_cameraslm_arg,
    _spot_hologram_shape,
)


def test_spot_hologram_shape_matches_native_slm_shape():
    raw_slm = SimpleNamespace(shape=(1080, 1920))
    assert _spot_hologram_shape(raw_slm) == (1080, 1920)


def test_lattice_builder_returns_expected_shape_and_center():
    args = SimpleNamespace(
        lattice_nx=5,
        lattice_ny=5,
        lattice_pitch_x=0.01,
        lattice_pitch_y=0.02,
        lattice_center_kx=0.005,
        lattice_center_ky=-0.004,
    )
    spots = _build_lattice_spot_kxy(args)
    assert spots.shape == (2, 25)
    assert np.isclose(spots[0].mean(), args.lattice_center_kx)
    assert np.isclose(spots[1].mean(), args.lattice_center_ky)


def test_lattice_builder_contains_expected_edge_coordinates():
    args = SimpleNamespace(
        lattice_nx=3,
        lattice_ny=3,
        lattice_pitch_x=0.01,
        lattice_pitch_y=0.01,
        lattice_center_kx=0.0,
        lattice_center_ky=0.0,
    )
    spots = _build_lattice_spot_kxy(args)
    xs = np.unique(np.round(spots[0], 6))
    ys = np.unique(np.round(spots[1], 6))
    assert np.allclose(xs, np.array([-0.01, 0.0, 0.01]))
    assert np.allclose(ys, np.array([-0.01, 0.0, 0.01]))


def test_deprecated_cameraslm_helper_wraps_raw_slm_with_pitch():
    raw_slm = SimpleNamespace(pitch=(8e-6, 8e-6), shape=(1080, 1920))
    wrapped = _spot_hologram_cameraslm_arg(raw_slm)
    assert wrapped.slm is raw_slm
    assert wrapped.pitch == raw_slm.pitch


def test_deprecated_cameraslm_helper_passthrough_for_cameraslm():
    camera_slm = SimpleNamespace(slm=SimpleNamespace(pitch=(8e-6, 8e-6)))
    assert _spot_hologram_cameraslm_arg(camera_slm) is camera_slm
