import numpy as np

from user_workflows.cli import build_parser
from user_workflows.commands.pattern import _build_laguerre_gaussian_target


def test_pattern_cli_supports_disabling_phase_depth_correction():
    parser = build_parser()
    args = parser.parse_args(["workflow", "pattern", "--no-phase-depth-correction", "--dry-run"])
    assert args.use_phase_depth_correction is False


def test_acquire_cli_supports_disabling_phase_depth_correction():
    parser = build_parser()
    args = parser.parse_args(["workflow", "acquire", "--no-phase-depth-correction", "--dry-run"])
    assert args.use_phase_depth_correction is False


def test_acquire_cli_supports_experimental_wgs_and_camera_phase_plot_flags():
    parser = build_parser()
    args = parser.parse_args([
        "workflow",
        "acquire",
        "--experimental-wgs-iters",
        "10",
        "--show-camera-phase-plot",
        "--save-camera-phase-plot",
        "user_workflows/output/camera_phase.png",
    ])
    assert args.experimental_wgs_iters == 10
    assert args.show_camera_phase_plot is True
    assert args.save_camera_phase_plot.endswith("camera_phase.png")


def test_acquire_cli_accepts_bootstrap_roots():
    parser = build_parser()
    args = parser.parse_args([
        "workflow",
        "acquire",
        "--repo-root",
        r"C:\\repo",
        "--sdk-root",
        r"C:\\sdk",
        "--dry-run",
        "--no-phase-depth-correction",
    ])
    assert args.repo_root == r"C:\\repo"
    assert args.sdk_root == r"C:\\sdk"


def test_laguerre_gaussian_target_radius_changes_p0_ring_size():
    small = _build_laguerre_gaussian_target((96, 96), l=1, p=0, radius_w=0.15)
    large = _build_laguerre_gaussian_target((96, 96), l=1, p=0, radius_w=0.45)
    yy, xx = np.indices(small.shape, dtype=float)
    r = np.sqrt((xx - 47.5) ** 2 + (yy - 47.5) ** 2)

    small_radius = float((small * r).sum() / small.sum())
    large_radius = float((large * r).sum() / large.sum())

    assert large_radius > small_radius * 2.0
