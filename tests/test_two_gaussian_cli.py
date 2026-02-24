from types import SimpleNamespace

from user_workflows.two_gaussian_wgs_test import _compute_spot_knm, build_parser


def test_two_gaussian_parser_accepts_legacy_separation_kxy_alias():
    parser = build_parser()
    args = parser.parse_args(["--separation-kxy", "0.03"])
    assert args.separation_kxy == 0.03


def test_compute_spot_knm_defaults_center_spots_in_bounds():
    args = SimpleNamespace(separation_knm=30.0, center_knm_x=0.0, center_knm_y=0.0)
    shape = (2048, 2048)
    spots = _compute_spot_knm(shape, args)
    assert spots.shape == (2, 2)
    assert spots[0, 0] >= 0
    assert spots[1, 0] >= 0
    assert spots[0, 1] < shape[1]
    assert spots[1, 1] < shape[0]
