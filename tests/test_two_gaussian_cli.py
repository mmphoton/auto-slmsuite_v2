from user_workflows.two_gaussian_wgs_test import build_parser


def test_two_gaussian_parser_accepts_legacy_separation_kxy_alias():
    parser = build_parser()
    args = parser.parse_args(["--separation-kxy", "0.03"])
    assert args.separation_kxy == 0.03
