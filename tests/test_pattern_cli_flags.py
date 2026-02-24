from user_workflows.cli import build_parser


def test_pattern_cli_supports_disabling_phase_depth_correction():
    parser = build_parser()
    args = parser.parse_args(["workflow", "pattern", "--no-phase-depth-correction", "--dry-run"])
    assert args.use_phase_depth_correction is False


def test_acquire_cli_supports_disabling_phase_depth_correction():
    parser = build_parser()
    args = parser.parse_args(["workflow", "acquire", "--no-phase-depth-correction", "--dry-run"])
    assert args.use_phase_depth_correction is False
