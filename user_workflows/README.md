# User workflow CLI and calibration persistence

Use the unified CLI entrypoint:

```bash
python user_workflows/cli.py workflow <subcommand> [...]
```

Available subcommands:
- `workflow calibrate`
- `workflow pattern`
- `workflow acquire`
- `workflow feedback`
- `workflow doctor`

Each subcommand supports:
- `--preset <name>` to load argument defaults from `user_workflows/presets.json`
- `--preset-file <path>` to use a different profile file
- `--dry-run` to validate config and file paths without touching hardware

## Calibration producer

```bash
python user_workflows/cli.py workflow calibrate \
  --factory my_lab.bootstrap:create_fourier_slm \
  --phase-lut /path/to/deep_1024.mat
```

## Pattern/acquisition workflows

```bash
# Display a pattern only
python user_workflows/cli.py workflow pattern --pattern laguerre-gaussian

# Display pattern and acquire frames
python user_workflows/cli.py workflow acquire --pattern single-gaussian --frames 3

# Run experimental camera feedback
python user_workflows/cli.py workflow feedback --pattern gaussian-lattice --feedback-iters 15
```

## Doctor checks

```bash
python user_workflows/cli.py workflow doctor
```

The doctor command checks:
- SDK import availability
- camera discoverability
- LUT file existence and shape
- calibration artifact presence and compatibility
- writable output directories

All failures include an explicit "Fix" command/next step.

## Backward compatibility

Legacy scripts are still available as wrappers:
- `python user_workflows/run_calibration.py ...`
- `python user_workflows/run_slm_andor.py ...`
