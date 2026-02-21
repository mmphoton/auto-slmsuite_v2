# User workflow CLI and calibration persistence

## Centralized run naming/output API
All workflow scripts now accept shared output naming controls:

- `--run-name`
- `--output-root`
- `--name-template`
- `--overwrite` / `--resume`

The naming template supports fields like:
`{date}_{run_name}_{pattern}_{camera}_{iter}`

Collisions auto-increment by default (`_001`, `_002`, ...).

## Calibration producer
Run `run_calibration.py` first to generate calibration artifacts consumed by camera workflows.

```bash
python user_workflows/run_calibration.py \
  --factory my_lab.bootstrap:create_fourier_slm \
  --phase-lut /path/to/deep_1024.mat \
  --run-name cal_session_a \
  --name-template "{date}_{run_name}_{pattern}_{camera}_{iter}" \
  --output-root user_workflows/output
```

## Persistent file layout
Calibration files for downstream consumers are still written under `--calibration-root` (default `user_workflows/calibrations`):

- `phase-depth-lut.npy`
- `fourier-calibration.h5`
- `wavefront-superpixel-calibration.h5`
- `source-phase-corrected.npy` (if available)
- `source-amplitude-corrected.npy`

Additionally, each run creates a run directory containing a manifest and produced run artifacts.

Example run tree:

```text
user_workflows/output/
└── 20260221_cal_session_a_calibration_none_000/
    ├── manifest.json
    ├── metrics.json
    ├── phase-depth-lut.npy
    └── source-amplitude-corrected.npy
```

## Consumer scripts
Consumer scripts should validate calibration files before hardware operations:

```python
from user_workflows.calibration_io import assert_required_calibration_files
assert_required_calibration_files("user_workflows/calibrations")
```

## Andor image acquisition + feedback workflow
Use `run_slm_andor.py` to:
- display configurable SLM analytical patterns,
- optionally acquire Andor full-frame images,
- optionally run experimental feedback optimization,
- save all run artifacts (`phase`, `frames`, `metrics`, `plots`, `manifest`) via `OutputManager`.

Example (camera + feedback + custom naming):

```bash
python user_workflows/run_slm_andor.py \
  --use-camera \
  --feedback \
  --feedback-iters 15 \
  --run-name lg_feedback \
  --output-root user_workflows/output \
  --name-template "{date}_{run_name}_{pattern}_{camera}_{iter}"
```

Resulting run tree (example):

```text
user_workflows/output/
└── 20260221_lg_feedback_laguerre-gaussian_andor_000/
    ├── manifest.json
    ├── metrics.json
    ├── phase.npy
    ├── frames/
    │   ├── frame_000.npy
    │   └── frame_001.npy
    └── plots/
        └── first_frame.png
```

### Pattern options in `run_slm_andor.py`
Select from `--pattern`:
- `single-gaussian`
- `double-gaussian`
- `gaussian-lattice`
- `laguerre-gaussian`
