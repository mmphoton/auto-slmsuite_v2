# User workflow files and calibration persistence

## First run without hardware (`--simulate`)
Use simulation mode to validate pattern generation, feedback loops, and artifact outputs without an Andor camera or physical SLM.

### 1) Simulated calibration
```bash
python user_workflows/run_calibration.py \
  --simulate \
  --seed 7 \
  --simulation-scenario two-spot-imbalance \
  --phase-lut deep_1024.mat
```

### 2) Simulated pattern + feedback run
```bash
python user_workflows/run_slm_andor.py \
  --simulate \
  --pattern double-gaussian \
  --feedback \
  --seed 7 \
  --simulation-scenario two-spot-imbalance \
  --lut-file deep_1024.mat
```

Simulation scenarios:
- `two-spot-imbalance` (biased two-spot response)
- `n-spot-lattice-nonuniform` (lattice intensity nonuniformity)
- `high-noise-failure` (strong dark/read noise failure mode)

Both simulated and hardware workflows write artifacts into the same calibration/output layout for parity.

## Calibration producer
Run `run_calibration.py` first to generate the calibration artifacts consumed by all other run scripts.

Hardware example:

```bash
python user_workflows/run_calibration.py \
  --factory my_lab.bootstrap:create_fourier_slm \
  --phase-lut /path/to/deep_1024.mat
```

## Persistent file layout
By default all outputs are written to:

- `user_workflows/calibrations/phase-depth-lut.npy` — validated phase-depth LUT used for phase correction.
- `user_workflows/calibrations/fourier-calibration.h5` — output of `FourierSLM.fourier_calibrate(...)` (or loaded equivalent).
- `user_workflows/calibrations/wavefront-superpixel-calibration.h5` — output of `wavefront_calibrate_superpixel(...)`.
- `user_workflows/calibrations/source-phase-corrected.npy` — processed source phase map (if available).
- `user_workflows/calibrations/source-amplitude-corrected.npy` — processed source amplitude for WGS initialization.
- `user_workflows/calibrations/andor_frames.npy` — default frame capture output for both hardware and simulation runs.
- `user_workflows/calibrations/run-metadata.npy` — seed/simulation metadata for reproducibility.

## Consumer scripts
Consumer scripts must validate these files before execution and fail with explicit instructions if missing.

- `test_working.py` now enforces this precondition via `user_workflows/calibration_io.py` and prints an actionable error telling you to run `python user_workflows/run_calibration.py`.

When adding future `run_*.py` scripts, import and call:

```python
from user_workflows.calibration_io import assert_required_calibration_files
assert_required_calibration_files("user_workflows/calibrations")
```

before interacting with hardware.

## Andor image acquisition + feedback workflow
Use `run_slm_andor.py` to:
- keep the Andor CCD cooled to `-65C` while connected,
- set shutter control to `auto`,
- acquire full camera images of displayed SLM patterns,
- optionally run camera-driven experimental feedback optimization,
- or run identical pattern/feedback flow with `--simulate`.

Hardware example:

```bash
python user_workflows/run_slm_andor.py \
  --use-camera \
  --feedback \
  --feedback-iters 15
```

Simulation example:

```bash
python user_workflows/run_slm_andor.py \
  --simulate \
  --feedback \
  --feedback-iters 15 \
  --simulation-scenario high-noise-failure
```

### Pattern options in `run_slm_andor.py`
You can select from four analytical pattern families using `--pattern`:

- `single-gaussian` (single focused Gaussian-like spot)
- `double-gaussian` (two Gaussian-like spots separated by `--double-sep-kxy`)
- `gaussian-lattice` (rectangular lattice of Gaussian-like spots)
- `laguerre-gaussian` (LG phase mode with `--lg-l`, `--lg-p`)
