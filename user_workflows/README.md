# User workflow files and calibration persistence

## Calibration producer
Run `run_calibration.py` first to generate the calibration artifacts consumed by all other run scripts.

Example:

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
- optionally run camera-driven experimental feedback optimization.

Example (with camera + feedback):

```bash
python user_workflows/run_slm_andor.py \
  --use-camera \
  --feedback \
  --feedback-iters 15 \
  --save-frames user_workflows/output/andor_frames.npy
```

Example (no camera, hold pattern indefinitely):

```bash
python user_workflows/run_slm_andor.py
```
