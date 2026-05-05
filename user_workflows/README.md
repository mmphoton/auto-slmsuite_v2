# User workflows: calibration, pattern generation, and acquisition

This directory contains operator-facing **CLI workflows** for:

- generating and persisting calibration artifacts,
- displaying SLM patterns,
- optional Andor image acquisition and feedback loops.

## Migration note
**GUI removed; all operations are command-line driven from the repository root.**

## Common operations (copy/paste from repo root)
Run all commands from the project root (`auto-slmsuite_v2`) so imports resolve consistently.

### 1) Calibration workflow (via unified CLI)
```bash
python user_workflows/cli.py workflow calibrate \
  --factory my_lab.bootstrap:create_fourier_slm \
  --phase-lut /path/to/deep_1024.mat \
  --run-name cal_session_a \
  --name-template "{date}_{run_name}_{pattern}_{camera}_{iter}" \
  --output-root user_workflows/output
```

### 2) Pattern display workflow (via unified CLI)
Default blaze grating in code is `kx=0.0`, `ky=0.02` unless you override it with flags.【see `--blaze-kx/--blaze-ky` defaults】

#### 2.1 One Gaussian spot (set radius + blaze)
```bash
python user_workflows/cli.py workflow pattern \
  --pattern single-gaussian \
  --single-kx 0.00 \
  --single-ky 0.01 \
  --single-radius-kxy 0.003 \
  --spot-radius-points 16 \
  --blaze-kx 0.0 \
  --blaze-ky 0.02
```

#### 2.2 Pair of Gaussian spots (set radius + separation + blaze)
```bash
python user_workflows/cli.py workflow pattern \
  --pattern double-gaussian \
  --double-center-kx 0.00 \
  --double-center-ky 0.02 \
  --double-sep-kxy 0.03 \
  --double-radius-kxy 0.004 \
  --spot-radius-points 16 \
  --blaze-kx 0.0 \
  --blaze-ky 0.02
```

#### 2.3 Gaussian lattice (set radius + lattice separation + blaze)
```bash
python user_workflows/cli.py workflow pattern \
  --pattern gaussian-lattice \
  --lattice-nx 5 \
  --lattice-ny 5 \
  --lattice-pitch-x 0.01 \
  --lattice-pitch-y 0.01 \
  --lattice-radius-kxy 0.003 \
  --spot-radius-points 16 \
  --blaze-kx 0.0 \
  --blaze-ky 0.02
```

#### 2.4 Laguerre-Gaussian mode (any order + set radius + blaze)
```bash
python user_workflows/cli.py workflow pattern \
  --pattern laguerre-gaussian \
  --lg-l 4 \
  --lg-p 1 \
  --lg-radius-w 0.30 \
  --blaze-kx 0.0 \
  --blaze-ky 0.02
```

### 3) Camera acquisition workflow (via unified CLI)
```bash
python user_workflows/cli.py workflow acquire \
  --pattern gaussian-lattice \
  --lattice-nx 8 \
  --lattice-ny 6
```

### 4) Direct calibration script
```bash
python user_workflows/run_calibration.py \
  --factory my_lab.bootstrap:create_fourier_slm \
  --phase-lut /path/to/deep_1024.mat \
  --run-name cal_session_a \
  --name-template "{date}_{run_name}_{pattern}_{camera}_{iter}" \
  --output-root user_workflows/output
```

### 6) Direct SLM/Andor runner script
```bash
python user_workflows/run_slm_andor.py \
  --pattern laguerre-gaussian \
  --lg-l 2 \
  --lg-p 1
```

### 7) Two-Gaussian setup validation script (optional experimental WGS)
```bash
python user_workflows/two_gaussian_wgs_test.py \
  --lut-file deep_1024.mat \
  --separation-knm 30 \
  --blaze-k 0.003 \
  --run-experimental-wgs
```

Pattern workflow always applies LUT depth correction; pass `--lut-file` to choose the calibration LUT explicitly.

Note: this script uses SLM-only `knm` coordinates (`--center-knm-x`, `--center-knm-y`, `--separation-knm`) so it runs without a camera wrapper during initial display; center values are offsets from the FFT center (0,0 by default), and the generated phase uses native SLM resolution for compatibility with blaze addition.

## Spyder troubleshooting note
If launching from Spyder, make sure you are executing from the **project root** and that the repo root is on `sys.path`; otherwise imports like `user_workflows.*` may fail.

Practical checks:

```python
import os, sys
print(os.getcwd())
print(sys.path[0])
```

If needed, set Spyder working directory to the repository root before running commands.

## Calibration artifacts and validation
By default, calibration outputs are written under `user_workflows/calibrations/`:

- `phase-depth-lut.npy`
- `fourier-calibration.h5`
- `wavefront-superpixel-calibration.h5`
- `source-phase-corrected.npy` (if available)
- `source-amplitude-corrected.npy` (if available)

Validate required artifacts before downstream workflows:

```python
from user_workflows.calibration_io import assert_required_calibration_files

assert_required_calibration_files("user_workflows/calibrations")
```

## Developer quick checks
```bash
python user_workflows/cli.py --help
python user_workflows/run_calibration.py --help
python user_workflows/run_slm_andor.py --help
```

## SDK/HEDS bootstrap helper
To locate repo + HEDS SDK paths programmatically (matching your requested setup), use `user_workflows.bootstrap.bootstrap_runtime(...)` before creating hardware objects.


If Holoeye import fails with `SDK HEDS not installed`, pass explicit bootstrap roots on the same CLI call:
```bash
python user_workflows/cli.py workflow acquire   --repo-root "C:\Users\admin\Desktop\SLM_Python\v4\auto-slmsuite_v2-main"   --sdk-root "C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.1.0"   --pattern laguerre-gaussian   --lg-l 9   --lg-p 0
```
