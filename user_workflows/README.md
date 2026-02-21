# User workflows: calibration, pattern generation, and migration guide

This directory contains the operator-facing workflows for:

- generating/persisting calibration artifacts,
- displaying analytical or holographically-optimized SLM patterns,
- optional Andor image acquisition and experimental feedback.

---

## Calibration producer
Run `run_calibration.py` first to generate the calibration artifacts consumed by downstream workflows.

```bash
python user_workflows/run_calibration.py \
  --factory my_lab.bootstrap:create_fourier_slm \
  --phase-lut /path/to/deep_1024.mat \
  --run-name cal_session_a \
  --name-template "{date}_{run_name}_{pattern}_{camera}_{iter}" \
  --output-root user_workflows/output
```

### Persistent file layout
By default all outputs are written to `user_workflows/calibrations/`:

- `phase-depth-lut.npy` — validated phase-depth LUT used for phase correction.
- `fourier-calibration.h5` — output of `FourierSLM.fourier_calibrate(...)` (or loaded equivalent).
- `wavefront-superpixel-calibration.h5` — output of `wavefront_calibrate_superpixel(...)`.
- `source-phase-corrected.npy` — processed source phase map (if available).
- `source-amplitude-corrected.npy` — processed source amplitude for WGS initialization.

Consumer workflows should validate these files before execution and fail with actionable instructions if missing.

```python
from user_workflows.calibration_io import assert_required_calibration_files

assert_required_calibration_files("user_workflows/calibrations")
```

---

## Workflow architecture (script vs pattern modules vs registry)
The pattern system is organized into three conceptual layers:

```text
run_slm_andor.py (workflow script)
    |
    |-- parses operator inputs / config
    |-- initializes hardware (SLM, optional camera)
    |-- loads calibration state
    v
Pattern registry (name -> pattern builder)
    |
    |-- validates pattern key
    |-- dispatches to selected pattern module(s)
    v
Pattern modules (single responsibility)
    |
    |-- single_gaussian
    |-- double_gaussian
    |-- gaussian_lattice
    |-- laguerre_gaussian
    |-- composite (ordered composition of modules)
    v
Final phase map -> depth correction -> set on SLM
```

Use this architecture to keep hardware orchestration in the workflow script and keep optical math in isolated pattern modules.

---

## Pattern examples (4 built-in patterns + composite mode)

### 1) Single Gaussian-like spot
```bash
python user_workflows/run_slm_andor.py \
  --pattern single-gaussian \
  --single-kx 0.00 \
  --single-ky 0.01
```

### 2) Double Gaussian-like spots
```bash
python user_workflows/run_slm_andor.py \
  --pattern double-gaussian \
  --double-center-kx 0.00 \
  --double-center-ky 0.00 \
  --double-sep-kxy 0.03
```

### 3) Gaussian lattice
```bash
python user_workflows/run_slm_andor.py \
  --pattern gaussian-lattice \
  --lattice-nx 8 \
  --lattice-ny 6 \
  --lattice-pitch-x 0.012 \
  --lattice-pitch-y 0.012
```

### 4) Laguerre-Gaussian
```bash
python user_workflows/run_slm_andor.py \
  --pattern laguerre-gaussian \
  --lg-l 2 \
  --lg-p 1
```

### 5) Composite mode (schema/config example)
Composite mode lets you stack multiple pattern modules in order (for example: LG phase + blaze + sparse spots). Example YAML-style config:

```yaml
pattern:
  mode: composite
  components:
    - type: laguerre_gaussian
      l: 2
      p: 1
    - type: single_gaussian
      kx: 0.00
      ky: 0.01
  combine: phase_add_mod_2pi
hologram:
  method: WGS-Kim
  maxiter: 30
```

---

## Add a new pattern in 3 steps
1. **Create the pattern module**
   - Add a module with a pure function that accepts validated parameters and returns phase (or target spot vectors).
   - Keep hardware access out of the module.

2. **Register the module**
   - Add an entry in the pattern registry (`pattern_name -> builder callable + schema metadata`).
   - Define default parameters and required keys.

3. **Expose/configure and test**
   - Wire the new pattern key into workflow configuration parsing.
   - Add one CLI/config example and a quick smoke check in local validation.

---

## CLI migration: old flags -> new schema/config fields
Use this table when migrating scripts from direct CLI arguments to declarative config.

| Old CLI flag | New config field |
|---|---|
| `--pattern` | `pattern.type` (or `pattern.mode` for composite) |
| `--single-kx` | `pattern.params.single_gaussian.kx` |
| `--single-ky` | `pattern.params.single_gaussian.ky` |
| `--double-center-kx` | `pattern.params.double_gaussian.center_kx` |
| `--double-center-ky` | `pattern.params.double_gaussian.center_ky` |
| `--double-sep-kxy` | `pattern.params.double_gaussian.sep_kxy` |
| `--lattice-nx` | `pattern.params.gaussian_lattice.nx` |
| `--lattice-ny` | `pattern.params.gaussian_lattice.ny` |
| `--lattice-pitch-x` | `pattern.params.gaussian_lattice.pitch_x` |
| `--lattice-pitch-y` | `pattern.params.gaussian_lattice.pitch_y` |
| `--lattice-center-kx` | `pattern.params.gaussian_lattice.center_kx` |
| `--lattice-center-ky` | `pattern.params.gaussian_lattice.center_ky` |
| `--lg-l` | `pattern.params.laguerre_gaussian.l` |
| `--lg-p` | `pattern.params.laguerre_gaussian.p` |
| `--blaze-kx` | `pattern.params.common.blaze_kx` |
| `--blaze-ky` | `pattern.params.common.blaze_ky` |
| `--holo-method` | `hologram.method` |
| `--holo-maxiter` | `hologram.maxiter` |
| `--use-camera` | `camera.enabled` |
| `--camera-serial` | `camera.serial` |
| `--exposure-s` | `camera.exposure_s` |
| `--frames` | `acquisition.frames` |
| `--feedback` | `feedback.enabled` |
| `--feedback-iters` | `feedback.iterations` |
| `--calibration-root` | `calibration.root` |
| `--save-frames` | `outputs.save_frames_npy` |
| `--lut-file` | `phase_lut.file` |
| `--lut-key` | `phase_lut.key` |

---

## Andor image acquisition + feedback workflow
Use `run_slm_andor.py` to:

- keep the Andor CCD cooled to `-65C` while connected,
- set shutter control to `auto`,
- acquire full camera images of displayed SLM patterns,
- optionally run camera-driven experimental feedback optimization.

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

Useful optimization knobs for spot-based patterns:

- `--holo-method` (default `WGS-Kim`)
- `--holo-maxiter` (default `30`)

---

## Developer quickstart (contributors)
For contributors making workflow or pattern changes:

1. **Set up and inspect**
   ```bash
   python -m pip install -e .
   python user_workflows/run_slm_andor.py --help
   ```
2. **Validate calibration preconditions**
   - Ensure `user_workflows/calibrations/` contains required artifacts, or run calibration first.
3. **Smoke-test a non-camera path**
   - Start with a simple pattern config/CLI using `--pattern single-gaussian`.
4. **Then test camera/feedback path on hardware**
   - Enable `--use-camera`, then optionally `--feedback`.
5. **Document every new pattern**
   - Add one minimal usage example and one migration-row entry in this README.
