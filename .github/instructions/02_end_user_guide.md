# End-User Guide: `auto-slmsuite_v2`

This guide explains how to run **simulation** and **hardware experiments**, tune **patterns/optimizations**, customize **file naming**, and perform **calibration**.

---

## 1) What this repository provides

Primary operator entrypoint:
- `python user_workflows/cli.py workflow <command>`

Available commands:
- `calibrate` — produce/refresh calibration artifacts.
- `pattern` — display a selected SLM pattern.
- `acquire` — display pattern and capture camera frames.
- `feedback` — run camera-driven iterative optimization.
- `doctor` — preflight checks (SDK, camera discoverability, LUT, calibration files, output directory).

Compatibility scripts also exist:
- `user_workflows/run_calibration.py`
- `user_workflows/run_slm_andor.py`

---

## 2) Quickstart command map

### 2.1 Inspect command help
```bash
python user_workflows/cli.py workflow --help
python user_workflows/cli.py workflow pattern --help
python user_workflows/cli.py workflow acquire --help
python user_workflows/cli.py workflow feedback --help
python user_workflows/cli.py workflow calibrate --help
python user_workflows/cli.py workflow doctor --help
```

### 2.2 Preflight check (recommended before hardware)
```bash
python user_workflows/cli.py workflow doctor \
  --lut-file deep_1024.mat \
  --lut-key deep \
  --calibration-root user_workflows/calibrations \
  --output-dir user_workflows/output
```

Tip: use `--dry-run` to skip camera-discovery checks when testing paths only.

---

## 3) Calibration workflow

Calibration is required for experiment-grade acquisition/feedback pipelines.

### 3.1 Run calibration
Use a lab-specific FourierSLM factory (`module:function`) and LUT path:

```bash
python user_workflows/cli.py workflow calibrate \
  --factory my_lab.bootstrap:create_fourier_slm \
  --phase-lut /path/to/deep_1024.mat \
  --phase-lut-key deep \
  --calibration-root user_workflows/calibrations
```

### 3.2 Expected calibration artifacts
`user_workflows/calibrations/` should contain:
- `fourier-calibration.h5`
- `wavefront-superpixel-calibration.h5`
- `source-amplitude-corrected.npy`

Additional artifacts may be produced (e.g., LUT copy, source phase).

### 3.3 Validate calibration before experiments
```bash
python user_workflows/cli.py workflow doctor \
  --calibration-root user_workflows/calibrations \
  --lut-file /path/to/deep_1024.mat
```

---

## 4) Pattern generation and parameter tuning

## 4.1 Pattern families
Use `workflow pattern` and choose one pattern:
- `single-gaussian`
- `double-gaussian`
- `gaussian-lattice`
- `laguerre-gaussian`

### A) Single Gaussian-like spot
```bash
python user_workflows/cli.py workflow pattern \
  --pattern single-gaussian \
  --single-kx 0.00 \
  --single-ky 0.01 \
  --lut-file /path/to/deep_1024.mat
```

### B) Double Gaussian-like spots
```bash
python user_workflows/cli.py workflow pattern \
  --pattern double-gaussian \
  --double-center-kx 0.00 \
  --double-center-ky 0.00 \
  --double-sep-kxy 0.03 \
  --holo-method WGS-Kim \
  --holo-maxiter 50 \
  --lut-file /path/to/deep_1024.mat
```

### C) Gaussian lattice
```bash
python user_workflows/cli.py workflow pattern \
  --pattern gaussian-lattice \
  --lattice-nx 8 \
  --lattice-ny 6 \
  --lattice-pitch-x 0.012 \
  --lattice-pitch-y 0.012 \
  --lattice-center-kx 0.0 \
  --lattice-center-ky 0.0 \
  --holo-method WGS-Kim \
  --holo-maxiter 60 \
  --lut-file /path/to/deep_1024.mat
```

### D) Laguerre-Gaussian
```bash
python user_workflows/cli.py workflow pattern \
  --pattern laguerre-gaussian \
  --lg-l 2 \
  --lg-p 1 \
  --blaze-kx 0.0 \
  --blaze-ky 0.0045 \
  --lut-file /path/to/deep_1024.mat
```

### 4.2 Practical tuning notes
- `--holo-maxiter`: more iterations generally improve match quality but take longer.
- `--holo-method`: algorithm choice for spot hologram optimization.
- `--double-sep-kxy`: directly controls pair spacing for two-spot patterns.
- `--lattice-nx/ny`: number of spots; increasing count increases optimization demand.
- `--lattice-pitch-x/y`: controls lattice spacing.
- `--blaze-kx/ky`: steering offsets for phase ramps.

---

## 5) Acquisition workflow (pattern + camera frames)

### 5.1 Acquire frames while holding a pattern
```bash
python user_workflows/cli.py workflow acquire \
  --pattern double-gaussian \
  --double-sep-kxy 0.03 \
  --camera-serial "" \
  --exposure-s 0.03 \
  --frames 10 \
  --save-frames user_workflows/output/run1/frames.npy \
  --calibration-root user_workflows/calibrations \
  --lut-file /path/to/deep_1024.mat
```

Behavior:
- Loads required calibration files.
- Displays selected pattern on SLM.
- Acquires `N` full-frame images.
- Optionally saves frame stack to `.npy`.

### 5.2 Safe config validation without hardware touch
```bash
python user_workflows/cli.py workflow acquire \
  --dry-run \
  --pattern single-gaussian \
  --calibration-root user_workflows/calibrations \
  --lut-file /path/to/deep_1024.mat
```

---

## 6) Feedback workflow (camera-driven optimization)

```bash
python user_workflows/cli.py workflow feedback \
  --pattern gaussian-lattice \
  --lattice-nx 6 \
  --lattice-ny 6 \
  --feedback-iters 20 \
  --exposure-s 0.03 \
  --calibration-root user_workflows/calibrations \
  --lut-file /path/to/deep_1024.mat
```

Behavior:
- Builds and displays initial pattern.
- Captures camera image and normalizes it as target.
- Runs iterative experimental optimization.
- Applies optimized phase to SLM.

---

## 7) Simulation workflows

The repository includes a simulated SLM/camera stack factory:
- `user_workflows/simulation/sim_factory.py`
- entry function: `build_simulated_fourier_slm(...)`

### 7.1 Supported simulation scenarios
- `two-spot-imbalance`
- `n-spot-lattice-nonuniform`
- `high-noise-failure`

### 7.2 Why simulation is useful
- Validate logic and tuning workflow without hardware risk.
- Reproduce behavior deterministically using `seed`.
- Test robustness against synthetic aberration/noise variations.

---

## 8) File and run naming customization

Run naming utilities live in:
- `user_workflows/io/run_naming.py`

Available naming flags:
- `--run-name` (human readable ID)
- `--output-root` (base output folder)
- `--name-template` (template fields)
- `--overwrite` / `--resume` (collision policy)

Supported template fields:
- `{date}`
- `{run_name}`
- `{pattern}`
- `{camera}`
- `{iter}`

Default template:
- `{date}_{run_name}_{pattern}_{camera}_{iter}`

### 8.1 Example naming template
```bash
--name-template "{date}_{run_name}_{pattern}_{camera}_{iter}"
```

### 8.2 Collision behavior
- Default: auto-increment suffixes (`_001`, `_002`, ...).
- `--overwrite`: reuse existing path and allow replacement.
- `--resume`: reuse existing path for continuation.

---

## 9) Output files, plots, and manifests

Output manager utilities are in:
- `user_workflows/io/output_manager.py`

Managed artifact helpers include:
- `save_phase(...)`
- `save_frame(...)`
- `save_metrics(...)`
- `save_plot(...)`
- `save_manifest(...)`

### 9.1 Naming files/plots as you wish
You can set filenames explicitly when saving artifacts (e.g. custom plot names) through output manager APIs in workflow scripts.

Recommended convention:
- Include `{run_name}`, pattern type, and index in file names.
- Keep plot files in `plots/` and numeric arrays in separate folders.
- Save a final `manifest.json` for run traceability.

---

## 10) Presets and profile-driven operation

### 10.1 CLI presets
- `--preset <name>`
- `--preset-file user_workflows/presets.json`

Preset loader behavior:
- Loads matching profile dict.
- Applies keys to valid CLI arguments.
- Throws a clear error for invalid keys.

### 10.2 Config profiles
Example profiles in:
- `user_workflows/config/profiles/quick_test.yaml`
- `user_workflows/config/profiles/two_spot_balance.yaml`
- `user_workflows/config/profiles/lattice_feedback.yaml`

Use profiles to standardize lab procedures and reduce operator mistakes.

---

## 11) Troubleshooting quick reference

Run:
```bash
python user_workflows/cli.py workflow doctor
```

Checks performed include:
- SDK import availability
- camera discoverability
- LUT existence/shape validity
- calibration artifact presence/compatibility
- output directory writability

Typical fixes:
- Install missing SDK package(s).
- Confirm camera connection/power.
- Regenerate calibration artifacts.
- Correct LUT path/key.
- Fix output folder permissions.

---

## 12) Suggested operator checklist (before every session)

1. Run `workflow doctor`.
2. Confirm LUT and calibration root are correct.
3. Start with `--dry-run` for modified configs.
4. Run pattern-only command to validate geometry.
5. Run acquisition and inspect saved frame arrays.
6. Run feedback when acquisition baseline is stable.
7. Save manifest and notes for reproducibility.

