# Functionality Priorities for Best User Experience

This roadmap is optimized for your stated outcomes:
1) run simulations and experiments,
2) tune pattern and optimization parameters,
3) control file/plot naming,
4) perform calibration reliably.

## Priority 1 — Fast start paths (Simulation first, then hardware)
- Learn the unified CLI commands in `user_workflows/cli.py` (`workflow calibrate|pattern|acquire|feedback|doctor`).
- Start with no-risk checks using `--dry-run` to validate paths/config before touching hardware.
- For simulation development, use the simulated stack factory in `user_workflows/simulation/sim_factory.py`.

Why this is first: users need quick success and confidence before advanced setup.

## Priority 2 — Calibration lifecycle (hard prerequisite for experiments)
- Understand calibration artifacts required by acquisition/feedback:
  - `fourier-calibration.h5`
  - `wavefront-superpixel-calibration.h5`
  - `source-amplitude-corrected.npy`
- Run calibration via `workflow calibrate` (or compatibility wrapper `user_workflows/run_calibration.py`).
- Validate with `workflow doctor` before experiments.

Why this is second: experiment and feedback commands depend on calibration presence.

## Priority 3 — Pattern families and optimization knobs
- Pattern families to master first:
  - `single-gaussian`
  - `double-gaussian`
  - `gaussian-lattice`
  - `laguerre-gaussian`
- Common optimization knobs:
  - `--holo-method`
  - `--holo-maxiter`
- Common steering knobs:
  - `--blaze-kx`, `--blaze-ky`

Why this is third: this is the main day-to-day control surface.

## Priority 4 — Acquisition and feedback closed loop
- Use `workflow acquire` to collect frame stacks and save `.npy` outputs.
- Use `workflow feedback` to run experimental iterative optimization (`--feedback-iters`).

Why this is fourth: once generation is stable, users typically move to data capture and closed-loop refinement.

## Priority 5 — Output naming, manifests, and reproducibility
- Configure run folder naming:
  - `--run-name`, `--output-root`, `--name-template`
  - collision policies: default auto-increment, or `--overwrite` / `--resume`
- Persist run metadata and artifacts with output manager manifesting.

Why this matters: traceability and auditability are essential in labs.

## Priority 6 — Presets/config-driven workflows for teams
- Use presets from `user_workflows/presets.json` via `--preset`.
- Use structured config profiles from `user_workflows/config/profiles/*.yaml`.

Why this is sixth: move from ad-hoc operation to repeatable, team-scale operation.

## Priority 7 — Diagnostics and troubleshooting playbook
- Treat `workflow doctor` as preflight before experimental sessions.
- Keep a quick “error -> fix” table from doctor outputs for operators.

Why this is last in learning order (but high operational importance): it is easiest to absorb after users know the normal flow.
