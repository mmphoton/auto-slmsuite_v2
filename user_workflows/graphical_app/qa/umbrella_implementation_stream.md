# Graphical App Umbrella Implementation Stream (Single Master Task)

## Milestone 0: Program Bootstrap (Gate Owner: Tech Lead)

**Umbrella branch:** `graphical-app-umbrella-rebuild`

This document is the single master task definition for the complete graphical app rebuild. Work must execute in strict milestone order and remain inside `user_workflows/graphical_app/` while preserving layered boundaries:

- `ui/`
- `app/`
- `devices/`
- `plotting/`
- `persistence/`

### Sequential execution policy (non-negotiable)

1. Exactly one milestone may be `IN PROGRESS` at a time.
2. A milestone is binary: `PASS` or `FAIL` (no partial completion state).
3. If any acceptance check fails, fix in place before any code for the next milestone.
4. Every milestone must produce:
   - implementation code,
   - tests,
   - signed checklist result in `qa/gui_capability_checklist.md`.

### Hard gate policy

Advancement requires all of the following:

- all acceptance checks for the current milestone marked `PASS`,
- required automated tests passing,
- checklist signed by implementer + reviewer,
- no unresolved blocking defects in milestone scope.

---


## Single Condensed Readiness Task (Perform At Once)

Execute one unified "GUI readiness closure" task that delivers all milestone acceptance outputs in a single coordinated implementation wave, with one final hard gate at the end.

**Task scope (must all be complete together):**
- freeze/validate app contracts and UI->controller boundaries,
- deliver multi-page shell + lifecycle + logging/error/progress UX,
- complete pattern/blaze/camera/plot/optimization/ratio/calibration/output flows,
- expand QA matrices and smoke coverage to executable parity checks,
- sign checklist only after all automated checks pass.

**Single-task acceptance gate:**
- `milestone_gate_report(smoke_suite_passed=True)` returns all `M1..M13` true,
- `release_freeze_ready(smoke_suite_passed=True)["release_freeze_ready"]` is true,
- full smoke suite passes,
- checklist and advancement log updated with evidence links/commands.

---

## Mandatory Milestones and Acceptance Gates

### Milestone 1 — Core contracts freeze

**Scope:** `app/state.py`, `app/interfaces.py`, `app/controller.py`

**Acceptance criteria:**
- Stable contracts exist for state, logs, progress, results, errors, mode, output/session metadata.
- UI calls controller only; UI->adapter direct calls are prohibited by design and tests.
- Contract tests freeze schema and behavior expectations.

**Gate checks:**
- targeted unit tests for contracts pass,
- smoke check validates UI/controller boundary,
- checklist signed PASS.

### Milestone 2 — Multi-page navigation shell

**Scope:** `ui/main_window.py`

**Acceptance criteria:**
- Single-page layout replaced with multi-page shell (Notebook/stacked pages + top navigation).
- Pages present:
  1. Device & Mode
  2. SLM Patterns & Blaze
  3. Camera & Telemetry
  4. Plot Workspace
  5. Optimization (WGS + Ratio)
  6. Calibration
  7. Session/Output/Recipes
  8. Logs/Diagnostics
- Relevant page internals support dockable/resizable subpanels.

**Gate checks:**
- UI smoke tests validate page presence and navigation,
- layout persistence tests include page/split state,
- checklist signed PASS.

### Milestone 3 — Device lifecycle completion

**Scope:** `devices/manager.py`

**Acceptance criteria:**
- discover/connect/reconnect/release implemented robustly.
- SLM and camera release independently.
- safe-stop transition checks enforced.
- explicit success/failure surfaced to UI.

**Gate checks:**
- simulation + hardware-mode lifecycle tests pass,
- reconnect and partial-release regression tests pass,
- checklist signed PASS.

### Milestone 4 — Logging, errors, progress UX

**Scope:** `app/`, `ui/Logs/Diagnostics page`

**Acceptance criteria:**
- structured log model implemented.
- global log page exposes filtered command history.
- errors provide actionable notifications.
- progress indicators for optimization/calibration/sequence.

**Gate checks:**
- log schema tests pass,
- error/progress UX smoke tests pass,
- checklist signed PASS.

### Milestone 5 — Pattern parity + schema-driven forms

**Scope:** `app/patterns.py`, `ui/pattern_form.py`

**Acceptance criteria:**
- freeform JSON pattern input removed.
- forms auto-generated from schema.
- 100% backend pattern parameters exposed in GUI controls.
- per-pattern presets + validation implemented.

**Gate checks:**
- pattern parity matrix row complete,
- form generation + validation tests pass,
- checklist signed PASS.

### Milestone 6 — Blaze grating global enforcement

**Scope:** `ui/SLM Patterns & Blaze`, `app/controller.py`

**Acceptance criteria:**
- blaze controls available on SLM page.
- blaze composition enforced across simulate/apply/queue controller paths.

**Gate checks:**
- cross-path blaze tests pass,
- UI control binding tests pass,
- checklist signed PASS.

### Milestone 7 — Camera controls page completion

**Scope:** `ui/Camera & Telemetry`, `devices/camera_settings.py`

**Acceptance criteria:**
- typed controls for exposure, gain, ROI, binning, trigger/shutter mode, FPS, acquisition mode.
- apply/read/revert workflow with validation.
- live temperature monitor + warning thresholds.

**Gate checks:**
- settings round-trip tests pass,
- telemetry warning tests pass,
- checklist signed PASS.

### Milestone 8 — Plot workspace completion

**Scope:** `plotting/backend.py`, `ui/Plot Workspace`

**Acceptance criteria:**
- interactive plot operations: zoom/pan/reset, axes, colormap, linear/log.
- pop-out editable windows with independent export (image + data + settings metadata).
- simulated phase and simulated intensity always visible.

**Gate checks:**
- plotting interactivity tests pass,
- export metadata tests pass,
- checklist signed PASS.

### Milestone 9 — WGS optimization completion

**Scope:** `optimization/runner.py`, optimization UI

**Acceptance criteria:**
- start/pause/resume/stop controls implemented.
- max-iteration cap + progress reporting.
- convergence + before/after phase/intensity plots for simulation and hardware modes.

**Gate checks:**
- optimization control tests pass,
- convergence output tests pass,
- checklist signed PASS.

### Milestone 10 — Gaussian ratio/lattice optimization completion

**Scope:** optimization target/config UI + objective logic

**Acceptance criteria:**
- N-beam and lattice ratio target configuration UI implemented.
- objective logic supports simulation and camera-feedback variants.
- export of ratio optimization results + metadata.

**Gate checks:**
- ratio objective tests pass,
- export tests pass,
- checklist signed PASS.

### Milestone 11 — Calibration page completion

**Scope:** calibration UI + calibration tooling

**Acceptance criteria:**
- dedicated calibration page with load/validate/apply.
- compatibility checks by mode/device.
- before/after metrics and explicit pass/fail status.

**Gate checks:**
- compatibility tests pass,
- metrics/pass-fail UI tests pass,
- checklist signed PASS.

### Milestone 12 — Reproducibility and outputs

**Scope:** `persistence/`, output/session UI

**Acceptance criteria:**
- output folder selector, naming template engine, collision policy, token preview.
- complete run/session snapshots include devices, parameters, blaze, calibration, optimizer, software version, telemetry.

**Gate checks:**
- snapshot completeness tests pass,
- naming collision behavior tests pass,
- checklist signed PASS.

### Milestone 13 — Hard QA gate (release blocker)

**Scope:** `qa/matrices.py`, `tests/test_graphical_app_smoke.py`

**Acceptance criteria:**
- backend-to-GUI and simulation/hardware parity matrices fully expanded.
- smoke tests cover critical flows: page navigation, layout persistence, plotting edits, optimization, calibration, error handling.
- release only if all matrix rows covered and all smoke tests pass.

**Gate checks:**
- full matrix shows all PASS,
- full smoke suite PASS,
- final checklist signed PASS.

---

## Definition of Done for Umbrella Task

The umbrella stream is complete only when milestones 1–13 are each marked PASS with signatures in `qa/gui_capability_checklist.md`, and no gate violations remain.
